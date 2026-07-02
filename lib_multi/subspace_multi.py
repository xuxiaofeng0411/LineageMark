import gc
import torch
import numpy as np
from .data_multi import get_loaders
from .utils_multi import get_blocks, find_layers, qweight2weight

# ----------------------------------
# Matrix-level DSSA statistics
# ----------------------------------
def _as_tensor_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def _flatten_last_dim(tensor):
    if tensor.dim() < 1:
        raise ValueError(f"Expected tensor with at least one dim, got shape={tuple(tensor.shape)}")
    return tensor.reshape(-1, tensor.shape[-1])

def _reshape_by_sample(tensor, batch_size):
    if tensor.dim() == 0:
        raise ValueError(f"Expected tensor with sample/token dims, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] == batch_size:
        return tensor.reshape(batch_size, -1, tensor.shape[-1])
    if tensor.shape[0] % batch_size == 0:
        return tensor.reshape(batch_size, -1, tensor.shape[-1])
    raise ValueError(
        f"Cannot split tensor shape={tuple(tensor.shape)} into batch_size={batch_size} samples"
    )

def _iter_calibration_batches(dataloader, nsamples, batch_size):
    batch_size = max(1, int(batch_size))
    pending = []
    used = 0
    for inp, _ in dataloader:
        if used >= nsamples:
            break
        pending.append(inp)
        used += 1
        if len(pending) == batch_size:
            yield torch.cat(pending, dim=0), len(pending)
            pending = []
    if pending:
        yield torch.cat(pending, dim=0), len(pending)

def _get_weight_matrix(layer, is_8bit=False):
    if is_8bit:
        return qweight2weight(layer).data.t()
    return layer.weight.data

def _init_matrix_stats(named_layers, is_8bit=False, stats_device="cpu"):
    stats = {}
    for name, layer in named_layers.items():
        weight = _get_weight_matrix(layer, is_8bit=is_8bit)
        d_out, d_in = weight.shape
        stats[name] = {
            "F": torch.zeros((d_out, d_out), device=stats_device, dtype=torch.float32),
            "C_A": torch.zeros((d_out, d_out), device=stats_device, dtype=torch.float32),
            "samples": 0,
            "tokens": 0,
            "weight_shape": (d_out, d_in),
        }
    return stats

def _init_block_group_stats(block_layer_specs, is_8bit=False, stats_device="cpu"):
    return {
        block_idx: _init_matrix_stats(named_layers, is_8bit=is_8bit, stats_device=stats_device)
        for block_idx, named_layers in block_layer_specs.items()
    }

def _accumulate_compression_variance_batch(C_A, activation_by_sample, device):
    """Accumulate C_A for a batch while preserving per-sample compression draws."""
    r = activation_by_sample.detach().to(device=device, dtype=torch.float32)
    r = r.reshape(r.shape[0], -1, r.shape[-1])
    batch_size, tokens_per_sample, d = r.shape
    total_tokens = batch_size * tokens_per_sample
    # Random projection: draw an independent projection for each sample, matching
    # the previous per-sample estimator while executing the batch as one kernel group.
    proj_dim = max(1, d // 2)
    P = torch.randn(batch_size, d, proj_dim, device=r.device, dtype=r.dtype)
    P = P / torch.norm(P, dim=1, keepdim=True).clamp_min(1e-12)
    r_projected = torch.bmm(torch.bmm(r, P), P.transpose(1, 2))
    diff = r - r_projected
    diff_2d = diff.reshape(-1, d)
    C_A += diff_2d.T @ diff_2d
    del P, r_projected, diff, diff_2d
    noise_std = 0.01 * torch.norm(r, dim=-1, keepdim=True)
    diff = torch.randn_like(r) * noise_std
    diff_2d = diff.reshape(-1, d)
    C_A += diff_2d.T @ diff_2d
    del noise_std, diff, diff_2d
    keep_mask = (torch.rand_like(r) > 0.1).to(dtype=r.dtype)
    diff = r * (1 - keep_mask)
    diff_2d = diff.reshape(-1, d)
    C_A += diff_2d.T @ diff_2d
    del keep_mask, diff, diff_2d, r
    return total_tokens, 3

def _compute_block_group_matrix_statistics(
    model,
    block_layer_specs,
    dataloader,
    device,
    nsamples,
    is_8bit=False,
    calib_batch_size=1,
):
    """
    Compute Fisher and C_A for all matrices in a group of blocks.

    A single forward/backward pass over one calibration batch now services all
    hooked matrices in the block group. Fisher and C_A use batched matrix
    operations equivalent to the previous per-sample accumulations.
    The requested calibration batch size is fixed; if it does not fit in GPU
    memory, lower dssa_calib_batch_size in the launch script.
    """
    stats = _init_block_group_stats(block_layer_specs, is_8bit=is_8bit, stats_device=device)
    captured = {}
    handles = []
    cache_setting = model.config.use_cache
    model.config.use_cache = False
    def add_hook(key):
        def hook_fn(_module, _input, output):
            captured[key] = _as_tensor_output(output)
        return hook_fn
    for block_idx, named_layers in block_layer_specs.items():
        for name, layer in named_layers.items():
            handles.append(layer.register_forward_hook(add_hook((block_idx, name))))
    requested_batch_size = max(1, int(calib_batch_size))
    n_processed = 0
    n_ops = 3
    def process_batch(inp_cpu, batch_size):
        captured.clear()
        inp_dev = None
        outputs = None
        grads = None
        activations = []
        try:
            inp_dev = inp_cpu.to(device)
            outputs = model(input_ids=inp_dev, labels=inp_dev)
            keys = []
            for block_idx, named_layers in block_layer_specs.items():
                for name in named_layers.keys():
                    key = (block_idx, name)
                    if key in captured:
                        keys.append(key)
                        activations.append(captured[key])
            # Hugging Face causal-LM loss is averaged over the batch. Multiplying
            # by batch_size restores the same per-sample gradient scale as the
            # original batch_size=1 estimator.
            grads = torch.autograd.grad(
                outputs.loss * batch_size,
                activations,
                retain_graph=False,
                allow_unused=True,
            )
            with torch.no_grad():
                for (block_idx, name), activation, grad in zip(keys, activations, grads):
                    if grad is None:
                        print(f"  Warning: no gradient captured for block {block_idx} [{name}], skipping batch.")
                        continue
                    stat = stats[block_idx][name]
                    d_out = stat["weight_shape"][0]
                    activation_by_sample = _reshape_by_sample(activation, batch_size)
                    grad_by_sample = _reshape_by_sample(grad, batch_size)
                    if activation_by_sample.shape[-1] != d_out or grad_by_sample.shape[-1] != d_out:
                        raise RuntimeError(
                            f"Block {block_idx} [{name}] output dim mismatch: "
                            f"weight d_out={d_out}, activation={activation_by_sample.shape[-1]}, "
                            f"grad={grad_by_sample.shape[-1]}"
                        )
                    G = grad_by_sample.detach().to(device=device, dtype=torch.float32).mean(dim=1)
                    stat["F"] += G.T @ G
                    stat["samples"] += batch_size
                    token_count, _ = _accumulate_compression_variance_batch(
                        stat["C_A"], activation_by_sample, device
                    )
                    stat["tokens"] += token_count
                    del G, activation_by_sample, grad_by_sample
        finally:
            del activations, grads, outputs, inp_dev
            captured.clear()
    try:
        for inp, batch_size in _iter_calibration_batches(dataloader, nsamples, requested_batch_size):
            process_batch(inp, batch_size)
            n_processed += batch_size
    finally:
        for handle in handles:
            handle.remove()
        model.config.use_cache = cache_setting
    if n_processed == 0:
        raise RuntimeError("No calibration samples were processed for matrix-level DSSA.")
    for block_idx, block_stats in stats.items():
        for name, stat in block_stats.items():
            if stat["samples"] == 0 or stat["tokens"] == 0:
                raise RuntimeError(f"No usable matrix DSSA statistics for block {block_idx} [{name}].")
            stat["F"] /= stat["samples"]
            stat["C_A"] /= (stat["tokens"] * n_ops)
    return stats

# ------------------------------------
# Stable subspace solvers
# ------------------------------------
_VALID_SUBSPACE_METHODS = ("full", "fisher_only", "ca_only")
def _symmetrize(matrix):
    return (matrix + matrix.T) / 2

def _normalize_columns(matrix):
    return matrix / torch.norm(matrix, dim=0, keepdim=True).clamp_min(1e-12)

def _solve_gevp(F, C_A, epsilon=1e-6, max_attempts=5):
    """
    Solve: F @ u = lambda * (C_A + epsilon*I) @ u.
    """
    d = F.shape[0]
    F = _symmetrize(F)
    C_A = _symmetrize(C_A)
    current_eps = epsilon
    for attempt in range(max_attempts):
        C_reg = C_A + current_eps * torch.eye(d, device=C_A.device, dtype=C_A.dtype)
        try:
            L = torch.linalg.cholesky(C_reg)
        except RuntimeError as exc:
            print(f"  Cholesky failed at eps={current_eps:.2e}: {exc}")
            current_eps *= 10
            continue
        X = torch.linalg.solve_triangular(L, F, upper=False)
        F_prime = torch.linalg.solve_triangular(L, X.T, upper=False)
        F_prime = _symmetrize(F_prime)
        eigvals, eigvecs_T = torch.linalg.eigh(F_prime)
        eigvals = torch.flip(eigvals, [0])
        eigvecs_T = torch.flip(eigvecs_T, [1])
        eigvecs = torch.linalg.solve_triangular(L.T, eigvecs_T, upper=True)
        if torch.isnan(eigvals).any() or torch.isnan(eigvecs).any():
            print(f"  NaN in eigenvalues/eigenvectors at eps={current_eps:.2e}, retrying...")
            current_eps *= 10
            continue
        print(f"  GEVP solved with eps={current_eps:.2e} (attempt {attempt + 1}).")
        return eigvals, eigvecs
    raise RuntimeError(
        f"GEVP failed after {max_attempts} attempts. "
        f"Check Fisher and C_A matrices for numerical issues."
    )

def _solve_fisher_evp(F):
    """Solve the Fisher-only ordinary eigenvalue problem and sort high to low."""
    F = _symmetrize(F)
    eigvals, eigvecs = torch.linalg.eigh(F)
    eigvals = torch.flip(eigvals, [0])
    eigvecs = torch.flip(eigvecs, [1])
    if torch.isnan(eigvals).any() or torch.isnan(eigvecs).any():
        raise RuntimeError("NaN in Fisher-only eigenvalues/eigenvectors.")
    print("  Fisher-only EVP solved; selecting top eigen-directions.")
    return eigvals, eigvecs

def _solve_ca_evp(C_A, epsilon=1e-6):
    """Solve the C_A-only ordinary eigenvalue problem and sort low to high."""
    d = C_A.shape[0]
    C_A = _symmetrize(C_A)
    C_reg = C_A + epsilon * torch.eye(d, device=C_A.device, dtype=C_A.dtype)
    eigvals, eigvecs = torch.linalg.eigh(C_reg)
    if torch.isnan(eigvals).any() or torch.isnan(eigvecs).any():
        raise RuntimeError("NaN in C_A-only eigenvalues/eigenvectors.")
    print("  C_A-only EVP solved; selecting lowest-variance eigen-directions.")
    return eigvals, eigvecs

def _topk_subspace(eigenvalues, eigenvectors, k=64):
    selected_indices = torch.arange(
        min(k, eigenvalues.shape[0]), device=eigenvalues.device
    )
    U = eigenvectors[:, selected_indices]
    U = _normalize_columns(U)
    return U, selected_indices

def _select_matrix_subspace(stat, subspace_method="full", k=64,
                            tau_lower=0.1, tau_upper=0.9, epsilon=1e-6):
    if subspace_method == "full":
        eigenvalues, eigenvectors = _solve_gevp(
            stat["F"], stat["C_A"], epsilon=epsilon
        )
        U, selected_indices = _spectral_truncation(
            eigenvalues, eigenvectors, k=k,
            tau_lower=tau_lower, tau_upper=tau_upper
        )
    elif subspace_method == "fisher_only":
        eigenvalues, eigenvectors = _solve_fisher_evp(stat["F"])
        U, selected_indices = _topk_subspace(eigenvalues, eigenvectors, k=k)
    elif subspace_method == "ca_only":
        eigenvalues, eigenvectors = _solve_ca_evp(stat["C_A"], epsilon=epsilon)
        U, selected_indices = _topk_subspace(eigenvalues, eigenvectors, k=k)
    else:
        raise ValueError(
            f"Unsupported subspace_method={subspace_method}. "
            f"Choose from {_VALID_SUBSPACE_METHODS}."
        )
    return eigenvalues, eigenvectors, U, selected_indices

# ----------------------------------
# Spectral truncation
# ----------------------------------
def _spectral_truncation(eigenvalues, eigenvectors, k=64, tau_lower=0.1, tau_upper=0.9):
    """
    Select the carrier subspace U from the GEVP solution.
    """
    lambda_1 = eigenvalues[0].item()
    lower = tau_lower * lambda_1
    upper = tau_upper * lambda_1
    eligible_mask = (eigenvalues >= lower) & (eigenvalues <= upper)
    eligible_indices = torch.where(eligible_mask)[0]
    if len(eligible_indices) == 0:
        print(f"  Warning: no eigenvalues in [{lower:.6f}, {upper:.6f}]. Falling back to top-{k} overall.")
        selected_indices = torch.arange(min(k, eigenvalues.shape[0]), device=eigenvalues.device)
    elif len(eligible_indices) < k:
        print(f"  Info: only {len(eligible_indices)} eigenvalues in range (requested {k}). Taking all eligible.")
        selected_indices = eligible_indices
    else:
        selected_indices = eligible_indices[:k]
    U = eigenvectors[:, selected_indices]
    U = _normalize_columns(U)
    return U, selected_indices

# ------------------------------------
# Matrix subspace to weight positions
# ------------------------------------
def _build_matrix_mask(layer, U, select_ratio=0.75, is_8bit=False):
    weight = _get_weight_matrix(layer, is_8bit=is_8bit)
    d_out, d_in = weight.shape
    if not (0 < select_ratio <= 1):
        raise ValueError(f"select_ratio must be in (0, 1], got {select_ratio}.")
    row_importance = torch.norm(U, dim=1)
    if row_importance.numel() != d_out:
        raise RuntimeError(
            f"Matrix subspace dim {row_importance.numel()} does not match weight rows {d_out}."
        )
    r_max = row_importance.max()
    if r_max > 0:
        row_importance = row_importance / r_max
    W_abs = torch.abs(weight).cpu().to(torch.float32)
    row_w = row_importance.cpu().to(torch.float32).reshape((-1, 1))
    W_metric = W_abs * row_w
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    n_select = max(1, int(W_metric.numel() * select_ratio))
    flat_metric = W_metric.reshape(-1)
    selected_flat = torch.sort(flat_metric, stable=True)[1][-n_select:]
    W_mask.reshape(-1).scatter_(0, selected_flat, True)
    per_row_selected = W_mask.sum(dim=1).cpu()
    return {
        "W_mask": W_mask,
        "row_importance": row_importance.cpu(),
        "per_row_selected": per_row_selected,
        "weight_shape": (d_out, d_in),
        "subspace_dim": U.shape[1],
        "subspace_scope": "matrix_output",
        "mask_selection": "global_metric_topk",
        "selected_count": int(n_select),
        "selected_ratio": float(select_ratio),
    }

def _resolve_block_chunk(dssa_block_chunk, num_blocks):
    if dssa_block_chunk is None or int(dssa_block_chunk) <= 0:
        return num_blocks
    return min(max(1, int(dssa_block_chunk)), num_blocks)

# ---------------------------
# Main entry point
# ---------------------------
def select_subspace(model, tokenizer, nsamples=512, seqlen=2048,
                    k=64, tau_lower=0.1, tau_upper=0.9, epsilon=1e-6,
                    select_ratio=0.75, is_8bit=False,
                    device=torch.device("cuda"), dataset_name="wikitext2", seed=42,
                    dssa_block_chunk=0, calib_batch_size=1,
                    subspace_method="full"):
    """
    Compute a separate DSSA carrier subspace for each linear weight matrix.
    The math is unchanged from matrix-level DSSA. dssa_block_chunk controls
    how many transformer blocks are hooked per pass, and calib_batch_size
    controls how many calibration samples are processed per forward/backward.
    """
    if subspace_method not in _VALID_SUBSPACE_METHODS:
        raise ValueError(
            f"Unsupported subspace_method={subspace_method}. "
            f"Choose from {_VALID_SUBSPACE_METHODS}."
        )
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    blocks = get_blocks(model)
    chunk_size = _resolve_block_chunk(dssa_block_chunk, len(blocks))
    print(
        "Matrix-level DSSA enabled. "
        f"block_chunk={chunk_size}/{len(blocks)}, "
        f"calib_batch_size={max(1, int(calib_batch_size))}; "
        f"subspace_method={subspace_method}; "
        "each matrix gets its own U."
    )
    model.eval()
    print(f"Loading calibration data ({dataset_name})...")
    dataloader, _ = get_loaders(
        dataset_name, nsamples=nsamples, seed=seed,
        seqlen=seqlen, tokenizer=tokenizer
    )
    all_layer_masks = {}
    matrix_subspaces = {}
    matrix_summaries = {}
    total_matrices = 0
    for chunk_start in range(0, len(blocks), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(blocks))
        block_layer_specs = {}
        for block_idx in range(chunk_start, chunk_end):
            named_layers = find_layers(blocks[block_idx])
            if named_layers:
                block_layer_specs[block_idx] = named_layers
            else:
                all_layer_masks[block_idx] = {}
                matrix_subspaces[block_idx] = {}
        if not block_layer_specs:
            continue
        chunk_matrix_count = sum(len(named_layers) for named_layers in block_layer_specs.values())
        print(
            f"Processing DSSA block chunk {chunk_start}-{chunk_end - 1} "
            f"({chunk_matrix_count} linear matrices)."
        )
        chunk_stats = _compute_block_group_matrix_statistics(
            model=model,
            block_layer_specs=block_layer_specs,
            dataloader=dataloader,
            device=device,
            nsamples=nsamples,
            is_8bit=is_8bit,
            calib_batch_size=calib_batch_size,
        )
        for block_idx, named_layers in block_layer_specs.items():
            block_masks = {}
            block_subspaces = {}
            for name, layer in named_layers.items():
                stat = chunk_stats[block_idx][name]
                d_out, d_in = stat["weight_shape"]
                print("\n" + "-" * 60)
                print(f"Solving matrix subspace for block {block_idx} [{name}] shape=({d_out},{d_in})")
                print("-" * 60)
                eigenvalues, eigenvectors, U, selected_indices = _select_matrix_subspace(
                    stat,
                    subspace_method=subspace_method,
                    k=k,
                    tau_lower=tau_lower,
                    tau_upper=tau_upper,
                    epsilon=epsilon,
                )
                assert U.shape[0] == d_out, f"U rows {U.shape[0]} != weight rows {d_out}"
                assert not torch.isnan(U).any(), f"NaN in U for block {block_idx} [{name}]"
                assert not torch.isinf(U).any(), f"Inf in U for block {block_idx} [{name}]"
                mask_info = _build_matrix_mask(
                    layer, U, select_ratio=select_ratio, is_8bit=is_8bit
                )
                block_masks[name] = mask_info
                block_subspaces[name] = {
                    "U": U.cpu(),
                    "eigenvalues": eigenvalues.cpu(),
                    "selected_indices": selected_indices.cpu(),
                    "subspace_method": subspace_method,
                    "weight_shape": (d_out, d_in),
                    "samples": stat["samples"],
                    "tokens": stat["tokens"],
                }
                n_masked = mask_info["W_mask"].sum().item()
                total = mask_info["W_mask"].numel()
                print(
                    f"  selected U dim={U.shape[1]} rows={d_out}; "
                    f"W_mask={n_masked}/{total} ({n_masked / total * 100:.1f}%) "
                    f"via global_metric_topk"
                )
                matrix_summaries[f"{block_idx}:{name}"] = {
                    "weight_shape": (d_out, d_in),
                    "subspace_dim": U.shape[1],
                    "subspace_method": subspace_method,
                    "samples": stat["samples"],
                    "tokens": stat["tokens"],
                    "lambda_max": float(eigenvalues[0].item()),
                }
                total_matrices += 1
                del eigenvectors
            all_layer_masks[block_idx] = block_masks
            matrix_subspaces[block_idx] = block_subspaces
        del chunk_stats
        gc.collect()
        torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print(f"Matrix-level DSSA complete: {total_matrices} matrices processed.")
    print("=" * 60)
    return {
        "subspace_scope": "matrix_output",
        "all_layer_masks": all_layer_masks,
        "matrix_subspaces": matrix_subspaces,
        "matrix_summaries": matrix_summaries,
        "mapping_config": {
            "k": k,
            "tau_lower": tau_lower,
            "tau_upper": tau_upper,
            "epsilon": epsilon,
            "select_ratio": select_ratio,
            "dataset": dataset_name,
            "seed": seed,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "dssa_block_chunk": dssa_block_chunk,
            "resolved_block_chunk": chunk_size,
            "subspace_method": subspace_method,
            "calib_batch_size": max(1, int(calib_batch_size)),
        },
    }
