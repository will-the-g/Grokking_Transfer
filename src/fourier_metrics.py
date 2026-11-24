"""Offline embedding Fourier metrics (spec section 3d).

Reads every saved checkpoint under `runs_nseed/` and writes one CSV row per
(seed, step) to `runs_nseed/fourier_metrics.csv`. No training happens here and
the audit split is not touched.

    python nseed_backbone\\fourier_metrics.py

Produces: the Fourier metrics of the checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import torch  # noqa: E402  (common binds the MSVC runtime first)

FIELDS = ['seed', 'name', 'step', 'kind', 'top5_share', 'freq_participation',
          'nonconst_share', 'peak_k', 'global_norm']


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else common.PARENT / path


def classify(name):
    if name.startswith('step_'):
        return 'source'
    if name.startswith('transfer_') or name.startswith('freeze_'):
        return 'transfer'
    return 'stage'


def collect(runs_dir):
    rows = []
    for ckpt_path in sorted(runs_dir.glob('**/ckpts/*.pt')):
        try:
            payload = common.load_checkpoint(ckpt_path)
        except Exception as exc:  # noqa: BLE001 - a broken checkpoint must not kill the sweep
            print(f'  [fourier] skipped {ckpt_path}: {exc}')
            continue
        state = payload.get('model')
        if state is None:
            continue
        embedding = state['token_embed.weight'][:common.P]
        metrics = common.fourier_metrics(embedding)
        seed = payload.get('seed')
        if seed is None:
            seed = _seed_from_path(ckpt_path)
        rows.append(dict(
            seed=seed, name=ckpt_path.stem, step=int(payload.get('step', -1)),
            kind=classify(ckpt_path.stem),
            top5_share=metrics['top5_share'],
            freq_participation=metrics['freq_participation'],
            nonconst_share=metrics['nonconst_share'],
            peak_k=metrics['peak_k'],
            global_norm=payload.get('global_norm', '')))
    return rows


def _seed_from_path(ckpt_path):
    for parent in ckpt_path.parents:
        if parent.name.startswith('seed_'):
            try:
                return int(parent.name.split('_')[-1])
            except ValueError:
                continue
    return ''


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import io
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    common.atomic_bytes(path, out.getvalue().encode())


def frequency_energies(embedding):
    """Per-frequency energy for k = 1..(p-1)//2, as in `embedding_spectrum`."""
    energy, _ = common.embedding_spectrum(embedding)
    return energy


def frequency_basis(p, k, device='cpu'):
    """Orthonormal basis of the 2-dim token subspace owned by frequency k.

    The subspace is spanned by cos(2*pi*k*a/p) and sin(2*pi*k*a/p) over the
    token index a; the scaling makes them unit vectors, and for odd p they are
    mutually orthogonal and orthogonal to the constant direction.
    """
    a = torch.arange(p, dtype=torch.float64, device=device)
    scale = math.sqrt(2.0 / p)
    cos = scale * torch.cos(2 * math.pi * k * a / p)
    sin = scale * torch.sin(2 * math.pi * k * a / p)
    return cos, sin


def project_out(embedding, ks):
    """Remove the frequencies in `ks` from the (p, d) embedding."""
    p, _ = embedding.shape
    work = embedding.to(torch.float64)
    out = work.clone()
    for k in ks:
        cos, sin = frequency_basis(p, k, work.device)
        out = out - torch.outer(cos, cos @ out) - torch.outer(sin, sin @ out)
    return out


def column_mean(embedding):
    """The DC component of the embedding: the mean over the token axis."""
    return embedding.to(torch.float64).mean(dim=0, keepdim=True)


def top_frequencies(embedding, coverage=0.90):
    """The smallest set of top-energy frequencies covering `coverage` of the
    non-DC energy, and the fraction of the non-DC energy they hold."""
    energy = frequency_energies(embedding)
    total = float(energy.sum())
    order = torch.argsort(energy, descending=True).tolist()
    chosen = []
    running = 0.0
    for index in order:
        chosen.append(index + 1)
        running += float(energy[index])
        if total > 0 and running >= coverage * total:
            break
    return chosen, (running / total if total > 0 else float('nan')), total


def non_dc_energy(embedding):
    """Total non-DC energy of the embedding (the same normalization as energy_k)."""
    energy = frequency_energies(embedding)
    return float(energy.sum())


def random_orthogonal_basis(p, dims, seed, device='cpu'):
    """A random orthonormal basis of `dims` vectors orthogonal to the constant one."""
    generator = torch.Generator().manual_seed(int(seed))
    gaussian = torch.randn(p, dims, generator=generator, dtype=torch.float64)
    constant = torch.full((p, 1), 1.0 / math.sqrt(p), dtype=torch.float64)
    gaussian = gaussian - constant @ (constant.T @ gaussian)
    basis, _ = torch.linalg.qr(gaussian)
    return basis.to(device)


def knockout(embedding, mode, top_k=None, seed=0):
    """Frequency knockouts on a numeric embedding (p, d).

    Returns (new_embedding, info) with info holding K, the fraction of non-DC
    energy removed and the embedding norm before and after.
    """
    work = embedding.to(torch.float64)
    before_norm = float(torch.linalg.norm(work))
    before_energy = non_dc_energy(work)
    dc = column_mean(work)
    dc_removed = work - dc

    if mode == 'key':
        ks = top_k if top_k is not None else top_frequencies(work)[0]
        projected = project_out(dc_removed, ks)
        result = projected + dc
        removed_fraction = ((before_energy - non_dc_energy(result)) / before_energy
                            if before_energy > 0 else float('nan'))
    elif mode == 'nonkey':
        ks = top_k if top_k is not None else top_frequencies(work)[0]
        all_ks = list(range(1, (work.shape[0] - 1) // 2 + 1))
        others = [k for k in all_ks if k not in set(ks)]
        result = project_out(dc_removed, others) + dc
        removed_fraction = ((before_energy - non_dc_energy(result)) / before_energy
                            if before_energy > 0 else float('nan'))
    elif mode == 'rand_dim':
        if top_k is None:
            top_k = top_frequencies(work)[0]
        basis = random_orthogonal_basis(work.shape[0], 2 * len(top_k), seed, work.device)
        projection = basis @ (basis.T @ dc_removed)
        result = dc_removed - projection + dc
        removed_fraction = ((before_energy - non_dc_energy(result)) / before_energy
                            if before_energy > 0 else float('nan'))
    elif mode == 'scale':
        if top_k is None:
            top_k = top_frequencies(work)[0]
        key_projected = project_out(dc_removed, top_k)
        key_energy = non_dc_energy(key_projected)
        current = non_dc_energy(dc_removed)
        factor = math.sqrt(key_energy / current) if current > 0 else 0.0
        result = dc_removed * factor + dc
        removed_fraction = ((before_energy - non_dc_energy(result)) / before_energy
                            if before_energy > 0 else float('nan'))
    else:
        raise ValueError(f'Unknown knockout mode {mode!r}')

    info = dict(K=len(top_k) if top_k is not None else 0,
                energy_removed_fraction=removed_fraction,
                norm_before=before_norm,
                norm_after=float(torch.linalg.norm(result)))
    return result.to(embedding.dtype), info


def transplant(state, donor_state):
    """Replace the numeric embedding rows of `state` with the donor's.

    Only token_embed.weight[:p] is swapped; every other tensor (the context
    rows, the equals row and all other parameters) stays with the recipient.
    """
    merged = {key: value.clone() for key, value in state.items()}
    donor_rows = donor_state['token_embed.weight'][:common.P]
    merged['token_embed.weight'][:common.P] = donor_rows.clone()
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='runs_nseed')
    parser.add_argument('--out', default='runs_nseed/fourier_metrics.csv')
    args = parser.parse_args(argv)

    runs_dir = project_path(args.runs)
    out_path = project_path(args.out)
    rows = collect(runs_dir)
    write_rows(out_path, rows)
    print(f'wrote {len(rows)} rows to {out_path}')
    for row in rows[:5]:
        print('  ', row)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
