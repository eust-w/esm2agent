"""
ESM-2 masked-marginal single-point mutation scan.

Reproduces the scoring logic of eust-w/esm2agent (mutation_prediction.py:
ESMMutationPredictor._predict_batch) so results stay comparable with the
earlier run in that repository:

  - model: esm2_t33_650M_UR50D
  - one <mask> per position, full-length context, one forward pass per position
  - softmax over the vocabulary at the masked position
  - keep the top-k non-wild-type amino acids per position
  - rank every (position, mutant) pair by that probability, take the top 100

Differences from the original, all deliberate:
  - runs on cuda/cpu instead of mps
  - loads the HuggingFace conversion of the same checkpoint (identical
    weights; dl.fbaipublicfiles.com sustained only ~43 kB/s from here)
  - also records the wild-type probability and llr = log p(mut) - log p(wt),
    the standard variant-effect score, and ranks by it as well
  - writes a full 20 x L LLR matrix, a heatmap, and run_meta.json for the
    PDF report
"""

import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    header, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:]
            else:
                seq.append(line)
    return header, "".join(seq)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def most_conserved(sequence, wt_prob, n=15, window=4):
    """Positions the model is least willing to change, straight from the scan.

    Deliberately not motif matching. An earlier version of this script carried
    hardcoded motif patterns and a guess about what they meant, and the guess
    was wrong for the first sequence it was run on. p(wt) is measured, so it
    cannot be wrong in that way: it says what the model will not substitute,
    and the biological reading is left to whoever knows the protein.
    """
    order = np.argsort(-wt_prob)[:n]
    out = []
    for i in sorted(order):
        s, e = max(0, i - window), min(len(sequence), i + window + 1)
        out.append({
            "position": int(i + 1),
            "residue": sequence[i],
            "wt_probability": float(wt_prob[i]),
            "context": f"{sequence[s:i]}[{sequence[i]}]{sequence[i+1:e]}",
        })
    return out


def draw_heatmap(llr_matrix, sequence, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = len(sequence)
    rows_per_band = 80
    bands = (L + rows_per_band - 1) // rows_per_band

    fig, axes = plt.subplots(bands, 1, figsize=(13, 2.55 * bands))
    if bands == 1:
        axes = [axes]

    vmax = float(np.nanmax(np.abs(llr_matrix)))
    vmax = min(vmax, 12.0)

    for b, ax in enumerate(axes):
        s, e = b * rows_per_band, min((b + 1) * rows_per_band, L)
        sub = llr_matrix[s:e].T  # 20 x window
        im = ax.imshow(sub, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(20))
        ax.set_yticklabels(AA20, fontsize=6.5)
        step = 5
        ticks = [i for i in range(e - s) if (s + i + 1) % step == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(s + i + 1) for i in ticks], fontsize=6, rotation=90)
        ax.set_xlabel(f"position {s+1}-{e}", fontsize=7)
        # Mark the wild-type residue in each column.
        for i, pos in enumerate(range(s, e)):
            ax.plot(i, AA20.index(sequence[pos]) if sequence[pos] in AA20 else 0,
                    marker="s", ms=1.6, color="black")
        ax.tick_params(length=1.5, pad=1)

    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
    cb.set_label("LLR = log p(mut) - log p(wt)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("ESM-2 masked-marginal LLR, all 20 substitutions per position "
                 "(black square = wild type)", fontsize=9, y=0.995)
    fig.savefig(path, dpi=185, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image as PILImage
    with PILImage.open(path) as im2:
        return im2.height / im2.width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="query.fasta")
    ap.add_argument("--model-dir", default="hf/esm2_t33_650M_UR50D")
    ap.add_argument("--model-id", default="esm2_t33_650M_UR50D")
    ap.add_argument("--top-k", type=int, default=5, help="mutants kept per position")
    ap.add_argument("--top-n", type=int, default=100, help="size of the final ranking")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 = default")
    ap.add_argument("--out-prefix", default="results/esm2")
    args = ap.parse_args()

    header, sequence = read_fasta(args.fasta)
    L = len(sequence)
    print(f"sequence: {header}")
    print(f"length:   {L} aa")

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:   {device}  (torch {torch.__version__}, threads {torch.get_num_threads()})")

    print(f"loading {args.model_dir} ...")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = EsmForMaskedLM.from_pretrained(args.model_dir).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params:   {n_params/1e6:.1f}M")

    aa_token_idx = {aa: tok.convert_tokens_to_ids(aa) for aa in AA20}
    assert all(i is not None and i >= 0 for i in aa_token_idx.values()), aa_token_idx

    enc = tok(sequence, return_tensors="pt")
    base_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    # Sanity-check the +1 BOS offset every index below relies on.
    for p in (1, L // 2, L):
        assert tok.convert_ids_to_tokens(int(base_ids[p])) == sequence[p - 1], \
            f"token offset mismatch at {p}"
    print("offset check: ok")

    rows = []
    llr_matrix = np.zeros((L, 20), dtype=np.float32)
    wt_prob = np.zeros(L, dtype=np.float64)
    positions = list(range(1, L + 1))
    t0 = time.time()

    for start in range(0, L, args.batch_size):
        chunk = positions[start : start + args.batch_size]

        ids = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        for row_i, pos in enumerate(chunk):
            ids[row_i, pos] = tok.mask_token_id  # +1 offset for BOS

        ids = ids.to(device)
        mask = attn.unsqueeze(0).repeat(len(chunk), 1).to(device)

        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        for row_i, pos in enumerate(chunk):
            wt = sequence[pos - 1]
            lp = logprobs[row_i, pos].cpu().numpy()
            lp_wt = float(lp[aa_token_idx[wt]])
            wt_prob[pos - 1] = float(np.exp(lp_wt))

            for j, aa in enumerate(AA20):
                llr_matrix[pos - 1, j] = float(lp[aa_token_idx[aa]]) - lp_wt

            cand = sorted(
                ((aa, float(lp[aa_token_idx[aa]])) for aa in AA20 if aa != wt),
                key=lambda kv: kv[1],
                reverse=True,
            )[: args.top_k]

            for aa, lp_mut in cand:
                rows.append({
                    "position": pos,
                    "original": wt,
                    "mutated_to": aa,
                    "mutation": f"{wt}{pos}{aa}",
                    "probability": float(np.exp(lp_mut)),
                    "log_probability": lp_mut,
                    "wt_probability": float(np.exp(lp_wt)),
                    "llr": lp_mut - lp_wt,
                })

        done = min(start + args.batch_size, L)
        el = time.time() - t0
        print(f"  {done}/{L} positions   {el:.0f}s elapsed, "
              f"~{el/done*(L-done):.0f}s left", flush=True)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    full = f"{args.out_prefix}_all_mutations.csv"
    df.to_csv(full, index=False)
    print(f"\nall {len(df)} candidates -> {full}")

    mat = pd.DataFrame(llr_matrix, columns=AA20)
    mat.insert(0, "wt", list(sequence))
    mat.insert(0, "position", range(1, L + 1))
    mat.to_csv(f"{args.out_prefix}_llr_matrix.csv", index=False)
    print(f"20 x {L} LLR matrix -> {args.out_prefix}_llr_matrix.csv")

    # The repo's ranking: raw masked probability of the mutant residue.
    top_prob = df.sort_values("probability", ascending=False).head(args.top_n).reset_index(drop=True)
    top_prob.insert(0, "rank", top_prob.index + 1)
    top_prob.to_csv(f"{args.out_prefix}_top{args.top_n}_by_probability.csv", index=False)

    # The standard variant-effect ranking.
    top_llr = df.sort_values("llr", ascending=False).head(args.top_n).reset_index(drop=True)
    top_llr.insert(0, "rank", top_llr.index + 1)
    top_llr.to_csv(f"{args.out_prefix}_top{args.top_n}_by_llr.csv", index=False)

    print("drawing heatmap ...")
    hm = f"{args.out_prefix}_heatmap.png"
    aspect = draw_heatmap(llr_matrix, sequence, hm)

    wpath = os.path.join(args.model_dir, "model.safetensors")
    meta = {
        "header": header,
        "sequence": sequence,
        "length": L,
        "model_id": args.model_id,
        "n_params": f"{n_params/1e6:.0f}M ({n_params:,})",
        "checkpoint_source": "huggingface.co/facebook/esm2_t33_650M_UR50D "
                             "(model.safetensors), the transformers conversion of the "
                             "fair-esm checkpoint",
        "weight_sha256": sha256(wpath) if os.path.exists(wpath) else "n/a",
        "device": f"{device} on {platform.processor() or platform.machine()}",
        "torch": torch.__version__,
        "n_positions": L,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "elapsed_s": elapsed,
        "heatmap_aspect": aspect,
        "conserved": most_conserved(sequence, wt_prob),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nscan finished in {elapsed:.1f}s\n")
    print("top 20 by LLR:")
    for _, r in top_llr.head(20).iterrows():
        print(f"  {int(r['rank']):3d}. {r['mutation']:<9s} llr={r['llr']:+7.3f}  "
              f"p={r['probability']:.4f}  p_wt={r['wt_probability']:.4f}")
    print("\ntop 20 by p(mut)  (the repo's ranking):")
    for _, r in top_prob.head(20).iterrows():
        print(f"  {int(r['rank']):3d}. {r['mutation']:<9s} p={r['probability']:.4f}  "
              f"p_wt={r['wt_probability']:.4f}  llr={r['llr']:+7.3f}")


if __name__ == "__main__":
    main()
