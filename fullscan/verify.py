"""
Sanity checks on the scan, run before trusting any ranking.

1. Wild-type recovery: masked ESM-2 650M should recover the native residue as
   top-1 at roughly half the positions of a natural protein. A low rate means
   the model, the tokenizer offset, or the weights are wrong.
2. Token offset: the residue the tokenizer puts at index p must be sequence[p-1].
3. Full distribution at the positions the ranking flags, so an extreme call can
   be inspected rather than taken on faith.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
MODEL_DIR = "hf/esm2_t33_650M_UR50D"


def read_fasta(path):
    seq = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith(">"):
            seq.append(line)
    return "".join(seq)


seq = read_fasta("query.fasta")
L = len(seq)
print(f"length {L}")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = EsmForMaskedLM.from_pretrained(MODEL_DIR).eval()

enc = tok(seq, return_tensors="pt")
ids0 = enc["input_ids"][0]
attn = enc["attention_mask"][0]

print("\n-- offset check --")
print(f"ids length {len(ids0)} for {L} residues (expect L+2 with BOS/EOS)")
print("bos:", tok.convert_ids_to_tokens(int(ids0[0])),
      " eos:", tok.convert_ids_to_tokens(int(ids0[-1])))
bad = [p for p in range(1, L + 1)
       if tok.convert_ids_to_tokens(int(ids0[p])) != seq[p - 1]]
print("positions whose token != residue:", bad if bad else "none")
print("mask_token_id:", tok.mask_token_id, tok.mask_token)

aa_idx = {aa: tok.convert_tokens_to_ids(aa) for aa in AA20}
print("amino-acid token ids:", aa_idx)

# ---- unmasked pass: does the model reconstruct the sequence it was given? ----
with torch.no_grad():
    lg = model(input_ids=ids0.unsqueeze(0), attention_mask=attn.unsqueeze(0)).logits[0]
pred_unmasked = "".join(
    AA20[int(np.argmax([lg[p, aa_idx[a]].item() for a in AA20]))] for p in range(1, L + 1)
)
acc_un = sum(a == b for a, b in zip(pred_unmasked, seq)) / L
print(f"\n-- unmasked identity (copying task, expect >0.95) -- {acc_un:.3f}")

# ---- masked wild-type recovery over every position ----
print("\n-- masked wild-type recovery --")
hits, top1 = 0, []
BS = 4
for s in range(0, L, BS):
    chunk = list(range(s + 1, min(s + BS, L) + 1))
    ids = ids0.unsqueeze(0).repeat(len(chunk), 1).clone()
    for r, p in enumerate(chunk):
        ids[r, p] = tok.mask_token_id
    with torch.no_grad():
        out = model(input_ids=ids,
                    attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1)).logits
    for r, p in enumerate(chunk):
        v = np.array([out[r, p, aa_idx[a]].item() for a in AA20])
        best = AA20[int(v.argmax())]
        top1.append(best)
        hits += best == seq[p - 1]
print(f"top-1 == wild type at {hits}/{L} positions = {hits/L:.3f}")
print("(ESM-2 650M on a natural protein: typically 0.45-0.65)")

# ---- full distribution at the flagged positions ----
for pos in (297, 301, 182, 181, 158, 287):
    ids = ids0.unsqueeze(0).clone()
    ids[0, pos] = tok.mask_token_id
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=attn.unsqueeze(0)).logits[0, pos]
    p = torch.softmax(o.float(), dim=-1)
    pairs = sorted(((a, float(p[aa_idx[a]])) for a in AA20),
                   key=lambda kv: kv[1], reverse=True)
    wt = seq[pos - 1]
    print(f"\nposition {pos} (wild type {wt}, p_wt={float(p[aa_idx[wt]]):.4f}) "
          f"context {seq[max(0,pos-6):pos+5]}")
    print("   " + "  ".join(f"{a}:{v:.3f}" for a, v in pairs[:8]))
