"""
Does mutation_prediction.py's amino-acid filter actually leak non-standard tokens?

The repo builds its candidate alphabet as:

    index_to_aa = {i: a for i, a in enumerate(alphabet.all_toks)
                   if a not in ('<cls>','<pad>','<eos>','<unk>','<mask>')}
    ...
    if len(aa) == 1: aa_probs[aa] = probs[token_idx]

The len == 1 test drops the multi-character specials but keeps the single
character ones ESM-2 also has: X B U Z O . - . This script re-scores the query
with that exact alphabet and counts how often one of them reaches the top 5,
i.e. how often the repo would emit a "mutation" to an unknown residue or a gap.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
# Single-character tokens ESM-2 has that are not amino acids.
NONSTD = ["X", "B", "U", "Z", "O", ".", "-"]
MODEL_DIR = "hf/esm2_t33_650M_UR50D"
TOP_K = 5


def read_fasta(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


seq = read_fasta("query.fasta")
L = len(seq)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = EsmForMaskedLM.from_pretrained(MODEL_DIR).eval()

# The repo's candidate set: every single-character token except the specials.
cand = sorted(AA20) + [c for c in NONSTD if tok.convert_tokens_to_ids(c) is not None]
idx = {c: tok.convert_tokens_to_ids(c) for c in cand}
print("candidate alphabet the repo scores over:", "".join(cand), f"({len(cand)} tokens)")

enc = tok(seq, return_tensors="pt")
ids0, attn = enc["input_ids"][0], enc["attention_mask"][0]

leaks, examples = 0, []
BS = 4
for s in range(0, L, BS):
    chunk = list(range(s + 1, min(s + BS, L) + 1))
    ids = ids0.unsqueeze(0).repeat(len(chunk), 1).clone()
    for r, p in enumerate(chunk):
        ids[r, p] = tok.mask_token_id
    with torch.no_grad():
        out = model(input_ids=ids,
                    attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1)).logits
    pr = torch.softmax(out.float(), dim=-1)
    for r, p in enumerate(chunk):
        wt = seq[p - 1]
        scored = sorted(((c, float(pr[r, p, idx[c]])) for c in cand if c != wt),
                        key=lambda kv: kv[1], reverse=True)[:TOP_K]
        hits = [(c, v) for c, v in scored if c not in AA20]
        if hits:
            leaks += 1
            if len(examples) < 12:
                examples.append((p, wt, scored))

print(f"\npositions whose top-{TOP_K} contains a non-amino-acid token: "
      f"{leaks}/{L} ({leaks/L:.1%})")
for p, wt, scored in examples:
    pretty = "  ".join(f"{'['+c+']' if c not in AA20 else c}:{v:.4f}" for c, v in scored)
    print(f"  pos {p:3d} (wt {wt}) -> {pretty}")
