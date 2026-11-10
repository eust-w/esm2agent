"""
Build the PDF report: method, provenance, and the top-100 mutation tables.

Reads the CSVs written by run_mutations.py. Nothing here recomputes scores,
so the PDF cannot disagree with the CSVs.
"""

import argparse
import datetime as dt
import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#1F4E79")
GRID = colors.HexColor("#B8C4D0")
HEADBG = colors.HexColor("#1F4E79")
ZEBRA = colors.HexColor("#F2F5F8")


def styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H", parent=ss["Title"], fontSize=17, leading=21,
                          textColor=ACCENT, spaceAfter=2 * mm))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], fontSize=9.5, leading=13,
                          textColor=colors.HexColor("#5A6672"), spaceAfter=6 * mm))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, leading=16,
                          textColor=ACCENT, spaceBefore=5 * mm, spaceAfter=2 * mm))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=14,
                          alignment=TA_LEFT, spaceAfter=2 * mm))
    ss.add(ParagraphStyle("Mono", parent=ss["Normal"], fontName="Courier",
                          fontSize=7.4, leading=9.6))
    ss.add(ParagraphStyle("Note", parent=ss["Normal"], fontSize=8.5, leading=12,
                          textColor=colors.HexColor("#7A3E00")))
    ss.add(ParagraphStyle("Cap", parent=ss["Normal"], fontSize=8, leading=11,
                          textColor=colors.HexColor("#5A6672"), spaceBefore=1 * mm))
    return ss


def kv_table(rows, w1=52 * mm, w2=116 * mm):
    t = Table([[Paragraph(f"<b>{k}</b>", _SS["Body"]), Paragraph(v, _SS["Body"])]
               for k, v in rows], colWidths=[w1, w2])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def rank_table(df, score_col):
    """Top-100 laid out as two side-by-side 50-row blocks.

    Columns are the same in both tables; only the sort differs. Position and
    wild-type are already spelled out in the mutation name (H297P), so they
    get no columns of their own, and the sort key is never repeated as a
    second column.
    """
    head = ["#", "Mutation", "LLR", "p(mut)", "p(wt)"]

    def block(sub):
        data = [head]
        for _, r in sub.iterrows():
            data.append([
                str(int(r["rank"])),
                r["mutation"],
                f"{r['llr']:+.3f}",
                f"{r['probability']:.4f}",
                f"{r['wt_probability']:.4f}",
            ])
        w = [7.5 * mm, 19 * mm, 15 * mm, 16 * mm, 16 * mm]
        t = Table(data, colWidths=w, repeatRows=1)
        st = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 6.8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("FONTSIZE", (0, 1), (-1, -1), 6.8),
            # Without an explicit LEADING the cells keep the 12pt default and
            # the 51-row block grows past the page, taking a whole page alone.
            ("LEADING", (0, 0), (-1, -1), 7.9),
            ("GRID", (0, 0), (-1, -1), 0.25, GRID),
            ("ALIGN", (0, 0), (-1, -1), "CENTRE"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
        ]
        # The sort column gets a tint so it is obvious which one ordered the table.
        sort_col = {"llr": 2, "probability": 3}[score_col]
        st.append(("BACKGROUND", (sort_col, 1), (sort_col, -1),
                   colors.HexColor("#E3ECF5")))
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (sort_col - 1, i), ZEBRA))
                st.append(("BACKGROUND", (sort_col + 1, i), (-1, i), ZEBRA))
        t.setStyle(TableStyle(st))
        return t

    left, right = block(df.iloc[:50]), block(df.iloc[50:100])
    outer = Table([[left, right]], colWidths=[84 * mm, 84 * mm])
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 3 * mm),
    ]))
    return outer


def wrap_seq(seq, per=60, group=10):
    out = []
    for i in range(0, len(seq), per):
        chunk = seq[i:i + per]
        spaced = " ".join(chunk[j:j + group] for j in range(0, len(chunk), group))
        out.append(f"{i+1:>4}  {spaced}")
    return "<br/>".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-meta", default="results/run_meta.json")
    ap.add_argument("--prefix", default="results/esm2")
    ap.add_argument("--fasta", default="query.fasta")
    ap.add_argument("--heatmap", default="results/esm2_heatmap.png")
    ap.add_argument("--out", default="results/ESM2_top100_report.pdf")
    args = ap.parse_args()

    global _SS
    _SS = styles()
    S = _SS

    meta = json.load(open(args.run_meta))
    by_prob = pd.read_csv(f"{args.prefix}_top100_by_probability.csv")
    by_llr = pd.read_csv(f"{args.prefix}_top100_by_llr.csv")
    allmut = pd.read_csv(f"{args.prefix}_all_mutations.csv")

    header = meta["header"]
    seq = meta["sequence"]

    doc = SimpleDocTemplate(
        args.out, pagesize=A4,
        leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=17 * mm, bottomMargin=16 * mm,
        title="ESM-2 single-point mutation scan", author="esm2agent reproduction",
    )
    E = []

    # ---------------- page 1: method and provenance ----------------
    E.append(Paragraph("ESM-2 single-point mutation scan", S["H"]))
    E.append(Paragraph(
        f"{header} &middot; {len(seq)} aa &middot; "
        f"generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", S["Sub"]))

    E.append(Paragraph("1. What was run", S["H2"]))
    E.append(Paragraph(
        "A masked-marginal scan over every residue of the query. For each position the "
        "residue is replaced by <font face='Courier'>&lt;mask&gt;</font>, the full-length "
        "sequence is passed through ESM-2, and the softmax over the 20 amino acids at that "
        "position gives a distribution. The scoring follows "
        "<font face='Courier'>mutation_prediction.py</font> in "
        "<font face='Courier'>eust-w/esm2agent</font> so the numbers stay comparable with "
        "the earlier run in that repository.", S["Body"]))

    E.append(kv_table([
        ("Model", meta["model_id"]),
        ("Parameters", meta["n_params"]),
        ("Checkpoint", meta["checkpoint_source"]),
        ("Weight SHA-256", f"<font face='Courier' size='8'>{meta['weight_sha256']}</font>"),
        ("Method", "masked-marginal, one forward pass per position, full-length context"),
        ("Forward passes", f"{meta['n_positions']} (one per residue), batch size {meta['batch_size']}"),
        ("Candidates kept", f"top {meta['top_k']} non-wild-type residues per position "
                            f"= {len(allmut)} total"),
        ("Device", meta["device"]),
        ("Runtime", f"{meta['elapsed_s']:.1f} s"),
        ("Reproduced from", "github.com/eust-w/esm2agent &middot; mutation_prediction.py"),
    ]))

    E.append(Paragraph("2. Two rankings, and which one to use", S["H2"]))
    E.append(Paragraph(
        "<b>By p(mut)</b> is the repository's original ranking: the raw masked probability of "
        "the mutant residue. It answers \"where does the model most confidently expect some "
        "other residue?\", so it is dominated by positions the model thinks are mis-assigned "
        "in the wild type, and it ignores how likely the wild-type residue was.", S["Body"]))
    E.append(Paragraph(
        "<b>By LLR</b> is log p(mut) &minus; log p(wt), the standard variant-effect score. It "
        "measures how much the model prefers the mutant <i>over the residue actually there</i>, "
        "which is the quantity that correlates with measured fitness in the ESM papers and in "
        "ProteinGym. Both tables are included; LLR is the one to act on.", S["Body"]))
    E.append(Paragraph(
        "Caveat worth stating plainly: a high LLR means the substitution looks natural to a "
        "language model trained on UniRef. That is evidence about evolutionary plausibility, "
        "not about activity, stability, or expression. Catalytic and binding residues are "
        "often highly conserved, so the model will correctly refuse to mutate them &mdash; but "
        "it will also happily propose substitutions that are plausible and functionally "
        "useless. These 100 are a shortlist to test, not a set of predictions.", S["Note"]))

    E.append(Paragraph("3. Query sequence", S["H2"]))
    E.append(Paragraph(wrap_seq(seq), S["Mono"]))

    if meta.get("checks"):
        E.append(Paragraph("4. Sanity checks run before trusting the ranking", S["H2"]))
        E.append(Paragraph(
            "H297P came out first by a wide margin, at a position that superficially looks "
            "catalytic. That is the signature of a mis-loaded model or an off-by-one in the "
            "token offset, so the run was checked before the tables were accepted.", S["Body"]))
        E.append(kv_table([(k, v) for k, v in meta["checks"]]))
        if meta.get("checks_verdict"):
            E.append(Spacer(1, 2 * mm))
            E.append(Paragraph(f"<b>Verdict.</b> {meta['checks_verdict']}", S["Body"]))

    if meta.get("conserved"):
        E.append(PageBreak())
        E.append(Paragraph("5. Positions the model will not substitute", S["H2"]))
        E.append(Paragraph(
            "The 15 residues with the highest p(wt): where the masked model puts almost all "
            "its mass back on the residue that is already there. These are the complement of "
            "the ranking &mdash; the positions to leave alone. Measured, not matched against a "
            "motif library, so there is no pattern-matching guess to be wrong about. What they "
            "mean biologically is for someone who knows the protein to say.", S["Body"]))
        data = [["Position", "Residue", "p(wt)", "Context (±4)"]]
        for c in meta["conserved"]:
            data.append([str(c["position"]), c["residue"],
                         f"{c['wt_probability']:.4f}", c["context"]])
        t = Table(data, colWidths=[20 * mm, 18 * mm, 20 * mm, 50 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.25, GRID),
            ("ALIGN", (0, 0), (2, -1), "CENTRE"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ] + [("BACKGROUND", (0, i), (-1, i), ZEBRA)
             for i in range(2, len(data), 2)]))
        E.append(t)

    # ---------------- heatmap ----------------
    if os.path.exists(args.heatmap):
        E.append(PageBreak())
        E.append(Paragraph("Per-position substitution landscape", S["H2"]))
        E.append(Paragraph(
            "LLR for all 20 amino acids at every position. Blue = the model prefers the "
            "substitution over the wild type; red = it prefers the wild type. Dark vertical "
            "bands are conserved positions.", S["Body"]))
        E.append(Image(args.heatmap, width=172 * mm,
                       height=172 * mm * meta["heatmap_aspect"]))

    # ---------------- ranking tables ----------------
    E.append(PageBreak())
    E.append(Paragraph("Top 100 by LLR = log p(mut) &minus; log p(wt)", S["H2"]))
    E.append(Paragraph(
        "The recommended ranking. Ranks 1&ndash;50 on the left, 51&ndash;100 on the right.",
        S["Body"]))
    E.append(rank_table(by_llr, "llr"))
    E.append(Paragraph(
        f"Full table of all {len(allmut)} candidates: "
        f"<font face='Courier' size='7.5'>{os.path.basename(args.prefix)}_all_mutations.csv</font>",
        S["Cap"]))

    E.append(PageBreak())
    E.append(Paragraph("Top 100 by p(mut) &mdash; the repository's original ranking", S["H2"]))
    E.append(Paragraph(
        "Included for continuity with the earlier esm2agent run. See section 2 for why this "
        "ranking is the weaker of the two.", S["Body"]))
    E.append(rank_table(by_prob, "probability"))

    def footer(canv, d):
        canv.saveState()
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(colors.HexColor("#8A939C"))
        canv.drawString(19 * mm, 10 * mm,
                        f"ESM-2 masked-marginal scan &bull; {meta['model_id']}"
                        .replace("&bull;", "·"))
        canv.drawRightString(A4[0] - 19 * mm, 10 * mm, f"{d.page}")
        canv.restoreState()

    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
