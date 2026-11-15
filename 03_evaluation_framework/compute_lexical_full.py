#!/usr/bin/env python3
"""Lexical evaluation of full-video narrations: ROUGE-L, chrF, BLEU-4, CIDEr, METEOR.

Compares model_response vs reference_narration in every *full_responses.jsonl
(results/ + results/archive/ for the timestamped run). Saves per-model corpus
metrics to results/lexical_full_summary.json and prints a comparison table.

Metric details (all use raw text, no timestamp stripping):
- ROUGE-L F1: rouge_score.RougeScorer(['rougeL'], use_stemmer=True), mean over samples (0-1)
- chrF: sacrebleu.sentence_chrf(hyp, [ref]).score per sample, mean (0-100)
- BLEU-4: sacrebleu.corpus_bleu(hyps, [refs]) corpus-level (0-100)
- CIDEr: pycocoevalcap Cider on the 15-doc corpus (0-10)
- METEOR: nltk.translate.meteor_score.meteor_score([ref.split()], hyp.split()), mean (0-1)
"""
import glob
import json
import os
import re

RESULTS_DIR = "results"
ARCHIVE_DIR = os.path.join(RESULTS_DIR, "archive")
OUT_JSON = os.path.join(RESULTS_DIR, "lexical_full_summary.json")


def load_pairs(path):
    refs, hyps, ids = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ref = (r.get("reference_narration") or "").strip()
            hyp = (r.get("model_response") or "").strip()
            if not ref:
                continue
            refs.append(ref)
            hyps.append(hyp if hyp else "")
            ids.append(r.get("record_id", ""))
    return ids, refs, hyps


def main():
    from rouge_score import rouge_scorer
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.cider.cider import Cider

    # discover files: prefer results/ over archive/ on tag collision
    found = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_full_responses.jsonl"))):
        tag = os.path.basename(path).replace("_full_responses.jsonl", "")
        found[tag] = path
    for path in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*_full_responses.jsonl"))):
        tag = os.path.basename(path).replace("_full_responses.jsonl", "")
        if tag not in found:
            found[tag] = path

    print(f"Found {len(found)} full-video response files")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = {}

    for tag in sorted(found):
        path = found[tag]
        ids, refs, hyps = load_pairs(path)
        n = len(refs)
        if n == 0:
            print(f"SKIP {tag}: no samples")
            continue

        # ROUGE-L (mean sentence F1) + also track P/R means
        rl_f, rl_p, rl_r = [], [], []
        for ref, hyp in zip(refs, hyps):
            if not hyp:
                rl_f.append(0.0); rl_p.append(0.0); rl_r.append(0.0)
                continue
            s = scorer.score(ref, hyp)["rougeL"]
            rl_f.append(s.fmeasure); rl_p.append(s.precision); rl_r.append(s.recall)
        rouge_l = sum(rl_f) / n

        # chrF: mean of sacrebleu.sentence_chrf (0-100)
        chrf_vals = []
        for ref, hyp in zip(refs, hyps):
            if not hyp:
                chrf_vals.append(0.0)
                continue
            chrf_vals.append(sacrebleu.sentence_chrf(hyp, [ref]).score)
        chrf = sum(chrf_vals) / n

        # BLEU-4 corpus-level (0-100)
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

        # METEOR: mean sentence-level (0-1), whitespace tokenized
        met_vals = []
        for ref, hyp in zip(refs, hyps):
            if not hyp:
                met_vals.append(0.0)
                continue
            try:
                met_vals.append(float(meteor_score([ref.split()], hyp.split())))
            except Exception:
                met_vals.append(0.0)
        meteor = sum(met_vals) / n

        # CIDEr corpus-level (0-10)
        gts = {str(i): [refs[i]] for i in range(n)}
        res = {str(i): [hyps[i]] for i in range(n)}
        try:
            cider, _ = Cider().compute_score(gts, res)
            cider = float(cider)
        except Exception as e:
            print(f"  CIDEr failed for {tag}: {e}")
            cider = 0.0

        # avg hypothesis/reference lengths for context
        avg_hyp_len = sum(len(h.split()) for h in hyps) / n
        avg_ref_len = sum(len(r.split()) for r in refs) / n

        # join LLM-judge overall if summary exists
        judge_overall = None
        for cand in (os.path.join(RESULTS_DIR, f"{tag}_full_summary.json"),
                     os.path.join(ARCHIVE_DIR, f"{tag}_full_summary.json")):
            if os.path.exists(cand):
                try:
                    with open(cand) as f:
                        judge_overall = json.load(f).get("narration", {}).get("avg_overall_score")
                except Exception:
                    pass
                break

        results[tag] = {
            "n": n,
            "rougeL_F": round(rouge_l, 4),
            "rougeL_P": round(sum(rl_p) / n, 4),
            "rougeL_R": round(sum(rl_r) / n, 4),
            "chrf": round(chrf, 2),
            "bleu4": round(bleu, 2),
            "cider": round(cider, 4),
            "meteor": round(meteor, 4),
            "avg_hyp_words": round(avg_hyp_len, 1),
            "avg_ref_words": round(avg_ref_len, 1),
            "judge_overall": judge_overall,
            "source": path,
        }
        print(f"  done {tag}: n={n} ROUGE-L={rouge_l:.4f} chrF={chrf:.2f} "
              f"BLEU-4={bleu:.2f} CIDEr={cider:.4f} METEOR={meteor:.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")

    # markdown-style table sorted by METEOR (most discriminative here)
    print("\n| Model | ROUGE-L F | chrF | BLEU-4 | CIDEr | METEOR | Judge |")
    print("|---|---|---|---|---|---|---|")
    for tag, m in sorted(results.items(), key=lambda kv: kv[1]["meteor"], reverse=True):
        j = f"{m['judge_overall']:.4f}" if m["judge_overall"] is not None else "n/a"
        print(f"| {tag} | {m['rougeL_F']:.4f} | {m['chrf']:.2f} | "
              f"{m['bleu4']:.2f} | {m['cider']:.4f} | {m['meteor']:.4f} | {j} |")


if __name__ == "__main__":
    main()
