"""Upload staged datasets and benchmark outputs to the Hugging Face Hub.

Covers:
  - Part I: Teacher reference reports, rendered markdown/json reports, memory trajectories
  - Part II: Candidate reports for all 7 models across settings s1 and s3
  - Part III: Video-grounded judge evaluations (v21full)

Usage:
  uv run python src/vlm_report_framework/upload_hf_dataset.py --part all
  uv run python src/vlm_report_framework/upload_hf_dataset.py --part evaluation
  uv run python src/vlm_report_framework/upload_hf_dataset.py --part candidates
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

REPO = Path.cwd()
DEFAULT_REPO = "shahedm2001/cataract-video-to-report"

CANDIDATE_MODELS = [
    "hulumed-4b",
    "hulumed-7b",
    "lingshu-7b",
    "qwen3vl-2b-grpo",
    "qwen3vl-2b-instruct",
    "qwen3vl-2b-sft-stage2",
    "qwen3vl-4b-instruct",
]


def upload_teacher_rendered(api, repo: str) -> None:
    src = REPO / "output-teacher" / "reports"
    if not src.exists():
        print(f"Skipping teacher reports: {src} not found")
        return
    print(f"Uploading teacher reports -> {repo}:teacher_reports/reports ...")
    t0 = time.time()
    url = api.upload_folder(
        folder_path=str(src),
        path_in_repo="teacher_reports/reports",
        repo_id=repo,
        repo_type="dataset",
        commit_message="Update teacher per-chunk raw JSON and rendered MD reports",
    )
    print(f"Teacher reports uploaded in {time.time()-t0:.1f}s: {url}")


def upload_candidates(api, repo: str) -> None:
    cand_dir = REPO / "output-candidates"
    if not cand_dir.exists():
        print(f"Skipping candidates: {cand_dir} not found")
        return
    for i, m in enumerate(CANDIDATE_MODELS, 1):
        m_dir = cand_dir / m
        if not m_dir.exists():
            print(f"[{i}/{len(CANDIDATE_MODELS)}] Model dir {m_dir} not found, skipping")
            continue
        print(f"[{i}/{len(CANDIDATE_MODELS)}] Uploading candidate reports for {m} ...")
        t0 = time.time()
        url = api.upload_folder(
            folder_path=str(m_dir),
            path_in_repo=f"output-candidates/{m}",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Add candidate reports for {m} (s1 and s3)",
        )
        print(f"[{i}/{len(CANDIDATE_MODELS)}] {m} uploaded in {time.time()-t0:.1f}s: {url}")


def upload_evaluation(api, repo: str, run_id: str = "v21full") -> None:
    eval_dir = REPO / "output-evaluation" / run_id
    if not eval_dir.exists():
        print(f"Skipping evaluation: {eval_dir} not found")
        return
    print(f"Uploading Tier J evaluation results -> {repo}:output-evaluation/{run_id} ...")
    t0 = time.time()
    url = api.upload_folder(
        folder_path=str(eval_dir),
        path_in_repo=f"output-evaluation/{run_id}",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add Tier J visual-3 evaluation results ({run_id})",
        ignore_patterns=["judgments-blind-visual1/judgments/*"],
    )
    print(f"Evaluation uploaded in {time.time()-t0:.1f}s: {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--part", choices=["all", "teacher", "candidates", "evaluation"],
                    default="all")
    ap.add_argument("--run-id", default="v21full")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    if args.part in ("all", "teacher"):
        upload_teacher_rendered(api, args.repo)
    if args.part in ("all", "candidates"):
        upload_candidates(api, args.repo)
    if args.part in ("all", "evaluation"):
        upload_evaluation(api, args.repo, run_id=args.run_id)


if __name__ == "__main__":
    main()
