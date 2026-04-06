#!/usr/bin/env python3
# Copyright 2024 (Author: Hotword FSA Experiment)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Grid search + Bayesian optimisation for the optimal hotword weight (alpha).

Workflow
--------
1. **Coarse grid search**: evaluate alpha in [0.0, 5.0] with step 0.5.
2. **Bayesian fine search**: use ``scikit-optimize`` (``skopt``) to refine
   the search around the grid optimum.

Objective (single-objective, lower is better)
----------------------------------------------
    objective(alpha) = -recall + beta * WER + gamma * FAR + penalty

Constraints (soft, implemented as penalty terms)
-------------------------------------------------
    recall  ≥ 0.8
    WER     ≤ 1.1 * baseline_WER
    FAR     ≤ 0.15

Usage::

    python search_hotword_weight.py \\
        --manifest-dir  data/tts_testset \\
        --checkpoint    exp/pretrained.pt \\
        --lang-dir      data/lang_char \\
        --hotword-fsa   data/lang_char/H_hotword.pt \\
        --output-dir    exp/weight_search \\
        --n-bayesian    30

Requirements (pip):
    scikit-optimize, matplotlib, torch, k2
"""

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from hotword_utils import (
    ALL_HOTWORDS,
    compute_far,
    compute_recall,
    compute_wer_cer,
    load_hotwords,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid + Bayesian search for optimal hotword weight (alpha).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/tts_testset",
        help="Directory with manifest.jsonl and text.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (optional; omit for mock evaluation).",
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_char",
        help="Language directory (tokens.txt, HLG.pt, etc.).",
    )
    parser.add_argument(
        "--hotword-fsa",
        type=str,
        default="data/lang_char/H_hotword.pt",
        help="Path to the hotword FSA.",
    )
    parser.add_argument(
        "--hotwords-file",
        type=str,
        default=None,
        help="Hotword list file (one per line).  Defaults to built-in list.",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.0,
        help="Minimum alpha for search.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=5.0,
        help="Maximum alpha for search.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.5,
        help="Step size for coarse grid search.",
    )
    parser.add_argument(
        "--n-bayesian",
        type=int,
        default=30,
        help="Number of Bayesian optimisation calls.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="WER penalty weight in objective.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="FAR penalty weight in objective.",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.8,
        help="Minimum recall constraint.",
    )
    parser.add_argument(
        "--far-threshold",
        type=float,
        default=0.15,
        help="Maximum FAR constraint.",
    )
    parser.add_argument(
        "--wer-rel-threshold",
        type=float,
        default=0.10,
        help="Maximum relative WER degradation w.r.t. baseline (alpha=0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/weight_search",
        help="Output directory for results, CSV, and plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use a mock evaluator (no model required).  "
            "Useful for testing the search logic."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Mock evaluator (for testing without a real model)
# ---------------------------------------------------------------------------

def _mock_evaluate(alpha: float, hotwords: List[str]) -> Tuple[float, float, float]:
    """Simulated evaluation: recall ↑ with alpha, WER ↑ with alpha, FAR ↑ with alpha.

    Args:
        alpha:    Hotword weight being evaluated.
        hotwords: Hotword list (unused in mock).

    Returns:
        (recall, wer, far) – all floats in [0, 1].
    """
    # Monotone model: recall saturates at ~0.95, WER degrades, FAR grows.
    recall = min(0.95, 0.3 + 0.5 * alpha / 5.0 + random.gauss(0, 0.02))
    wer = max(0.0, 0.12 - 0.005 * alpha + random.gauss(0, 0.005))
    far = min(1.0, 0.02 + 0.08 * alpha / 5.0 + random.gauss(0, 0.01))
    return (
        max(0.0, min(1.0, recall)),
        max(0.0, wer),
        max(0.0, min(1.0, far)),
    )


# ---------------------------------------------------------------------------
# Real evaluator (wraps decode_with_hotword logic)
# ---------------------------------------------------------------------------

def _real_evaluate(
    alpha: float,
    entries: List[Dict],
    hotwords: List[str],
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    """Run decoding with the given alpha and return (recall, wer, far).

    This function imports ``decode_with_hotword.decode_manifest`` so as to
    avoid code duplication.

    Args:
        alpha:    Hotword weight.
        entries:  Manifest entries (loaded once).
        hotwords: Hotword strings.
        args:     Original argument namespace (checkpoint, lang-dir, etc.).

    Returns:
        (recall, wer, far).
    """
    import types

    # Build a temporary args namespace for decode_manifest
    decode_args = types.SimpleNamespace(
        checkpoint=args.checkpoint,
        lang_dir=args.lang_dir,
        hotword_weight=alpha,
        method="nbest",
        num_paths=100,
        beam_size=4,
        device=args.device,
        max_duration=30.0,
    )

    try:
        from decode_with_hotword import decode_manifest, evaluate
    except ImportError as exc:
        raise ImportError(
            "decode_with_hotword.py must be in the Python path."
        ) from exc

    hyps = decode_manifest(entries, decode_args, hotwords, hotword_fsa=None)
    metrics = evaluate(entries, hyps, hotwords, alpha)
    return metrics["recall"], metrics["wer"], metrics["far"]


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

class ObjectiveFunction:
    """Callable objective for Bayesian optimisation.

    Args:
        evaluate_fn:       Function(alpha) → (recall, wer, far).
        baseline_wer:      WER at alpha=0 (constraint reference).
        beta:              WER penalty weight.
        gamma:             FAR penalty weight.
        recall_threshold:  Minimum recall constraint.
        far_threshold:     Maximum FAR constraint.
        wer_rel_threshold: Maximum relative WER degradation.
    """

    def __init__(
        self,
        evaluate_fn,
        baseline_wer: float,
        beta: float = 0.5,
        gamma: float = 0.3,
        recall_threshold: float = 0.8,
        far_threshold: float = 0.15,
        wer_rel_threshold: float = 0.10,
    ):
        self.evaluate_fn = evaluate_fn
        self.baseline_wer = baseline_wer
        self.beta = beta
        self.gamma = gamma
        self.recall_threshold = recall_threshold
        self.far_threshold = far_threshold
        self.wer_rel_threshold = wer_rel_threshold
        self.history: List[Dict] = []

    def __call__(self, params: list) -> float:
        alpha = float(params[0])
        recall, wer, far = self.evaluate_fn(alpha)

        # Constraint penalties
        penalty = 0.0
        if recall < self.recall_threshold:
            penalty += 1000.0 * (self.recall_threshold - recall)
        if self.baseline_wer > 0:
            wer_limit = self.baseline_wer * (1 + self.wer_rel_threshold)
            if wer > wer_limit:
                penalty += 1000.0 * (wer - wer_limit)
        if far > self.far_threshold:
            penalty += 1000.0 * (far - self.far_threshold)

        score = -recall + self.beta * wer + self.gamma * far + penalty

        record = {
            "alpha": alpha,
            "recall": recall,
            "wer": wer,
            "far": far,
            "penalty": penalty,
            "score": score,
        }
        self.history.append(record)
        logger.info(
            "alpha=%.3f  recall=%.3f  wer=%.3f  far=%.3f  penalty=%.1f  score=%.4f",
            alpha, recall, wer, far, penalty, score,
        )
        return score


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    objective: ObjectiveFunction,
    alpha_min: float,
    alpha_max: float,
    step: float,
) -> Tuple[float, float]:
    """Evaluate objective at evenly-spaced alpha values.

    Args:
        objective:  Callable(params) → float.
        alpha_min:  Start of search range.
        alpha_max:  End of search range.
        step:       Step size.

    Returns:
        (best_alpha, best_score).
    """
    import numpy as np

    alphas = list(np.arange(alpha_min, alpha_max + 1e-9, step))
    logger.info("Grid search over %d alpha values: %s", len(alphas), alphas)

    best_alpha, best_score = None, float("inf")
    for alpha in alphas:
        score = objective([alpha])
        if score < best_score:
            best_score = score
            best_alpha = alpha

    logger.info("Grid search best: alpha=%.3f  score=%.4f", best_alpha, best_score)
    return best_alpha, best_score


# ---------------------------------------------------------------------------
# Bayesian optimisation
# ---------------------------------------------------------------------------

def bayesian_search(
    objective: ObjectiveFunction,
    alpha_min: float,
    alpha_max: float,
    n_calls: int,
    seed: int,
    x0: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Run Bayesian optimisation using skopt's gp_minimize.

    Args:
        objective:  Callable(params) → float.
        alpha_min:  Search space lower bound.
        alpha_max:  Search space upper bound.
        n_calls:    Total number of objective evaluations.
        seed:       Random seed.
        x0:         Initial point(s) to evaluate first (warm start).

    Returns:
        (best_alpha, best_score).
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        logger.warning(
            "scikit-optimize not installed.  Skipping Bayesian search.  "
            "Install with: pip install scikit-optimize"
        )
        return None, None

    space = [Real(alpha_min, alpha_max, name="alpha")]
    x0_list = [[x0]] if x0 is not None else None

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=seed,
        x0=x0_list,
        verbose=True,
    )

    best_alpha = float(result.x[0])
    best_score = float(result.fun)
    logger.info("Bayesian search best: alpha=%.4f  score=%.4f", best_alpha, best_score)
    return best_alpha, best_score


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_results(history: List[Dict], output_dir: Path) -> None:
    """Save search history to CSV and produce a matplotlib plot.

    Args:
        history:    List of dicts with keys alpha, recall, wer, far, score.
        output_dir: Directory to write outputs.
    """
    # Sort by alpha for plotting
    history_sorted = sorted(history, key=lambda x: x["alpha"])

    # CSV
    csv_path = output_dir / "search_results.csv"
    if history_sorted:
        fieldnames = list(history_sorted[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_sorted)
        logger.info("Search results written to: %s", csv_path)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        alphas = [r["alpha"] for r in history_sorted]
        recalls = [r["recall"] for r in history_sorted]
        wers = [r["wer"] for r in history_sorted]
        fars = [r["far"] for r in history_sorted]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(alphas, recalls, "o-", color="green", label="Recall")
        axes[0].axhline(0.8, color="green", linestyle="--", alpha=0.5, label="Recall ≥ 0.8")
        axes[0].set_ylabel("Recall")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(alphas, wers, "o-", color="blue", label="WER (CER)")
        axes[1].set_ylabel("WER (CER)")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(alphas, fars, "o-", color="red", label="FAR")
        axes[2].axhline(0.15, color="red", linestyle="--", alpha=0.5, label="FAR ≤ 0.15")
        axes[2].set_ylabel("False Alarm Rate")
        axes[2].set_xlabel("Alpha (hotword weight)")
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle("Hotword Weight Search Results", fontsize=14)
        plt.tight_layout()
        plot_path = output_dir / "search_results.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        logger.info("Plot saved to: %s", plot_path)
    except Exception as exc:
        logger.warning("Could not generate plot: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hotwords = load_hotwords(args.hotwords_file)
    logger.info("Hotwords: %s", hotwords)

    # ------------------------------------------------------------------
    # Choose evaluator
    # ------------------------------------------------------------------
    if args.mock or args.checkpoint is None:
        logger.info("Using mock evaluator (no model inference).")
        evaluate_fn = lambda alpha: _mock_evaluate(alpha, hotwords)
    else:
        import json as _json

        manifest_path = Path(args.manifest_dir) / "manifest.jsonl"
        entries: List[Dict] = []
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(_json.loads(line))
        evaluate_fn = lambda alpha: _real_evaluate(alpha, entries, hotwords, args)

    # ------------------------------------------------------------------
    # Baseline (alpha = 0)
    # ------------------------------------------------------------------
    logger.info("Computing baseline (alpha=0) …")
    baseline_recall, baseline_wer, baseline_far = evaluate_fn(0.0)
    logger.info(
        "Baseline → recall=%.3f  wer=%.3f  far=%.3f",
        baseline_recall, baseline_wer, baseline_far,
    )

    # ------------------------------------------------------------------
    # Build objective
    # ------------------------------------------------------------------
    objective = ObjectiveFunction(
        evaluate_fn=evaluate_fn,
        baseline_wer=baseline_wer,
        beta=args.beta,
        gamma=args.gamma,
        recall_threshold=args.recall_threshold,
        far_threshold=args.far_threshold,
        wer_rel_threshold=args.wer_rel_threshold,
    )

    # ------------------------------------------------------------------
    # Stage 1: Grid search
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 1: Coarse grid search")
    logger.info("=" * 60)
    grid_best_alpha, grid_best_score = grid_search(
        objective, args.alpha_min, args.alpha_max, args.grid_step
    )

    # ------------------------------------------------------------------
    # Stage 2: Bayesian optimisation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: Bayesian optimisation (n_calls=%d)", args.n_bayesian)
    logger.info("=" * 60)
    bayes_best_alpha, bayes_best_score = bayesian_search(
        objective,
        args.alpha_min,
        args.alpha_max,
        args.n_bayesian,
        args.seed,
        x0=grid_best_alpha,
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    best_alpha = bayes_best_alpha if bayes_best_alpha is not None else grid_best_alpha
    best_record = min(
        (r for r in objective.history if abs(r["alpha"] - best_alpha) < 0.01),
        key=lambda r: r["score"],
        default=None,
    )

    logger.info("=" * 60)
    logger.info("FINAL RESULT")
    logger.info("  Best alpha    : %.4f", best_alpha)
    if best_record:
        logger.info("  Recall        : %.4f (%.2f%%)", best_record["recall"], best_record["recall"] * 100)
        logger.info("  WER (CER)     : %.4f (%.2f%%)", best_record["wer"], best_record["wer"] * 100)
        logger.info("  FAR           : %.4f (%.2f%%)", best_record["far"], best_record["far"] * 100)
    logger.info("=" * 60)

    # Save everything
    save_results(objective.history, output_dir)

    summary = {
        "best_alpha": best_alpha,
        "grid_best_alpha": grid_best_alpha,
        "bayes_best_alpha": bayes_best_alpha,
        "baseline_wer": baseline_wer,
        "baseline_recall": baseline_recall,
        "baseline_far": baseline_far,
        "best_metrics": best_record,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Summary saved to: %s", summary_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
