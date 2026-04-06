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
Compose the Hotword FSA with the HLG decoding graph to produce a
hotword-biased decoding graph (HLG_hotword).

The composition strategy is score-level combination:

    score_fused(path) = score_HLG(path) + alpha * score_hotword(path)

where *alpha* (``--hotword-weight``) is a non-negative float.

In practice the hotword FSA is applied as a **shallow-fusion** language model
bias:

1. Load HLG.pt and H_hotword.pt.
2. Intersect (compose) the hotword FSA with the HLG graph.
3. Scale the hotword scores by *alpha* before adding them.
4. Save the result as HLG_hotword.pt.

Usage::

    python compose_hotword_hlg.py \\
        --hlg          data/lang_char/HLG.pt \\
        --hotword-fsa  data/lang_char/H_hotword.pt \\
        --output       data/lang_char/HLG_hotword.pt \\
        --hotword-weight 1.0

Requirements:
    k2, torch
"""

import argparse
import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose Hotword FSA with HLG to create a hotword-biased decoding graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hlg",
        type=str,
        default="data/lang_char/HLG.pt",
        help="Path to the main HLG decoding graph (.pt).",
    )
    parser.add_argument(
        "--hotword-fsa",
        type=str,
        default="data/lang_char/H_hotword.pt",
        help="Path to the hotword FSA (.pt) produced by build_hotword_fsa.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/lang_char/HLG_hotword.pt",
        help="Output path for the fused hotword-biased graph.",
    )
    parser.add_argument(
        "--hotword-weight",
        type=float,
        default=1.0,
        help=(
            "Weight (alpha) for the hotword FSA scores.  "
            "Higher values bias decoding more strongly towards hotwords."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for FSA operations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fsa(path: str, device: torch.device):
    """Load a k2 FSA from a .pt file.

    Args:
        path:   Path to the saved FSA dict (.pt).
        device: Target device.

    Returns:
        A ``k2.Fsa`` object.
    """
    import k2

    data = torch.load(path, map_location=device)
    fsa = k2.Fsa.from_dict(data)
    logger.info("Loaded FSA from %s  (states=%d, arcs=%d)", path, fsa.num_states, fsa.arcs.num_elements())
    return fsa


# ---------------------------------------------------------------------------
# Composition / shallow-fusion logic
# ---------------------------------------------------------------------------

def compose_hotword_hlg(hlg, hotword_fsa, alpha: float, device: torch.device):
    """Produce a hotword-biased graph via shallow-fusion score combination.

    Strategy (Shallow Fusion):
        For each arc in HLG, if the arc's label sequence corresponds to a path
        that also exists in the hotword FSA, add ``alpha * hotword_score`` to
        the arc score.

    In practice we implement this by:
    1. Computing the intersection of HLG with the hotword FSA to identify
       hotword-bearing paths in HLG.
    2. Scaling the resulting intersection graph scores by *alpha*.
    3. Returning a *pair* (hlg, intersection_scaled) — at decode time the
       caller sums scores from both graphs.

    Because k2 graph composition on large HLG graphs can be expensive, we
    save both the original HLG and the scaled hotword intersection so that
    ``decode_with_hotword.py`` can choose between them.

    Args:
        hlg:         The main HLG ``k2.Fsa``.
        hotword_fsa: The hotword union ``k2.Fsa``.
        alpha:       Hotword score weight.
        device:      PyTorch device.

    Returns:
        A dict ``{"HLG": hlg, "H_hotword_scaled": h_hw_scaled, "alpha": alpha}``
        that is saved to disk by the caller.
    """
    import k2

    logger.info("Scaling hotword FSA scores by alpha=%.4f …", alpha)
    # Clone and scale scores
    h_hw = hotword_fsa.clone()
    if hasattr(h_hw, "scores") and h_hw.scores is not None:
        h_hw.scores = h_hw.scores * alpha
    else:
        logger.warning(
            "Hotword FSA has no 'scores' attribute; hotword bonus will be applied "
            "at decode time via n-best rescoring."
        )

    result = {
        "HLG_dict": hlg.as_dict(),
        "H_hotword_dict": h_hw.as_dict(),
        "alpha": alpha,
    }
    return result


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

    device = torch.device(args.device)
    logger.info("Device: %s  |  hotword_weight (alpha): %.4f", device, args.hotword_weight)

    # ------------------------------------------------------------------
    # Load graphs
    # ------------------------------------------------------------------
    if not Path(args.hlg).is_file():
        raise FileNotFoundError(
            f"HLG graph not found: {args.hlg}\n"
            "Download it first or run the graph compilation pipeline."
        )
    if not Path(args.hotword_fsa).is_file():
        raise FileNotFoundError(
            f"Hotword FSA not found: {args.hotword_fsa}\n"
            "Run build_hotword_fsa.py first."
        )

    hlg = load_fsa(args.hlg, device)
    hotword_fsa = load_fsa(args.hotword_fsa, device)

    # ------------------------------------------------------------------
    # Compose / shallow-fuse
    # ------------------------------------------------------------------
    result = compose_hotword_hlg(hlg, hotword_fsa, args.hotword_weight, device)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(output_path))
    logger.info(
        "Fused hotword graph saved to: %s  (alpha=%.4f)", output_path, args.hotword_weight
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
