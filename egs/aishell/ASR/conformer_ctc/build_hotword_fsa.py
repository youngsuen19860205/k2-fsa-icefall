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
Build a Hotword FSA (H_hotword) for CTC+FSA hotword-biased decoding.

This script reads a list of hotwords (hardcoded or from file), tokenises each
hotword character-by-character using ``data/lang_char/tokens.txt``, builds a
linear FSA for each hotword, merges them via k2.union, and saves the result to
``data/lang_char/H_hotword.pt``.

Optionally, a DOT visualisation of the FSA is also written.

Usage::

    python build_hotword_fsa.py \\
        --tokens   data/lang_char/tokens.txt \\
        --output   data/lang_char/H_hotword.pt \\
        --dot-out  data/lang_char/H_hotword.dot \\
        [--hotwords-file hotwords.txt]

Requirements:
    k2, torch
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional

import torch

from hotword_utils import (
    ALL_HOTWORDS,
    build_linear_fsa,
    load_hotwords,
    load_token_table,
    text_to_token_ids,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Hotword FSA from a list of hotwords and a token table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_char/tokens.txt",
        help="Path to the character-level token table (tokens.txt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/lang_char/H_hotword.pt",
        help="Output path for the saved hotword FSA (.pt file).",
    )
    parser.add_argument(
        "--dot-out",
        type=str,
        default=None,
        help="Optional path to write a DOT visualisation of the hotword FSA.",
    )
    parser.add_argument(
        "--hotwords-file",
        type=str,
        default=None,
        help=(
            "Path to a plain-text file with one hotword per line.  "
            "When omitted, the built-in hotword list is used."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for FSA construction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


# ---------------------------------------------------------------------------
# Core FSA building logic
# ---------------------------------------------------------------------------

def build_hotword_union_fsa(
    hotwords: List[str],
    token_table: dict,
    device: torch.device,
):
    """Build a union FSA for all hotwords.

    Each hotword is converted to a linear FSA.  All linear FSAs are then
    merged via ``k2.union`` to form a single FSA that accepts any hotword.

    Args:
        hotwords:    List of hotword strings.
        token_table: Token → id mapping.
        device:      PyTorch device.

    Returns:
        A ``k2.Fsa`` representing the union of all hotword linear FSAs.

    Raises:
        RuntimeError: If no valid hotword FSAs could be built.
    """
    import k2

    valid_fsas = []
    for hw in hotwords:
        token_ids = text_to_token_ids(hw, token_table)
        if not token_ids:
            logger.warning("Hotword '%s' produced no token ids; skipping.", hw)
            continue
        fsa = build_linear_fsa(token_ids, device)
        valid_fsas.append(fsa)
        logger.debug("Built linear FSA for '%s': %d tokens.", hw, len(token_ids))

    if not valid_fsas:
        raise RuntimeError(
            "No valid hotword FSAs could be built.  "
            "Check that the token table covers the hotword characters."
        )

    logger.info("Merging %d hotword FSAs via k2.union …", len(valid_fsas))
    if len(valid_fsas) == 1:
        union_fsa = valid_fsas[0]
    else:
        fsa_vec = k2.create_fsa_vec(valid_fsas)
        union_fsa = k2.union(fsa_vec)

    # Add epsilon self-loops so that blank frames can be consumed without
    # leaving the current state (useful when composing with CTC lattices).
    union_fsa = k2.add_epsilon_self_loops(union_fsa)

    logger.info(
        "Hotword union FSA built: %d states, %d arcs.",
        union_fsa.num_states,
        union_fsa.arcs.num_elements(),
    )
    return union_fsa


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

    # Fix random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Step 1: Load token table
    # ------------------------------------------------------------------
    tokens_path = Path(args.tokens)
    if not tokens_path.is_file():
        raise FileNotFoundError(
            f"Token table not found: {tokens_path}\n"
            "Please run the data preparation pipeline first, or specify "
            "--tokens pointing to an existing tokens.txt."
        )
    token_table = load_token_table(tokens_path)

    # ------------------------------------------------------------------
    # Step 2: Load hotwords
    # ------------------------------------------------------------------
    hotwords = load_hotwords(args.hotwords_file)
    logger.info("Hotwords (%d): %s", len(hotwords), hotwords)

    # ------------------------------------------------------------------
    # Step 3: Build union FSA
    # ------------------------------------------------------------------
    hotword_fsa = build_hotword_union_fsa(hotwords, token_table, device)

    # ------------------------------------------------------------------
    # Step 4: Save FSA
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hotword_fsa.as_dict(), str(output_path))
    logger.info("Hotword FSA saved to: %s", output_path)

    # ------------------------------------------------------------------
    # Step 5: Optional DOT visualisation
    # ------------------------------------------------------------------
    if args.dot_out:
        try:
            import k2
            dot_str = k2.to_dot(hotword_fsa)
            dot_path = Path(args.dot_out)
            dot_path.parent.mkdir(parents=True, exist_ok=True)
            dot_path.write_text(dot_str)
            logger.info("DOT visualisation written to: %s", dot_path)
        except Exception as exc:
            logger.warning("Could not write DOT file: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
