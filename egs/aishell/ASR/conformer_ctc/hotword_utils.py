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
Utility functions for hotword-biased CTC+FSA decoding experiments.

This module provides:
  - Hotword list loading and management
  - Hotword detection in hypothesis text
  - Recall / False Alarm Rate / WER (CER) computation
  - Token table loading
  - Text-to-token-id conversion
  - Linear FSA construction helpers

Requirements:
  - k2
  - torch
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hotword / phrase lists used across all experiment scripts
# ---------------------------------------------------------------------------
DEFAULT_HOTWORDS: List[str] = [
    "礼冥酒",
    "樟烘烘",
    "棋贸恙",
    "膏痞岔",
    "洪西效应",
    "西门污痕",
]

DEFAULT_PHRASES: List[str] = [
    "打给",
    "打电话给",
    "扣",
]

ALL_HOTWORDS: List[str] = DEFAULT_HOTWORDS + DEFAULT_PHRASES


# ---------------------------------------------------------------------------
# Hotword management
# ---------------------------------------------------------------------------

def load_hotwords(source: Union[str, Path, List[str], None] = None) -> List[str]:
    """Load hotwords from a file path or return a default list.

    Args:
        source: Either a file path (one hotword per line) or a pre-built list
                of strings.  When *None* the built-in ``ALL_HOTWORDS`` list is
                returned.

    Returns:
        A list of hotword strings (deduplicated, order preserved).
    """
    if source is None:
        logger.info("Using built-in hotword list (%d entries).", len(ALL_HOTWORDS))
        return list(ALL_HOTWORDS)

    if isinstance(source, list):
        return list(source)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Hotword file not found: {path}")

    hotwords: List[str] = []
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            word = line.strip()
            if word and word not in seen:
                hotwords.append(word)
                seen.add(word)
    logger.info("Loaded %d hotwords from %s.", len(hotwords), path)
    return hotwords


# ---------------------------------------------------------------------------
# Hotword detection
# ---------------------------------------------------------------------------

def detect_hotwords(text: str, hotwords: List[str]) -> List[str]:
    """Return every hotword that appears in *text*.

    Args:
        text:     The hypothesis (or reference) string.
        hotwords: List of hotword strings.

    Returns:
        A list of hotwords found in *text* (may contain duplicates if the same
        hotword appears multiple times).
    """
    found: List[str] = []
    for hw in hotwords:
        count = text.count(hw)
        found.extend([hw] * count)
    return found


def contains_hotword(text: str, hotwords: List[str]) -> bool:
    """Return True if *text* contains at least one hotword."""
    return any(hw in text for hw in hotwords)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_recall(
    refs: List[str],
    hyps: List[str],
    hotwords: List[str],
) -> float:
    """Compute hotword recall over a paired list of references and hypotheses.

    Recall = (number of hotword occurrences correctly recognised) /
             (total number of hotword occurrences in references)

    Args:
        refs:     Reference transcripts (one per utterance).
        hyps:     Hypothesis transcripts (one per utterance).
        hotwords: List of hotword strings.

    Returns:
        Recall in [0, 1].  Returns 0.0 when there are no hotwords in refs.
    """
    assert len(refs) == len(hyps), "refs and hyps must have the same length"
    total_ref = 0
    total_hit = 0
    for ref, hyp in zip(refs, hyps):
        for hw in hotwords:
            ref_count = ref.count(hw)
            hyp_count = hyp.count(hw)
            total_ref += ref_count
            total_hit += min(ref_count, hyp_count)
    if total_ref == 0:
        logger.warning("No hotwords found in references; recall is undefined (returning 0.0).")
        return 0.0
    return total_hit / total_ref


def compute_far(
    normal_refs: List[str],
    normal_hyps: List[str],
    hotwords: List[str],
) -> float:
    """Compute False Alarm Rate (FAR) on utterances that contain no hotwords.

    FAR = (number of non-hotword utterances where a hotword was falsely
           detected) / (total number of non-hotword utterances)

    Args:
        normal_refs: Reference transcripts that do **not** contain hotwords.
        normal_hyps: Corresponding hypothesis transcripts.
        hotwords:    List of hotword strings.

    Returns:
        FAR in [0, 1].  Returns 0.0 when *normal_refs* is empty.
    """
    assert len(normal_refs) == len(normal_hyps)
    if not normal_refs:
        return 0.0
    false_alarms = sum(
        1
        for ref, hyp in zip(normal_refs, normal_hyps)
        if not contains_hotword(ref, hotwords) and contains_hotword(hyp, hotwords)
    )
    return false_alarms / len(normal_refs)


def compute_wer_cer(refs: List[str], hyps: List[str]) -> float:
    """Compute Character Error Rate (CER) for Chinese text.

    CER = (S + D + I) / N  where S/D/I are substitutions/deletions/insertions
    and N is the total number of characters in the references.

    Args:
        refs: Reference strings.
        hyps: Hypothesis strings.

    Returns:
        CER in [0, ∞).
    """
    assert len(refs) == len(hyps)
    total_chars = 0
    total_edits = 0
    for ref, hyp in zip(refs, hyps):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        total_chars += len(ref_chars)
        total_edits += _edit_distance(ref_chars, hyp_chars)
    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


def _edit_distance(seq1: list, seq2: list) -> int:
    """Standard dynamic-programming edit distance."""
    m, n = len(seq1), len(seq2)
    # Use two rows to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


# ---------------------------------------------------------------------------
# Token table
# ---------------------------------------------------------------------------

def load_token_table(tokens_file: Union[str, Path]) -> Dict[str, int]:
    """Load a token → id mapping from a tokens.txt file.

    Expected format (one entry per line)::

        <eps>  0
        <blk>  1
        一     2
        ...

    Args:
        tokens_file: Path to the token table file.

    Returns:
        A ``dict`` mapping token string → integer id.
    """
    token_table: Dict[str, int] = {}
    with open(tokens_file, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                token, idx = parts[0], int(parts[1])
                token_table[token] = idx
    logger.info("Loaded %d tokens from %s.", len(token_table), tokens_file)
    return token_table


def text_to_token_ids(text: str, token_table: Dict[str, int]) -> List[int]:
    """Convert a string of characters to a list of token ids.

    Unknown characters are silently skipped with a warning.

    Args:
        text:        Input string (character-level, e.g. Chinese text).
        token_table: Mapping produced by :func:`load_token_table`.

    Returns:
        A list of integer token ids.
    """
    ids: List[int] = []
    for ch in text:
        if ch in token_table:
            ids.append(token_table[ch])
        else:
            logger.warning("Token '%s' not found in token table; skipping.", ch)
    return ids


# ---------------------------------------------------------------------------
# Linear FSA construction
# ---------------------------------------------------------------------------

def build_linear_fsa(
    token_ids: List[int],
    device: torch.device,
):
    """Build a linear (chain) FSA for a single token sequence.

    The resulting FSA has states 0, 1, …, n where n is the final (accepting)
    state.  Arc labels correspond to *token_ids*.

    Args:
        token_ids: Sequence of integer token ids for one hotword.
        device:    PyTorch device for the resulting FSA.

    Returns:
        A ``k2.Fsa`` object representing the linear path.

    Raises:
        ImportError: If k2 is not installed.
        ValueError:  If *token_ids* is empty.
    """
    try:
        import k2  # noqa: F401
    except ImportError as exc:
        raise ImportError("k2 is required for build_linear_fsa") from exc

    if not token_ids:
        raise ValueError("token_ids must be non-empty")

    import k2

    # k2.linear_fsa accepts a list-of-lists (batch dimension)
    fsa = k2.linear_fsa([token_ids], device=device)
    return fsa


# ---------------------------------------------------------------------------
# N-best rescoring helpers
# ---------------------------------------------------------------------------

def rescore_nbest_with_hotwords(
    hyps_list: List[List[str]],
    hyps_scores: "torch.Tensor",
    hotwords: List[str],
    alpha: float,
) -> List[str]:
    """Rescore an n-best list by adding a hotword bonus.

    For each hypothesis the bonus is:
        ``alpha * (total number of hotword occurrences in hypothesis)``

    Args:
        hyps_list:   List of hypotheses; each hypothesis is itself a list of
                     token/character strings.
        hyps_scores: 1-D tensor of scores (log-probs or similar), one per hyp.
        hotwords:    List of hotword strings.
        alpha:       Weight for the hotword bonus.

    Returns:
        The best hypothesis (as a list of strings) after rescoring.
    """
    bonuses: List[float] = []
    for hyp in hyps_list:
        hyp_str = "".join(hyp)
        bonus = sum(hyp_str.count(hw) for hw in hotwords) * alpha
        bonuses.append(bonus)
    bonus_tensor = torch.tensor(bonuses, dtype=hyps_scores.dtype, device=hyps_scores.device)
    new_scores = hyps_scores + bonus_tensor
    best_idx = int(new_scores.argmax().item())
    return hyps_list[best_idx]
