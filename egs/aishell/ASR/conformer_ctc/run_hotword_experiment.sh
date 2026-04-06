#!/usr/bin/env bash
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

# ---------------------------------------------------------------------------
# run_hotword_experiment.sh
#
# One-click runner for the complete CTC+FSA hotword decoding experiment.
#
# Stages:
#   1  Build hotword FSA (H_hotword.pt)
#   2  Generate TTS test set
#   3  Baseline decoding (no hotword bias)
#   4  Hotword-biased decoding (initial alpha = 1.0)
#   5  Grid + Bayesian search for optimal alpha
#   6  Final decoding with best alpha + report
#
# Usage:
#   bash run_hotword_experiment.sh [options]
#
# Options:
#   --stage       <int>   First stage to run (default: 1)
#   --stop-stage  <int>   Last stage to run  (default: 6)
#   --checkpoint  <path>  Model checkpoint path (required for stages 3-6)
#   --lang-dir    <path>  Language directory    (default: data/lang_char)
#   --exp-dir     <path>  Experiment directory  (default: exp/hotword_exp)
#   --tts-backend <str>   edge-tts | pyttsx3   (default: edge-tts)
#   --alpha       <float> Initial hotword weight (default: 1.0)
#   --n-bayesian  <int>   Bayesian optimisation calls (default: 30)
#   --mock                Use mock evaluator (no model needed)
#   --device      <str>   cpu | cuda            (default: cpu)
# ---------------------------------------------------------------------------

set -eou pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
stage=1
stop_stage=6
checkpoint=""
lang_dir="data/lang_char"
exp_dir="exp/hotword_exp"
tts_dir="data/tts_testset"
tts_backend="edge-tts"
alpha=1.0
n_bayesian=30
mock=false
device="cpu"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)       stage="$2";       shift 2 ;;
    --stop-stage)  stop_stage="$2";  shift 2 ;;
    --checkpoint)  checkpoint="$2";  shift 2 ;;
    --lang-dir)    lang_dir="$2";    shift 2 ;;
    --exp-dir)     exp_dir="$2";     shift 2 ;;
    --tts-backend) tts_backend="$2"; shift 2 ;;
    --alpha)       alpha="$2";       shift 2 ;;
    --n-bayesian)  n_bayesian="$2";  shift 2 ;;
    --mock)        mock=true;        shift   ;;
    --device)      device="$2";      shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${exp_dir}"
mkdir -p "${lang_dir}"

echo "============================================================"
echo "CTC+FSA Hotword Experiment"
echo "  stage       = ${stage}"
echo "  stop_stage  = ${stop_stage}"
echo "  checkpoint  = ${checkpoint}"
echo "  lang_dir    = ${lang_dir}"
echo "  exp_dir     = ${exp_dir}"
echo "  tts_backend = ${tts_backend}"
echo "  alpha       = ${alpha}"
echo "  mock        = ${mock}"
echo "  device      = ${device}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage 1: Build Hotword FSA
# ---------------------------------------------------------------------------
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Stage 1: Build Hotword FSA"
  echo "------------------------------------------------------------"

  hotword_fsa="${lang_dir}/H_hotword.pt"
  hotword_dot="${lang_dir}/H_hotword.dot"

  python build_hotword_fsa.py \
    --tokens    "${lang_dir}/tokens.txt" \
    --output    "${hotword_fsa}" \
    --dot-out   "${hotword_dot}" \
    --device    "${device}"

  echo "Stage 1 done: ${hotword_fsa}"
fi

# ---------------------------------------------------------------------------
# Stage 2: Generate TTS Test Set
# ---------------------------------------------------------------------------
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Stage 2: Generate TTS Test Set"
  echo "------------------------------------------------------------"

  python generate_tts_testset.py \
    --output-dir  "${tts_dir}" \
    --tts-backend "${tts_backend}"

  echo "Stage 2 done: TTS test set in ${tts_dir}"
fi

# ---------------------------------------------------------------------------
# Stage 3: Baseline Decoding (alpha = 0, no hotword bias)
# ---------------------------------------------------------------------------
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Stage 3: Baseline Decoding (alpha=0)"
  echo "------------------------------------------------------------"

  baseline_dir="${exp_dir}/baseline"

  decode_args=(
    --manifest-dir   "${tts_dir}"
    --lang-dir       "${lang_dir}"
    --hotword-weight 0.0
    --output-dir     "${baseline_dir}"
    --device         "${device}"
  )

  if [ -n "${checkpoint}" ]; then
    decode_args+=(--checkpoint "${checkpoint}")
  fi

  python decode_with_hotword.py "${decode_args[@]}"

  echo "Stage 3 done: baseline results in ${baseline_dir}"
fi

# ---------------------------------------------------------------------------
# Stage 4: Hotword-biased Decoding (initial alpha)
# ---------------------------------------------------------------------------
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Stage 4: Hotword-biased Decoding (alpha=${alpha})"
  echo "------------------------------------------------------------"

  hw_dir="${exp_dir}/hotword_alpha_${alpha}"

  decode_args=(
    --manifest-dir   "${tts_dir}"
    --lang-dir       "${lang_dir}"
    --hotword-fsa    "${lang_dir}/H_hotword.pt"
    --hotword-weight "${alpha}"
    --output-dir     "${hw_dir}"
    --device         "${device}"
  )

  if [ -n "${checkpoint}" ]; then
    decode_args+=(--checkpoint "${checkpoint}")
  fi

  python decode_with_hotword.py "${decode_args[@]}"

  echo "Stage 4 done: hotword decode results in ${hw_dir}"
fi

# ---------------------------------------------------------------------------
# Stage 5: Grid + Bayesian Search for Optimal Alpha
# ---------------------------------------------------------------------------
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Stage 5: Grid + Bayesian Search for Optimal Alpha"
  echo "------------------------------------------------------------"

  search_dir="${exp_dir}/weight_search"

  search_args=(
    --manifest-dir "${tts_dir}"
    --lang-dir     "${lang_dir}"
    --hotword-fsa  "${lang_dir}/H_hotword.pt"
    --output-dir   "${search_dir}"
    --n-bayesian   "${n_bayesian}"
    --device       "${device}"
  )

  if [ -n "${checkpoint}" ]; then
    search_args+=(--checkpoint "${checkpoint}")
  fi

  if [ "${mock}" = true ]; then
    search_args+=(--mock)
  fi

  python search_hotword_weight.py "${search_args[@]}"

  # Extract best alpha from summary
  if [ -f "${search_dir}/summary.json" ]; then
    best_alpha=$(python -c "
import json
with open('${search_dir}/summary.json') as f:
    d = json.load(f)
print(d['best_alpha'])
")
    echo "Best alpha found: ${best_alpha}"
  else
    best_alpha="${alpha}"
    echo "Summary not found; using default alpha=${alpha}"
  fi

  echo "Stage 5 done: search results in ${search_dir}"
fi

# ---------------------------------------------------------------------------
# Stage 6: Final Decoding with Best Alpha + Report
# ---------------------------------------------------------------------------
if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Stage 6: Final Decoding with Best Alpha"
  echo "------------------------------------------------------------"

  # Try to read best alpha from Stage 5 output
  search_dir="${exp_dir}/weight_search"
  if [ -f "${search_dir}/summary.json" ]; then
    best_alpha=$(python -c "
import json
with open('${search_dir}/summary.json') as f:
    d = json.load(f)
print(d['best_alpha'])
")
  else
    best_alpha="${alpha}"
    echo "Warning: no search summary found; using alpha=${alpha}"
  fi

  final_dir="${exp_dir}/final_best_alpha"

  decode_args=(
    --manifest-dir   "${tts_dir}"
    --lang-dir       "${lang_dir}"
    --hotword-fsa    "${lang_dir}/H_hotword.pt"
    --hotword-weight "${best_alpha}"
    --output-dir     "${final_dir}"
    --device         "${device}"
  )

  if [ -n "${checkpoint}" ]; then
    decode_args+=(--checkpoint "${checkpoint}")
  fi

  python decode_with_hotword.py "${decode_args[@]}"

  echo ""
  echo "============================================================"
  echo "EXPERIMENT COMPLETE"
  echo "  Results    : ${final_dir}"
  echo "  Search log : ${search_dir}"
  echo "  Best alpha : ${best_alpha}"
  echo "============================================================"
fi
