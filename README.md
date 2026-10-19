# FuncFlow

**Hierarchical Structural Conditioning for Controllable Protein Backbone Generation with SE(3) Flow Matching**

FuncFlow extends [FrameFlow](https://github.com/microsoft/protein-frame-flow) with hierarchical structural conditioning for controllable protein backbone generation.

The framework introduces three levels of structural information:

- **M1 — Residue-level conditioning:** incorporates DSSP-derived secondary-structure labels (helix, strand, and coil).
- **M2 — SSE graph-to-node conditioning:** represents secondary-structure elements (SSEs) as a graph and incorporates graph-derived features into residue-level representations.
- **M3 — SSE graph-to-pair conditioning:** additionally models relationships between SSEs to guide higher-order structural organization.

## Overview

FuncFlow aims to improve structural controllability while preserving the geometric modeling capabilities of SE(3) flow matching.

The framework evaluates both local secondary-structure consistency and higher-order relationships, including SSE contacts and relative orientations.

## Preliminary Results

The following results were independently re-evaluated using 600 previously generated E7 backbone structures, comprising 40 validation profiles and five samples per profile for each conditioning setting.

| Conditioning | DSSP Accuracy | SSE Contact F1 | Orientation Accuracy |
|---|---:|---:|---:|
| Correct | 0.8776 | 0.6861 | 0.7146 |
| Shuffled | 0.8500 | 0.2543 | 0.4267 |
| None | 0.8605 | 0.3067 | 0.4375 |

These results establish reproducibility of the evaluation on saved structures, not end-to-end reproducibility of model training and generation.

## Reproducibility

Evaluation requires matching generated structures to the validation profiles used during inference.

For the E7 experiments, the original evaluation used:

- 40 validation profiles
- 5 generated structures per profile
- `max_num_res: 512`
- The original validation metadata and filtering configuration

Using different metadata or filtering parameters may change the correspondence between generated structures and target conditions, resulting in incorrect evaluation scores.

## Installation

FuncFlow builds on the original FrameFlow codebase. Environment setup and dependency instructions will be finalized after compatibility checks.

## Training and Inference

Training and inference configurations are provided in `configs/`. Model checkpoints and any required datasets must be supplied separately.

Executable reproduction commands will be documented after verification against the released source code.

# Run from the FuncFlow repository root.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Set this to the E7 checkpoint on your system.
CKPT="/path/to/e7/last.ckpt"

# Correct pair conditioning
SSE_PAIR_MODE=correct \
PAIRED_INFERENCE_SEED=123 \
python experiments/inference_se3_flows.py \
  --config-name=inference_e7_paper \
  inference.ckpt_path="$CKPT"

# Shuffled pair conditioning
SSE_PAIR_MODE=shuffled \
PAIRED_INFERENCE_SEED=123 \
python experiments/inference_se3_flows.py \
  --config-name=inference_e7_paper \
  inference.ckpt_path="$CKPT"

# Pair conditioning disabled
SSE_PAIR_MODE=none \
PAIRED_INFERENCE_SEED=123 \
python experiments/inference_se3_flows.py \
  --config-name=inference_e7_paper \
  inference.ckpt_path="$CKPT"

## Acknowledgements

FuncFlow is based on the Microsoft FrameFlow implementation. We acknowledge the original FrameFlow authors and the OpenFold project.

Please retain the original license, copyright notices, and third-party attribution.

## License

This repository retains the original FrameFlow MIT license and the existing OpenFold attribution in `NOTICE.md`.

## Citation

Please cite the original FrameFlow publications when using the underlying implementation. A FuncFlow-specific citation will be added when its publication details are available.
