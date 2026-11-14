MiniQwen Spatial Grounding

Investigating spatial grounding and coordinate-conditioned visual feature selection in a lightweight vision-language model.

Overview

This project investigates whether a lightweight ResNet18–Qwen2.5-1.5B-Instruct architecture can reliably associate spatial coordinates with object attributes in synthetic CLEVR scenes.

The experiments examine full-grid visual representations, bilinear sampling, unconstrained learned attention, and spatially biased attention.

Key Results

Metric	Result
Final correct-coordinate accuracy	62.12%
Wrong-projector accuracy	20.17%
Alternative-target accuracy	56.31%
Coordinate sensitivity gap	41.96 percentage points
Wrong-condition prediction agreement	98.52%

The final evaluation used 5,658 questions across 498 CLEVR validation images.

The results indicate that coordinate-conditioned visual selection substantially influences target recognition under spatially biased attention. Cross-phase architectural comparisons remain subject to differences in experimental configurations.

Architecture

* Vision encoder: Pretrained ResNet18 (frozen)
* Language model: Qwen2.5-1.5B-Instruct (frozen backbone)
* Adaptation: LoRA (rank 8, alpha 16)
* Visual projection: Trainable projection and spatial attention modules
* Dataset: CLEVR

Installation

Install the Python dependencies:

python -m pip install -r requirements.txt

The CLEVR dataset and pretrained model weights must be obtained separately.

Research Report

See report.md for the complete methodology, experimental results, ablation studies, statistical analysis, and limitations.

Reproducibility

The repository contains training, evaluation, and statistical analysis scripts corresponding to the experimental phases.

Some scripts require configuration of local dataset paths, model paths, and checkpoint locations before execution.

Limitations

The experiments use a limited training scale and synthetic CLEVR scenes. Architectural comparisons across phases are not fully controlled, and the validation split was reused during development.

License

MIT License.