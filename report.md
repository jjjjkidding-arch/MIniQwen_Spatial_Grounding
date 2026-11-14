Investigating Spatial Grounding in a Lightweight Vision-Language Model

Project 1 — MiniQwen Spatial Grounding

Abstract

Reliable spatial grounding requires vision-language models (VLMs) to associate linguistic instructions and spatial coordinates with corresponding visual entities. However, incorporating visual representations into a language model does not necessarily ensure accurate object localization or coordinate-sensitive predictions. This study investigates spatial grounding in a lightweight VLM combining a pretrained ResNet18 visual encoder, Qwen2.5-1.5B-Instruct, and trainable visual projection modules with Low-Rank Adaptation (LoRA).

Using synthetic CLEVR scenes, we progressively evaluate full-grid visual representations, bilinear feature sampling, unconstrained learned attention, and spatially biased attention. Initial experiments reveal weak image-specific grounding, while unconstrained attention exhibits highly concentrated spatial distributions and negligible coordinate sensitivity. Introducing an explicit coordinate-distance penalty is associated with improved attention localization and substantially stronger coordinate-conditioned object recognition.

The final spatially biased model achieves 62.12% accuracy across 5,658 color and shape questions from 498 validation images. A controlled intervention that changes projector coordinates while preserving textual coordinates reduces original-target accuracy to 20.17%, yielding a 41.96 percentage-point difference (95% confidence interval [39.84, 44.10]). Under the same intervention, alternative-target accuracy reaches 56.31%, indicating systematic changes in object selection rather than a simple degradation in performance.

These findings suggest that explicit spatial inductive biases can facilitate coordinate-conditioned visual selection in lightweight VLMs. Nevertheless, differences in experimental configurations and limited task coverage warrant further controlled evaluation.

Keywords: Visual Grounding, Vision-Language Models, Spatial Attention, Inductive Bias, Coordinate Conditioning
⸻

1. Introduction

1.1 Background and Motivation

Visual grounding refers to the ability of an artificial intelligence system to associate linguistic expressions, concepts, or instructions with corresponding entities or regions in visual input[1]. Rather than merely recognizing objects within an image, a visually grounded model must establish meaningful connections between language and visual information. 

Besides that, spatial understanding is particularly important in applications requiring interaction with physical environments. In robotics and autonomous navigation, intelligent systems must identify objects, estimate their locations, and reason about spatial relationships to support manipulation, obstacle avoidance, and task planning. 

Recent advances in vision-language models (VLMs) have enabled the integration of visual representations with large language models, allowing systems to answer questions about images and perform multimodal tasks.However, successful image-level recognition does not necessarily imply reliable spatial grounding. A model may identify the objects present in an image while struggling to associate a specific location with the correct object.

This distinction motivates the investigation of spatial feature selection mechanisms in lightweight VLM architectures, in particular, understanding how visual features are selected and conditioned on spatial information may help identify architectural limitations and inform the design of more reliable multimodal systems.

1.2 Problem Statement

VLMs typically connect visual encoders and language models through intermediate bridging projection, however, visual information does not guarantee that the model can reliably link specific spatial coordinates with corresponding objects.

As a result, this project investigates coordinate-conditioned object-attribute recognition using synthetic CLEVR scenes [2]. This examines whether different visual feature selection mechanisms, especially learned attention and explicit spatial inductive bias, will improve model's sensitivity to object coordinates. 

Through controlled coordinate intervention experiments, the study distinguishes general visual dependence from spatially specific grounding and evaluates the contribution of the visual projection pathway to object selection


1.3 Research Questions

* RQ1: Can a lightweight VLM reliably associate spatial coordinates with object attributes?
* RQ2: How do different visual feature-selection mechanisms affect coordinate sensitivity?
* RQ3: Does explicit spatial bias improve localization and coordinate-conditioned recognition?
* RQ4: To what extent do projector coordinates influence predictions independently of textual coordinates?

1.4 Contributions

This study presents a systematic investigation of spatial grounding in a lightweight vision-language model, with three principal contributions:

1. Architectural investigation: Multiple visual-language bridging mechanisms is tested from MLP-based projection and coordinated-conditioned feature selection to learned single-head attention and spatially biasd attention.

2. Component-level diagnostics: 
 We examine the capabilities of the pretrained ResNet18 visual encoder and Qwen language model independently to investigate potential sources of grounding failure and distinguish component-level limitations from visual-language integration issues

3. Controlled evaluation of spatial grounding: Visual-input and coordinate-intervention experiments are conducted, analyze attention localization, and estimate statistical uncertainty using image-clustered bootstrapping. These experiments provide evidence that coordinate-conditioned visual selection substantially influences object-attribute predictions under spatially biased attention.


⸻

2. Methodology

2.1 Dataset and Task Construction

Experiments were conducted using the CLEVR dataset [2], which contains synthetic scenes with objects varying in color, shape, size, material, and spatial arrangement.

The initial evaluation used relational questions to assess general spatial grounding. Subsequent experiments focused on coordinate-conditioned color and shape recognition, where the model was given normalized image coordinates and asked to identify the corresponding object’s attribute.

The final evaluation consisted of 5,658 questions across 498 validation images, with alternative object coordinates used to test spatial sensitivity.


2.2 Base Model Architecture

The model integrates a pretrained ResNet18 visual encoder with Qwen2.5-1.5B-Instruct through a trainable visual projection module.

ResNet18 extracts spatial features from input images, which are transformed by the projector into embeddings compatible with Qwen’s hidden dimension. The language model backbone and visual encoder remain frozen, while the projector and LoRA adapters (rank = 8, alpha = 16) are trained.

This modular architecture enables comparisons of different spatial feature-selection mechanisms while retaining the same underlying vision-language framework.
2.3 Spatial Feature-Selection Mechanisms

2.3.1 Full-Grid Representation

The full-grid approach preserves spatial features extracted by ResNet18, allowing visual information from multiple image regions to be represented. Target coordinates are provided to condition the model’s predictions on a specified location.

2.3.2 Bilinear Sampling

Bilinear sampling extracts a feature vector at a specified coordinate by interpolating between neighboring positions in the spatial feature grid. This provides a direct, differentiable mechanism for coordinate-conditioned visual feature selection.

2.3.3 Learned Attention

A single-head attention mechanism uses learned query, key, and value projections to aggregate spatial visual features. Attention weights are computed through query-key similarity, allowing the model to select relevant image regions without an explicit spatial locality constraint.

2.3.4 Spatially Biased Attention

Spatially biased attention introduces an explicit distance penalty into the attention scores:

$$
s_i = \frac{q^\top k_i}{\sqrt{d}} - \lambda |p_i-c|^2
$$

$$
a_i = \operatorname{softmax}(s)_i
$$

Here, (p_i) denotes the position of spatial feature (i), (c) is the target coordinate, and (\lambda) controls the strength of the spatial bias. This mechanism encourages attention toward features near the requested location while retaining learned feature-dependent attention.

2.4 Training Configuration

The pretrained ResNet18 encoder and Qwen2.5-1.5B backbone were kept frozen, while the visual projection modules and LoRA parameters were trained. Table 1 summarizes the training configuration of the final spatially biased attention model. Earlier experimental configurations are provided in Appendix A.

Table 1. Training configuration for Phase 2B.1

|Hyperparameter              |Value                       |
|----------------------------|----------------------------|
|Vision encoder              |Pretrained ResNet18 (frozen)|
|Language model              |Qwen2.5-1.5B-Instruct       |
|Language model adaptation   |LoRA (rank = 8, alpha = 16) |
|Spatial bias coefficient (λ)|10                          |
|Projector learning rate     |2 × 10⁻⁴                    |
|LoRA learning rate          |2 × 10⁻⁵                    |
|Batch size                  |2                           |
|Training epochs             |1                           |
|Training scenes             |2,000                       |
|Validation scenes           |500                         |

2.5 Evaluation Protocol

Model performance was evaluated using candidate-answer accuracy under different visual and coordinate conditions. These included correct visual input, mismatched or zero visual input, and target-altered coordinates modifications.

The final evaluation additionally isolated the projector-coordinate pathway by changing projector coordinates while keeping textual coordinates fixed. Statistical uncertainty was estimated using image-clustered bootstrap resampling with 5,000 iterations and 95% confidence intervals.


⸻

3. Experimental Progression

3.1 Phase 1 — Initial Spatial Grounding

Objective: Establish a baseline for visual grounding and determine whether the model meaningfully utilizes image information.

Method: Evaluated the initial ResNet18–projector–Qwen architecture on CLEVR relational questions, comparing predictions under correct, shuffled, and zero visual inputs. Additional component-level diagnostics examined ResNet18’s visual representation capabilities and Qwen’s task performance.

Observations: Correct-image accuracy reached 45.80%, compared with 45.47% for shuffled images and 35.41% for zero visual input. These results indicated some visual dependence but weak image-specific grounding, motivating further investigation of spatial feature-selection mechanisms.


3.2 Phase 2A — Explicit Coordinate Conditioning

Objective: Determine whether explicit spatial coordinates improve object-specific visual grounding.

Method: Introduced coordinate-conditioned visual representations, including full-grid features and bilinear sampling, to associate target locations with ResNet18 features.

Observations: The full-grid model achieved 31.46% correct-coordinate accuracy versus 31.19% under wrong coordinates, indicating limited coordinate sensitivity despite improved performance over zero-visual input (22.99%).

3.3 Phase 2B — Unbiased Learned Attention

Objective: Investigate whether learned attention can select spatially relevant visual features without an explicit locality prior.

Method: Implemented single-head attention over ResNet18 spatial features using trainable query, key, and value projections.

Observations: Attention concentrated near image corners, with low normalized entropy (0.0157). Correct-coordinate accuracy was 23.98%, compared with 24.20% under wrong coordinates, indicating negligible spatial sensitivity.

3.4 Phase 2B.1 — Spatially Biased Attention

Objective: Evaluate whether an explicit spatial inductive bias improves attention localization and coordinate sensitivity.

Method: Introduced a coordinate-distance penalty into attention scores, using a fixed spatial bias coefficient of λ = 10.

Observations: Attention localization improved substantially, with normalized entropy increasing to 0.6673 and centroid distance decreasing to 0.1625. Correct-coordinate accuracy reached 62.12%, compared with 20.04% under the original wrong-coordinate condition.

3.5 Final Coordinate-Pathway Ablation

Objective: Isolate the influence of projector coordinates from textual coordinates on object selection.

Method: Changed the projector’s target coordinates while keeping the textual coordinates fixed, and compared predictions against both original and alternative object attributes.

Observations: Original-target accuracy decreased to 20.17%, while alternative-target accuracy reached 56.31%. Predictions agreed 98.52% of the time between the projector-only and combined wrong-coordinate conditions, suggesting that object selection was primarily controlled by the projector pathway.
⸻

4. Results

### 4.1 Cross-Phase Performance

**Table 2. Accuracy across experimental phases**

| Phase | Correct (%) | Wrong (%) | Zero Visual (%) |
|---|---:|---:|---:|
| Phase 1 — Initial Grounding | 45.80 | 45.47 | 35.41 |
| Phase 2A — Full-Grid | 31.46 | 31.19 | 22.99 |
| Phase 2B — Unbiased Attention | 23.98 | 24.20 | 22.25 |
| Phase 2B.1 — Spatially Biased Attention | 62.12 | 20.04 | 22.36 |

The initial architectures demonstrated limited sensitivity to spatial interventions, with correct and wrong conditions producing similar accuracy. Spatially biased attention exhibited substantially greater coordinate sensitivity.

These results are descriptive rather than strictly controlled cross-phase comparisons, as the initial evaluation used relational questions while subsequent phases focused on coordinate-conditioned color and shape recognition.
### 4.2 Attention Localization

**Table 3. Attention localization diagnostics**

| Metric | Unbiased Attention | Spatially Biased Attention |
|---|---:|---:|
| Normalized attention entropy | 0.0157 | 0.6673 |
| Centroid distance | 0.7447 | 0.1625 |
| Peak distance | — | 0.2715 |
| Target attention mass | — | 0.0412 |

Unbiased attention exhibited highly concentrated attention patterns, frequently selecting image corners regardless of target coordinates. Introducing a spatial distance penalty increased attention distribution entropy and reduced centroid distance from the target location, indicating improved spatial localization.

### 4.3 Color and Shape Recognition

**Table 4. Category-specific accuracy for Phase 2B.1**

| Condition | Color (%) | Shape (%) |
|---|---:|---:|
| Correct coordinates | 50.44 | 73.81 |
| Both wrong — original target | 13.89 | 26.19 |
| Both wrong — alternative target | 45.92 | 67.13 |
| Zero visual token | 12.34 | 32.38 |

Shape recognition achieved higher accuracy than color recognition across the evaluated conditions. Under wrong-coordinate interventions, predictions showed substantially greater agreement with alternative-target attributes than with original-target attributes.

### 4.4 Coordinate-Pathway Ablation
To isolate the contribution of projector coordinates, the final evaluation independently manipulated projector and textual coordinate inputs.
**Table 5. Original-target accuracy under coordinate interventions**

| Condition | Accuracy (%) |
|---|---:|
| Correct projector + correct text | 62.12 |
| Wrong projector + correct text | 20.17 |
| Wrong projector + wrong text | 20.04 |
| Zero visual token | 22.36 |

Changing only the projector coordinates reduced original-target accuracy by 41.96 percentage points. The nearly identical performance of the two wrong-projector conditions suggests that projector coordinates exerted a substantially greater influence on object selection than textual coordinate changes.
### 4.5 Alternative-Target Switching

**Table 6. Alternative-target recognition and prediction agreement**

Predictions under wrong-projector conditions were additionally evaluated against the attributes of the alternative object.

| Metric | Result (%) |
|---|---:|
| Wrong projector + correct text — alternative target | 56.31 |
| Wrong projector + wrong text — alternative target | 56.52 |
| Agreement between wrong-projector conditions | 98.52 |
| Alternative − original accuracy gap | 36.14 |

Alternative-target accuracy exceeded original-target accuracy under the projector-only intervention, indicating that changing projector coordinates systematically redirected predictions toward the alternative object.

The 98.52% prediction agreement between wrong-projector conditions further suggests that changing textual coordinates had limited additional influence when projector coordinates were held fixed.

### 4.6 Statistical Uncertainty

Statistical uncertainty was estimated using 5,000 image-clustered bootstrap resamples across 498 validation images, preserving within-image dependencies among questions.

**Table 7. Image-clustered bootstrap 95% confidence intervals**

| Metric | Estimate (%) | 95% CI (%) |
|---|---:|---:|
| Correct-coordinate accuracy | 62.12 | [60.86, 63.38] |
| Wrong-projector — original target | 20.17 | [18.65, 21.73] |
| Wrong-projector — alternative target | 56.31 | [53.71, 58.85] |
| Both wrong — original target | 20.04 | [18.52, 21.62] |
| Both wrong — alternative target | 56.52 | [53.92, 59.12] |
| Zero-visual accuracy | 22.36 | [21.32, 23.39] |
| Wrong-condition prediction agreement | 98.52 | [98.02, 98.95] |
| Correct − wrong-projector gap | 41.96 | [39.84, 44.10] |
| Alternative − original gap | 36.14 | [32.37, 39.84] |

The primary coordinate-sensitivity effect was 41.96 percentage points (95% CI [39.84, 44.10]). The interval excludes zero, supporting a substantial and consistently positive effect of correct projector coordinates on original-target recognition.

Similarly, the alternative-target advantage of 36.14 percentage points (95% CI [32.37, 39.84]) supports the interpretation that projector-coordinate interventions produce systematic target switching rather than merely degrading prediction accuracy.
⸻

5. Discussion

5.1 Why Unbiased Attention Struggled

The unbiased attention mechanism exhibited highly concentrated attention distributions, frequently selecting image corners regardless of target coordinates. This suggests that learned query-key similarity alone was insufficient to establish reliable spatial localization under the experimental training conditions. Consequently, the model showed minimal coordinate sensitivity despite receiving spatial visual features.

5.2 The Role of Spatial Inductive Bias

Introducing an explicit coordinate-distance penalty encouraged attention toward spatial features near the requested location. Compared with unbiased attention, the spatially biased model exhibited improved localization and substantially greater coordinate sensitivity.

These findings suggest that spatial inductive biases can facilitate coordinate-conditioned feature selection in lightweight VLMs, particularly when limited training data or optimization constraints make learning spatial relationships from scratch difficult. However, the experiments do not isolate the contribution of the spatial penalty from all other differences between training configurations.

5.3 Projector Coordinates Versus Text Coordinates

The final ablation demonstrated that changing projector coordinates alone substantially altered object-attribute predictions, while additional changes to textual coordinates had little effect. The high agreement between the two wrong-projector conditions suggests that object selection was primarily determined by the coordinate-conditioned visual projection pathway.

This indicates that spatial information was effectively incorporated into the visual representation, although it does not establish that the language model generally ignores textual spatial instructions.

5.4 Implications for Lightweight VLM Design

The experiments highlight the importance of visual feature-selection mechanisms in spatial grounding. Simply providing spatial features or introducing unconstrained attention did not ensure coordinate-sensitive predictions, whereas incorporating a spatial locality prior was associated with substantially improved performance.

More broadly, the results suggest that carefully designed inductive biases may provide an effective alternative to relying entirely on learned spatial representations, particularly in computationally constrained multimodal architectures. Further controlled experiments would be required to establish how well these findings generalize to more complex visual environments and relational reasoning tasks.
⸻

6. Limitations and Future Work

6.1 Experimental Limitations

Several limitations should be considered when interpreting the results.

Training scale: The final spatially biased attention model was trained on 2,000 CLEVR scenes for one epoch, limiting the ability to assess convergence and performance at larger training scales. The final evaluation contained 5,658 questions across 498 images, providing substantial question coverage but limited scene diversity.

Cross-phase comparability: Experimental phases differed in task formulation, architecture, and training configuration. Consequently, performance differences cannot be attributed exclusively to individual architectural modifications.

Validation reuse: The CLEVR validation split was repeatedly used during model development and evaluation. Although image-clustered bootstrapping estimates sampling uncertainty, it does not eliminate potential selection bias arising from repeated validation reuse.

Architectural confounds: The experiments did not independently isolate all architectural components, including projector design, attention mechanisms, and spatial bias strength. Improvements may therefore reflect interactions among these components.

Task coverage: Evaluation primarily focused on synthetic CLEVR scenes and coordinate-conditioned color and shape recognition. These tasks do not fully capture relational reasoning, natural-image complexity, or spatial grounding in real-world environments.

6.2 Future Directions

Future research could extend the investigation in several directions.

Scaling experiments: Evaluate larger training datasets, longer training schedules, and additional random seeds to determine whether coordinate sensitivity and attention localization improve with scale.

Controlled architectural ablations: Compare unbiased and spatially biased attention under identical training configurations, systematically varying spatial bias strength, projector architecture, and visual feature-selection mechanisms.

Relational reasoning: Extend coordinate-conditioned attribute recognition to multi-object relationships, compositional spatial queries, and more complex reasoning tasks.

Stronger visual representations: Investigate higher-resolution visual features and alternative pretrained encoders to assess whether improved spatial representations enhance grounding performance.

Generalization evaluation: Introduce untouched test sets, more diverse synthetic scenes, and natural-image benchmarks to examine robustness beyond the development distribution.
⸻

7. Conclusion

This study investigated spatial grounding in a lightweight ResNet18–Qwen vision-language model through progressively modified visual feature-selection mechanisms.

Initial architectures exhibited weak spatial sensitivity, while unconstrained attention suffered from attention collapse. Introducing an explicit spatial inductive bias substantially improved coordinate-conditioned object recognition, achieving 62.12% accuracy and a 41.96 percentage-point advantage over the wrong-projector condition.

Controlled coordinate interventions further demonstrated that the visual projection pathway played a dominant role in target selection. Overall, the findings highlight the importance of spatial inductive biases in lightweight VLMs, while motivating further investigation into scalability, relational reasoning, and generalization beyond synthetic environments.
⸻

## References

[1] Kazemzadeh, S., et al. (2014).
    ReferItGame: Referring to Objects in Photographs
    of Natural Scenes. EMNLP.

[2] Johnson, J., et al. (2017).
    CLEVR: A Diagnostic Dataset for Compositional
    Language and Elementary Visual Reasoning. CVPR.
    https://doi.org/10.1109/CVPR.2017.215



Appendix A — Implementation Details

The implementation uses a pretrained ResNet18 visual encoder and Qwen2.5-1.5B-Instruct, connected through trainable visual projection modules. ResNet18 produces a 7 × 7 spatial feature grid with 512-dimensional features, which are projected into Qwen’s 1,536-dimensional embedding space.

The visual encoder and language model backbone remain frozen, while the projector and LoRA adapters are optimized. The final model uses spatially biased attention with a fixed locality coefficient of λ = 10. Training hyperparameters are summarized in Table 1.

Appendix B — Additional Results

Supplementary analyses include attention localization diagnostics, component-level evaluations, and representative model predictions.

The initial experiments revealed weak image-specific grounding, while unconstrained attention exhibited concentration near image corners. Spatially biased attention improved localization and coordinate sensitivity, as quantified in Section 4.

Representative attention heatmaps and example predictions may be included here to illustrate these behaviors.

Appendix C — Reproducibility

Experiments use the CLEVR dataset and a pretrained ResNet18–Qwen2.5-1.5B-Instruct architecture implemented in PyTorch.

The final evaluation consists of 5,658 color and shape questions across 498 validation images. Coordinate interventions compare correct, projector-only wrong, combined wrong, and zero-visual conditions. Statistical uncertainty is estimated using 5,000 image-clustered bootstrap resamples.

Source code, checkpoint references, dependencies, and execution instructions should be documented in the accompanying GitHub repository.