# Spiking-PointNet Ablation Studies

## Introduction

This repository reproduces and extends the **Spiking-PointNet** model — a biologically inspired spiking neural network (SNN) version of the classical [PointNet](https://arxiv.org/abs/1612.00593) architecture — designed for efficient processing of 3D point-cloud data such as those in ModelNet40.  

Unlike conventional PointNet, which uses continuous activations and dense matrix operations, Spiking-PointNet replaces them with event-driven spiking neurons that emit discrete pulses when membrane potentials exceed a threshold. This yields sparse, temporally distributed computation that more closely resembles the operation of biological neurons and allows lower-power inference on neuromorphic hardware.

The **original Spiking-PointNet paper** proposed:
- A spiking conversion of PointNet layers using membrane integration and surrogate gradient training.
- Multi-timestep inference to capture temporal coding in static point clouds.
- Comparable accuracy to dense PointNet while achieving significant theoretical energy savings.

---

## Motivation

The Spiking-PointNet paper identified two fundamental challenges in adapting spiking neural networks (SNNs) to 3D point clouds:
1. **The optimization difficulty of large-timestep SNNs due to exploding or vanishing surrogate gradients.**
2. **The high computational cost of PointNet-style architectures when scaled to spiking domains.**
To address these open issues and understand the design trade-offs, our experiments are divided into two categories: Ablation Studies and Hyperparameter Studies.

### 🧩 **Ablation Studies**
These experiments modify the *structure or mechanism* of the model to isolate the importance of specific components proposed or discussed in the original paper.

#### 1. Temporal Integration Sensitivity  
**(Notebook 1 – Ablation on Timesteps)**  
The original work explicitly highlights that *training with large time steps leads to optimization instability*, and therefore introduces the “**trained-less but learning-more**” paradigm — training with one timestep but inferring with multiple ones.  
Our experiment systematically varies \\(T ∈ \\{1,2,4,8\\}\\) to test whether the same performance gains (ensemble-like improvement) still hold, and to quantify where diminishing returns occur.

#### 2. Premapping Impact  
**(Notebook 2 – Ablation on Premap Layer)**  
While the original paper focused on optimizing PointNet’s structure for efficiency, it did not explore early feature remapping before spiking conversion.  
We hypothesize that a lightweight *premap (linear or MLP) transform* can improve the representation stability of input coordinates prior to LIF integration, especially when used under the single-timestep training regime.  
This experiment tests whether such pre-activation normalization helps convergence and accuracy.

#### 3. Feature Transform Regularization  
**(Notebook 5 – Ablation on Feature Transform)**  
The Spiking-PointNet retains PointNet’s use of a **spatial transformer network (STN)** for rotation invariance.  
However, its necessity for spiking neurons was never empirically verified.  
We therefore remove the STN to quantify its real contribution under spiking dynamics and evaluate whether geometric invariance remains preserved without it.

---

### ⚙️ **Hyperparameter Studies**
These experiments focus on *training stability and optimization behavior* of spiking neurons, guided by the original hyperparameter selections in the paper.

#### 4. Temperature Scaling  
**(Notebook 3 – Surrogate Gradient Temperature)**  
In Section 3.3 of the paper, the authors analyze the surrogate gradient parameter \\(k\\), corresponding to the *inverse temperature* controlling spike smoothness.  
They found that **\\(k = 5\\)** (moderate temperature) provides the best trade-off — sharper gradients cause explosion, while smaller \\(k\\) causes vanishing gradients.  
We replicate and extend this by sweeping across multiple temperature values (equivalent to varying \\(k\\)) to confirm the robustness of the surrogate-gradient regime.

#### 5. Membrane Decay Rate  
**(Notebook 4 – Membrane Decay/Leak)**  
The Spiking-PointNet paper adopts the **Leaky Integrate-and-Fire (LIF)** neuron model with leak coefficient \\(λ ≈ 0.2 – 0.25\\) as standard.  
However, the paper does not perform a sensitivity analysis on this decay term.  
We therefore vary the leak/decay rate to study how long-term membrane retention affects accuracy and temporal ensemble effects, especially under their “membrane potential perturbation” enhancement.

---

### 🎯 **Why These Experiments Matter**

Together, these controlled studies directly probe the key mechanisms that the original authors only partially explored:

- **Ablation** reveals which architectural elements (temporal depth, premap, feature transform) truly matter for efficient spiking-point processing.  
- **Hyperparameter** sweeps validate the theoretical design decisions (temperature = k = 5; leak ≈ 0.2–0.25) proposed in the paper and test their generalization under different regimes.

This combination not only replicates the core claims of *Spiking-PointNet* but extends them into a deeper quantitative understanding of stability, accuracy, and neuromorphic efficiency.

---

## Dataset

All experiments are conducted on the **ModelNet40 Normal Resampled** dataset:

👉 [Download here from Kaggle](https://www.kaggle.com/datasets/quynguyen03/modelnet40-normal-resampled)

It contains 40 object categories, each represented as uniformly sampled 3D point clouds with normals.  
We preprocess each sample to \(N=1024\) points and normalize to unit scale.

---

## Methodology

### Experimental Design

We provide **six Jupyter notebooks** corresponding to each experimental phase:

| Notebook | Description |
|-----------|--------------|
| `0_baseline_reproduction.ipynb` | Reproduction of the original PointNet baseline and Spiking-PointNet model to verify correctness. |
| `1_ablation_timesteps.ipynb` | Sweeps over timestep values \(T \in \{1,2,4,8\}\) to observe accuracy vs. temporal depth. |
| `2_ablation_premap.ipynb` | Tests with and without the premap (input feature mapping layer). |
| `3_ablation_temperature.ipynb` | Studies surrogate gradient temperature scaling \(τ\) and its effect on spike sparsity and convergence. |
| `4_ablation_decay.ipynb` | Varies membrane potential decay (leak rate) across runs to analyze temporal memory retention. |
| `5_ablation_feature_transform.ipynb` | Examines removing PointNet’s feature transform regularizer to see its effect on geometric invariance. |

### Implementation Highlights

- **Framework:** PyTorch  
- **Model Wrapping:** All SNN models are implemented via a `SpikeModel` wrapper around the baseline PointNet backbone.  
- **Training:** Uses Adam optimizer with gradient clipping and validation-based checkpointing.  
- **Visualization:** Each notebook generates accuracy/loss curves, summary tables, and comparison bar charts saved under `../log/figures/`.  
- **Automation:** Unified experiment runner (`run_all_studies()`) manages training, evaluation, and caching.

---

## Results and Discussion

After analyzing all experiment logs and plots:

### 1. Baseline vs. Spiking
- The spiking version achieves **comparable final test accuracy** (~88–90%) to the dense baseline (~90–91%), confirming effective surrogate training.
- Training is slower in convergence but notably more stable after 30 epochs, showing consistent upward accuracy trends.

### 2. Timestep Ablation
- Increasing timesteps from **1 → 4** improves accuracy, suggesting temporal integration helps smooth noisy spike activations.  
- Beyond 4 timesteps, accuracy plateaus or slightly declines, likely due to accumulated membrane leakage noise and over-integration.

### 3. Premap Ablation
- Removing the **premap layer** marginally decreases performance but reduces computational overhead.  
- The gain from premap becomes significant when paired with higher timesteps, indicating interaction between spatial encoding and temporal coding.

### 4. Temperature Ablation
- A temperature range of **2.0–3.0** provides the best trade-off between gradient smoothness and spike sparsity.  
- Low temperature (≤1.5) causes vanishing gradients; high temperature (>4.0) destabilizes learning.  
- The training curves show smoother progression for moderate temperatures, aligning with known surrogate-gradient theory.

### 5. Decay Ablation
- Membrane decay around **0.9–0.95** yields best performance — preserving temporal memory without excessive accumulation.  
- Extreme settings (0.5 or 1.0) cause under- or over-firing neurons, confirming the importance of calibrated temporal leakage.

### 6. Feature Transform Ablation
- Removing PointNet’s **feature transform (STN)** reduces invariance to rotations and degrades accuracy by 1–2%.  
- The effect is magnified in spiking models, where spatial perturbations propagate through time.

### 7. Energy Efficiency Insight
- Operation count estimates show **up to 70% theoretical energy reduction** vs. dense PointNet for equivalent accuracy when using 4 timesteps.  
- This validates the promise of spiking architectures for neuromorphic 3D perception.

---

## Conclusion

Our comprehensive ablation reveals that:
- Spiking-PointNet retains PointNet-level accuracy with drastically lower compute costs.
- Optimal configuration lies around **T=4**, **decay≈0.9**, **temperature=2–3**, with premap and feature transforms enabled.
- These experiments clarify how each architectural and temporal factor affects convergence, generalization, and energy efficiency.

The notebooks serve as a reproducible, modular benchmark for future research on **spiking 3D perception networks**, bridging the gap between neuroscience-inspired computation and geometric deep learning.

---

## Usage

1. **Install dependencies**
   ```bash
   pip install torch numpy tqdm matplotlib
   ```

2. **Download dataset**
   ```bash
   wget https://www.kaggle.com/datasets/quynguyen03/modelnet40-normal-resampled
   ```

3. **Run experiments**
   Open any notebook (`0_baseline_reproduction.ipynb` → `5_ablation_feature_transform.ipynb`) and execute sequentially.

4. **Results**
   Figures and metrics will be saved to:
   ```
   /log/figures/<experiment_name>/
   ```

---

## Citation

If you use this work, please cite:
> Ren et al., “Spiking PointNet: Spiking Neural Network for 3D Point Cloud Processing,” *arXiv:2310.07189*, 2023.  
> This repository: “Spiking-PointNet Ablation Studies — Extended Reproduction and Analysis,” 2025.
