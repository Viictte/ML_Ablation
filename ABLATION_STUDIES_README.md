# Spiking-PointNet Ablation Studies

This document provides comprehensive documentation for the ablation studies on Spiking-PointNet.

## Project Setup

### Environment
- Python 3.12.8
- PyTorch 2.9.0 (CPU version)
- NumPy 2.3.4
- tqdm, h5py, matplotlib, nbformat

### Repository
- Cloned from: https://github.com/DayongRen/Spiking-PointNet
- Location: `/home/ubuntu/Spiking-PointNet`

### Dataset
The ModelNet40 dataset should be downloaded and extracted to `data/modelnet40_normal_resampled/`.

Download link: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

**Note**: The dataset download may fail due to server connectivity issues. You may need to download it manually and extract it to the correct location.

## Ablation Studies

### Ablation Study 2.1: Multiple Timesteps vs. Single Timestep

#### Motivation
The original paper's claim that single-timestep training outperforms multi-timestep training is counterintuitive. Typically, longer temporal windows should enrich spiking dynamics. This ablation re-examines whether the observed superiority arises from optimization difficulty, overfitting, or an intrinsic property of SpikePoint's architecture.

#### Experimental Plan
1. Use the official `train_classification.py` script with identical hyperparameters
2. Modify the training flag `--step` from 1 to 4 while keeping `--spike` active
3. Train both models on ModelNet10 and ModelNet40 datasets
4. Evaluate both models at test timesteps 1, 2, and 4

#### Implementation Details

**Training Configuration:**
- Single Timestep (step=1):
  ```bash
  python train_classification.py --model pointnet_cls --log_dir ablation_step1_modelnet40 --spike --step 1 --num_category 40
  ```

- Multiple Timesteps (step=4):
  ```bash
  python train_classification.py --model pointnet_cls --log_dir ablation_step4_modelnet40 --spike --step 4 --num_category 40
  ```

**Testing Configuration:**
For cross-evaluation, test both models at different timesteps (1, 2, 4):
```bash
python test_classification.py --log_dir ablation_step1_modelnet40 --spike --step 1 --num_category 40
python test_classification.py --log_dir ablation_step1_modelnet40 --spike --step 2 --num_category 40
python test_classification.py --log_dir ablation_step1_modelnet40 --spike --step 4 --num_category 40
```

#### Expected Insights
This experiment reveals whether the single-step training advantage is a result of:
- **Gradient stability**: Single timestep may have more stable gradients
- **Over-regularization**: Multi-timestep may overfit to temporal patterns
- **Optimization difficulty**: Longer temporal windows increase complexity

If multi-step training improves or converges more slowly, we can better understand the trade-off between temporal resolution and optimization complexity in SNNs.

#### Key Metrics to Track
1. Training accuracy curves for both configurations
2. Test accuracy at different timesteps (1, 2, 4)
3. Convergence speed (epochs to best accuracy)
4. Final best accuracy comparison

---

### Ablation Study 2.2: Feature Extractor vs. Identity Mapping

#### Motivation
SpikePoint employs an identity mapping instead of PointNet's T-Net to reduce overhead. However, this may limit representational flexibility. Introducing a lightweight linear projection layer could uncover whether early-stage learnable transformations improve accuracy or harm efficiency.

#### Experimental Plan
1. Modify `models/pointnet_cls.py` by replacing the identity mapping with a simple linear layer: `h_i = W * x_i + b`
2. Add an argument `--use_premap` in the training script to toggle between modes
3. Train and evaluate both configurations on the same datasets with fixed timesteps

#### Implementation Details

**Modified Model Architecture:**
Create a new model class `PointNetWithPremap` that includes an optional learnable premap layer:

```python
class PointNetWithPremap(nn.Module):
    def __init__(self, k=40, normal_channel=True, use_premap=False):
        super(PointNetWithPremap, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        
        self.use_premap = use_premap
        
        # Add learnable premap if requested
        if self.use_premap:
            self.premap = nn.Linear(channel, channel)
        else:
            self.premap = None
        
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        # ... rest of the architecture
```

**Training Configuration:**
- Identity Mapping (baseline):
  ```bash
  python train_classification.py --model pointnet_cls --log_dir ablation_identity_modelnet40 --spike --step 1 --num_category 40
  ```

- Learnable Premap:
  ```bash
  python train_classification.py --model pointnet_cls --log_dir ablation_premap_modelnet40 --spike --step 1 --num_category 40 --use_premap
  ```

#### Expected Insights
This ablation will quantify the performance trade-off between learnability and structural simplicity:

- **If learnable premap improves accuracy significantly**: It suggests that identity mapping sacrifices valuable spatial adaptation. The additional parameters (channel × channel + channel) provide meaningful benefit.

- **If accuracy remains similar**: It validates SpikePoint's minimalist design philosophy for neuromorphic efficiency. The identity mapping maintains performance without added complexity.

#### Key Metrics to Track
1. Final accuracy comparison (instance and class accuracy)
2. Parameter count difference
3. Training efficiency (epochs to convergence)
4. Computational overhead

---

### Ablation Study 2.3: Temperature Parameter Analysis

#### Motivation
The temperature parameter in LIF (Leaky Integrate-and-Fire) neurons controls the steepness of the spike activation function. The current implementation uses a default temperature of 5.0. This ablation examines how different temperature values affect gradient flow, spiking dynamics, and model performance.

#### Experimental Plan
1. Test temperature values: [1.0, 3.0, 5.0, 7.0, 10.0]
2. Keep all other hyperparameters constant (step=1, spike=True)
3. Train models on ModelNet40 dataset
4. Compare accuracy, convergence speed, and training stability

#### Expected Insights
- Lower temperatures (e.g., 1.0) produce sharper activation functions, potentially leading to gradient issues
- Higher temperatures (e.g., 10.0) produce smoother activations, which may improve gradient flow but reduce spiking behavior
- Optimal temperature balances gradient flow with spiking dynamics

---

### Ablation Study 2.4: Decay Rate Analysis

#### Motivation
The decay rate in LIF neurons controls the temporal integration properties and membrane potential leakage. The current implementation uses a fixed decay rate of 0.25 in the `mem_update` function. This ablation examines how different decay rates affect temporal dynamics and model performance.

#### Experimental Plan
1. Test decay rate values: [0.1, 0.25, 0.5, 0.75]
2. Modify the `mem_update` function to use different decay rates
3. Keep all other hyperparameters constant (step=1, temp=5.0, spike=True)
4. Train models on ModelNet40 dataset
5. Compare accuracy and temporal integration behavior

#### Expected Insights
- Lower decay rates (e.g., 0.1) mean slower leakage and longer memory
- Higher decay rates (e.g., 0.75) mean faster leakage and shorter memory
- Optimal decay rate balances temporal integration with responsiveness to new inputs

---

### Ablation Study 2.5: Feature Transform Analysis

#### Motivation
PointNet uses a feature transformation network (T-Net) to learn a 64-dimensional feature space transformation. The current implementation uses `feature_transform=True` in the PointNetEncoder. This ablation examines whether this feature transformation is necessary or if it can be removed to simplify the model while maintaining performance.

#### Experimental Plan
1. Compare two configurations:
   - With feature transform: `feature_transform=True` (baseline)
   - Without feature transform: `feature_transform=False`
2. Keep all other hyperparameters constant (step=1, temp=5.0, spike=True)
3. Train models on ModelNet40 dataset
4. Compare accuracy, parameter count, and training efficiency

#### Expected Insights
- If accuracy remains similar without feature transform, it adds unnecessary complexity
- If accuracy drops significantly, it validates the importance of learnable feature space transformations
- Quantifies the performance-complexity trade-off

---

### Ablation Study 2.6: Data Augmentation Analysis

#### Motivation
The current training pipeline uses three data augmentation techniques: random point dropout, random scale, and random shift. This ablation examines the contribution of each augmentation technique individually and in combination to understand which augmentations are most beneficial for model performance.

#### Experimental Plan
1. Test different augmentation combinations:
   - No augmentation (baseline)
   - Only dropout
   - Only scale
   - Only shift
   - Dropout + Scale
   - Dropout + Shift
   - Scale + Shift
   - All augmentations (current default)
2. Keep all other hyperparameters constant (step=1, temp=5.0, spike=True)
3. Train models on ModelNet40 dataset
4. Compare accuracy and generalization

#### Expected Insights
- Reveals which augmentation techniques contribute most to model performance
- Some augmentations may be redundant or even harmful
- Quantifies individual augmentation contributions
- Helps optimize the augmentation pipeline for efficiency

---

## Jupyter Notebooks

### Available Notebooks

1. **0_baseline_reproduction.ipynb**: Complete baseline reproduction of Spiking-PointNet
   - Reproduces the original paper's results
   - Includes training and testing code
   - Visualization of training progress

2. **1_ablation_timesteps.ipynb**: Ablation Study 2.1 implementation
   - Compares single timestep (step=1) vs multiple timesteps (step=4)
   - Cross-evaluation at different test timesteps
   - Comprehensive analysis and visualization

3. **2_ablation_premap.ipynb**: Ablation Study 2.2 implementation
   - Compares identity mapping vs learnable premap
   - Parameter count analysis
   - Performance trade-off evaluation

4. **3_ablation_temperature.ipynb**: Ablation Study 2.3 implementation
   - Temperature parameter analysis for LIF neurons
   - Tests temperature values: [1.0, 3.0, 5.0, 7.0, 10.0]
   - Reveals optimal temperature for gradient flow and spiking dynamics

5. **4_ablation_decay.ipynb**: Ablation Study 2.4 implementation
   - Decay rate analysis for membrane potential leakage
   - Tests decay values: [0.1, 0.25, 0.5, 0.75]
   - Examines temporal integration properties

6. **5_ablation_feature_transform.ipynb**: Ablation Study 2.5 implementation
   - Feature transform necessity analysis
   - Compares with vs without 64-dimensional feature transformation
   - Parameter count and efficiency trade-off evaluation

7. **6_ablation_augmentation.ipynb**: Ablation Study 2.6 implementation
   - Data augmentation contribution analysis
   - Tests different combinations of dropout, scale, and shift
   - Quantifies individual augmentation contributions

### Running the Notebooks

1. Ensure the dataset is downloaded and extracted to `data/modelnet40_normal_resampled/`
2. Start Jupyter notebook:
   ```bash
   cd /home/ubuntu/Spiking-PointNet/notebooks
   jupyter notebook
   ```
3. Open the desired notebook and run cells sequentially

**Note**: Training takes significant time (200 epochs). Consider reducing the number of epochs for initial testing.

---

## Suggestions and Recommendations

### Additional Ablation Studies to Consider

1. **Batch Size Sensitivity**
   - Test different batch sizes (16, 24, 32, 48) to understand memory-accuracy trade-offs
   - Particularly relevant for neuromorphic hardware deployment

2. **Temperature Parameter Analysis**
   - The `temp` parameter (default 5.0) controls the spike activation function
   - Ablate across values: [1.0, 3.0, 5.0, 7.0, 10.0]
   - May reveal optimal settings for gradient flow

3. **Decay Rate in LIF Neurons**
   - Current decay rate is fixed at 0.25 in `mem_update` function
   - Test values: [0.1, 0.25, 0.5, 0.75]
   - Affects temporal integration properties

4. **Feature Transform vs No Feature Transform**
   - The paper uses `feature_transform=True` in PointNetEncoder
   - Compare with `feature_transform=False` to quantify the benefit of the 64-dimensional feature transformation

5. **Data Augmentation Ablation**
   - Current augmentations: random dropout, random scale, shift
   - Test removing each augmentation individually
   - Quantify contribution of each augmentation technique

### Optimization Suggestions

1. **Learning Rate Schedule**
   - Current: StepLR with step_size=20, gamma=0.7
   - Consider: CosineAnnealingLR or ReduceLROnPlateau for potentially better convergence

2. **Early Stopping**
   - Implement early stopping based on validation accuracy
   - Can significantly reduce training time without sacrificing performance

3. **Mixed Precision Training**
   - Use PyTorch's automatic mixed precision (AMP)
   - Can speed up training on compatible hardware

### Experimental Design Improvements

1. **Multiple Random Seeds**
   - Run each experiment with 3-5 different random seeds
   - Report mean and standard deviation for robust conclusions

2. **Statistical Significance Testing**
   - Use paired t-tests to determine if accuracy differences are statistically significant
   - Important for making strong claims about ablation results

3. **Computational Cost Analysis**
   - Track training time, memory usage, and FLOPs
   - Essential for neuromorphic deployment considerations

### ModelNet10 Experiments

For faster iteration and initial validation, consider running all ablation studies on ModelNet10 first:
- 10 classes instead of 40
- Faster training (fewer samples)
- Good proxy for ModelNet40 trends

Command example:
```bash
python train_classification.py --model pointnet_cls --log_dir ablation_step1_modelnet10 --spike --step 1 --num_category 10
```

---

## Expected Results (Based on Paper)

### ModelNet40 Results from Paper

| Configuration | Training Steps | Test Steps | Instance Accuracy |
|--------------|----------------|------------|-------------------|
| Vanilla SNN  | 4              | 1          | 85.59%            |
| Vanilla SNN  | 4              | 2          | 86.58%            |
| Vanilla SNN  | 4              | 4          | 86.70%            |
| Ours (MPP)   | 1              | 1          | 87.72%            |
| Ours (MPP)   | 1              | 2          | 88.46%            |
| Ours (MPP)   | 1              | 4          | **88.61%**        |

### Key Observations from Paper
1. Single-timestep training (step=1) outperforms multi-timestep training (step=4)
2. Models can be tested at different timesteps than they were trained
3. Testing with more timesteps generally improves accuracy
4. The best result (88.61%) uses single-timestep training with 4-timestep testing

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 16`
   - Use CPU mode: `--use_cpu`

2. **Dataset Not Found**
   - Ensure dataset is extracted to correct location
   - Check path: `data/modelnet40_normal_resampled/`

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes the repository root

4. **Slow Training**
   - Use GPU if available
   - Reduce number of workers: `num_workers=2`
   - Consider using pre-processed data: `--process_data`

---

## File Structure

```
Spiking-PointNet/
├── data/
│   └── modelnet40_normal_resampled/  # Dataset location
├── data_utils/
│   └── ModelNetDataLoader.py
├── models/
│   ├── pointnet_cls.py
│   ├── pointnet_utils.py
│   ├── spike_layer_without_MPR.py
│   └── spike_model.py
├── notebooks/
│   ├── 0_baseline_reproduction.ipynb
│   ├── 1_ablation_timesteps.ipynb  # To be created
│   └── 2_ablation_premap.ipynb     # To be created
├── log/
│   └── classification/  # Training logs and checkpoints
├── provider.py
├── train_classification.py
├── test_classification.py
└── ABLATION_STUDIES_README.md  # This file
```

---

## References

1. **Paper**: Spiking PointNet: Spiking Neural Networks for Point Clouds
   - Authors: Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo
   - Conference: NeurIPS 2023
   - URL: https://openreview.net/forum?id=Ev2XuqvJCy

2. **Original Repository**: https://github.com/DayongRen/Spiking-PointNet

3. **ModelNet Dataset**: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

---

## Contact and Support

For questions or issues with the ablation studies, please refer to:
- Original paper: https://openreview.net/forum?id=Ev2XuqvJCy
- GitHub repository: https://github.com/DayongRen/Spiking-PointNet

---

## Acknowledgments

This ablation study framework is built upon the official Spiking-PointNet implementation. The codebase is inspired by the Re-Loss library.
