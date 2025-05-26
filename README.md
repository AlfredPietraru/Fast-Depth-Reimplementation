# FastDepth Reimplementation

This project is a reimplementation of the **FastDepth** paper for real-time monocular depth estimation. The pipeline is designed using the latest versions of Python and PyTorch (as of May 2025):

- **Python:** 3.12.3  
- **PyTorch:** 2.6.0+cu118

## Project Overview

The goal of this project is to replicate the training and ONNX conversion pipeline of FastDepth, adapting it to modern PyTorch APIs. The model uses **MobileNetV2** as a lightweight encoder and a custom decoder as proposed in the original paper.

- **Original GitHub Repo:** [https://github.com/dwofk/fast-depth](https://github.com/dwofk/fast-depth)  
- **Official Project Website:** [https://fastdepth.mit.edu/](https://fastdepth.mit.edu/)


## Implementation Highlights

- **Backbone:** MobileNetV2 (pretrained on ImageNet).
- **Dataset:** [NYU Depth V2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) from Kaggle.
- **Training:** Pipeline supports training and validation using PyTorch DataLoaders.
- **Export:** Model can be exported to **ONNX** format for deployment in libraries like OpenCV.

## Training Configuration

| Hyperparameter      | Value                                    |
|---------------------|------------------------------------------|
| Optimizer           | SGD (momentum=0.9, weight_decay=1e-4)    |
| Learning Rate       | 1e-3                                     |
| Batch Size          | 8                                        |
| Epochs              | 10 (initial run)                         |
| Scheduler           | ReduceLROnPlateau (disabled via `patience=30`) |

> Note: The learning rate scheduler was mostly unused as the model converged early. In future updates, a flag will be added to explicitly enable/disable it.

## Loss Function

The loss function is a weighted combination of several components, inspired by more recent work on depth estimation ([arXiv:2404.07686](https://arxiv.org/abs/2404.07686)):

- **Mean Absolute Error (MAE)**
- **Gradient Edge Loss**
- **Structural Similarity (SSIM) Loss**

This composite approach balances per-pixel accuracy with structural coherence and edge sharpness.


## Results

- **Training time:** ~1 hour for 10 epochs on a consumer-grade GPU.
- **Accuracy:** ~50–55% depending on scene complexity.
- **Sample Output:** See `output.png` in this repository.

> A full training run (≈60 epochs) is expected to take around 5–6 hours.


## Files
- `fast_depth.ipynb`: Training script with custom loss and evaluation.
- `fast_depth.onnx`: Converts the model to ONNX format.
- `output.png`: Sample prediction.


## References
1. Wofk, D., Ma, F., Yang, T. J., Karaman, S., & Sze, V. (2019). FastDepth: Fast monocular depth estimation on embedded systems. *IEEE ICRA*. [[Link](https://fastdepth.mit.edu/)]
2. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*. [[Link](https://arxiv.org/abs/1801.04381)]
3. Hafeez, M. A., et al. (2024). Depth Estimation using Weighted-loss and Transfer Learning. *arXiv:2404.07686*. [[Link](https://arxiv.org/abs/2404.07686)]


## Future Work
- Add flag for enabling/disabling learning rate scheduler.
- Explore mixed-precision training for improved performance.
- Enhance ONNX export compatibility with OpenCV DNN module.
- Experiment with alternative decoders for reduced model size and latency.