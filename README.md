#FakeFinder: AI vs. Real Image Classification

A PyTorch-based computer vision pipeline that utilizes transfer learning to distinguish between authentic photographs and AI-generated images.


# Objective

With the rapid advancement of generative AI, reliably detecting synthetic media has become a critical challenge. This project implements a deep learning binary classifier trained on the "AI_VS_Real" dataset to accurately identify the origin of an image. The pipeline handles custom data loading, image transformation, model fine-tuning, and performance evaluation.


# Architecture: Why MobileNetV3?

This project leverages **MobileNetV3** for transfer learning rather than heavier models like ResNet50 or VGG16. The rationale for this architectural choice includes:
* **Computational Efficiency:** MobileNetV3 is optimized for high performance with a low parameter count. It utilizes depthwise separable convolutions and squeeze-and-excitation modules, making it highly efficient for inference without requiring massive GPU resources. 
* **Feature Extraction:** By freezing the early layers trained on ImageNet, the model successfully retains robust, generalized feature extractors (like edge and texture detection) while the modified classification head learns the specific artifacts unique to AI-generated images.


# Repository Structure

* **`FakeFinder_TransferLearning.ipynb`**: The primary Jupyter Notebook containing the end-to-end data preparation, model initialization, and training narrative.
* **`helper_utils.py`**: A modular, standalone Python script containing the core backend logic, including custom `DataLoader` inspection, TorchMetrics integration, and Matplotlib prediction visualizations.
* **`images/`**: Contains sample images used for local inference testing.

# Model Performance
The model was evaluated using `torchmetrics` to ensure a comprehensive understanding of its classification capabilities beyond standard accuracy.

**Best Val Accuracy:** 86.55%
**Best Val Precision:** 0.8667
**Best Val Recall:** 0.8655
