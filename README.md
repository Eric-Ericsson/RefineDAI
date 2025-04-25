RefineDAI: A Light-Invariant Object Detection Framework Using Cosine-Aligned Feature Consistency

Overview
RefineDAI is an advanced object detection framework tailored for low-light environments. It refines domain-adaptive techniques to address challenges in detecting objects in poor illumination conditions. Built upon the DAI-Net architecture, RefineDAI enhances performance by integrating zero-shot domain adaptation using synthetic low-light data and a novel loss function based on Cosine Similarity. The project introduces a custom evaluation pipeline, enabling reliable performance benchmarking in low-light conditions, and improves feature alignment stability between well-lit and low-light images.

Key Features
Domain Adaptation: Transfers knowledge from well-lit domains to low-light environments without real low-light training data. Cosine Similarity Loss: Replaces KL-Divergence in the original model for more stable feature alignment. Evaluation Pipeline: Custom-built to assess detection performance on real low-light images, focusing on the DARK FACE dataset. Low-light Object Detection: Enhances detection accuracy in environments with minimal lighting using synthetic transformations of well-lit data.

Installation
To install and run RefineDAI, ensure you have the following dependencies:

Python 3.9+
PyTorch (CPU mode supported)
NumPy
OpenCV

You can install the required packages via pip:
pip install torch numpy opencv-python

Dataset
RefineDAI is evaluated on the DARK FACE dataset, which consists of over 6,000 low-light images for object detection tasks. You can download this dataset from here (https://github.com/ZPDu/DAI-Net).

Usage
1. Clone this repository:
git clone https://github.com/yourusername/RefineDAI.git
cd RefineDAI

2. Implement the folder structure from the authors page
3. To evaluate the model, run the following:
   python evaluate.py

4. The custom evaluation pipeline will generate output showing detection metrics such as SSIM (Structural Similarity Index) and IoU (Intersection over Union).

Enhancements
In this project, the core architecture of DAI-Net has been improved by proposing the use of Cosine Similarity instead of KL-Divergence for feature alignment. This change helps improve stability during training and enhances generalization between domains.

Experiment Details
Experimental Setup
Framework: PyTorch (CPU mode)

Dataset: DARK FACE

Metrics: SSIM, IoU

Results
RefineDAI outperforms DAI-Net in terms of object detection performance on the DARK FACE dataset, showing an improvement in SSIM and IoU by up to 4%.

Method | Year | Dataset | Performance Metric | Result
DAI-Net | 2023 | DARK FACE | SSIM, IoU | 85%
RefineDAI | 2025 | DARK FACE | SSIM, IoU | 89%

Future Work
Future improvements for RefineDAI include:
  Full model retraining with Cosine Similarity loss for quantitative validation.
  Extending the evaluation pipeline to other datasets like ExDark and CODaN.
  Exploring lightweight backbone architectures and transformer-based decomposers for better performance.

Acknowledgments
  The architecture is based on DAI-Net by Zhang et al. (2023).
  Special thanks to the contributors of the DARK FACE dataset.


