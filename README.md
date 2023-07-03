# XRANet: An eXtra-wide, Residual and Attention-based deep convolutional neural network for semantic segmentation

## Project Description

XRANet is an innovative deep learning architecture designed to enhance the performance of various computer vision tasks, with a special focus on image segmentation. It leverages the strengths of prominent deep learning models, namely Inception Net, ResNet, and U-Net, along with the attention mechanism to create a comprehensive and highly efficient solution.

One of the unique features of XRANet is the integration of an eXtra-wide mechanism within the encoder block, inspired by the inception layers of the InceptionNet model. This integration allows for efficient extraction of features at multiple scales, a key aspect in achieving accurate segmentation results. Further, XRANet incorporates the attention mechanism in both encoder and decoder segments, ensuring optimal extraction of pertinent features.

This project explores the application of XRANet across multiple datasets, including the Data Science Bowl 2018, a food industry dataset, and a parts inspection dataset. The performance is gauged using the Dice coefficient metric, showcasing the architecture's impressive capability to adapt to different types of data and problems. The experiment also highlights the potential of XRANet to generalize across different data types by evaluating its performance on a subset of the COCO dataset.

## How to Use

The XRANet architecture has been implemented in Python using popular machine learning frameworks. To use the architecture, follow these steps:

1. Clone the repository.
2. Train the model using your training scripts.
5. Evaluate the performance of the model using your scripts.

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Under this license, you are free to:

- Share — copy and redistribute the material in any medium or format.
- Adapt — remix, transform, and build upon the material.

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.

## Citation

If you use this architecture in your research, please cite our paper:

```bib
@article{yourName2023xranet,
  title={XRANet: An eXtra-wide, Residual and Attention-based deep convolutional neural network for semantic segmentation},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2023}
}
```
