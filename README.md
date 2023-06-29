# Solution for LPCVC 2023

## Team Members
Student at Telecom Paris - First-year final project
- [Fan Hong](github_profile_link)
- [Noé Vernier](github_profile_link)
- [Ulysse Ristorcelli](github_profile_link)
- [Jacques Sun](github_profile_link)

## Project Description

The LOW-POWER COMPUTER VISION CHALLENGE 2023 project is a competition focused on achieving efficient and accurate understanding of disaster scenes using low-power edge devices for computer vision. The main objective of this competition is to improve semantic segmentation on an embedded device (NVIDIA Jetson Nano 2GB Developer Kit) using PyTorch models. Participants will develop models capable of automatically analyzing images captured by unmanned aerial vehicles (UAVs) in disaster-stricken areas.

UAVs equipped with inexpensive sensors provide imagery of disaster areas that are difficult to access for humans. However, their processing capability is limited due to energy-constrained resources and low-compute devices, resulting in delays in analysis and longer response times for relief operations. The competition aims to promote the use of on-device computer vision on UAVs, addressing challenges related to power consumption and latency.

## Objective

The main objective of this project is to develop efficient semantic segmentation models for automatically analyzing disaster scenes from UAV-captured images. We will use the NVIDIA Jetson Nano 2GB Developer Kit as the embedded device to run the developed models.

## Requirements

- Python 3.6 
- PyTorch 1.11.1
- NVIDIA Jetson Nano 2GB Developer Kit

## Installation

1. Clone this GitHub repository to your local machine:

```bash
git clone https://github.com/SunJacques/LPCVC-2023.git
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this competition consists of 1,700 samples of images collected by UAVs in disaster-affected areas. The dataset will be provided to the participants and should be placed in the `data/` directory at the root of the project.

## Model Training

To train the model, run the training script `train.py`:

```bash
python train.py
```

The trained model will be saved in the `checkpoints/` directory.

## Model Evaluation

To evaluate the model, run the evaluation script `evaluate.py`:

```bash
python evaluate.py
```

The evaluation results will be displayed on the screen.

## Results

| Architecure                 | Backbone       | Inference time on A100 (ms) | Dice |Inference time on Jetson Nano (ms) | Score|
|-----------------------------|----------------|-----------------------------|------|-----------------------------------|------|
| UNet (128, no augmentation) | /              | 0.87                        | 0.43 | /                                 | /    |
| UNet (128)                  | /              | 0.85                        | 0.52 | /                                 | /    |
| UNet (128)                  | ImageNet       | 0.55                        | 0.55 | /                                 | /    |
| UNet (256)                  | ImageNet       | 0.65                        | 0.6  | /                                 | /    |
| UNet (256)                  | MobileNetV2    | 0.94                        | 0.65 | /                                 | /    |
| UNet (128)                  | EfficientNetB4 | 1.4                         | 0.6  | /                                 | /    |
| FPN (128)                   | EfficientNetB3 | 0.8                         | 0.63 | 36                                | 18   |
| FPN (256)                   | EfficientNetB3 | 1.7                         | 0.65 | /                                 | /    |
| FPN (256)                   | EfficientNetB0 | 1.1                         | 0.63 | /                                 | /    |
| FPN (256)                   | MobileNetV2    | 0.8                         | 0.65 | 50                                | 13   |
| FastCNN (256)               | /              | 0.37                        | 0.4  | /                                 | /    |
| PSPNet (256)                | MobileNetV2    | 0.28                        | 0.55 | 14                                | 39   |
| PSPNET (256)                | MobileNetV3    | 1                           | 0.56 | /                                 | /    |

## License

This project is distributed under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
