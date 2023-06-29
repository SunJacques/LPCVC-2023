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
- JetPack SDK 4.6.3

## Installation

1. Clone this GitHub repository to your local machine:

```bash
git clone https://github.com/SunJacques/LPCVC-2023.git
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Install JetPack on your NVIDIA Jetson Nano device to ensure you have TensorRT installed. You can follow the official NVIDIA JetPack installation guide for detailed instructions on how to install JetPack on your Jetson Nano.

## Dataset

The dataset used in this competition consists of 1,700 samples of images collected by UAVs in disaster-affected areas. The dataset will be provided to the participants and should be placed in the `dataset/` directory at the root of the project.

```bash
├── dataset
│   ├── train
│   │    ├── IMG
│   │    │   ├── train_0000.png
│   │    │   ├── train_0001.png
│   │    │   └── ...
│   │    └── GT
│   │        ├── train_0000.png
│   │        ├── train_0001.png
│   │        └── ...
│   │   
│   │   
│   └── val
│       ├── IMG
│       │   ├── val_0000.png
│       │   ├── val_0001.png
│       │   └── ...
│       └── GT
│           ├── val_0000.png
│           ├── val_0001.png
│           └── ...
├── README.md
├── train.py
└── ...
```
The training and validation data can be downloaded from [here](https://www.google.com/url?q=https://drive.google.com/drive/folders/1h4AyYiFY-kCU3KT-guP_QTVAcONn7VUD&sa=D&source=editors&ust=1688031905668033&usg=AOvVaw3CtjWrC3FrrQjCh8nlYZ7o).
## Model Training

To train the model, run the training script `train.py`:

```bash
python train.py --datapath dataset/ --epochs 100
```

The trained model will be saved in the `checkpoints/` directory.

## Model Evaluation

To evaluate the model, run the evaluation script `evaluate.py`:

```bash
python evaluate.py --modelpath checkpoint/model.pth --datapath dataset/
```

The evaluation results will be displayed on the terminal.

You can also evaluate the model on the NVIDIA Jetson Nano by using the `evaluate_nano.py` script:

```bash
python evaluate_nano.py --modelpath checkpoint/model.pth --datapath dataset/
```

Make sure you have installed JetPack on your Jetson Nano to ensure compatibility with TensorRT.

## License

This project is distributed under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)