# RL-EMPC-RampMetering

Using Reinforcement learning (RL) based event-triggered Model predictive control (EMPC) to solve the ramp metering problem.

# File Explain

## src: core code

agent: RL agent

mpc: mpc controller

opt: optimization controller

simulation: model and env

## docs: related documents

## tests: test code

## examples: example code


# Track the Deer 
This repository is a part of the research : 
**A Comprehensive Evaluation of YOLO-based Deer Detection Performance on Edge Devices**

This github repository contains the source code used for training and evaluation of YOLO models, exporting them to ONNX and using the ONNX models for inference testing on edge devices; NVIDIA Jetson AGX Xavier and Raspberry Pi 5.

## Directory Tree (Main Structure)

```
track-the-deer/
├── cfg/                # Configuration files for training and datasets
├── data/               # Datasets and raw data
├── experiments/        # Training outputs and logs
├── inference/          # Inference scripts for different devices
├── notebooks/          # Jupyter notebooks and analysis utilities
├── onnx_models_op13/   # Exported ONNX models
├── output_images/      # Images generated from inference and annotation
├── runs/               # Detection results
├── src/                # Source code for training and testing
├── requirements.txt    # Python dependencies
├── Makefile            # Build and automation commands
└── README.md           # Project documentation
```


## NOTES

### Dataset
For dataset, refer to https://www.kaggle.com/datasets/winnerbishal/deer-cameratraps .

### How To

The structure streamlines the training process. For instance, models can be trained with ease using commands like `make run CFG=cfg/run_Lila.v1.3k.yaml` . The content of [cfg/run_Lila.v1.3k.yaml](./cfg/run_Lila.v1.3k.yaml) is as follows :

```yaml
model: yolov9s.pt 
data: cfg/Lila.v1.3k.data.yaml
epochs: 100
batch: 48
imgsz: 640
workers: 32
project: experiments
name: Lila.v1.3k_y9c_ep100
```

After training, `make run export_onnx` to generate ONNX models.

Inference in NVIDIA Jetson AGX Xavier : [jetson_cuda_inference.py](./inference/jetson_cuda_inf.py)

Inference in Raspberry Pi : [pi_inference.py](./inference/pi_inference.py)

Inference is performed on image batches while recording the predictions, inference, pre-processing and post-processing times, and other performance metrics. Finally, a csv file is returned.




