
# Brain Tumor Detection Project

This project uses a **YOLOv8** deep learning model to classify brain tumor types from MRI images into four classes: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**. The dataset was sourced from **Kaggle**, and **Roboflow** was used for train-test-validation splitting and image resizing.

## Project Overview

Brain tumor detection is a crucial task in medical imaging. This project utilizes **YOLOv8**, a cutting-edge model for object detection and classification, to detect and classify brain tumors in MRI scans. The model is deployed on Hugging Face for real-time inference and evaluation.

### Key Features
- **Four-class classification**: Glioma, Meningioma, Pituitary, and No Tumor.
- **Model**: YOLOv8 for accurate image classification.
- **Dataset**: Sourced from Kaggle, with image processing handled through Roboflow.

## Live Demo

The model is deployed and available for testing on **Hugging Face**. You can access the live demo here:
[Brain Tumor Detection on Hugging Face](https://huggingface.co/spaces/SameenKhurram/BrainTumor)

## Dataset

The dataset was sourced from **Kaggle** and contains MRI images with bounding boxes for brain tumor classification. You can access the dataset here:
[Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

### Dataset Processing with Roboflow

- **Roboflow** was used for:
  - Splitting the data into training, validation, and test sets.
  - Resizing images to **224x224** for model input compatibility.

The dataset was structured as follows after processing with Roboflow:


## Model

The **YOLOv8** model was used for its speed and accuracy in detecting and classifying brain tumors from MRI images.

### Model Training

- **Architecture**: YOLOv8
- **Classes**:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor

The model was trained using the processed dataset from Roboflow, and the best-performing weights are stored in `best.pt`.

### Performance Metrics

The model's performance was evaluated using:
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

These metrics assess the model’s accuracy in correctly classifying the brain tumor types.

## Results
The model achieves high accuracy in detecting and classifying brain tumors into the four specified categories. The deployed version on Hugging Face allows for real-time interaction with the model.

## Usage

### Installation

To replicate this project locally, install the following dependencies:

```bash
pip install ultralytics
pip install roboflow
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Validate the model
results = model.val(data='path_to_your_yaml.yaml')

# Print the results
print(results)

