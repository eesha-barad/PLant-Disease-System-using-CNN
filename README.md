# Plant Disease Detection System

## Overview

This project is a deep learning-based Plant Disease Detection System using a Convolutional Neural Network (CNN). It aims to assist farmers in identifying plant diseases through image classification, enabling timely interventions and preventing crop loss.

## Features

Dataset Exploration: Analyzes and visualizes the dataset.

Data Preprocessing: Applies image augmentation and normalization techniques.

CNN Model Training: Uses TensorFlow and Keras to train a deep learning model.

Performance Evaluation: Assesses model accuracy and loss.

Deployment: Can be integrated into mobile applications for real-time disease detection.

## Installation

Install dependencies using:
```sh
pip install tensorflow numpy matplotlib opencv-python
```

## Dataset Structure

The dataset follows this format:
```sh
/dataset
    /train
        /class_1
        /class_2
        ...
    /validation
        /class_1
        /class_2
        ...
    /test
        /class_1
        /class_2
        ...
```

## Model Architecture

The CNN model follows this structure:

Convolutional Layers (Conv2D): Extracts features from images.

MaxPooling Layers: Reduces dimensionality while retaining important features.

Flatten Layer: Converts feature maps into a dense vector.

Fully Connected Layers (Dense Layers): Classifies images into different disease categories.

Dropout Layers: Prevents overfitting by randomly deactivating neurons during training.

## Training and Performance

The model was trained using TensorFlow/Keras and achieved the following results:

Test Accuracy: 96%

Test Loss: 0.112

## Usage

Clone the repository:
```sh
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

Run the Jupyter Notebook:
```sh
jupyter notebook model.ipynb
```

## Results

The model shows strong generalization capabilities with minimal overfitting.

Performance evaluation metrics confirm its robustness in detecting plant diseases.

## Future Improvements

Expand the dataset to include more plant species.

Optimize model architecture for better generalization.

Develop a mobile app for real-time disease detection.

## Contributing

Contributions are welcome! Open an issue or submit a pull request.
