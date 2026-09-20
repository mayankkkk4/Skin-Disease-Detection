# 🩺 Skin Disease Detection Using Machine Learning

This project identifies 7 types of skin diseases using advanced Convolutional Neural Networks (CNN) combined with an ensemble model that integrates the strengths of VGG16, InceptionV3, and DenseNet architectures to improve detection accuracy. The system processes dermatoscopic images to recognize patterns and classify diseases effectively.

To make the model accessible, a simple Tkinter-based GUI is implemented allowing users to upload skin images easily and receive diagnostic predictions.

The dataset used is the HAM10000 dataset from Harvard Dataverse, containing over 10,000 labeled images of skin lesions.

## 🌟 Overview

Many people suffer from various skin diseases. Traditional diagnosis is often time-consuming and requires expert knowledge. This project uses deep learning to automatically analyze skin images and help with early disease detection.

## 🔍 Features

Detects 7 skin disease types:

  - Actinic Keratoses
  
  - Basal Cell Carcinoma
  
  - Benign Keratosis
  
  - Dermatofibroma
  
  - Melanoma
  
  - Melanocytic Nevi
  
  - Vascular Lesions

- Custom CNN model from scratch

- Ensemble of VGG16, InceptionV3 & DenseNet models for better accuracy

## 📚 Dataset

Uses the HAM10000 dataset from Harvard Dataverse

Contains 10,015 labeled dermatoscopic skin images.

## 🛠️ Requirements

🐍 Language: Python

📝 IDE: Jupyter Notebook

📦 Libraries: pandas, numpy, matplotlib, seaborn, sklearn, tensorflow, keras, PIL, tkinter

## 🧰 Methodology

  📥 Data Collection from HAM10000

  🧹 Data Cleaning & Augmentation

  ✂️ Split dataset into training, validation & test sets

  🏗️ Build CNN architecture with convolution, pooling, normalization & dropout layers

  🚂 Train CNN and pre-trained models (VGG16, DenseNet, InceptionV3)

  🤝 Combine pre-trained models with ensemble weighted average

  📊 Evaluate models using accuracy, precision, recall & F1-score

  🖼️ Developed a simple Tkinter GUI for users to upload skin images and get predictions

## ⚙️ Usage

Run the Jupyter notebooks for data preprocessing, model training, and evaluation.
