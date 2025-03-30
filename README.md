# -Leukemia-detection-from-blood-images-using-CNN-model-MobileNet-
"Leukemia detection from blood images using MobileNet" aims to develop an AI-powered system for automatic leukemia diagnosis. By leveraging  convolutional neural network architecture, the system analyzes blood images to detect abnormal white blood cells, enabling early diagnosis and treatment.
BloodWise - AI-Powered Blood Cell Cancer Classification

Problem Statement:

Early detection of blood cell cancer (Acute Lymphoblastic Leukemia - ALL) is crucial for effective treatment. Manual diagnosis through microscopic examination is time-consuming and requires expertise. This project leverages AI and deep learning techniques to automate and enhance the accuracy of blood cell cancer classification.

Team Members:

1. P. DILEEP KRISHNA


2. K. MAHARSHI


3. D. AJAY DHEERAJ


4. T. HARSHA VARDHAN



Demo Video:https://drive.google.com/drive/folders/1dLbKVNec-dhU0A2aMMugwjHe2fgE0Bsy?usp=sharing

Watch Here

Overview:Blood Cell Cancer Classification Model
This repository contains a deep learning model for classifying blood cell cancer images. The model is built using TensorFlow and Keras, and utilizes the MobileNet architecture.

Key Features:
1. Data Preprocessing: The code includes data preprocessing steps, such as image resizing and data augmentation.
2. Model Training: The model is trained using a dataset of blood cell cancer images, with a custom top-3 accuracy metric.
3. Model Evaluation: The model is evaluated on a validation dataset, with metrics including accuracy, top-2 accuracy, and top-3 accuracy.
4. Model Conversion: The trained model is converted to a TensorFlow Lite (TFLite) model for deployment on mobile devices.

Files:
1. bloodwisepnormalmodel.h5: The trained Keras model.
2. BloodWise.tflite: The converted TFLite model.
3. code.py: The Python code for training and evaluating the model.

Requirements:
1. TensorFlow: Version 2.x
2. Keras: Version 2.x
3. NumPy: Version 1.x
4. Pandas: Version 1.x

BloodWise is a deep learning-based model that classifies blood cell images to detect potential signs of Acute Lymphoblastic Leukemia (ALL). The project utilizes a MobileNet-based Convolutional Neural Network (CNN) for high accuracy and efficiency in medical image classification.

Features:

Preprocessing of Blood Cell Images: Uses TensorFlow's MobileNet for feature extraction.

Deep Learning Model: A CNN model trained on blood cell images for classification.

Automated Classification: Classifies images into normal and cancerous cells.

Data Augmentation: Enhances model performance by artificially expanding the dataset.

Model Deployment: Converts the trained model to TensorFlow Lite for mobile deployment.


Installation:

Clone the Repository:

git clone <repository_link>
cd BloodWise

Install Dependencies:

pip install -r requirements.txt

Run the Model Training:

python train.py

Convert Model to TensorFlow Lite:

python convert_to_tflite.py

Deployment:

The trained model can be deployed on cloud servers or mobile devices using TensorFlow Lite.

The final .tflite model is optimized for edge computing, allowing offline classification.


Open-Source Libraries & APIs Used:

TensorFlow/Keras: Deep learning framework for training and deploying the model.

NumPy & Pandas: Data manipulation and preprocessing.

Scikit-learn: Evaluation metrics and data handling.

Matplotlib & Seaborn: Visualization of training results and model performance.

OpenCV: Image processing and augmentation.


License:

This project is open-source and available under the MIT License.

Contact:

For any queries, contact: 6303501533
Email: dileepkrishnapentela@gmail.com
GitHub: dileepkrishna1106
