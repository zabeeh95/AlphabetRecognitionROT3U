🔠 Handwritten Alphabet Recognition (A–Z) using CNN

A deep learning–based system for recognizing handwritten English alphabets (A–Z) using Convolutional Neural Networks (
CNNs).
The project leverages TensorFlow/Keras for model training and OpenCV for image preprocessing and real-time prediction.

📌 Project Overview

Handwritten character recognition is a fundamental problem in computer vision and pattern recognition with applications
in:

Optical Character Recognition (OCR)

Automated form processing

Assistive technologies

Human–computer interaction

This project focuses on recognizing uppercase handwritten alphabets (A–Z) from grayscale images of size 28×28 pixels.

🚀 Features

✅ Recognition of 26 English alphabets (A–Z)

✅ CNN-based deep learning model

✅ OpenCV preprocessing (grayscale, thresholding, resizing)

✅ Supports image-based prediction

✅ Trained on a large-scale handwritten dataset

✅ Modular and easy-to-extend codebase

🧠 Model Architecture

The CNN architecture consists of:

Convolution + ReLU layers

MaxPooling layers for spatial reduction

Fully connected Dense layers

Softmax output layer for multi-class classification

Output layer:

Dense(26, activation="softmax")

This corresponds to 26 alphabet classes (A–Z).

📂 Project Structure
alphabet-recognition/
│
├── data/ # Dataset (ignored in GitHub)
├── model.keras # Trained CNN model
├── train.py # Model training script
├── predict.py # Image prediction script
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files & folders
└── README.md # Project documentation

📊 Dataset Description

Total samples: 372,450

Image size: 28 × 28 (grayscale)

Labels: 0–25 mapped to A–Z

Pixel intensity range: 0–255

Note: Dataset is not included in the repository due to size constraints.

🖼️ Image Preprocessing Pipeline

Gaussian Blur (noise reduction)

Grayscale conversion

Binary thresholding

Resizing to 28×28

Normalization (/255.0)

Reshaping to (1, 28, 28, 1)

This preprocessing matches the training pipeline, ensuring accurate predictions.

🔤 Label Mapping
word_dict = {
0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F',
6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L',
12:'M', 13:'N', 14:'O', 15:'P', 16:'Q',
17:'R', 18:'S', 19:'T', 20:'U', 21:'V',
22:'W', 23:'X', 24:'Y', 25:'Z'
}

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Predict a handwritten alphabet
python predict.py

Make sure the input image:

Is a single alphabet

Has a clear background

Is centered in the image

📈 Results

High accuracy on clean handwritten samples

Robust performance across multiple alphabet styles

Works best when preprocessing matches training conditions

🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Pandas

📌 Future Improvements

🔹 Support for lowercase alphabets

🔹 Real-time webcam recognition

🔹 Data augmentation

🔹 Model optimization and pruning

🔹 Deployment as a web or mobile app

📜 License

This project is released for educational and research purposes.

👤 Author

Zabeeh Ullah Noor
Computer Vision & Deep Learning Enthusiast