# MNIST Digit Classifier with CNN

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## 🔍 Project Overview

- 📚 Dataset: MNIST (70,000 grayscale images of handwritten digits)
- 🧠 Model: CNN with two Conv2D and MaxPooling layers, followed by Dense layers
- 🏆 Achieved **98.55%** test accuracy
- 🖼️ Visualizations: Training history, confusion matrix, misclassified digits

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/mnist_cnn_classifier.git
cd mnist_cnn_classifier

#2. Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

#3. Install dependencies
pip install -r requirements.txt

#4. Launch the notebook
code .
# Then open: notebooks/mnist_cnn.ipynb

#📊 Results
Test Accuracy: 98.55%
Misclassification example: True = 3, Predicted = 5
Final Predictions:
Predictions:    [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 5 4]
Ground Truth:   [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]

📁 Folder Structure
mnist_cnn_classifier/
├── notebooks/           # Jupyter notebooks
│   └── mnist_cnn.ipynb
├── data/                # Optional: store downloaded or processed data
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── venv/                # Virtual environment (not pushed to Github

📌 Dependencies
Python 3.11
TensorFlow
NumPy
Matplotlib
Seaborn
scikit-learn
Jupyter

📚 References
MNIST Dataset
Convolutional Neural Networks
TensorFlow Documentation
Dropout: A Simple Way to Prevent Neural Networks from Overfitting

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

Made with ❤️ by [akakinad]