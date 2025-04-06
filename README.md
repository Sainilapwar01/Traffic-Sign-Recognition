
# Traffic Sign Recognition System using CNN

This project implements a **real-time traffic sign recognition system** using a Convolutional Neural Network (CNN) and OpenCV. It detects and classifies traffic signs from live webcam feed, making it suitable for self-driving car prototypes and driver-assistance systems.

---

## Dataset

The model is trained on the **[German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)** dataset from Kaggle. This dataset includes:

- Over 50,000 labeled images
- 43 different traffic sign classes
- Varying lighting conditions, sizes, and angles
- CSV file for labels and metadata

### Preprocessing Steps
- Grayscale conversion
- Histogram equalization
- Normalization (scaling pixel values)
- Resizing to 32x32 pixels

---

## Model Architecture

The CNN model built using Keras includes:

- Convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- Dense (fully connected) layers
- Softmax output layer for classification

The trained model is saved as `model_trained.p` using Python's `pickle` module.

---

## Project Structure

```
.
├── Abstract.pdf            # Project overview and explanation
├── demo.py                 # Model training and saving
├── TSR.py                  # Dataset preprocessing and model setup
├── TSR_Test.py             # Real-time traffic sign detection via webcam
├── model_trained.p         # Trained CNN model
├── labels.csv              # Traffic sign class labels
└── README.md               # Project documentation
```

---

## Real-Time Detection

The `TSR_Test.py` script:

- Captures live frames from a webcam
- Preprocesses each frame
- Predicts the traffic sign using the trained model
- Displays the predicted class and probability

### To Run:
```bash
python TSR_Test.py
```

Press `q` to quit the webcam window.

---

## Class Labels

Class labels (0 to 42) are mapped to human-readable traffic sign names using a function. Examples include:

- `0`: Speed Limit 20 km/h
- `14`: Stop
- `17`: No Entry
- `25`: Road Work
- `28`: Children Crossing
- ...

See `TSR_Test.py` for the full label mapping.

---

## Installation

Install required Python libraries:
```bash
pip install numpy opencv-python keras tensorflow
Libraries Used
numpy: Array operations and preprocessing
opencv-python: Image and video processing
keras: Deep learning model building
tensorflow: Backend for Keras
matplotlib: Plotting data and accuracy/loss curves
scikit-learn: Accuracy score and model evaluation
pandas: Handling CSV and label data
pickle / pickle5: Saving and loading trained models
```

---



---
