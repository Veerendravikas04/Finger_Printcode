# Fingerprint Prediction Using CNN

## Overview
This project focuses on fingerprint prediction using a Convolutional Neural Network (CNN). The dataset was collected from Kaggle, and after preprocessing, around 700 images per group were used for training. The model was optimized with dropout layers to prevent overfitting, achieving over 90% accuracy on both training and test data.

## Dataset
- The dataset was sourced from Kaggle.
- Preprocessing steps included resizing, normalization, and augmentation.
- A balanced subset of 700 images per category was selected for training.

## Model Architecture
- CNN-based architecture with multiple convolutional layers.
- Dropout layers were added to prevent overfitting.
- Evaluated train and test accuracy to check model performance.

## Training Process
- The model was trained on preprocessed fingerprint images.
- Overfitting and underfitting were analyzed, and dropout layers were added accordingly.
- Achieved over 90% accuracy on both train and test datasets.

## Testing with Unseen Data
- The model was tested on unseen fingerprint images.
- It accurately predicted the fingerprint categories.

## Results
- Train Accuracy: **90%+**
- Test Accuracy: **90%+**
- Successfully classified unseen fingerprint images.

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/Veerendravikas04/Finger_Printcode.git
   cd fingerprint-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train.py
   ```
4. Test the model on unseen fingerprints:
   ```sh
   python test.py --image path/to/image
   ```

## Dependencies
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib

## Author
[Veerendra]  


