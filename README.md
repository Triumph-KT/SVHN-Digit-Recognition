# SVHN Digit Recognition with Convolutional Neural Networks (CNN)

Built and optimized convolutional neural network models to classify digits (0–9) from street-level house number images in the Street View House Numbers (SVHN) dataset, supporting automated address transcription in real-world scenarios.

## Project Overview

This project applies deep learning techniques to the SVHN dataset, which contains images of house numbers collected from Google Street View. The goal is to accurately recognize individual digits from real-world scenes using convolutional neural networks (CNNs).

## Dataset

The SVHN dataset contains over **60,000 labeled digit images**, each with the following characteristics:
- Grayscale images of size **32x32 pixels**.
- Labeled digits from **0 to 9**.
- Real-world street view images cropped to individual digits.

For computational efficiency, a subset of the dataset was used.

## Objectives

- Preprocess image data and apply normalization and reshaping for CNN compatibility.
- Build, train, and optimize CNN models for digit classification.
- Compare model architectures to minimize overfitting and maximize test accuracy.
- Evaluate model performance with accuracy, precision, recall, and confusion matrices.
- Provide recommendations for improving model robustness and future enhancements.

## Methods

- Data preprocessing:
  - Reshaping and normalizing image data.
  - One-hot encoding target labels.
- Model building:
  - Two CNN architectures with:
    - **LeakyReLU activations**
    - **Batch Normalization**
    - **Dropout layers**
  - Optimization with the **Adam optimizer**.
- Model training:
  - Trained over 20–30 epochs.
  - Validation split to monitor overfitting.
- Model evaluation:
  - Test set accuracy.
  - Precision, recall, and F1-score.
  - Confusion matrix analysis.

## Results

- Achieved **91% accuracy** on the test set.
- First CNN model showed overfitting; improved with a second, deeper CNN architecture.
- Validation accuracy stabilized after 15 epochs with minimal overfitting.
- **Recall** exceeded **88%** for most digits, with the highest recall of **95%** for digit **0**.
- Misclassification patterns identified common confusions (e.g., digits **3**, **5**, and **8**).
- The model generalizes well, outperforming baseline feedforward networks.

## Business/Scientific Impact

- Supports automated digit extraction from urban street scenes for applications in:
  - Map services (e.g., Google Maps).
  - Address validation.
  - Building identification.
- Demonstrates the power of CNNs for real-world image classification.
- Provides a scalable foundation for integrating street-level data into geographic information systems (GIS).

## Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy
- Pandas

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/svhn-digit-recognition.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to preprocess the data, train the CNN models, and evaluate performance.

## Future Work

- Apply **data augmentation** to increase training data diversity.
- Perform **hyperparameter tuning** to optimize CNN architecture and training parameters.
- Explore **transfer learning** from larger pre-trained models for potential accuracy improvements.
