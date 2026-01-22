# Face Mask Detection using Convolutional Neural Networks (CNN)

## Project Overview
This project implements a Convolutional Neural Network (CNN) to detect whether a person in an image is wearing a face mask or not. The model is trained on a custom dataset of images with and without face masks, processed and prepared within a Google Colab environment.

## Dataset
The dataset used for this project is the **'Face Mask Dataset'** from Kaggle, available at `omkargurav/face-mask-dataset`. It consists of two main categories:
-   `with_mask`: Images of people wearing face masks.
-   `without_mask`: Images of people not wearing face masks.

### Dataset Statistics
-   Number of images with mask: 3725
-   Number of images without mask: 3828
-   Total images: 7553

## Methodology

### 1. Data Collection and Preparation
-   **Kaggle API Setup**: Configured Kaggle API to download the dataset directly into the Colab environment.
-   **Dataset Download & Extraction**: Downloaded and extracted the `face-mask-dataset.zip` file.

### 2. Image Processing
-   **Labeling**: Created numerical labels for the two classes: `1` for 'with mask' and `0` for 'without mask'.
-   **Resizing**: All images were resized to a uniform dimension of (128x128) pixels to ensure consistent input for the CNN.
-   **Color Conversion**: Converted images to RGB format.
-   **Numpy Array Conversion**: Converted image data into NumPy arrays for efficient processing by the neural network.

### 3. Data Splitting and Scaling
-   **Train-Test Split**: The dataset was split into training and testing sets with a ratio of 80:20 using `sklearn.model_selection.train_test_split`.
-   **Scaling**: Pixel values (0-255) were scaled to a range of 0-1 by dividing by 255 to normalize the input data for the neural network.

### 4. Model Architecture (CNN)
-   A Sequential CNN model was built using `tf.keras.Sequential`.
-   **Convolutional Layers**: Two `Conv2D` layers with 32 and 64 filters, respectively, and ReLU activation.
-   **Pooling Layers**: Two `MaxPooling2D` layers to reduce dimensionality.
-   **Flatten Layer**: To convert the 2D feature maps into a 1D vector.
-   **Dense Layers**: Two dense layers with 128 and 64 units, respectively, with ReLU activation.
-   **Dropout Layers**: `Dropout` layers (0.5) were included after dense layers to prevent overfitting.
-   **Output Layer**: A dense output layer with 2 units (for two classes) and `softmax` activation.

### 5. Model Compilation and Training
-   **Optimizer**: Adam optimizer was used.
-   **Loss Function**: `sparse_categorical_crossentropy` was chosen as the loss function suitable for multi-class classification with integer labels.
-   **Metrics**: Accuracy was used to monitor model performance.
-   **Training**: The model was trained for 10 epochs with a validation split of 0.1.

### 6. Model Evaluation
-   The trained model was evaluated on the unseen test set to determine its generalization performance.

### 7. Predictive System
-   A simple predictive system was implemented to take a new image as input, preprocess it, and predict whether a person in the image is wearing a mask or not.

## Results
-   **Training Accuracy**: Achieved high training accuracy (e.g., around 97-98%) after 10 epochs.
-   **Validation Accuracy**: Validation accuracy was also consistently high (e.g., around 93-94%).
-   **Test Accuracy**: The model achieved a test accuracy of approximately **94.04%**.

## How to Run

1.  **Clone the Repository**: Clone this GitHub repository to your local machine or open it directly in Google Colab.

2.  **Kaggle API Key**: You need a `kaggle.json` file containing your Kaggle API credentials. Place this file in your Colab environment or ensure it's configured in your local `~/.kaggle/` directory.

3.  **Install Dependencies**: The project uses several Python libraries. Ensure they are installed (e.g., `pip install kaggle tensorflow keras numpy matplotlib opencv-python-headless pillow scikit-learn`).

4.  **Run the Notebook**: Execute the cells in the provided Jupyter Notebook (`.ipynb` file) sequentially. This will:
    -   Download and extract the dataset.
    -   Preprocess the images.
    -   Build, train, and evaluate the CNN model.
    -   Allow you to test the predictive system with an example image.

5.  **Predictive System Usage**: When prompted, enter the path to an image file to get a prediction on whether a person in that image is wearing a mask.

## Dependencies
-   `kaggle`
-   `tensorflow`
-   `keras`
-   `numpy`
-   `matplotlib`
-   `opencv-python` (cv2)
-   `Pillow` (PIL)
-   `scikit-learn`

```
