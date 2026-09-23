# Brain Tumor Detection Using CNN

A deep learning project for classifying brain MRI images using Convolutional Neural Networks (CNNs) with TensorFlow/Keras. The project explores a custom CNN architecture and a MobileNetV2-based transfer learning approach, with the trained model integrated into a Streamlit web application for image prediction.

> **Disclaimer:** This project is developed for educational purposes and should not be used as a medical diagnostic system or as a replacement for professional medical evaluation.

## 📌 Project Overview

The objective of this project is to build an image classification model capable of identifying the class of a brain MRI image.

The notebook contains two model approaches:

1. A custom CNN model
2. A MobileNetV2-based transfer learning model

The MobileNetV2-based model was saved and used for the application.

## 🚀 Features

- Brain MRI image classification
- Custom CNN implementation
- MobileNetV2 transfer learning
- Image augmentation
- Training and validation
- Early stopping
- Model saving in H5 format
- Streamlit web interface
- Custom application styling

## 🧠 Model Architecture

### Custom CNN

The custom CNN consists of:

- Convolutional layers
- Max pooling layers
- Flatten layer
- Fully connected dense layers
- Dropout
- Softmax output layer

Architecture:

    Input Image
         ↓
    Conv2D (32 filters, 3×3)
         ↓
    MaxPooling2D (2×2)
         ↓
    Conv2D (32 filters, 3×3)
         ↓
    MaxPooling2D (2×2)
         ↓
    Flatten
         ↓
    Dense (64, ReLU)
         ↓
    Dense (32, ReLU)
         ↓
    Dense (16, ReLU)
         ↓
    Dropout (0.3)
         ↓
    Dense (4, Softmax)

### MobileNetV2

The second approach uses MobileNetV2 initialized with ImageNet weights.

    Input Image (256×256×3)
         ↓
    MobileNetV2
         ↓
    GlobalAveragePooling2D
         ↓
    Dense (64, ReLU)
         ↓
    Dense (32, ReLU)
         ↓
    Dense (16, ReLU)
         ↓
    Dropout (0.3)
         ↓
    Dense (4, Softmax)
         ↓
    Prediction

## 📊 Dataset

The dataset was divided into training, validation, and testing sets using `split-folders`.

    Training   : 80%
    Validation : 10%
    Testing    : 10%

Dataset distribution recorded in the notebook:

    Training   : 5617 images
    Validation : 701 images
    Testing    : 705 images
    Total      : 7023 images
    Classes    : 4

The dataset was split using:

    splitfolders.ratio(
        data_path,
        output="final_data",
        seed=1234,
        ratio=(0.8, 0.1, 0.1)
    )

## 🖼️ Image Preprocessing

Training and testing data were processed using Keras `ImageDataGenerator`.

Training preprocessing and augmentation:

    ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True
    )

Validation preprocessing:

    ImageDataGenerator(
        rescale=1./255
    )

The model uses:

    Image Size: 256 × 256
    Channels: 3

## ⚙️ Training Configuration

The custom CNN was compiled using:

    Optimizer: Adam
    Learning Rate: 0.001
    Loss: categorical_crossentropy
    Metric: accuracy

The MobileNetV2-based model was compiled using:

    Optimizer: Adam
    Learning Rate: 0.0001
    Loss: categorical_crossentropy
    Metric: accuracy

Training configuration:

    Batch Size: 16
    Maximum Epochs: 50
    Early Stopping Patience: 5

Early stopping was configured with:

    monitor="val_loss"
    patience=5
    restore_best_weights=True

## 📈 Training Results

### Custom CNN

The custom CNN achieved its highest recorded validation accuracy at:

    Validation Accuracy: 87.16%
    Epoch: 25

The training was stopped after epoch 30.

### MobileNetV2

The MobileNetV2-based model achieved its highest recorded validation accuracy at:

    Validation Accuracy: 99.00%
    Epoch: 18
    Validation Loss: 0.0398

Training continued until epoch 23, where early stopping terminated the run.

> These values are the recorded validation results from the project notebook. They are not a guarantee of performance on unseen clinical data.

## 💾 Model Saving

The trained MobileNetV2-based model was saved in two formats:

    brain_tumor.h5
    brain_tumor.keras

The application uses:

    models/brain_tumor.h5

## 🌐 Streamlit Application

The trained model is integrated into a Streamlit application.

The application allows the user to:

- Upload a brain MRI image
- Preprocess the uploaded image
- Pass the image to the trained model
- Display the predicted class

## 📂 Project Structure

    brain-tumor-detection-using-cnn/
    │
    ├── .streamlit/
    │   └── config.toml
    │
    ├── assets/
    │   └── style.css
    │
    ├── models/
    │   └── brain_tumor.h5
    │
    ├── screenshots/
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── model.py
    │   ├── preprocess.py
    │   └── ui.py
    │
    ├── app.py
    ├── brain_tumor.ipynb
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- NumPy
- Pandas
- Pillow
- Split-Folders

## 📦 Installation

Clone the repository:

    git clone https://github.com/saif-mohammed9505/brain-tumor-detection-using-cnn.git

Move into the project directory:

    cd brain-tumor-detection-using-cnn

Create a virtual environment:

    python -m venv venv

Activate the environment on Windows:

    venv\Scripts\activate

Install the required dependencies:

    pip install -r requirements.txt

## ▶️ Run the Application

Start the Streamlit application using:

    streamlit run app.py

The application will open in your default web browser.

## 📓 Jupyter Notebook

The complete model development and training process is available in:

    brain_tumor.ipynb

The notebook contains:

- Dataset splitting
- Image generators
- Custom CNN construction
- CNN compilation
- CNN training
- Early stopping
- MobileNetV2 initialization
- Transfer learning model construction
- Model training
- Model saving

## 🔄 Project Workflow

    Brain MRI Dataset
            ↓
    Dataset Splitting
            ↓
    Image Preprocessing
            ↓
    Data Augmentation
            ↓
    CNN Model
            ↓
    Model Training
            ↓
    Validation
            ↓
    Trained Model
            ↓
    Streamlit Application
            ↓
    MRI Image Upload
            ↓
    Prediction

## 📸 Screenshots

Application screenshots are included in the `screenshots/` directory.

They demonstrate the Streamlit interface and prediction workflow.

## 🔮 Future Improvements

- Evaluate the model on a separate held-out test set and report test metrics
- Add confusion matrix and classification report
- Add precision, recall, and F1-score
- Add prediction confidence visualization
- Perform hyperparameter tuning
- Compare additional CNN architectures
- Improve dataset balancing and preprocessing
- Add explainability techniques such as Grad-CAM
- Deploy the Streamlit application online

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

Predictions from the model should not be considered medical diagnoses. Brain MRI interpretation should be performed by qualified healthcare professionals using appropriate clinical information and diagnostic procedures.

## ⭐ Project

**Brain Tumor Detection Using CNN**

Built with Python, TensorFlow/Keras, MobileNetV2, and Streamlit.
## 👨‍💻 Author

**Saif Mohammed**

GitHub:

https://github.com/saif-mohammed9505

---

## License

This project is intended for educational and experimental purposes.
