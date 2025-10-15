# BISINDO Sign Language Recognition

This project focuses on the recognition of BISINDO (Indonesian Sign Language) alphabet signs using various deep learning models. The primary objective is to develop and compare different models for their accuracy and performance in classifying images of sign language gestures.

---

## Dataset

The dataset used in this project is the **BISINDO Sign Language Dataset**, which contains images of the BISINDO alphabet signs.

* **Source**: The original dataset can be found at [https://doi.org/10.17632/ywnjpbcz8m.1](https://doi.org/10.17632/ywnjpbcz8m.1).
* **Ready-to-Use Dataset**: A pre-processed and cleaned version of the dataset, as used in the testing notebooks, is available for download from [Google Drive](https://drive.google.com/file/d/1cr_7QR9IfJZ1ivGeE5vYkVRSd6mIKC_Y/view?usp=sharing).

### Data Preparation

The raw dataset underwent a series of preprocessing steps to prepare it for model training:

1.  **Analysis and Cleaning**: The dataset was first analyzed to identify any issues such as inaccessible images or inconsistencies in image resolutions. In total, 297 inaccessible images were identified and removed from the original dataset of 4763 images.

2.  **Dataset Splitting**: The cleaned dataset was then split into training, validation, and testing sets using a stratified approach to ensure a balanced distribution of classes in each set. The final distribution of images is as follows:
    * **Training Set**: 3,126 images
    * **Validation Set**: 670 images
    * **Test Set**: 670 images

---

## Models

Three different deep learning models were trained and evaluated for this sign language recognition task:

* **YOLOv8n-cls**: A state-of-the-art object detection model from the YOLO family, used here for classification.
* **MobileNetV2**: A lightweight and efficient convolutional neural network designed for mobile and embedded vision applications.
* **VGG19**: A deep convolutional neural network known for its simplicity and effectiveness in image classification tasks.

---

## Usage

The provided Jupyter notebooks cover the entire workflow of this project, from data preparation to model evaluation. To replicate the results, you can follow the steps outlined in the notebooks:

1.  **`SLR_Dataset_Analyzing_&_Cleaning.ipynb`**: Use this notebook to analyze the raw dataset and clean it by removing any inaccessible images.
2.  **`SLR_Dataset_Splitting.ipynb`**: After cleaning, run this notebook to split the dataset into the training, validation, and testing sets.
3.  **`Colab_Script_to_Create_Label_Directories_for_YOLO.ipynb`**: If you plan to train the YOLOv8 model with your own dataset, this script will help you create the necessary label directory structure.
4.  **`Testing_YOLOv8.ipynb`**, **`Testing_MobileNetV2.ipynb`**, **`Testing_VGG19.ipynb`**: These notebooks contain the code for training and evaluating each of the three models.
5.  **`Inference_Time_Comparison_of_Models.ipynb`**: This notebook can be used to compare the inference times of the trained models.

---

## Results

The performance of the three models was evaluated based on their accuracy on the test set and their inference time per image.

### Accuracy

| Model | Accuracy on Test Set |
| :--- | :--- |
| **YOLOv8n-cls** | **93%** |
| MobileNetV2 | 59% |
| VGG19 | 55% |

### Inference Time and Model Parameters

| Model | Parameters | Inference Time (ms/image) |
| :--- | :--- | :--- |
| MobileNetV2 | 2,929,242 | 3.0759 |
| VGG19 | 20,302,426 | 8.5736 |
| **YOLOv8n-cls** | **1,468,186** | **5.2247** |

Based on these results, **YOLOv8n-cls** emerges as the best-performing model for this task, achieving the highest accuracy with a relatively low number of parameters and a fast inference time.
