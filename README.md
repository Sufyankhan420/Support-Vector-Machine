# Support-Vector-Machine
# Cat and Dog Image Classification using Support Vector Machine (SVM)

This repository contains a project that implements a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The classifier leverages image preprocessing techniques and SVM's classification capabilities to identify the images with high accuracy.

![Cat and Dog Image Classification](https://your-image-link.jpg)

## Project Overview
This project loads and preprocesses a balanced dataset of cat and dog images, splits it into training and validation sets, trains an SVM classifier, and then evaluates its performance. The classifier also predicts labels on a separate test dataset and outputs a CSV file compatible with Kaggle submissions.

## Dataset
The dataset used in this project can be found on [Kaggle's Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) competition page. Ensure you download the data and place it in the appropriate directories as specified.

- **Training Directory**: `D:/cat and dog/train/`
- **Test Directory**: `D:/cat and dog/test1/`
- **Sample Submission File**: `D:/cat and dog/sampleSubmission.csv`

## Project Structure
├── data/ │ ├── train/ # Training images │ ├── test1/ # Test images │ └── sampleSubmission.csv # Sample submission file ├── src/ │ ├── load_images.py # Image loading and preprocessing functions │ ├── svm_classifier.py # SVM model training and evaluation │ └── visualize_predictions.py # Visualization of predictions ├── README.md └── submission.csv # Output file for predictions

## Features
1. **Image Preprocessing**: Images are resized, normalized, and shuffled for balanced data loading.
2. **SVM Classifier**: Trained on flattened image data for simplicity.
3. **Evaluation**: Confusion matrix and classification report for performance evaluation.
4. **Submission File Generation**: Outputs predictions in a format compatible with Kaggle.

## Requirements
Install the necessary packages with:

pip install -r requirements.txt

Key libraries:

numpy
opencv-python
pandas
scikit-learn
matplotlib

# Usage
1. Clone the Repository
git clone https://github.com/yourusername/cat-dog-svm-classifier.git
cd cat-dog-svm-classifier
2. Run the Classifier
Run the following command to load, preprocess, train, and evaluate the SVM model:

python src/svm_classifier.py
3. Generate Submission File
A submission file, submission.csv, will be generated for Kaggle submissions after running the classifier on the test dataset.r

# Results
The model achieves a balanced accuracy on the validation set. Further tuning of parameters, such as the kernel type, may improve the results.

Metric	Value
Accuracy	60%    //you can make it more accurate//
Precision (Cat)	around 60%  
Precision (Dog)	 around 60%
Confusion Matrix	
Sample Visualization
Here are a few predictions by the model:


License
This project is licensed under the MIT License - see the LICENSE file for details.



