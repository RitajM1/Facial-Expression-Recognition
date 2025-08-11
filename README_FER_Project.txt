Facial Expression Recognition Using VGG16, PCA, and SVM/Random Forest

Project Summary

We developed a hybrid deep learning and machine learning pipeline to classify human emotions from facial images using the FER-2013 dataset. The system uses a pre-trained VGG16 CNN for deep feature extraction, applies PCA for dimensionality reduction, and classifies emotions using Support Vector Machine (SVM) and Random Forest (RF) classifiers.

- Best Accuracy: 53.9% (VGG16 + PCA + SVM)
- Dataset: FER-2013 Facial Expression Recognition Dataset
- Models Used: VGG16, PCA, SVM, Random Forest
- Tools: Python, TensorFlow/Keras, scikit-learn, NumPy, OpenCV, Matplotlib, Seaborn

Pipeline

1. Data loading & preprocessing (resize to 224×224, RGB conversion, normalization)
2. Feature extraction using pre-trained VGG16 (top layers removed)
3. Dimensionality reduction with PCA (500 components)
4. Classification using:
   - SVM (Linear Kernel) with hyperparameter tuning
   - Random Forest (100 estimators)
5. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix

Dataset

We used the publicly available FER-2013 dataset:
https://www.kaggle.com/datasets/msambare/fer2013


Results

Model                   | Train Accuracy | Test Accuracy | Weighted F1 Score | Notes
------------------------|---------------|---------------|-------------------|--------------------
VGG16 + PCA + SVM       | 62.2%         | 53.9%         | 0.5318            | Best generalization
VGG16 + RF (no PCA)     | 99.9%         | 53.1%         | 0.5165            | Overfitted

opencv-python
joblib
tqdm
