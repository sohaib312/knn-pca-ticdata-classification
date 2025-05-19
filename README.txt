 KNN Classification with PCA on TIC 2000 Dataset

This project demonstrates a machine learning pipeline that applies K-Nearest Neighbors (KNN) classification and Principal Component Analysis (PCA) for dimensionality reduction on the TIC 2000 dataset.

 📌 Objective
To classify the target variable using image-like structured tabular data. The project includes:
- Data loading and preprocessing
- Exploratory data visualization
- Dimensionality reduction using PCA
- Classification using KNN
- Accuracy evaluation and visualization

 📂 Dataset
The dataset contains rows where each feature vector represents 9x9 (81) pixel grayscale image data. It includes:
- `tic_2000_train_data.csv` (features + labels)
- `tic_2000_target_data.csv` (test data)

 🧪 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Plotly
- Google Colab

 📊 Workflow Summary

1. Data Extraction: Read and unzip the provided dataset.
2. Visualization: Reshape selected rows into 9x9 images using `matplotlib`.
3. Preprocessing:
   - Standardization via `StandardScaler`
   - PCA to reduce dimensionality
4. Model Training:
   - KNN classifier from `sklearn.neighbors`
   - Evaluated accuracy before and after PCA
5. Results:
   - Visualize 2D PCA-projected data using Plotly
   - Explore how varying the number of components affects performance

 📈 Accuracy vs PCA Components
The script iteratively tests PCA components from 1 to 9 and prints the resulting classification accuracy.

 📊 Visualizations
- 9x9 grayscale representation of sample input
- 2D PCA scatter plot colored by label
- Cumulative explained variance ratio plot
