<div align="center">
  <img src="assets/logo.png" alt="House Rate Predictor Logo" width="200" />
</div>

# House Rate Predictor 🏠📈

A machine learning project built with Python and XGBoost to predict house prices based on various features like crime rate, number of rooms, and accessibility. The model processes a housing dataset (provided in `HousingData.csv`), handles missing values optimally, and implements gradient boosting regression to achieve accurate predictions.

## 🚀 Key Features
- **Data Preprocessing & Cleaning**: Imputes missing values intelligently (using median for skewed continuous variables and mode for categorical/binary features like `CHAS`).
- **Exploratory Data Analysis (EDA)**: Visualizes feature relationships via a correlation heatmap.
- **XGBoost Regressor**: Uses `XGBRegressor` from the `xgboost` library for robust and accurate predictions.
- **Hyperparameter Tuning**: Showcases performance enhancements by fine-tuning tree parameters (e.g., depth, learning rate, subsampling).
- **Performance Metrics**: Evaluates the model using R-squared ($R^2$) Score and Mean Absolute Error (MAE).

## 📊 Dataset Overview
The project relies on `HousingData.csv`. Some of the key features included are:
- `CRIM`: Per capita crime rate by town
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq.ft.
- `INDUS`: Proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- `LSTAT`: % lower status of the population
- `MEDV`: Median value of owner-occupied homes (Target Variable)

## 💻 Tech Stack
- **Python 3.x**
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Train-test splitting and model evaluation metrics
- **XGBoost**: Advanced gradient boosting algorithm for regression

## 📸 Screenshots & Visualizations

### 1. Correlation Matrix
A heatmap illustrating the correlation between all features and the target variable (`MEDV`), helping to identify which variables most strongly influence house prices.
![Correlation Matrix](assets/Screenshot%202026-07-25%20122950.png)

### 2. Training Data: Actual vs Predicted
Scatter plot comparing the actual house prices with the model's predicted prices on the training dataset.
![Training Predictions](assets/Screenshot%202026-07-25%20123001.png)

### 3. Testing Data: Actual vs Predicted
Scatter plot evaluating the model's generalization by comparing actual vs predicted prices on unseen testing data.
![Testing Predictions](assets/Screenshot%202026-07-25%20123010.png)

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pratyush06-aec/house-rate-predictor.git
   cd house-rate-predictor
   ```

2. **Install the dependencies:**
   Make sure you have an environment setup, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model:**
   Execute the main python script to train the model and view the evaluation metrics/plots:
   ```bash
   python main.py
   ```

## 📈 Future Scope
- Implement Cross-Validation and Grid Search for automated hyperparameter tuning.
- Wrap the model in a web application (e.g., using Flask, FastAPI, or Streamlit) for user-friendly predictions.
- Expand the dataset to include more modern housing features and updated valuations.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
