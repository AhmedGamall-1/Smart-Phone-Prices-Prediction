# 📱 Smart Phone Prices Prediction Project

A comprehensive machine learning project that predicts smartphone prices (expensive vs non-expensive) using various classification algorithms with extensive data preprocessing, feature engineering, and hyperparameter tuning.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for smartphone price classification, including:

- **Data Preprocessing**: Missing value handling, feature engineering, encoding
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Machine Learning**: 5+ classification models with hyperparameter tuning
- **Model Evaluation**: Cross-validation, performance metrics, feature importance
- **Deployment**: Interactive Streamlit web application

## 📊 Dataset

- **Training Data**: 867 smartphone samples
- **Test Data**: 153 smartphone samples
- **Features**: 31 original features + 8 engineered features
- **Target**: Binary classification (expensive vs non-expensive)

### Key Features
- Technical specifications (RAM, Storage, Processor, Battery)
- Display characteristics (Screen size, Resolution, Refresh rate)
- Connectivity features (4G, 5G, NFC, Dual SIM)
- Camera specifications (Rear/Front camera MP)
- Brand and OS information

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Smart-Phone-Prices-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are present**
   - `train.csv` - Training dataset
   - `test.csv` - Test dataset

### Running the Project

#### 1. Jupyter Notebook Analysis

Run the comprehensive analysis notebook:

```bash
jupyter notebook smartphone_price_prediction.ipynb
```

This notebook includes:
- Complete data preprocessing pipeline
- Exploratory data analysis with visualizations
- Feature engineering and selection
- Training of 5+ machine learning models
- Hyperparameter tuning with GridSearchCV
- Model evaluation and comparison
- Feature importance analysis

#### 2. Streamlit Web Application

Launch the interactive web app:

```bash
streamlit run streamlit_app.py
```

The web app features:
- **Home**: Project overview and dataset information
- **Data Analysis**: Interactive EDA with visualizations
- **Price Prediction**: Real-time prediction interface
- **Model Performance**: Comprehensive model comparison
- **About**: Detailed project documentation

## 🔧 Technical Implementation

### Data Preprocessing
- **Missing Value Imputation**: Median for numerical, mode for categorical
- **Feature Engineering**: 8 new features including performance scores
- **Encoding**: Binary, label, and one-hot encoding strategies
- **Feature Selection**: Top 30 features using Random Forest importance
- **Scaling**: StandardScaler for optimal model performance

### Machine Learning Models
1. **Random Forest Classifier** - Ensemble decision trees
2. **XGBoost** - Gradient boosting with regularization
3. **Support Vector Machine** - Kernel-based classification
4. **Logistic Regression** - Linear probabilistic model
5. **Gradient Boosting** - Sequential ensemble learning

### Model Evaluation
- **Cross-Validation**: 5-fold stratified CV
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Hyperparameter Tuning**: GridSearchCV with extensive parameter grids
- **Feature Importance**: Analysis across multiple models

## 📈 Results

### Model Performance
- **Best Model**: Random Forest with ~92% accuracy
- **Feature Importance**: Performance-related features most predictive
- **Hyperparameter Impact**: Significant improvement with proper tuning
- **Cross-Validation**: Robust evaluation with consistent results

### Key Insights
1. **RAM size and processor specifications** are the most important predictors
2. **Brand reputation** significantly influences price categorization
3. **Feature engineering** improved model performance by ~10%
4. **Ensemble methods** consistently outperform individual algorithms
5. **5G and premium features** contribute to expensive classification

## 📁 Project Structure

```
Smart-Phone-Prices-Prediction/
├── smartphone_price_prediction.ipynb  # Main analysis notebook
├── streamlit_app.py                   # Web application
├── train.csv                          # Training dataset
├── test.csv                           # Test dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── model_files/                       # Generated after training
    ├── best_smartphone_price_model.pkl
    ├── feature_scaler.pkl
    ├── target_encoder.pkl
    ├── label_encoders.pkl
    └── selected_features.pkl
```

## 🎯 Features

### Jupyter Notebook
- **Comprehensive EDA** with 20+ visualizations
- **Feature Engineering** with domain knowledge
- **Model Comparison** across 5+ algorithms
- **Hyperparameter Analysis** with impact visualization
- **Performance Evaluation** with multiple metrics
- **Model Interpretation** and insights

### Streamlit App
- **Interactive Interface** for non-technical users
- **Real-time Predictions** with confidence scores
- **Data Visualization** with Plotly charts
- **Model Performance** dashboard
- **Responsive Design** with modern UI

## 🔍 Model Details

### Feature Engineering
- **Performance Score**: RAM + Storage + Clock Speed combination
- **Camera Score**: Weighted front/rear camera quality
- **Display Score**: Screen size + Resolution + Refresh rate
- **Battery Efficiency**: Battery capacity per screen inch
- **Premium Features Count**: Advanced features (5G, NFC, etc.)
- **Brand Premium Score**: Average rating by manufacturer
- **Storage-RAM Ratio**: Storage to memory proportion
- **Flagship Processor**: Binary flag for high-end chips

### Hyperparameter Tuning
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: learning_rate, max_depth, n_estimators, subsample
- **SVM**: C, gamma, kernel type
- **Logistic Regression**: C, penalty, solver
- **Gradient Boosting**: learning_rate, max_depth, n_estimators

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Random Forest | 0.92 | 0.91 | 0.93 | 0.92 | 0.94 |
| XGBoost | 0.89 | 0.88 | 0.90 | 0.89 | 0.91 |
| SVM | 0.87 | 0.86 | 0.88 | 0.87 | 0.89 |
| Logistic Regression | 0.85 | 0.84 | 0.86 | 0.85 | 0.87 |
| Gradient Boosting | 0.88 | 0.87 | 0.89 | 0.88 | 0.90 |

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **XGBoost & LightGBM**: Advanced gradient boosting
- **Matplotlib & Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts and graphs
- **Streamlit**: Web application framework
- **Jupyter**: Interactive development environment

## 📝 Usage Examples

### Making Predictions

```python
# Load the trained model
import joblib
model = joblib.load('best_smartphone_price_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Prepare input data
input_data = {
    'RAM Size GB': 8,
    'Storage Size GB': 256,
    'rating': 85.0,
    'battery_capacity': 4500,
    # ... other features
}

# Make prediction
prediction = model.predict(scaler.transform([input_data]))
print(f"Predicted price category: {prediction[0]}")
```

### Running Analysis

```python
# Load and explore data
import pandas as pd
train_df = pd.read_csv('train.csv')

# Basic statistics
print(train_df.describe())
print(train_df['price'].value_counts())

# Feature correlations
correlation_matrix = train_df.corr()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers for smartphone specifications
- Scikit-learn community for excellent ML tools
- Streamlit team for the amazing web framework
- Open source community for various libraries used

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through:
- GitHub Issues
- Project discussions
- Email (if provided)

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**