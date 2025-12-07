
# Customer Churn Prediction 📊

A comprehensive machine learning project that predicts customer churn in the telecommunications industry using multiple supervised learning algorithms.

## 🎯 Project Overview

This project implements a complete machine learning pipeline to predict customer churn, helping telecommunications companies identify at-risk customers before they leave. By analyzing customer demographics, account information, and service usage patterns, the model provides actionable insights for customer retention strategies.

- **Dataset**: IBM Telco Customer Churn (7,043 customers, 21 features)
- **Problem Type**: Binary Classification
- **Best Model**: Logistic Regression (84.36% ROC-AUC, 80.09% recall)
- **Key Strength**: Identifies 80% of at-risk customers with excellent ranking ability
- **Business Impact**: Early identification of at-risk customers enables targeted retention campaigns

## 🧠 Machine Learning Models

We implemented and compared six different supervised learning algorithms with comprehensive evaluation metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Type |
|-------|----------|-----------|--------|----------|---------|------|
| **Logistic Regression** | **74.73%** | **51.52%** | **80.09%** | **62.70%** | **84.36%** | Linear |
| Gradient Boosting | 79.50% | 64.80% | 49.68% | 56.24% | 83.83% | Ensemble |
| Random Forest | 78.93% | 61.82% | 53.75% | 57.50% | 82.45% | Ensemble |
| Support Vector Machine | 74.67% | 51.48% | 78.37% | 62.14% | 82.36% | Kernel-based |
| K-Nearest Neighbors | 74.62% | 52.19% | 50.96% | 51.57% | 75.92% | Instance-based |
| Decision Tree | 72.52% | 48.84% | 76.45% | 59.60% | 78.92% | Tree-based |

### Performance Trade-offs

- **Best Overall Model**: Logistic Regression - Superior ROC-AUC (84.36%) and recall (80.09%)
- **Highest Accuracy**: Gradient Boosting (79.50%) - Most correct predictions
- **Best Recall**: Logistic Regression (80.09%) - Catches 80% of churning customers
- **Best Precision**: Gradient Boosting (64.80%) - Fewer false alarms
- **Fewest Missed Churners**: Logistic Regression (93 false negatives vs 235 for Gradient Boosting)

## ✨ Key Features

### Data Preprocessing
- Automated missing value detection and handling
- Feature encoding using Label Encoding for categorical variables
- Data standardization with StandardScaler for improved model performance
- Train-test split with stratified sampling (75/25 ratio)

### Exploratory Data Analysis
- Comprehensive distribution analysis of all features
- Correlation matrix to identify feature relationships
- Churn rate analysis across different customer segments
- Interactive visualizations for better insights

### Model Training & Evaluation
- Hyperparameter tuning for optimal performance
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- Cross-validation to ensure model robustness
- Side-by-side model comparison

### Visualizations
- Model performance comparison charts
- ROC curves with AUC scores
- Confusion matrices for error analysis
- Feature importance rankings
- Customer churn distribution plots

## 🛠️ Technologies & Libraries

```
Python 3.12.7
Jupyter Notebook / JupyterLab
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Capojayjay/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open the main notebook**
   - Navigate to 'CS205-Customer_Churn_Project.ipynb' in the Jupyter interface

3. **Execute the analysis**
   - Run all cells sequentially: **Cell** → **Run All**
   - Or run cells individually to explore step-by-step

## 📁 Project Structure

```
customer-churn-prediction/
│
├── CS205-Customer_Churn_Project.ipynb              # Main analysis notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── images/                             # Visualization outputs (optional)
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
└── models/                             # Saved models (optional)
    └── best_model.pkl
```

## 📊 Dataset Details

**Source**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Features Description

**Customer Demographics**
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether customer is a senior citizen (0/1)
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)

**Account Information**
- `tenure`: Number of months with the company
- `Contract`: Contract type (Month-to-month, One year, Two year)
- `PaperlessBilling`: Paperless billing enabled (Yes/No)
- `PaymentMethod`: Payment method used

**Services Subscribed**
- `PhoneService`: Phone service subscription
- `MultipleLines`: Multiple phone lines
- `InternetService`: Type of internet service
- `OnlineSecurity`: Online security add-on
- `OnlineBackup`: Online backup add-on
- `DeviceProtection`: Device protection add-on
- `TechSupport`: Tech support subscription
- `StreamingTV`: TV streaming service
- `StreamingMovies`: Movie streaming service

**Billing**
- `MonthlyCharges`: Monthly bill amount
- `TotalCharges`: Total amount charged

**Target Variable**
- `Churn`: Whether customer churned (Yes/No)

## 🔍 Key Insights

Our analysis revealed several important patterns:

1. **Contract Duration**: Month-to-month contracts have a 42% churn rate compared to 11% for two-year contracts
2. **Customer Tenure**: New customers (0-12 months) are 3x more likely to churn than long-term customers
3. **Service Adoption**: Customers with tech support and online security services show 15-20% lower churn rates
4. **Payment Method**: Electronic check users have higher churn rates than automatic payment users
5. **Monthly Charges**: Customers paying over $70/month have significantly higher churn probability

## 📈 Model Performance Analysis

### Why Logistic Regression is the Best Model

**Logistic Regression emerged as the top performer** for customer churn prediction due to:

1. **Best ROC-AUC Score (84.36%)**
   - Superior ability to rank customers by churn probability
   - Critical for prioritizing retention efforts
   - Outperforms all other models in discriminating between churners and non-churners

2. **Highest Recall (80.09%)**
   - Identifies 80% of customers who will actually churn
   - Only 93 false negatives (missed churners) - lowest among all models
   - In customer retention, missing a churning customer is costlier than a false alarm

3. **Best F1-Score (62.70%)**
   - Optimal balance between precision and recall
   - More reliable overall performance than higher-accuracy models

4. **Business Value**
   - Provides interpretable coefficients showing which factors drive churn
   - Fast predictions enable real-time customer risk scoring
   - Probability outputs allow flexible threshold tuning for different business scenarios

### Alternative Model Considerations

**Gradient Boosting** (79.50% accuracy, 64.80% precision)
- Higher accuracy but misses 235 churners (vs 93 for Logistic Regression)
- Better precision if retention campaigns are very expensive
- Trade-off: Catches fewer at-risk customers overall

**Random Forest** (78.93% accuracy, 57.50% F1-score)
- Balanced middle-ground option
- More complex and less interpretable than Logistic Regression
- No significant advantage over Logistic Regression for this dataset

### Business Recommendation

**Primary Model**: **Logistic Regression**
- Use for customer risk scoring and retention campaign targeting
- Best overall performance for identifying at-risk customers
- Interpretable results help understand churn drivers

**Secondary Use Case**: **Gradient Boosting**
- Consider only when retention costs are extremely high
- Use for high-confidence, precision-focused interventions

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- **Data Science Fundamentals**: Data cleaning, preprocessing, and feature engineering
- **Machine Learning**: Implementation of multiple supervised learning algorithms
- **Model Evaluation**: Comprehensive performance analysis using various metrics
- **Data Visualization**: Creating informative plots and charts for stakeholder communication
- **Python Programming**: Efficient use of pandas, scikit-learn, and visualization libraries
- **Research Methodology**: Following systematic ML workflow from problem definition to deployment

## 🔮 Future Enhancements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add SHAP values for better model interpretability
- [ ] Create a web application for real-time churn prediction
- [ ] Implement automated retraining pipeline
- [ ] Add customer lifetime value (CLV) predictions
- [ ] Develop retention strategy recommendations based on predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Capojayjay** - [GitHub](https://github.com/Capojayjay)
- **RomiCocaPermont** - [GitHub](https://github.com/RomiCocaPermont)

## 🙏 Acknowledgments

- Dataset provided by IBM and hosted on Kaggle
- scikit-learn documentation and community
- Jupyter Project for the excellent notebook environment
- All contributors and supporters of this project

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out via GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**

**🔗 Connect with us on LinkedIn and let's discuss data science!**
