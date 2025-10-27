# Customer Churn Prediction 📊

Machine learning research project that predicts customer churn using 6 supervised learning algorithms.

## 🎯 Project Overview

This research implementation demonstrates the complete machine learning workflow:
- **Dataset**: Telco Customer Churn (7,043 customers, 21 features)
- **Objective**: Predict which customers will leave the telecommunications company
- **Approach**: Compare 6 supervised learning algorithms
- **Best Result**: Logistic Regression achieved 80.30% accuracy

## 🧠 Machine Learning Models

| Model | Accuracy | Type |
|-------|----------|------|
| Logistic Regression | 80.30% | Linear |
| Gradient Boosting | 79.78% | Ensemble |
| Support Vector Machine | 79.44% | Kernel-based |
| Random Forest | 79.05% | Ensemble |
| Decision Tree | 75.47% | Tree-based |
| K-Nearest Neighbors | 74.56% | Instance-based |

## 📈 Project Features

✅ **Data Preprocessing**
- Missing value handling
- Feature encoding (Label Encoding)
- Data normalization (StandardScaler)

✅ **Exploratory Data Analysis**
- Distribution analysis
- Correlation studies
- Churn rate visualization

✅ **Model Training & Evaluation**
- Train-test split (75/25)
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- Cross-model comparison

✅ **Visualization**
- Performance comparison charts
- ROC curves
- Confusion matrices
- Feature importance plots

## 🛠️ Technologies Used
```
Python 3.x
Jupyter Notebook
scikit-learn
pandas
numpy
matplotlib
seaborn
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook churn_prediction.ipynb
```

### 4. Run all cells
- Click **"Cell"** → **"Run All"**
- View the results and visualizations

## 📁 Project Structure
```
customer-churn-prediction/
├── churn_prediction.ipynb                      # Main Jupyter notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Dataset
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
└── LICENSE                                     # MIT License
```

## 📊 Dataset Information

**Source**: IBM Telco Customer Churn Dataset

**Features**:
- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services (phone, internet, streaming, security)
- Billing (monthly charges, total charges)
- **Target**: Churn (Yes/No)

## 🔍 Key Findings

1. **Contract Type**: Month-to-month contracts have highest churn rate
2. **Tenure**: Longer customer relationships reduce churn probability
3. **Services**: Customers with tech support and online security churn less
4. **Model Performance**: Logistic Regression provides best predictive accuracy

## 📝 Research Methodology

This project follows standard machine learning research practices:

1. **Problem Definition** - Binary classification (churn/no churn)
2. **Data Collection** - Real-world telecommunications dataset
3. **Data Preprocessing** - Cleaning, encoding, scaling
4. **Exploratory Analysis** - Understanding patterns and relationships
5. **Model Training** - 6 different supervised learning algorithms
6. **Evaluation** - Comprehensive metrics and visualizations
7. **Comparison** - Identify best-performing model

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning workflow
- Data preprocessing techniques
- Multiple algorithm implementation
- Model evaluation and comparison
- Data visualization skills
- Python and scikit-learn proficiency

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 👤 Author

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset: IBM Telco Customer Churn
- Libraries: scikit-learn, pandas, matplotlib, seaborn
- Platform: Jupyter Notebook

---

⭐ If you found this project helpful, please give it a star!
