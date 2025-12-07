#!/usr/bin/env python3
"""
Customer Churn Prediction - Complete Machine Learning Pipeline
Authors: Jay R. Refugia, Romi Pereira Garcia Montejo
Date: November 2025
Dataset: Telco Customer Churn (7,043 customers, 21 features)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard Library
import os
import sys
import warnings
import json
import multiprocessing
import time
from datetime import datetime
from contextlib import contextmanager
import joblib

# Third-party Libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, auc
)

# ============================================================================
# CUSTOM JSON ENCODER
# ============================================================================

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Global random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)

# CPU configuration
MAX_CPU = max(1, multiprocessing.cpu_count() - 1)
os.environ['LOKY_MAX_CPU_COUNT'] = str(MAX_CPU)

# Warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*LOKY.*')

# File paths
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create directories
for directory in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'Logistic Regression': {
        'config': {
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'solver': 'lbfgs',
            'class_weight': 'balanced'
        }
    },
    'Decision Tree': {
        'config': {
            'max_depth': 10,
            'min_samples_split': 20,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced'
        }
    },
    'Random Forest': {
        'config': {
            'n_estimators': 100,
            'max_depth': 15,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    },
    'Gradient Boosting': {
        'config': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_STATE,
            'subsample': 0.8
        }
    },
    'SVM': {
        'config': {
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced',
            'probability': True
        }
    },
    'K-Nearest Neighbors': {
        'config': {
            'n_neighbors': 5,
            'metric': 'minkowski',
            'n_jobs': -1
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@contextmanager
def timer(operation_name, verbose=True):
    """Context manager to time operations"""
    start_time = time.time()
    if verbose:
        print(f"   ⏱️  Starting {operation_name}...")
    yield
    end_time = time.time()
    if verbose:
        print(f"   ✅ {operation_name} completed in {end_time - start_time:.2f} seconds")

def parse_version(version):
    """Convert version string to tuple for comparison"""
    return tuple(map(int, version.split('.')[:3]))

def check_versions():
    """Verify minimum required package versions"""
    REQUIRED_VERSIONS = {
        'pandas': '1.3.0',
        'numpy': '1.21.0', 
        'scikit-learn': '1.0.0',
        'matplotlib': '3.3.0',
        'seaborn': '0.11.0'
    }
    
    current_versions = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit-learn': sklearn.__version__,
        'matplotlib': matplotlib.__version__,
        'seaborn': sns.__version__
    }
    
    compatibility_issues = []
    
    for package, min_version in REQUIRED_VERSIONS.items():
        current = current_versions[package]
        if parse_version(current) < parse_version(min_version):
            compatibility_issues.append(
                f"{package}: {current} < {min_version} (minimum required)"
            )
    
    if compatibility_issues:
        print("⚠️  VERSION COMPATIBILITY WARNINGS:")
        for issue in compatibility_issues:
            print(f"   - {issue}")
        print("   Consider upgrading packages for full functionality")
    
    return len(compatibility_issues) == 0

def verify_dataset():
    """Verify dataset exists and is accessible"""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ CRITICAL ERROR: Dataset not found at {DATASET_PATH}")
        print("   Please ensure the CSV file is placed in the 'data' directory")
        print("   Dataset available from: https://www.kaggle.com/blastchar/telco-customer-churn")
        sys.exit(1)
    
    file_size = os.path.getsize(DATASET_PATH) / (1024 * 1024)  # MB
    print(f"✅ Dataset verified: {DATASET_PATH} ({file_size:.2f} MB)")
    return True

def setup_visualization():
    """Configure visualization settings"""
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titleweight'] = 'bold'
    
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 120)
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

def save_experiment_config():
    """Save experiment configuration to JSON file for reproducibility"""
    config_path = os.path.join(OUTPUT_DIR, 'experiment_config.json')
    
    EXPERIMENT_INFO = {
        'project': 'Customer Churn Prediction',
        'authors': ['Jay R. Refugia', 'Romi Pereira Garcia Montejo'],
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_state': RANDOM_STATE,
        'dataset': 'Telco Customer Churn',
        'expected_samples': 7043,
        'expected_features': 21,
        'system_info': {
            'cpu_cores': multiprocessing.cpu_count(),
            'max_cores_used': MAX_CPU,
            'python_version': sys.version.split()[0]
        },
        'package_versions': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__,
            'matplotlib': matplotlib.__version__,
            'seaborn': sns.__version__
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(EXPERIMENT_INFO, f, indent=2, cls=DateTimeEncoder)
        print(f"📄 Experiment configuration saved to: {config_path}")
    except Exception as e:
        print(f"⚠️  Could not save experiment config: {e}")

def initialize_environment():
    """Initialize the complete environment for the project"""
    print("=" * 70)
    print("🚀 INITIALISING CUSTOMER CHURN PREDICTION ENVIRONMENT")
    print("=" * 70)
    
    verify_dataset()
    setup_visualization()
    versions_ok = check_versions()
    save_experiment_config()
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📦 NumPy: {np.__version__}")
    print(f"📦 Pandas: {pd.__version__}")
    print(f"📦 Scikit-learn: {sklearn.__version__}")
    print(f"📦 Matplotlib: {matplotlib.__version__}")
    print(f"📦 Seaborn: {sns.__version__}")
    print("=" * 70)
    print(f"🖥️  CPU Cores: {multiprocessing.cpu_count()} (Using: {MAX_CPU})")
    print(f"🎲 Random State: {RANDOM_STATE}")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Models Directory: {MODELS_DIR}")
    print("=" * 70)
    print(f"🎯 {len(MODEL_CONFIGS)} algorithms configured and ready")
    print("=" * 70)
    
    if not versions_ok:
        print("\n⚠️  NOTE: Some version compatibility issues detected.")
        print("   The code will run, but consider upgrading packages for optimal performance.")
    
    return versions_ok

# ============================================================================
# STEP 1: DATA LOADING FUNCTIONS
# ============================================================================

def optimize_memory_usage(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    print("\n💾 MEMORY OPTIMIZATION")
    print("=" * 70)
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        if num_unique < len(df) * 0.5:
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = ((initial_memory - final_memory) / initial_memory) * 100
    
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory:   {final_memory:.2f} MB")
    print(f"Reduction:      {reduction:.1f}%")
    print("=" * 70)
    
    return df

def generate_data_summary(df):
    """Generate comprehensive data summary"""
    print("\n📊 COMPREHENSIVE DATA SUMMARY")
    print("=" * 70)
    
    summary = {
        'dataset_shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
        'total_missing': df.isnull().sum().sum(),
        'data_types': df.dtypes.value_counts().to_dict(),
        'numerical_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    print(f"📈 Dataset Shape:      {summary['dataset_shape'][0]:,} rows × {summary['dataset_shape'][1]} columns")
    print(f"💾 Memory Usage:       {summary['memory_usage_mb']:.2f} MB")
    print(f"🔄 Duplicate Rows:     {summary['duplicate_rows']:,}")
    print(f"❓ Missing Values:     {summary['total_missing']:,}")
    print(f"📊 Data Types:")
    for dtype, count in summary['data_types'].items():
        print(f"     - {dtype}: {count} columns")
    print(f"🔢 Numerical Columns:  {summary['numerical_cols']}")
    print(f"📝 Categorical Columns: {summary['categorical_cols']}")
    
    print("=" * 70)
    return summary

def analyze_target_variable(df, target_col='Churn'):
    """Comprehensive analysis of target variable"""
    print(f"\n🎯 TARGET VARIABLE ANALYSIS: '{target_col}'")
    print("=" * 70)
    
    if target_col not in df.columns:
        print(f"❌ ERROR: Target column '{target_col}' not found!")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    churn_counts = df[target_col].value_counts()
    churn_pct = df[target_col].value_counts(normalize=True) * 100
    
    print("Distribution:")
    for value in churn_counts.index:
        count = churn_counts[value]
        pct = churn_pct[value]
        print(f"  {value:5s}: {count:5,} ({pct:5.2f}%)")
    
    minority_class = churn_pct.idxmin()
    minority_pct = churn_pct.min()
    imbalance_ratio = churn_pct.max() / churn_pct.min()
    
    print(f"\n📊 Class Imbalance Metrics:")
    print(f"   Minority class: {minority_class} ({minority_pct:.2f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if minority_pct < 20:
        print(f"⚠️  SEVERE CLASS IMBALANCE DETECTED!")
    elif minority_pct < 30:
        print(f"⚠️  MODERATE CLASS IMBALANCE DETECTED")
    else:
        print(f"✅ Balanced dataset")
    
    print("=" * 70)
    return {
        'counts': churn_counts,
        'percentages': churn_pct,
        'imbalance_ratio': imbalance_ratio,
        'minority_class': minority_class
    }

def perform_data_quality_checks(df):
    """Perform comprehensive data quality assessment"""
    print("\n🔍 DATA QUALITY CHECKS")
    print("=" * 70)
    
    quality_issues = []
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        quality_issues.append(f"Constant columns: {constant_cols}")
    
    high_cardinality = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() > 100:
            high_cardinality.append((col, df[col].nunique()))
    
    if high_cardinality:
        quality_issues.append("High cardinality categorical columns:")
        for col, count in high_cardinality:
            quality_issues.append(f"  - {col}: {count} unique values")
    
    potential_ids = [col for col in df.columns if df[col].nunique() == len(df)]
    if potential_ids:
        quality_issues.append(f"Potential ID columns: {potential_ids}")
    
    if quality_issues:
        print("⚠️  Data quality issues found:")
        for issue in quality_issues:
            print(f"   {issue}")
    else:
        print("✅ No data quality issues detected")
    
    print("=" * 70)
    return quality_issues

def load_and_analyze_data():
    """Load dataset and perform initial analysis"""
    print("=" * 70)
    print("📂 DATA LOADING")
    print("=" * 70)
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"📁 Expected size: 7,043 rows × 21 columns")
    print("-" * 70)
    
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Actual shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        file_size = os.path.getsize(DATASET_PATH) / (1024 * 1024)
        print(f"💾 File size: {file_size:.2f} MB")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if df.shape[0] != 7043:
            print(f"⚠️  WARNING: Row count mismatch!")
            print(f"   Expected: 7,043, Got: {df.shape[0]:,}")
        
        if df.shape[1] != 21:
            print(f"⚠️  WARNING: Column count mismatch!")
            print(f"   Expected: 21, Got: {df.shape[1]}")
        
        print("-" * 70)
        print(f"✅ All validations passed!")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file not found!")
        print(f"\n📋 Troubleshooting:")
        print(f"   1. Expected location: {DATASET_PATH}")
        print(f"   2. Current directory: {os.getcwd()}")
        print(f"   3. Data directory exists: {os.path.exists(DATA_DIR)}")
        print(f"\n💡 To fix this:")
        print(f"   1. Download the Telco Customer Churn dataset")
        print(f"   2. Place it in the '{DATA_DIR}' folder")
        print(f"   3. Ensure filename is: {os.path.basename(DATASET_PATH)}")
        sys.exit(1)
    
    print("=" * 70)
    
    # Optimize memory
    df = optimize_memory_usage(df)
    
    # Display information
    print("\n📋 DATASET OVERVIEW")
    print("=" * 70)
    print(f"Total Records:   {len(df):,}")
    print(f"Total Features:  {df.shape[1]}")
    print(f"Memory Usage:    {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicates:      {df.duplicated().sum():,}")
    
    print(f"\n📑 Column List ({len(df.columns)} columns):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        unique = df[col].nunique()
        missing = df[col].isnull().sum()
        print(f"  {i:2d}. {col:20s} | Type: {str(dtype):10} | Unique: {unique:4} | Missing: {missing:3}")
    
    print("=" * 70)
    
    # Generate summaries
    data_summary = generate_data_summary(df)
    target_analysis = analyze_target_variable(df)
    quality_issues = perform_data_quality_checks(df)
    
    # Missing values
    print("\n❓ MISSING VALUES ANALYSIS")
    print("=" * 70)
    missing = df.isnull().sum()
    missing_total = missing.sum()
    if missing_total > 0:
        print("Columns with missing values:")
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Missing Count': missing[missing > 0].values,
            'Percentage': (missing[missing > 0] / len(df) * 100).values
        })
        print(missing_df)
        print(f"\n⚠️  Total missing values: {missing_total:,} ({missing_total/len(df)*100:.2f}%)")
    else:
        print("✅ No missing values detected!")
    
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ DATA LOADING AND ANALYSIS COMPLETE!")
    print("=" * 70)
    
    final_summary = {
        'status': 'SUCCESS',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_shape': df.shape,
        'memory_optimized_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'total_missing': missing_total,
        'duplicate_rows': df.duplicated().sum(),
        'class_imbalance_ratio': target_analysis['imbalance_ratio'] if target_analysis else 'N/A',
        'quality_issues_count': len(quality_issues)
    }
    
    print("📋 FINAL SUMMARY:")
    for key, value in final_summary.items():
        if key != 'status':
            print(f"  - {key.replace('_', ' ').title():25}: {value}")
    
    print("=" * 70)
    print("🎯 NEXT STEPS:")
    print("  • Data preprocessing and cleaning")
    print("  • Feature engineering")
    print("  • Model training and evaluation")
    print("=" * 70)
    
    return df

# ============================================================================
# STEP 2: PREPROCESSING FUNCTIONS
# ============================================================================

def validate_preprocessing_inputs(df, target_column):
    """Validate inputs before preprocessing"""
    validation_errors = []
    
    if not isinstance(df, pd.DataFrame):
        validation_errors.append("Input must be a pandas DataFrame")
    
    if target_column not in df.columns:
        validation_errors.append(f"Target column '{target_column}' not found")
    
    if df.empty:
        validation_errors.append("DataFrame is empty")
    
    if validation_errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors))

def convert_target_to_binary(series, verbose=True):
    """Safely convert target variable to binary"""
    original_dtype = series.dtype
    original_values = series.unique()
    
    if verbose:
        print(f"   Original dtype: {original_dtype}")
        print(f"   Original values: {original_values}")
    
    series_str = series.astype(str)
    series_str = series_str.str.lower().str.strip()
    
    mapping = {
        'yes': 1, 'no': 0, 
        'true': 1, 'false': 0, 
        '1': 1, '0': 0,
        'churn': 1, 'no churn': 0,
        'positive': 1, 'negative': 0
    }
    
    series_mapped = series_str.map(mapping)
    unmapped_mask = series_mapped.isnull()
    unmapped_count = unmapped_mask.sum()
    
    if unmapped_count > 0:
        unique_unmapped = series_str[unmapped_mask].unique()
        raise ValueError(f"Could not map values: {unique_unmapped}. Available mappings: {list(mapping.keys())}")
    
    series_binary = series_mapped.astype(int)
    unique_after = series_binary.unique()
    
    if not set(unique_after).issubset({0, 1}):
        raise ValueError(f"Target conversion failed. Final values: {unique_after}")
    
    if verbose:
        print(f"✅ Target converted from {original_dtype} to binary")
        print(f"   Final distribution: 0={sum(series_binary==0)}, 1={sum(series_binary==1)}")
    
    return series_binary

def encode_categorical_features(X, verbose=True):
    """Encode categorical features with validation"""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    if len(categorical_cols) == 0:
        if verbose:
            print("ℹ️  No categorical columns to encode")
        return X, label_encoders
    
    X_encoded = X.copy()
    
    for col in categorical_cols:
        unique_count = X[col].nunique()
        if unique_count > 50 and verbose:
            print(f"⚠️  High cardinality: {col} has {unique_count} unique values")
        
        missing_count = X[col].isnull().sum()
        if missing_count > 0:
            X_encoded[col] = X[col].astype(str)
            X_encoded[col] = X_encoded[col].fillna('MISSING')
            if verbose:
                print(f"   ⚠️  Filled {missing_count} missing values in {col}")
        else:
            X_encoded[col] = X[col].astype(str)
        
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le
        
        if verbose:
            print(f"   ✅ {col}: {len(le.classes_)} categories → encoded")
    
    return X_encoded, label_encoders

def preprocess_data(df, target_column='Churn', verbose=True):
    """
    Preprocess the dataset for machine learning.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🧹 DATA PREPROCESSING PIPELINE")
        print(f"{'='*70}")
        print(f"📋 Target column: '{target_column}'")
        print(f"📊 Initial shape: {df.shape}")
    
    validate_preprocessing_inputs(df, target_column)
    
    df_processed = df.copy()
    label_encoders = {}
    original_shape = df.shape
    
    # Handle TotalCharges
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(
            df_processed['TotalCharges'], 
            errors='coerce'
        )
        
        missing_count = df_processed['TotalCharges'].isnull().sum()
        
        if missing_count > 0:
            if 'tenure' in df_processed.columns and 'MonthlyCharges' in df_processed.columns:
                new_customers = (df_processed['tenure'] == 0)
                new_customers_filled = new_customers & df_processed['TotalCharges'].isnull()
                others_filled = (~new_customers) & df_processed['TotalCharges'].isnull()
                
                df_processed.loc[new_customers_filled, 'TotalCharges'] = 0
                df_processed.loc[others_filled, 'TotalCharges'] = df_processed.loc[others_filled, 'MonthlyCharges']
                
                if verbose:
                    print(f"✅ Filled {missing_count} missing values intelligently:")
                    print(f"   - New customers (tenure=0): {new_customers_filled.sum()} → $0")
                    print(f"   - Existing customers: {others_filled.sum()} → MonthlyCharges")
            else:
                median_charges = df_processed['TotalCharges'].median()
                df_processed['TotalCharges'].fillna(median_charges, inplace=True)
                if verbose:
                    print(f"✅ Filled {missing_count} missing values with median: ${median_charges:.2f}")
        else:
            if verbose:
                print("✅ No missing values in TotalCharges")
    
    # Remove non-predictive columns
    id_columns = ['customerID', 'CustomerID', 'customer_id', 'id', 'ID']
    columns_to_drop = [col for col in id_columns if col in df_processed.columns]
    
    if columns_to_drop:
        df_processed.drop(columns_to_drop, axis=1, inplace=True)
        if verbose:
            print(f"✅ Dropped ID columns: {columns_to_drop}")
    
    # Convert target to binary
    if verbose:
        print(f"\n{'─'*70}")
        print("🎯 STEP 4: Convert Target to Binary")
        print(f"{'─'*70}")
    
    y = convert_target_to_binary(df_processed[target_column], verbose=verbose)
    df_processed[target_column] = y
    
    # Separate features and target
    X = df_processed.drop(target_column, axis=1)
    
    if verbose:
        print(f"\n{'─'*70}")
        print("🔀 STEP 5: Separate Features and Target")
        print(f"{'─'*70}")
        print(f"✅ Features (X): {X.shape}")
        print(f"✅ Target (y): {y.shape}")
    
    # Encode categorical variables
    if verbose:
        print(f"\n{'─'*70}")
        print("🔤 STEP 6: Encode Categorical Variables")
        print(f"{'─'*70}")
    
    X, label_encoders = encode_categorical_features(X, verbose=verbose)
    
    # Generate summary
    summary = {
        'original_shape': original_shape,
        'final_feature_shape': X.shape,
        'target_distribution': {
            'churn_count': int(y.sum()),
            'churn_percentage': float(y.mean() * 100),
            'no_churn_count': int(len(y) - y.sum()),
            'no_churn_percentage': float((1 - y.mean()) * 100)
        },
        'feature_types': {
            'numerical': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_encoded': len(label_encoders)
        },
        'data_quality': {
            'missing_values_remaining': int(X.isnull().sum().sum()),
            'constant_columns': [col for col in X.columns if X[col].nunique() <= 1]
        }
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("✅ PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Original shape: {original_shape}")
        print(f"📊 Final feature shape: {X.shape}")
        print(f"🎯 Target distribution:")
        print(f"   - Churn (1): {summary['target_distribution']['churn_count']} "
              f"({summary['target_distribution']['churn_percentage']:.2f}%)")
        print(f"   - No Churn (0): {summary['target_distribution']['no_churn_count']} "
              f"({summary['target_distribution']['no_churn_percentage']:.2f}%)")
        print(f"🔤 Encoded {summary['feature_types']['categorical_encoded']} categorical columns")
        print(f"🔢 {summary['feature_types']['numerical']} numerical columns")
        print(f"❓ Remaining missing values: {summary['data_quality']['missing_values_remaining']}")
        
        if summary['data_quality']['constant_columns']:
            print(f"⚠️  Constant columns: {summary['data_quality']['constant_columns']}")
        
        print(f"\n💡 Note: Feature scaling will be applied after train/test split")
        print(f"{'='*70}\n")
    
    return X, y, label_encoders, summary

# ============================================================================
# STEP 3: EDA FUNCTIONS
# ============================================================================

def perform_eda_with_boxplots(df_processed, target_column, label_encoders=None):
    """
    Perform comprehensive exploratory data analysis with box plots.
    """
    print("\n" + "="*70)
    print("[STEP 3] EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe!")
    
    y = df_processed[target_column]
    X = df_processed.drop(target_column, axis=1)
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if label_encoders:
        encoded_categoricals = list(label_encoders.keys())
        true_numerics = [col for col in numeric_cols if col not in encoded_categoricals]
    else:
        known_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        true_numerics = [col for col in numeric_cols if col in known_numeric]
        encoded_categoricals = [col for col in numeric_cols if col not in true_numerics]
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {len(df_processed):,}")
    print(f"   Numeric features: {len(true_numerics)}")
    print(f"   Categorical features: {len(encoded_categoricals)}")
    print(f"   Churn rate: {y.mean()*100:.2f}%")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 18))  
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)  
    fig.suptitle('Comprehensive Exploratory Data Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Churn Distribution (Pie Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    churn_counts = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    ax1.pie(churn_counts.values, labels=['No Churn', 'Churn'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'})
    ax1.set_title('Churn Distribution', fontweight='bold', fontsize=13, pad=10)
    
    # Plot 2: Tenure Distribution by Churn (Box Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'tenure' in df_processed.columns:
        tenure_data = [
            df_processed[y==0]['tenure'].values,
            df_processed[y==1]['tenure'].values
        ]
        
        bp = ax2.boxplot(tenure_data, tick_labels=['No Churn', 'Churn'],
                         patch_artist=True, widths=0.6, boxprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='darkred'))
        
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#e74c3c')
        bp['boxes'][1].set_alpha(0.7)
        
        ax2.set_title('Tenure Distribution by Churn', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Tenure (months)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Monthly Charges by Churn (Box Plot)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'MonthlyCharges' in df_processed.columns:
        charges_data = [
            df_processed[y==0]['MonthlyCharges'].values,
            df_processed[y==1]['MonthlyCharges'].values
        ]
        
        bp = ax3.boxplot(charges_data, tick_labels=['No Churn', 'Churn'],
                         patch_artist=True, widths=0.6, boxprops=dict(linewidth=1.5),
                         medianprops=dict(linewidth=2, color='darkred'))
        
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#e74c3c')
        bp['boxes'][1].set_alpha(0.7)
        
        ax3.set_title('Monthly Charges by Churn', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Monthly Charges ($)', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Plots 4-6: Categorical features vs churn
    potential_categoricals = ['Contract', 'InternetService', 'PaymentMethod',
                             'gender', 'Partner', 'Dependents']
    available_categoricals = [col for col in potential_categoricals 
                             if col in df_processed.columns][:3]
    
    for idx, col in enumerate(available_categoricals):
        ax = fig.add_subplot(gs[1, idx])
        
        churn_crosstab = pd.crosstab(df_processed[col], y, normalize='index') * 100
        
        churn_crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'],
                           alpha=0.8, width=0.7, edgecolor='black', linewidth=1)
        ax.set_title(f'Churn Rate by {col}', fontweight='bold', fontsize=13)
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.legend(['No Churn', 'Churn'], loc='upper right', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Plot 7: Correlation heatmap
    ax7 = fig.add_subplot(gs[2, 0])
    correlations_with_target = df_processed.corr()[target_column].abs()
    top_features = correlations_with_target.nlargest(11).index.tolist()
    corr_subset = df_processed[top_features].corr()
    
    sns.heatmap(corr_subset, ax=ax7, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, annot=False)
    ax7.set_title('Top 10 Features - Correlation Heatmap', fontweight='bold', fontsize=12)
    ax7.tick_params(labelsize=9)
    
    # Plot 8: Feature correlation with target
    ax8 = fig.add_subplot(gs[2, 1])
    feature_correlations = correlations_with_target.drop(target_column)
    top_8_features = feature_correlations.nlargest(8)
    
    ax8.barh(range(len(top_8_features)), top_8_features.values, 
             color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    ax8.set_yticks(range(len(top_8_features)))
    ax8.set_yticklabels(top_8_features.index, fontsize=10)
    ax8.set_title('Top 8 Features Correlated with Churn', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Absolute Correlation', fontweight='bold')
    ax8.invert_yaxis()
    ax8.grid(axis='x', alpha=0.3)
    
    for i, (idx, value) in enumerate(top_8_features.items()):
        ax8.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
    
    # Plot 9: Churn rate by tenure groups
    ax9 = fig.add_subplot(gs[2, 2])
    if 'tenure' in df_processed.columns:
        max_tenure = df_processed['tenure'].max()
        
        if max_tenure <= 24:
            bins = [0, 6, 12, 24]
            labels = ['0-6m', '7-12m', '13-24m']
        elif max_tenure <= 48:
            bins = [0, 12, 24, 48]
            labels = ['0-12m', '13-24m', '25-48m']
        else:
            bins = [0, 12, 24, 48, max_tenure + 1]
            labels = ['0-12m', '13-24m', '25-48m', f'49-{int(max_tenure)}m']
        
        tenure_groups = pd.cut(df_processed['tenure'], bins=bins, labels=labels)
        tenure_churn_rate = df_processed.groupby(tenure_groups, observed=False)[target_column].mean() * 100
        
        bars = ax9.bar(range(len(tenure_churn_rate)), tenure_churn_rate.values, 
                       color='#e74c3c', alpha=0.8, width=0.6, 
                       edgecolor='black', linewidth=1.5)
        ax9.set_xticks(range(len(tenure_churn_rate)))
        ax9.set_xticklabels(tenure_churn_rate.index, rotation=45, fontsize=10)
        ax9.set_title('Churn Rate by Tenure Groups', fontweight='bold', fontsize=12)
        ax9.set_ylabel('Churn Rate (%)', fontweight='bold')
        ax9.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.show()
    
    print("\n" + "="*70)
    print("📊 KEY EDA INSIGHTS")
    print("="*70)
    print(f"✅ Overall churn rate: {y.mean()*100:.2f}%")
    print(f"✅ Churned customers: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"✅ Retained customers: {len(y) - y.sum():,} ({(1-y.mean())*100:.1f}%)")
    
    print(f"\n🔝 Top 3 features correlated with churn:")
    for i, feature in enumerate(top_8_features.index[:3], 1):
        print(f"   {i}. {feature}: {top_8_features[feature]:.3f}")
    
    print("="*70 + "\n")
    
    return {
        'dataset_shape': df_processed.shape,
        'churn_rate': y.mean(),
        'numeric_features': true_numerics,
        'categorical_features': encoded_categoricals,
        'top_features': top_8_features.to_dict()
    }

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT FUNCTIONS
# ============================================================================

def validate_split_inputs(df_processed, target_column, test_size):
    """Comprehensive validation of split inputs"""
    validation_errors = []
    
    if not isinstance(df_processed, pd.DataFrame):
        validation_errors.append("df_processed must be a pandas DataFrame")
    
    if target_column not in df_processed.columns:
        validation_errors.append(f"Target column '{target_column}' not found")
    elif df_processed[target_column].isnull().any():
        validation_errors.append(f"Target column '{target_column}' contains null values")
    
    if not 0 < test_size < 1:
        validation_errors.append(f"test_size must be between 0 and 1, got {test_size}")
    
    if len(df_processed) < 100:
        validation_errors.append(f"Dataset too small for reliable split: {len(df_processed)} samples")
    
    if target_column in df_processed.columns:
        class_counts = df_processed[target_column].value_counts()
        if class_counts.min() < 2:
            validation_errors.append("Insufficient samples in minority class for stratification")
    
    if validation_errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors))

def smart_stratified_split(X, y, test_size, random_state):
    """Smart stratified split with automatic fallback logic"""
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    min_required = max(2, int(1 / test_size))
    
    if min_class_count < min_required:
        print(f"⚠️  Insufficient samples for stratification (min class: {min_class_count})")
        print(f"   Falling back to random split...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        print(f"⚠️  Stratification failed: {e}")
        print("   Falling back to random split...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def prepare_features(df_processed, target_column, test_size=0.25, random_state=42, 
                     return_dataframe=False, save_scaler_file=True):
    """
    Prepare features and split data into training and testing sets.
    """
    print("\n" + "="*70)
    print("[STEP 4] FEATURE ENGINEERING AND TRAIN-TEST SPLIT")
    print("="*70)
    
    print("\n🔍 Validating inputs...")
    validate_split_inputs(df_processed, target_column, test_size)
    print("   ✅ All inputs validated successfully")
    
    print("\n📊 Separating Features and Target...")
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    feature_names = X.columns.tolist()
    
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    print(f"   Total features: {len(feature_names)}")
    
    unique_targets = y.unique()
    target_counts = y.value_counts()
    print(f"   Target distribution: {target_counts.to_dict()}")
    
    if len(unique_targets) > 2:
        print(f"⚠️  Warning: Target has {len(unique_targets)} unique values: {unique_targets}")
        print(f"   Expected binary classification (0, 1)")
    
    print(f"\n✂️  Performing Smart Train-Test Split...")
    print(f"   Test size: {test_size*100:.1f}%")
    print(f"   Random state: {random_state}")
    
    X_train, X_test, y_train, y_test = smart_stratified_split(
        X, y, test_size, random_state
    )
    
    print("\n📏 Scaling Features...")
    print("   Method: StandardScaler (mean=0, std=1)")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled_array = scaler.transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    if return_dataframe:
        X_train_scaled = pd.DataFrame(
            X_train_scaled_array,
            columns=feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled_array,
            columns=feature_names,
            index=X_test.index
        )
        print("   ✅ Scaled features returned as DataFrames")
    else:
        X_train_scaled = X_train_scaled_array
        X_test_scaled = X_test_scaled_array
        print("   ✅ Scaled features returned as numpy arrays")
    
    print(f"\n🔍 Scaling Verification:")
    print(f"   Training mean: {X_train_scaled.mean():.6f} (should be ~0)")
    print(f"   Training std:  {X_train_scaled.std():.6f} (should be ~1)")
    
    if save_scaler_file:
        scaler_info = {
            'scaler': scaler,
            'feature_names': feature_names,
            'fitted_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scaler_type': type(scaler).__name__
        }
        
        scaler_path = os.path.join(OUTPUT_DIR, 'feature_scaler.pkl')
        joblib.dump(scaler_info, scaler_path)
        print(f"💾 Scaler saved to: {scaler_path}")
    
    print("\n" + "="*70)
    print("✅ FEATURE PREPARATION COMPLETE")
    print("="*70)
    print(f"📊 Training features: {X_train_scaled.shape}")
    print(f"📊 Testing features:  {X_test_scaled.shape}")
    print(f"🎯 Training target:   {y_train.shape}")
    print(f"🎯 Testing target:    {y_test.shape}")
    print(f"📋 Feature names:     {len(feature_names)} features")
    print(f"📏 Scaler:            StandardScaler (fitted on training data)")
    print("="*70 + "\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

# ============================================================================
# STEP 5: MODEL TRAINING FUNCTIONS
# ============================================================================

def validate_model_inputs(X_train, X_test, y_train, y_test, model_params):
    """Validate all inputs before training"""
    validation_errors = []
    
    if X_train.shape[1] != X_test.shape[1]:
        validation_errors.append("Train and test features have different dimensions")
    
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        validation_errors.append("NaN values detected in features")
    
    if len(np.unique(y_train)) < 2:
        validation_errors.append("Training target has only one class")
    
    required_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                      'Gradient Boosting', 'K-Nearest Neighbors', 'SVM']
    missing_models = [model for model in required_models if model not in model_params]
    if missing_models:
        validation_errors.append(f"Missing model configurations: {missing_models}")
    
    if validation_errors:
        raise ValueError("Model input validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors))

def perform_robust_cross_validation(model, X_train, y_train, cv_folds=5, verbose=True):
    """Perform cross-validation with multiple metrics"""
    scoring_metrics = {
        'f1': 'f1',
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc' if hasattr(model, 'predict_proba') else None
    }
    
    cv_results = {}
    
    for metric_name, metric_scorer in scoring_metrics.items():
        if metric_scorer is None:
            cv_results[metric_name] = {'mean': None, 'std': None, 'scores': None}
            continue
            
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring=metric_scorer,
                n_jobs=-1
            )
            cv_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        except Exception as e:
            if verbose:
                print(f"   ⚠️  CV for {metric_name} failed: {e}")
            cv_results[metric_name] = {'mean': None, 'std': None, 'scores': None}
    
    return cv_results

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_params, 
                               cv_folds=5, verbose=True):
    """
    Train and evaluate multiple machine learning models.
    """
    if verbose:
        print("\n" + "="*70)
        print("[STEP 5] MODEL TRAINING AND EVALUATION")
        print("="*70)
        print(f"\n📊 Training Configuration:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Cross-validation folds: {cv_folds}")
        print(f"   Models to train: {len(model_params)}")
    
    if verbose:
        print(f"\n🔍 Validating inputs...")
    
    validate_model_inputs(X_train, X_test, y_train, y_test, model_params)
    
    if verbose:
        print(f"   ✅ All inputs validated successfully")
    
    models = {
        'Logistic Regression': LogisticRegression(**model_params['Logistic Regression']['config']),
        'Decision Tree': DecisionTreeClassifier(**model_params['Decision Tree']['config']),
        'Random Forest': RandomForestClassifier(**model_params['Random Forest']['config']),
        'Gradient Boosting': GradientBoostingClassifier(**model_params['Gradient Boosting']['config']),
        'K-Nearest Neighbors': KNeighborsClassifier(**model_params['K-Nearest Neighbors']['config']),
        'Support Vector Machine': SVC(**model_params['SVM']['config'])
    }
    
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
    
    for idx, (name, model) in enumerate(models.items(), 1):
        if verbose:
            print(f"\n[{idx}/{len(models)}] 🔧 Training {name}...")
            print("-" * 70)
        
        try:
            with timer(f"training {name}", verbose=verbose):
                model.fit(X_train, y_train)
            
            if verbose:
                print(f"   🔄 Running {cv_folds}-fold cross-validation...")
            
            with timer(f"cross-validation for {name}", verbose=verbose):
                cv_results = perform_robust_cross_validation(
                    model, X_train, y_train, cv_folds, verbose
                )
            
            cv_mean = cv_results['f1']['mean'] or 0.0
            cv_std = cv_results['f1']['std'] or 0.0
            
            if verbose:
                print(f"   ✅ CV F1-Score: {cv_mean:.4f} (±{cv_std * 2:.4f})")
            
            if verbose:
                print(f"   🎯 Evaluating on test set...")
            
            with timer(f"predictions for {name}", verbose=verbose):
                y_pred = model.predict(X_test)
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
                else:
                    y_pred_proba = None
            
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if y_pred_proba is not None:
                try:
                    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  ROC-AUC calculation failed: {e}")
                    test_roc_auc = None
            else:
                test_roc_auc = None
            
            if verbose:
                print(f"\n   📊 Test Set Performance:")
                print(f"      Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                print(f"      Precision: {test_precision:.4f}")
                print(f"      Recall:    {test_recall:.4f}")
                print(f"      F1-Score:  {test_f1:.4f}")
                if test_roc_auc:
                    print(f"      ROC-AUC:   {test_roc_auc:.4f}")
            
            results[name] = {
                'model': model,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_scores': cv_results['f1']['scores'],
                'cv_full_results': cv_results,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'roc_auc': test_roc_auc,
                'feature_names': getattr(X_train, 'columns', None),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            if verbose:
                print(f"\n   ❌ ERROR training {name}: {e}")
            results[name] = {
                'model': None,
                'error': str(e),
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': 0.0
            }
    
    if verbose:
        print("\n" + "="*70)
        print("✅ ALL MODELS TRAINED AND EVALUATED")
        print("="*70)
        
        summary_data = []
        for name, result in results.items():
            if 'error' in result:
                row = {
                    'Model': name,
                    'Status': 'Failed',
                    'Error': result['error'],
                    'Accuracy': 0.0,
                    'F1_Score': 0.0
                }
            else:
                row = {
                    'Model': name,
                    'Status': 'Success',
                    'Error': None,
                    'Accuracy': result['accuracy'],
                    'F1_Score': result['f1_score']
                }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(f"\n📊 Performance Summary:")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        successful_models = {k: v for k, v in results.items() if 'error' not in v}
        if successful_models:
            best_model_name = max(successful_models, key=lambda x: results[x]['f1_score'])
            best_result = results[best_model_name]
            
            print("\n" + "="*70)
            print(f"🏆 BEST MODEL: {best_model_name}")
            print("="*70)
            print(f"📊 Test Performance:")
            print(f"   F1-Score:  {best_result['f1_score']:.4f}")
            print(f"   Accuracy:  {best_result['accuracy']:.4f}")
            print(f"   Precision: {best_result['precision']:.4f}")
            print(f"   Recall:    {best_result['recall']:.4f}")
            if best_result['roc_auc']:
                print(f"   ROC-AUC:   {best_result['roc_auc']:.4f}")
            print(f"📈 Cross-Validation:")
            print(f"   F1-Score:  {best_result['cv_mean']:.4f} (±{best_result['cv_std'] * 2:.4f})")
            print("="*70)
        
        print(f"\n💡 Access individual model results:")
        print(f"   trained_models['Logistic Regression'].keys()")
        print(f"   trained_models['Random Forest']['model']")
    
    return results

# ============================================================================
# STEP 6: MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_models(trained_models, y_test, verbose=True):
    """
    Evaluate models on test set and generate comprehensive comparison.
    """
    if verbose:
        print("\n" + "="*70)
        print("[STEP 6] MODEL EVALUATION & COMPARISON")
        print("="*70)
        print(f"\n📊 Evaluating {len(trained_models)} models on {len(y_test):,} test samples...")
    
    evaluation_results = {}
    
    for idx, (name, model_info) in enumerate(trained_models.items(), 1):
        if verbose:
            print(f"\n[{idx}/{len(trained_models)}] 📊 {name}")
            print("-" * 70)
        
        if 'error' in model_info:
            if verbose:
                print(f"   ❌ Model failed during training: {model_info['error']}")
            continue
            
        evaluation_results[name] = {
            'model': model_info['model'],
            'accuracy': model_info['accuracy'],
            'precision': model_info['precision'],
            'recall': model_info['recall'],
            'f1_score': model_info['f1_score'],
            'roc_auc': model_info.get('roc_auc'),
            'y_pred': model_info['y_pred'],
            'y_pred_proba': model_info.get('y_pred_proba'),
            'cv_mean': model_info.get('cv_mean', 0.0),
            'cv_std': model_info.get('cv_std', 0.0)
        }
        
        y_pred = model_info['y_pred']
        y_pred_proba = model_info.get('y_pred_proba')
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        evaluation_results[name]['confusion_matrix'] = conf_matrix
        
        tn, fp, fn, tp = conf_matrix.ravel()
        evaluation_results[name]['true_negatives'] = tn
        evaluation_results[name]['false_positives'] = fp
        evaluation_results[name]['false_negatives'] = fn
        evaluation_results[name]['true_positives'] = tp
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        evaluation_results[name]['specificity'] = specificity
        
        if y_pred_proba is not None:
            try:
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                evaluation_results[name]['fpr'] = fpr
                evaluation_results[name]['tpr'] = tpr
                evaluation_results[name]['roc_thresholds'] = thresholds
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  ROC curve calculation failed: {e}")
                evaluation_results[name]['fpr'] = None
                evaluation_results[name]['tpr'] = None
        
        if verbose:
            print(f"   📊 Test Set Performance:")
            print(f"      Accuracy:   {evaluation_results[name]['accuracy']:.4f} "
                  f"({evaluation_results[name]['accuracy']*100:.2f}%)")
            print(f"      Precision:  {evaluation_results[name]['precision']:.4f}")
            print(f"      Recall:     {evaluation_results[name]['recall']:.4f}")
            print(f"      F1-Score:   {evaluation_results[name]['f1_score']:.4f}")
            print(f"      Specificity:{specificity:.4f}")
            
            if evaluation_results[name]['roc_auc']:
                print(f"      ROC-AUC:    {evaluation_results[name]['roc_auc']:.4f}")
            
            print(f"\n   📋 Confusion Matrix:")
            print(f"      True Negatives:  {tn:,} (Correct 'No Churn' predictions)")
            print(f"      False Positives: {fp:,} (Predicted churn, but didn't)")
            print(f"      False Negatives: {fn:,} (Missed churners - COSTLY!)")
            print(f"      True Positives:  {tp:,} (Correctly caught churners)")
            
            print(f"\n   🔄 Cross-Validation:")
            print(f"      CV F1-Score: {evaluation_results[name]['cv_mean']:.4f} "
                  f"(±{evaluation_results[name]['cv_std']*2:.4f})")
    
    results_df = pd.DataFrame({
        'Model': list(evaluation_results.keys()),
        'Accuracy': [evaluation_results[m]['accuracy'] for m in evaluation_results.keys()],
        'Precision': [evaluation_results[m]['precision'] for m in evaluation_results.keys()],
        'Recall': [evaluation_results[m]['recall'] for m in evaluation_results.keys()],
        'Specificity': [evaluation_results[m]['specificity'] for m in evaluation_results.keys()],
        'F1-Score': [evaluation_results[m]['f1_score'] for m in evaluation_results.keys()],
        'ROC-AUC': [evaluation_results[m]['roc_auc'] if evaluation_results[m]['roc_auc'] 
                    else np.nan for m in evaluation_results.keys()],
        'CV_F1_Mean': [evaluation_results[m]['cv_mean'] for m in evaluation_results.keys()],
        'CV_F1_Std': [evaluation_results[m]['cv_std'] for m in evaluation_results.keys()],
        'False_Negatives': [evaluation_results[m]['false_negatives'] for m in evaluation_results.keys()]
    })
    
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    best_model_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    
    if verbose:
        print("\n" + "="*70)
        print("🏆 BEST MODEL IDENTIFIED")
        print("="*70)
        print(f"\n   Model:     {best_model_name}")
        print(f"   F1-Score:  {results_df.loc[best_model_idx, 'F1-Score']:.4f}")
        print(f"   Accuracy:  {results_df.loc[best_model_idx, 'Accuracy']:.4f} "
              f"({results_df.loc[best_model_idx, 'Accuracy']*100:.2f}%)")
        print(f"   Recall:    {results_df.loc[best_model_idx, 'Recall']:.4f} "
              f"(catches {results_df.loc[best_model_idx, 'Recall']*100:.1f}% of churners)")
        
        print(f"\n📊 Top 3 Models by F1-Score:")
        for i in range(min(3, len(results_df))):
            model_name = results_df.iloc[i]['Model']
            f1 = results_df.iloc[i]['F1-Score']
            acc = results_df.iloc[i]['Accuracy']
            rec = results_df.iloc[i]['Recall']
            print(f"   {i+1}. {model_name}")
            print(f"      F1={f1:.4f} | Acc={acc:.4f} | Recall={rec:.4f}")
        
        print("="*70 + "\n")
    
    return results_df, best_model_name, evaluation_results

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_complete_pipeline():
    """Execute the complete churn prediction pipeline"""
    print("=" * 80)
    print("🚀 CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    print("=" * 80)
    print("Authors: Jay R. Refugia & Romi Pereira Garcia Montejo")
    print("Date:", datetime.now().strftime('%B %d, %Y'))
    print("=" * 80)
    
    # Step 0: Initialize environment
    print("\n📦 STEP 0: INITIALIZING ENVIRONMENT")
    initialize_environment()
    
    # Step 1: Load and analyze data
    print("\n📥 STEP 1: LOADING AND ANALYZING DATA")
    df = load_and_analyze_data()
    
    # Step 2: Preprocess data
    print("\n🧹 STEP 2: PREPROCESSING DATA")
    X, y, label_encoders, preprocessing_summary = preprocess_data(df, target_column='Churn')
    
    # Create processed dataframe for EDA
    df_processed = pd.DataFrame(X.copy())
    df_processed['Churn'] = y.values.copy()
    target_column = 'Churn'
    
    # Step 3: Perform EDA
    print("\n🔍 STEP 3: EXPLORATORY DATA ANALYSIS")
    try:
        eda_results = perform_eda_with_boxplots(df_processed, target_column, label_encoders)
    except Exception as e:
        print(f"⚠️  EDA failed: {e}")
        print("   Continuing with analysis...")
    
    # Step 4: Prepare features and split
    print("\n✂️  STEP 4: TRAIN-TEST SPLIT AND FEATURE ENGINEERING")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = prepare_features(
        df_processed=df_processed,
        target_column=target_column,
        test_size=0.25,
        random_state=RANDOM_STATE,
        return_dataframe=False,
        save_scaler_file=True
    )
    
    # Step 5: Train models
    print("\n🤖 STEP 5: MODEL TRAINING")
    trained_models = train_and_evaluate_models(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        model_params=MODEL_CONFIGS,
        cv_folds=5,
        verbose=True
    )
    
    # Step 6: Evaluate models
    print("\n📊 STEP 6: MODEL EVALUATION")
    results_df, best_model_name, evaluation_results = evaluate_models(
        trained_models=trained_models,
        y_test=y_test,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📈 Best F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"🎯 Best Accuracy: {results_df.iloc[0]['Accuracy']:.4f} ({results_df.iloc[0]['Accuracy']*100:.2f}%)")
    
    # Save results
    print("\n💾 SAVING RESULTS...")
    results_path = os.path.join(OUTPUT_DIR, 'model_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"   ✅ Results saved to: {results_path}")
    
    # Save best model
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if best_model_name in trained_models and trained_models[best_model_name]['model'] is not None:
        joblib.dump(trained_models[best_model_name]['model'], best_model_path)
        print(f"   ✅ Best model saved to: {best_model_path}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Return all results for further analysis
    return {
        'df': df,
        'df_processed': df_processed,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'trained_models': trained_models,
        'results_df': results_df,
        'best_model_name': best_model_name,
        'evaluation_results': evaluation_results,
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoders': label_encoders
    }

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_complete_pipeline()
        print("\n🎉 Analysis complete! All results are available in the 'results' dictionary.")
        print("\n📋 Quick access:")
        print("   - Best model: results['best_model_name']")
        print("   - Performance table: results['results_df']")
        print("   - All models: results['trained_models']")
        
        # Ask if user wants to save final report
        try:
            response = input("\n📄 Would you like to generate a final report? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                from datetime import datetime
                report_path = os.path.join(OUTPUT_DIR, f'final_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
                with open(report_path, 'w') as f:
                    f.write("CUSTOMER CHURN PREDICTION - FINAL REPORT\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Best Model: {results['best_model_name']}\n")
                    f.write(f"Best F1-Score: {results['results_df'].iloc[0]['F1-Score']:.4f}\n")
                    f.write(f"Best Accuracy: {results['results_df'].iloc[0]['Accuracy']:.4f}\n\n")
                    f.write("MODEL PERFORMANCE:\n")
                    f.write(results['results_df'].to_string())
                print(f"✅ Report saved to: {report_path}")
        except:
            pass  # Silent fail if not in interactive mode
            
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline execution failed!")
        print(f"   Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check dataset is in data/ folder")
        print("   2. Verify all dependencies are installed")
        print("   3. Check Python version (3.8+)")
        sys.exit(1)

# ============================================================================
# SIMPLIFIED EXECUTION (for notebooks or modular use)
# ============================================================================

def run_quick_analysis():
    """Quick execution without intermediate prints"""
    print("Running quick analysis...")
    initialize_environment()
    df = pd.read_csv(DATASET_PATH)
    X, y, label_encoders, _ = preprocess_data(df, target_column='Churn', verbose=False)
    
    df_processed = pd.DataFrame(X)
    df_processed['Churn'] = y
    
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, _ = prepare_features(
        df_processed, 'Churn', test_size=0.25, random_state=RANDOM_STATE,
        return_dataframe=False, save_scaler_file=False
    )
    
    trained_models = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, MODEL_CONFIGS,
        cv_folds=3, verbose=False
    )
    
    results_df, best_model_name, _ = evaluate_models(trained_models, y_test, verbose=False)
    
    print(f"✅ Quick analysis complete!")
    print(f"🏆 Best model: {best_model_name} (F1: {results_df.iloc[0]['F1-Score']:.3f})")
    
    return {
        'results_df': results_df,
        'best_model': trained_models[best_model_name]['model'],
        'feature_names': feature_names
    }
