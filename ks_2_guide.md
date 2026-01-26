# Kickstarter Success Prediction - Complete ML Project Guide

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Team Workflow](#3-team-workflow)
4. [Phase 1: Data Loading & Initial Exploration](#4-phase-1-data-loading--initial-exploration)
5. [Phase 2: Exploratory Data Analysis (EDA)](#5-phase-2-exploratory-data-analysis-eda)
6. [Phase 3: Data Cleaning](#6-phase-3-data-cleaning)
7. [Phase 4: Feature Engineering](#7-phase-4-feature-engineering)
8. [Phase 5: Data Preparation (Train-Test Split)](#8-phase-5-data-preparation-train-test-split)
9. [Phase 6: Model Development](#9-phase-6-model-development)
10. [Phase 7: Model Comparison & Selection](#10-phase-7-model-comparison--selection)
11. [Phase 8: Final Presentation](#11-phase-8-final-presentation)
12. [Visualization Catalog](#12-visualization-catalog)
13. [Code Templates](#13-code-templates)
14. [Checklist](#14-checklist)

---

## 1. Project Overview

### Goal
Build a machine learning model to predict whether a Kickstarter project will be **successful** or **failed** before it is released.

### Dataset Columns
| Column | Description |
|--------|-------------|
| `ID` | Unique project identifier |
| `name` | Project name |
| `category` | Subcategory of the project |
| `main_category` | Main category |
| `currency` | Currency of the goal |
| `deadline` | Project deadline date |
| `goal` | Goal amount in original currency |
| `launched` | Launch date |
| `pledged` | Amount pledged in original currency |
| `state` | Project state (successful, failed, canceled, etc.) |
| `backers` | Number of backers |
| `country` | Country of origin |
| `usd_pledged` | USD conversion by Kickstarter |
| `usd_pledged_real` | USD conversion by Fixer.io API |
| `usd_goal_real` | Goal in USD (Fixer.io API) |

### Target Variable
- **Binary Classification**: `state` → `successful` (1) vs `failed` (0)
- **Exclude**: `canceled`, `live`, `suspended`, `undefined` states

---

## 2. Repository Structure

```
kickstarter-ml-project/
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
│
├── data/
│   ├── raw/                       # Original data (DO NOT MODIFY)
│   │   └── ks-projects-201801.csv
│   ├── processed/                 # Cleaned data
│   │   └── ks_cleaned.csv
│   └── features/                  # Feature-engineered data
│       └── ks_features.csv
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_train_test_split.ipynb
│   ├── 06_model_development.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_final_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Data loading utilities
│   │   └── cleaner.py             # Data cleaning functions
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py             # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Model training
│   │   └── evaluator.py           # Model evaluation
│   └── visualization/
│       ├── __init__.py
│       └── plots.py               # Plotting functions
│
├── models/
│   └── saved/                     # Saved model files
│       └── best_model.pkl
│
├── reports/
│   ├── figures/                   # Generated plots
│   └── presentation/              # Final presentation
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── Makefile
```

---

## 3. Team Workflow

### Branch Strategy
```
main (protected)
  └── develop (integration branch)
        ├── feature/data-loading
        ├── feature/eda
        ├── feature/data-cleaning
        ├── feature/feature-engineering
        ├── feature/train-test-split
        ├── feature/model-development
        ├── feature/model-comparison
        └── feature/presentation
```

### Collaboration Pattern (All 3 Developers on Each Task)
Since all team members want to work together on each stage:

1. **Mob Programming Sessions**
   - One driver (types code)
   - Two navigators (review, suggest, research)
   - Rotate every 15-30 minutes
   - Use VS Code Live Share or similar

2. **Pair + Reviewer Pattern**
   - Two developers pair program
   - Third reviews and approves PR
   - Rotate pairs each phase

3. **Divide & Conquer within Phase**
   - Split each phase into 3 sub-tasks
   - Each developer takes one sub-task
   - Sync and merge at phase end

### Communication
- Daily standup (15 min)
- Use GitHub Issues for task tracking
- Use PR comments for code review
- Document decisions in `DECISIONS.md`

---

## 4. Phase 1: Data Loading & Initial Exploration

### Step-by-Step Tasks

#### 4.1 Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 4.2 Load Raw Data
```python
# Sub-tasks for 3 developers:
# Dev 1: Write loader function with error handling
# Dev 2: Add data validation checks
# Dev 3: Create initial data documentation

import pandas as pd
import numpy as np
from pathlib import Path

def load_kickstarter_data(filepath: str) -> pd.DataFrame:
    """Load Kickstarter dataset with proper encoding and parsing."""
    df = pd.read_csv(
        filepath,
        encoding='utf-8',
        parse_dates=['deadline', 'launched'],
        low_memory=False
    )
    return df
```

#### 4.3 Initial Data Inspection Checklist
- [ ] Check DataFrame shape (rows, columns)
- [ ] Display first/last 5 rows
- [ ] Check column names and data types
- [ ] Identify numeric vs categorical columns
- [ ] Check memory usage
- [ ] Get basic statistics with `.describe()`
- [ ] Get info with `.info()`

#### 4.4 Document Initial Findings
Create `data/README.md` with:
- Data source
- Number of records
- Column descriptions
- Initial observations
- Data quality issues spotted

#### 4.5 Attention Points
⚠️ **Watch for:**
- Encoding issues (special characters in project names)
- Date parsing errors
- Multiple CSV files (may need concatenation)
- Large file size (consider chunked loading)
- Duplicate rows

---

## 5. Phase 2: Exploratory Data Analysis (EDA)

### EDA Strategy Framework

#### 5.1 Univariate Analysis (Single Variable)

##### For Numerical Variables:
```python
# Sub-tasks:
# Dev 1: Central tendency (mean, median, mode)
# Dev 2: Dispersion (std, variance, range, IQR)
# Dev 3: Distribution shape (skewness, kurtosis)

numerical_cols = ['goal', 'pledged', 'backers', 'usd_pledged_real', 'usd_goal_real']

for col in numerical_cols:
    print(f"\n=== {col} ===")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Std: {df[col].std():.2f}")
    print(f"Min: {df[col].min():.2f}")
    print(f"Max: {df[col].max():.2f}")
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurtosis():.2f}")
    print(f"Q1: {df[col].quantile(0.25):.2f}")
    print(f"Q3: {df[col].quantile(0.75):.2f}")
```

##### For Categorical Variables:
```python
categorical_cols = ['main_category', 'category', 'currency', 'country', 'state']

for col in categorical_cols:
    print(f"\n=== {col} ===")
    print(f"Unique values: {df[col].nunique()}")
    print(f"Most common: {df[col].mode()[0]}")
    print(f"Value counts:\n{df[col].value_counts().head(10)}")
```

#### 5.2 Bivariate Analysis (Two Variables)

##### Numerical vs Target:
```python
# Success rate analysis
df['is_successful'] = (df['state'] == 'successful').astype(int)

# Compare distributions
successful = df[df['is_successful'] == 1]['usd_goal_real']
failed = df[df['is_successful'] == 0]['usd_goal_real']

print(f"Successful projects - Median goal: ${successful.median():,.0f}")
print(f"Failed projects - Median goal: ${failed.median():,.0f}")
```

##### Categorical vs Target:
```python
# Success rate by category
success_by_category = df.groupby('main_category')['is_successful'].agg(['mean', 'count'])
success_by_category.columns = ['success_rate', 'total_projects']
success_by_category = success_by_category.sort_values('success_rate', ascending=False)
```

#### 5.3 Multivariate Analysis

##### Correlation Matrix:
```python
# Only for numerical columns
correlation_matrix = df[numerical_cols].corr()

# Find highly correlated features
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })
```

#### 5.4 Temporal Analysis
```python
# Extract time features
df['launch_year'] = df['launched'].dt.year
df['launch_month'] = df['launched'].dt.month
df['launch_day'] = df['launched'].dt.day
df['launch_dayofweek'] = df['launched'].dt.dayofweek
df['launch_hour'] = df['launched'].dt.hour

# Success rate over time
yearly_success = df.groupby('launch_year')['is_successful'].mean()
monthly_success = df.groupby('launch_month')['is_successful'].mean()
```

#### 5.5 EDA Checklist
- [ ] Target variable distribution (class balance)
- [ ] Missing values analysis
- [ ] Outlier detection
- [ ] Distribution of numerical features
- [ ] Frequency of categorical features
- [ ] Correlation analysis
- [ ] Feature vs target relationships
- [ ] Temporal patterns
- [ ] Geographic patterns
- [ ] Category-specific patterns

#### 5.6 Key Questions to Answer
1. What is the overall success rate?
2. Which categories have highest/lowest success rates?
3. What goal amounts are most likely to succeed?
4. Does project duration affect success?
5. Are there temporal trends (year, month, day)?
6. Which countries have best success rates?
7. Is there a relationship between goal and backers?

---

## 6. Phase 3: Data Cleaning

### Data Cleaning Strategy

#### 6.1 Handle Target Variable
```python
# Sub-tasks:
# Dev 1: Filter target states
# Dev 2: Create binary target
# Dev 3: Validate class distribution

# Keep only successful and failed projects
valid_states = ['successful', 'failed']
df_clean = df[df['state'].isin(valid_states)].copy()

# Create binary target
df_clean['target'] = (df_clean['state'] == 'successful').astype(int)

print(f"Original rows: {len(df)}")
print(f"After filtering: {len(df_clean)}")
print(f"Class distribution:\n{df_clean['target'].value_counts(normalize=True)}")
```

#### 6.2 Handle Missing Values

##### Strategy Decision Tree:
```
Missing Value Strategy:
│
├── < 5% missing → Consider dropping rows
│
├── 5-30% missing
│   ├── Numerical → Impute (median/mean/mode)
│   └── Categorical → Impute (mode) or create "Unknown" category
│
├── > 30% missing → Consider dropping column
│
└── MCAR/MAR/MNAR analysis → Choose appropriate strategy
```

```python
# Analyze missing values
missing_stats = pd.DataFrame({
    'missing_count': df_clean.isnull().sum(),
    'missing_percent': (df_clean.isnull().sum() / len(df_clean) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_percent', ascending=False)

# Imputation strategies
from sklearn.impute import SimpleImputer

# Numerical imputation
num_imputer = SimpleImputer(strategy='median')
df_clean[numerical_cols] = num_imputer.fit_transform(df_clean[numerical_cols])

# Categorical imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])
```

#### 6.3 Handle Duplicates
```python
# Check for duplicates
print(f"Duplicate rows: {df_clean.duplicated().sum()}")
print(f"Duplicate IDs: {df_clean['ID'].duplicated().sum()}")

# Remove duplicates
df_clean = df_clean.drop_duplicates(subset='ID', keep='first')
```

#### 6.4 Handle Outliers

##### Detection Methods:
```python
# Method 1: IQR Method
def detect_outliers_iqr(df, column, threshold=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Method 2: Z-Score Method
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df[z_scores > threshold]

# Method 3: Percentile Method
def detect_outliers_percentile(df, column, lower=0.01, upper=0.99):
    lower_bound = df[column].quantile(lower)
    upper_bound = df[column].quantile(upper)
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
```

##### Handling Strategies:
```python
# Strategy 1: Cap outliers (Winsorization)
def cap_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower, upper)
    return df

# Strategy 2: Log transformation (for skewed data)
df_clean['log_goal'] = np.log1p(df_clean['usd_goal_real'])

# Strategy 3: Remove outliers (use carefully!)
# Only if outliers are errors, not legitimate extreme values
```

#### 6.5 Fix Data Types
```python
# Ensure correct data types
df_clean['ID'] = df_clean['ID'].astype(str)
df_clean['deadline'] = pd.to_datetime(df_clean['deadline'])
df_clean['launched'] = pd.to_datetime(df_clean['launched'])

# Convert categories to categorical type for memory efficiency
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype('category')
```

#### 6.6 Data Cleaning Checklist
- [ ] Filter valid target states
- [ ] Analyze missing values pattern
- [ ] Apply imputation strategies
- [ ] Remove/handle duplicates
- [ ] Detect outliers
- [ ] Apply outlier handling strategy
- [ ] Fix data types
- [ ] Validate data integrity
- [ ] Document all transformations
- [ ] Save cleaned data

---

## 7. Phase 4: Feature Engineering

### Feature Engineering Strategies

#### 7.1 Feature Categories

##### A. Temporal Features
```python
# Sub-tasks:
# Dev 1: Date-based features
# Dev 2: Duration features
# Dev 3: Cyclical encoding

# Date-based features
df_clean['launch_year'] = df_clean['launched'].dt.year
df_clean['launch_month'] = df_clean['launched'].dt.month
df_clean['launch_day'] = df_clean['launched'].dt.day
df_clean['launch_dayofweek'] = df_clean['launched'].dt.dayofweek
df_clean['launch_hour'] = df_clean['launched'].dt.hour
df_clean['launch_quarter'] = df_clean['launched'].dt.quarter

# Duration features
df_clean['campaign_duration_days'] = (df_clean['deadline'] - df_clean['launched']).dt.days

# Is weekend launch?
df_clean['is_weekend_launch'] = df_clean['launch_dayofweek'].isin([5, 6]).astype(int)

# Cyclical encoding for months (preserves January-December continuity)
df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['launch_month'] / 12)
df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['launch_month'] / 12)
```

##### B. Text-based Features
```python
# Project name features
df_clean['name_length'] = df_clean['name'].str.len()
df_clean['name_word_count'] = df_clean['name'].str.split().str.len()
df_clean['has_exclamation'] = df_clean['name'].str.contains('!').astype(int)
df_clean['has_question'] = df_clean['name'].str.contains('\?').astype(int)
df_clean['name_uppercase_ratio'] = df_clean['name'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
)
```

##### C. Goal-based Features
```python
# Goal transformations
df_clean['log_goal'] = np.log1p(df_clean['usd_goal_real'])
df_clean['sqrt_goal'] = np.sqrt(df_clean['usd_goal_real'])

# Goal binning
df_clean['goal_category'] = pd.cut(
    df_clean['usd_goal_real'],
    bins=[0, 1000, 5000, 10000, 50000, 100000, float('inf')],
    labels=['micro', 'small', 'medium', 'large', 'xlarge', 'mega']
)
```

##### D. Category-based Features
```python
# Category success rate (target encoding - careful with leakage!)
# Must be calculated on training set only and applied to test set
category_success = df_train.groupby('main_category')['target'].mean()
df_clean['category_success_rate'] = df_clean['main_category'].map(category_success)

# Category project count (popularity)
category_counts = df_clean['main_category'].value_counts()
df_clean['category_project_count'] = df_clean['main_category'].map(category_counts)
```

##### E. Country-based Features
```python
# Country success rate
country_success = df_train.groupby('country')['target'].mean()
df_clean['country_success_rate'] = df_clean['country'].map(country_success)

# Is US project?
df_clean['is_us'] = (df_clean['country'] == 'US').astype(int)
```

#### 7.2 Feature Encoding Strategies

##### One-Hot Encoding (for nominal categories):
```python
# For low cardinality categories
df_encoded = pd.get_dummies(
    df_clean,
    columns=['main_category', 'country'],
    drop_first=True,  # Avoid multicollinearity
    dtype=int
)
```

##### Label Encoding (for ordinal categories):
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_clean['goal_category_encoded'] = le.fit_transform(df_clean['goal_category'])
```

##### Target Encoding (use with caution - risk of data leakage):
```python
# Only calculate on training data!
from category_encoders import TargetEncoder

te = TargetEncoder(cols=['main_category', 'category'])
# Fit only on training data
te.fit(X_train, y_train)
X_train_encoded = te.transform(X_train)
X_test_encoded = te.transform(X_test)
```

#### 7.3 Feature Selection Strategies

##### A. Filter Methods:
```python
# Correlation with target
correlations = df_clean.corrwith(df_clean['target']).abs().sort_values(ascending=False)

# Variance threshold
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

##### B. Wrapper Methods:
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=15
)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_].tolist()
```

##### C. Embedded Methods:
```python
# Feature importance from tree-based models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

#### 7.4 Feature Engineering Checklist
- [ ] Create temporal features
- [ ] Create text-based features
- [ ] Create goal-based features
- [ ] Create category aggregations
- [ ] Apply appropriate encoding
- [ ] Handle high cardinality categories
- [ ] Check for feature leakage
- [ ] Perform feature selection
- [ ] Document all new features
- [ ] Save feature-engineered data

---

## 8. Phase 5: Data Preparation (Train-Test Split)

### Train-Test Split Strategies

#### 8.1 Basic Random Split
```python
from sklearn.model_selection import train_test_split

# Standard 80-20 split
X = df_clean.drop(columns=['target', 'state', 'ID', 'name'])
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class balance
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train class balance: {y_train.value_counts(normalize=True).to_dict()}")
print(f"Test class balance: {y_test.value_counts(normalize=True).to_dict()}")
```

#### 8.2 Train-Validation-Test Split
```python
# 70-15-15 split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 15/85
)

print(f"Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
```

#### 8.3 Time-based Split (Recommended for Temporal Data)
```python
# Sort by launch date
df_sorted = df_clean.sort_values('launched')

# Use last 20% as test (future prediction scenario)
split_idx = int(len(df_sorted) * 0.8)

df_train = df_sorted.iloc[:split_idx]
df_test = df_sorted.iloc[split_idx:]

X_train = df_train.drop(columns=['target', 'state', 'ID', 'name'])
y_train = df_train['target']
X_test = df_test.drop(columns=['target', 'state', 'ID', 'name'])
y_test = df_test['target']

print(f"Train date range: {df_train['launched'].min()} to {df_train['launched'].max()}")
print(f"Test date range: {df_test['launched'].min()} to {df_test['launched'].max()}")
```

#### 8.4 Cross-Validation Setup
```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Stratified K-Fold (maintains class balance)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time Series Split (for temporal data)
tscv = TimeSeriesSplit(n_splits=5)
```

#### 8.5 Feature Scaling

##### When to Scale:
- **Always scale for:** Linear Regression, Logistic Regression, SVM, KNN, Neural Networks
- **No scaling needed for:** Tree-based models (RF, XGBoost, LightGBM)

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: mean=0, std=1 (best for normally distributed data)
scaler = StandardScaler()

# MinMaxScaler: range [0, 1] (best for bounded data)
scaler = MinMaxScaler()

# RobustScaler: uses median and IQR (best for data with outliers)
scaler = RobustScaler()

# IMPORTANT: Fit on training data only!
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler, no fit!
```

#### 8.6 Handle Class Imbalance

##### Check Imbalance:
```python
print(f"Class distribution:\n{y_train.value_counts()}")
imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Imbalance ratio: {imbalance_ratio:.2f}")
```

##### Strategies:
```python
# Strategy 1: Class weights in model
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Strategy 2: SMOTE (Synthetic Minority Over-sampling)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Strategy 3: Random Undersampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
```

#### 8.7 Train-Test Split Checklist
- [ ] Remove non-feature columns (ID, name, target)
- [ ] Choose split strategy (random vs temporal)
- [ ] Apply stratification if needed
- [ ] Check class balance in splits
- [ ] Apply feature scaling (fit on train only!)
- [ ] Address class imbalance if needed
- [ ] Set up cross-validation
- [ ] Save preprocessed data
- [ ] Document preprocessing pipeline

---

## 9. Phase 6: Model Development

### Model Development Strategies

#### 9.1 Baseline Model
```python
# Always start with a simple baseline!
from sklearn.dummy import DummyClassifier

# Strategy: most_frequent (always predicts majority class)
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_accuracy = baseline.score(X_test, y_test)
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
```

#### 9.2 Model Selection

##### A. Logistic Regression (Interpretable baseline)
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
lr.fit(X_train_scaled, y_train)
```

##### B. Decision Tree (Interpretable, prone to overfitting)
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train, y_train)
```

##### C. Random Forest (Robust, good default choice)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
```

##### D. Gradient Boosting (XGBoost)
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

##### E. LightGBM (Fast, handles large datasets)
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    class_weight='balanced',
    random_state=42
)
lgb_model.fit(X_train, y_train)
```

##### F. CatBoost (Handles categorical features natively)
```python
from catboost import CatBoostClassifier

cb_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)
cb_model.fit(X_train, y_train)
```

#### 9.3 Hyperparameter Tuning

##### Grid Search:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

##### Randomized Search (faster for large grids):
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

##### Optuna (Advanced optimization):
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }
    
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
```

#### 9.4 Model Development Checklist
- [ ] Create baseline model
- [ ] Train multiple model types
- [ ] Apply cross-validation
- [ ] Perform hyperparameter tuning
- [ ] Save model configurations
- [ ] Document training process
- [ ] Track experiments (MLflow/W&B)

---

## 10. Phase 7: Model Comparison & Selection

### Model Evaluation Strategies

#### 10.1 Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'PR-AUC': average_precision_score(y_test, y_prob) if y_prob is not None else None
    }
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    if metrics['ROC-AUC']:
        print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        print(f"PR-AUC:    {metrics['PR-AUC']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Failed', 'Successful']))
    
    return metrics
```

#### 10.2 Which Metric to Prioritize?

| Business Goal | Priority Metric |
|---------------|-----------------|
| Minimize false positives (don't predict success for failures) | **Precision** |
| Minimize false negatives (catch all successes) | **Recall** |
| Balance precision and recall | **F1-Score** |
| Overall correctness | **Accuracy** (only if balanced classes) |
| Ranking/probability quality | **ROC-AUC** or **PR-AUC** |

For Kickstarter prediction:
- **If advising creators**: High recall (don't miss potential successes)
- **If advising investors**: High precision (don't fund failures)
- **General purpose**: F1-Score or ROC-AUC

#### 10.3 Compare All Models

```python
# Store all models
models = {
    'Logistic Regression': lr,
    'Decision Tree': dt,
    'Random Forest': rf,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cb_model
}

# Evaluate all models
results = []
for name, model in models.items():
    metrics = evaluate_model(model, X_test, y_test, name)
    results.append(metrics)

# Create comparison DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string(index=False))
```

#### 10.4 Statistical Comparison

```python
from scipy import stats

# Compare best models using paired t-test on cross-validation scores
from sklearn.model_selection import cross_val_score

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=10, scoring='f1')
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='f1')

t_stat, p_value = stats.ttest_rel(cv_scores_rf, cv_scores_xgb)
print(f"RF mean F1: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"XGB mean F1: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant!")
else:
    print("No significant difference between models.")
```

#### 10.5 Feature Importance Analysis

```python
# For tree-based models
def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=300)
    plt.show()

# For Logistic Regression (coefficients)
def plot_lr_coefficients(model, feature_names, top_n=20):
    coef = model.coef_[0]
    indices = np.argsort(np.abs(coef))[-top_n:]
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if c < 0 else 'green' for c in coef[indices]]
    plt.barh(range(len(indices)), coef[indices], color=colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression Coefficients')
    plt.tight_layout()
    plt.savefig('reports/figures/lr_coefficients.png', dpi=300)
    plt.show()
```

#### 10.6 Save Best Model

```python
import joblib

# Save the best model
best_model = rf  # or whichever performed best
joblib.dump(best_model, 'models/saved/best_model.pkl')

# Save the scaler if used
joblib.dump(scaler, 'models/saved/scaler.pkl')

# Save feature names
import json
with open('models/saved/feature_names.json', 'w') as f:
    json.dump(list(X_train.columns), f)
```

#### 10.7 Model Comparison Checklist
- [ ] Calculate all relevant metrics
- [ ] Create comparison table
- [ ] Perform statistical tests
- [ ] Analyze feature importance
- [ ] Check for overfitting (train vs test scores)
- [ ] Validate on holdout set
- [ ] Document final model choice
- [ ] Save best model and artifacts

---

## 11. Phase 8: Final Presentation

### Presentation Structure

#### Slide 1: Title
- Project Name: "Kickstarter Success Prediction"
- Team Members
- Date

#### Slide 2: Problem Statement
- Goal: Predict project success before launch
- Business value: Help creators optimize projects

#### Slide 3: Dataset Overview
- Source: Kickstarter
- Size: X projects, Y features
- Time period covered
- Key columns

#### Slide 4: Data Quality
- Missing values summary
- Outliers found
- Cleaning steps applied

#### Slide 5: Exploratory Analysis - Key Insights
- Overall success rate: X%
- Top performing categories
- Goal amount patterns
- Include 2-3 key visualizations

#### Slide 6: Feature Engineering
- New features created
- Feature selection results
- Top predictive features

#### Slide 7: Modeling Approach
- Models tested
- Cross-validation strategy
- Hyperparameter tuning

#### Slide 8: Results - Model Comparison
- Comparison table
- Best model performance
- Key metrics (F1, ROC-AUC)

#### Slide 9: Best Model Deep Dive
- Confusion matrix
- ROC curve
- Feature importance

#### Slide 10: Business Insights
- What makes a project successful?
- Actionable recommendations
- Limitations

#### Slide 11: Future Work
- Additional data sources
- Model improvements
- Deployment considerations

#### Slide 12: Q&A

---

## 12. Visualization Catalog

### EDA Visualizations

#### 1. Target Distribution
```python
fig, ax = plt.subplots(figsize=(8, 6))
df['state'].value_counts().plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'])
ax.set_title('Distribution of Project States')
ax.set_xlabel('State')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/target_distribution.png', dpi=300)
```

#### 2. Success Rate by Category
```python
fig, ax = plt.subplots(figsize=(12, 6))
success_rate = df.groupby('main_category')['is_successful'].mean().sort_values(ascending=True)
success_rate.plot(kind='barh', ax=ax, color='#3498db')
ax.set_xlabel('Success Rate')
ax.set_title('Success Rate by Main Category')
ax.axvline(x=df['is_successful'].mean(), color='red', linestyle='--', label='Overall Average')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/success_by_category.png', dpi=300)
```

#### 3. Goal Distribution (Log Scale)
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
axes[0].hist(df['usd_goal_real'], bins=50, color='#9b59b6', edgecolor='white')
axes[0].set_xlabel('Goal (USD)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Goal Distribution (Original)')

# Log-transformed
axes[1].hist(np.log1p(df['usd_goal_real']), bins=50, color='#1abc9c', edgecolor='white')
axes[1].set_xlabel('Log(Goal + 1)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Goal Distribution (Log-transformed)')

plt.tight_layout()
plt.savefig('reports/figures/goal_distribution.png', dpi=300)
```

#### 4. Box Plot: Goal by Success Status
```python
fig, ax = plt.subplots(figsize=(10, 6))
df_filtered = df[df['usd_goal_real'] < df['usd_goal_real'].quantile(0.95)]  # Remove extreme outliers
df_filtered.boxplot(column='usd_goal_real', by='state', ax=ax)
ax.set_xlabel('State')
ax.set_ylabel('Goal (USD)')
ax.set_title('Goal Distribution by Project State')
plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.savefig('reports/figures/goal_by_state_boxplot.png', dpi=300)
```

#### 5. Correlation Heatmap
```python
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=300)
```

#### 6. Temporal Analysis
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Success rate by year
yearly = df.groupby('launch_year')['is_successful'].mean()
axes[0, 0].plot(yearly.index, yearly.values, marker='o', color='#3498db')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Success Rate')
axes[0, 0].set_title('Success Rate by Launch Year')

# Success rate by month
monthly = df.groupby('launch_month')['is_successful'].mean()
axes[0, 1].bar(monthly.index, monthly.values, color='#2ecc71')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Success Rate')
axes[0, 1].set_title('Success Rate by Launch Month')
axes[0, 1].set_xticks(range(1, 13))

# Success rate by day of week
daily = df.groupby('launch_dayofweek')['is_successful'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1, 0].bar(days, daily.values, color='#e74c3c')
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Success Rate')
axes[1, 0].set_title('Success Rate by Day of Week')

# Projects over time
monthly_count = df.groupby([df['launched'].dt.year, df['launched'].dt.month]).size()
axes[1, 1].plot(range(len(monthly_count)), monthly_count.values, color='#9b59b6')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Number of Projects')
axes[1, 1].set_title('Projects Launched Over Time')

plt.tight_layout()
plt.savefig('reports/figures/temporal_analysis.png', dpi=300)
```

#### 7. Geographic Analysis
```python
fig, ax = plt.subplots(figsize=(14, 8))
country_stats = df.groupby('country').agg({
    'is_successful': ['mean', 'count']
}).round(3)
country_stats.columns = ['success_rate', 'count']
country_stats = country_stats[country_stats['count'] >= 100].sort_values('success_rate', ascending=True)

colors = plt.cm.RdYlGn(country_stats['success_rate'])
bars = ax.barh(country_stats.index, country_stats['success_rate'], color=colors)
ax.set_xlabel('Success Rate')
ax.set_title('Success Rate by Country (min 100 projects)')
ax.axvline(x=df['is_successful'].mean(), color='black', linestyle='--', label='Overall Average')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/success_by_country.png', dpi=300)
```

### Model Evaluation Visualizations

#### 8. Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Failed', 'Successful'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('reports/figures/confusion_matrix.png', dpi=300)
```

#### 9. ROC Curve
```python
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('reports/figures/roc_curves.png', dpi=300)
```

#### 10. Precision-Recall Curve
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

fig, ax = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('reports/figures/pr_curves.png', dpi=300)
```

#### 11. Feature Importance
```python
fig, ax = plt.subplots(figsize=(10, 8))
importance = best_model.feature_importances_
indices = np.argsort(importance)[-20:]
ax.barh(range(len(indices)), importance[indices], color='#3498db')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Feature Importance')
ax.set_title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('reports/figures/feature_importance.png', dpi=300)
```

#### 12. Learning Curves
```python
from sklearn.model_selection import learning_curve

fig, ax = plt.subplots(figsize=(10, 6))

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='f1', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

ax.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training Score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#3498db')
ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Cross-Validation Score')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#e74c3c')

ax.set_xlabel('Training Set Size')
ax.set_ylabel('F1 Score')
ax.set_title('Learning Curves')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('reports/figures/learning_curves.png', dpi=300)
```

#### 13. Model Comparison Bar Chart
```python
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, results_df[metric], width, label=metric)

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('reports/figures/model_comparison.png', dpi=300)
```

---

## 13. Code Templates

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
catboost>=1.2
imbalanced-learn>=0.10.0
optuna>=3.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
jupyter>=1.0.0
jupyterlab>=4.0.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.1.0
isort>=5.12.0
pre-commit>=3.4.0
joblib>=1.3.0
category_encoders>=2.6.0
```

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="kickstarter-ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.10",
)
```

### Makefile
```makefile
.PHONY: install test lint format clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info

data:
	python src/data/loader.py

train:
	python src/models/trainer.py

evaluate:
	python src/models/evaluator.py
```

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg
*.egg-info/
dist/
build/
venv/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (large files)
data/raw/*.csv
data/raw/*.zip
*.csv

# Models
models/saved/*.pkl
models/saved/*.joblib

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
```

---

## 14. Checklist

### Pre-Project
- [ ] Set up repository structure
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Configure Git workflow
- [ ] Assign initial tasks

### Phase 1: Data Loading
- [ ] Load dataset
- [ ] Initial inspection
- [ ] Document data source

### Phase 2: EDA
- [ ] Univariate analysis
- [ ] Bivariate analysis
- [ ] Multivariate analysis
- [ ] Temporal analysis
- [ ] Geographic analysis
- [ ] Document insights

### Phase 3: Data Cleaning
- [ ] Handle missing values
- [ ] Remove duplicates
- [ ] Handle outliers
- [ ] Fix data types
- [ ] Filter target variable
- [ ] Save cleaned data

### Phase 4: Feature Engineering
- [ ] Create temporal features
- [ ] Create text features
- [ ] Create goal features
- [ ] Encode categorical variables
- [ ] Feature selection
- [ ] Save feature data

### Phase 5: Train-Test Split
- [ ] Choose split strategy
- [ ] Apply stratification
- [ ] Scale features
- [ ] Handle class imbalance
- [ ] Set up cross-validation

### Phase 6: Model Development
- [ ] Train baseline model
- [ ] Train multiple models
- [ ] Hyperparameter tuning
- [ ] Track experiments

### Phase 7: Model Comparison
- [ ] Calculate all metrics
- [ ] Statistical comparison
- [ ] Feature importance
- [ ] Select best model
- [ ] Save final model

### Phase 8: Presentation
- [ ] Create slides
- [ ] Add visualizations
- [ ] Prepare conclusions
- [ ] Practice presentation

---

## Quick Reference

### Key Python Imports
```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
```

### Important Commands
```bash
# Run tests
pytest tests/ -v

# Format code
black src/ notebooks/

# Check code style
flake8 src/

# Generate documentation
pdoc --html src/
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Team: [Your Team Name]*