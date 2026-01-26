# 🎯 Kickstarter Success Prediction - Complete ML Project Roadmap

## 📊 Project Goal
**Predict whether a Kickstarter project will be successful BEFORE it launches**

**Target Variable:** `state` (successful/failed)

---

## 🗺️ Project Phases Overview

```
Phase 1: Setup & Data Understanding (Week 1)
Phase 2: Exploratory Data Analysis (Week 1-2)
Phase 3: Data Cleaning & Preprocessing (Week 2)
Phase 4: Feature Engineering (Week 3)
Phase 5: Model Development (Week 4-5)
Phase 6: Model Evaluation & Selection (Week 5-6)
Phase 7: Final Model & Presentation (Week 6)
```

---

# 📋 PHASE 1: SETUP & DATA UNDERSTANDING

## Step 1.1: Repository Setup
- [ ] Create GitHub repository with proper structure
- [ ] Initialize `.gitignore` (data/, models/, .ipynb_checkpoints)
- [ ] Create `requirements.txt` with all dependencies
- [ ] Set up virtual environment
- [ ] Create `README.md` with project description
- [ ] Create `.env` file for any API keys (if needed)

## Step 1.2: Data Acquisition
- [ ] Download Kickstarter dataset (CSV file)
- [ ] Place in `data/raw/` directory
- [ ] Document data source in `data/README.md`
- [ ] Check file size and format
- [ ] Verify data integrity (no corruption)

```python
# Initial data load
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/kickstarter_data.csv')
```

## Step 1.3: First Look at Data (5 minutes)
- [ ] Print dataset shape: `df.shape`
- [ ] Display first 10 rows: `df.head(10)`
- [ ] Display last 10 rows: `df.tail(10)`
- [ ] Check column names: `df.columns.tolist()`
- [ ] Check data types: `df.dtypes`
- [ ] Get basic info: `df.info()`
- [ ] Quick statistics: `df.describe()`

## Step 1.4: Understand the Problem
- [ ] Identify target variable: `state` column
- [ ] Check unique values in target: `df['state'].value_counts()`
- [ ] Determine if binary or multi-class classification
- [ ] Decide which states to keep (successful/failed) vs exclude (canceled, suspended, live)
- [ ] Document business logic: What counts as "success"?

**⚠️ IMPORTANT DECISIONS:**
```python
# Which states to include?
# successful, failed - YES
# canceled, suspended, undefined, live - MAYBE EXCLUDE
# Document your reasoning!
```

---

# 🔍 PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)

## Step 2.1: Data Quality Assessment

### A. Missing Values Analysis
- [ ] Count missing values per column: `df.isnull().sum()`
- [ ] Calculate missing percentage: `(df.isnull().sum() / len(df)) * 100`
- [ ] Identify columns with >50% missing (consider dropping)
- [ ] Identify columns with <5% missing (easy to impute)
- [ ] Check if missing values are random or systematic
- [ ] Create missing value heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Missing values heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('reports/figures/01_missing_values_heatmap.png')
```

### B. Duplicate Records
- [ ] Check for duplicate rows: `df.duplicated().sum()`
- [ ] Check for duplicate project IDs: `df['ID'].duplicated().sum()` (if ID column exists)
- [ ] Investigate duplicates if found
- [ ] Decide on removal strategy

### C. Data Types Verification
- [ ] Check if numeric columns are actually numeric
- [ ] Check if date columns are in correct format
- [ ] Check if categorical columns are strings
- [ ] Identify mixed-type columns
- [ ] Plan type conversions

**Expected Kickstarter columns:**
```
- name: object (text)
- category: object (categorical)
- main_category: object (categorical)
- currency: object (categorical)
- deadline: datetime
- goal: float (numeric)
- launched: datetime
- pledged: float (numeric)
- state: object (target - categorical)
- backers: int (numeric)
- country: object (categorical)
- usd_pledged: float (numeric)
- usd_pledged_real: float (numeric)
- usd_goal_real: float (numeric)
```

## Step 2.2: Target Variable Analysis

- [ ] Count distribution: `df['state'].value_counts()`
- [ ] Calculate percentages: `df['state'].value_counts(normalize=True) * 100`
- [ ] Check for class imbalance (is it 50/50 or 90/10?)
- [ ] Plot target distribution (bar chart)
- [ ] Decide if you need to handle imbalance (SMOTE, class weights, undersampling)

```python
# Target distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
df['state'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Target Variable Distribution (Counts)', fontsize=14)
axes[0].set_ylabel('Count')
axes[0].set_xlabel('State')

# Percentage plot
(df['state'].value_counts(normalize=True) * 100).plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
axes[1].set_title('Target Variable Distribution (%)', fontsize=14)
axes[1].set_ylabel('Percentage')
axes[1].set_xlabel('State')

plt.tight_layout()
plt.savefig('reports/figures/02_target_distribution.png')
```

**⚠️ CLASS IMBALANCE STRATEGIES:**
- If 60/40 → OK, proceed normally
- If 70/30 → Use stratified split, consider class_weight='balanced'
- If 80/20+ → SMOTE, undersampling, or ensemble methods

## Step 2.3: Numerical Features Analysis

### A. Univariate Analysis (Each feature separately)

For each numerical column (`goal`, `pledged`, `backers`, `usd_goal_real`, `usd_pledged_real`):

- [ ] Calculate statistics: mean, median, std, min, max, quartiles
- [ ] Plot histogram
- [ ] Plot boxplot (check for outliers)
- [ ] Plot distribution by target variable (successful vs failed)
- [ ] Check for skewness: `df['column'].skew()`
- [ ] Check for kurtosis: `df['column'].kurt()`
- [ ] Identify if log transformation needed (if highly skewed)

```python
# Analyze each numerical feature
numerical_features = ['goal', 'pledged', 'backers', 'usd_pledged_real', 'usd_goal_real']

for col in numerical_features:
    print(f"\n{'='*50}")
    print(f"Analysis of: {col}")
    print(f"{'='*50}")
    
    # Statistics
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    axes[0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'{col} - Histogram')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')
    
    # Boxplot
    axes[1].boxplot(df[col].dropna())
    axes[1].set_title(f'{col} - Boxplot')
    axes[1].set_ylabel(col)
    
    # Distribution by target
    for state in df['state'].unique():
        data = df[df['state'] == state][col].dropna()
        axes[2].hist(data, alpha=0.5, label=state, bins=30)
    axes[2].set_title(f'{col} - Distribution by State')
    axes[2].set_xlabel(col)
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'reports/figures/03_{col}_analysis.png')
    plt.close()
```

### B. Outlier Detection

- [ ] Define outlier threshold (IQR method: Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- [ ] Count outliers per column
- [ ] Visualize outliers with boxplots
- [ ] Decide: keep, cap, or remove outliers
- [ ] **For Kickstarter:** Very high goals might be legitimate, investigate!

```python
# Outlier detection using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return outliers

# Check all numerical features
for col in numerical_features:
    outliers = detect_outliers_iqr(df, col)
```

### C. Correlation Analysis

- [ ] Calculate correlation matrix: `df[numerical_features].corr()`
- [ ] Plot correlation heatmap
- [ ] Identify highly correlated features (>0.8 or <-0.8)
- [ ] Check correlation with target variable
- [ ] Decide which correlated features to keep/remove

```python
# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig('reports/figures/04_correlation_heatmap.png')
```

**⚠️ KICKSTARTER-SPECIFIC INSIGHTS:**
- `usd_pledged_real` vs `pledged` → Likely highly correlated (keep one)
- `usd_goal_real` vs `goal` → Likely highly correlated (keep one)
- `backers` vs `pledged` → Probably correlated (but both useful)

## Step 2.4: Categorical Features Analysis

### A. Univariate Analysis

For each categorical column (`category`, `main_category`, `currency`, `country`, `state`):

- [ ] Count unique values: `df[col].nunique()`
- [ ] Display value counts: `df[col].value_counts()`
- [ ] Calculate percentages
- [ ] Plot bar chart (top 10-20 categories if many)
- [ ] Identify rare categories (<1% occurrence)
- [ ] Decide on grouping strategy for rare categories

```python
categorical_features = ['main_category', 'category', 'currency', 'country']

for col in categorical_features:
    print(f"\n{'='*50}")
    print(f"Analysis of: {col}")
    print(f"{'='*50}")
    print(f"Unique values: {df[col].nunique()}")
    print(f"\nTop 10 values:")
    print(df[col].value_counts().head(10))
    print(f"\nPercentage distribution (top 10):")
    print((df[col].value_counts(normalize=True).head(10) * 100).round(2))
    
    # Plot top 20 categories
    plt.figure(figsize=(14, 6))
    df[col].value_counts().head(20).plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title(f'{col} - Top 20 Categories', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'reports/figures/05_{col}_distribution.png')
    plt.close()
```

### B. Relationship with Target Variable

- [ ] Calculate success rate per category
- [ ] Create crosstab: `pd.crosstab(df['category'], df['state'], normalize='index')`
- [ ] Plot success rate by category (horizontal bar chart)
- [ ] Identify categories with highest/lowest success rates
- [ ] **This is GOLD for feature engineering!**

```python
# Success rate by category
for col in categorical_features:
    # Calculate success rate
    success_rate = df.groupby(col)['state'].apply(
        lambda x: (x == 'successful').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    print(f"\n{col} - Success Rate (%):")
    print(success_rate.head(10))
    
    # Plot
    plt.figure(figsize=(12, 8))
    success_rate.head(20).plot(kind='barh', color='green', alpha=0.7, edgecolor='black')
    plt.title(f'Success Rate by {col} (Top 20)', fontsize=14)
    plt.xlabel('Success Rate (%)')
    plt.ylabel(col)
    plt.axvline(x=df['state'].value_counts(normalize=True)['successful']*100, 
                color='red', linestyle='--', label='Overall Success Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/06_{col}_success_rate.png')
    plt.close()
```

**⚠️ KEY INSIGHTS TO FIND:**
- Which categories have highest success rate?
- Which countries fund more successfully?
- Does currency matter?
- Are there "doomed" categories?

## Step 2.5: Temporal Analysis (Dates)

### A. Date Feature Extraction
- [ ] Convert `launched` and `deadline` to datetime
- [ ] Extract: year, month, day, day_of_week, hour
- [ ] Calculate campaign duration: `deadline - launched`
- [ ] Check for future dates (data errors)
- [ ] Check for impossible dates (year < 2009, Kickstarter founded)

```python
# Date preprocessing
df['launched'] = pd.to_datetime(df['launched'])
df['deadline'] = pd.to_datetime(df['deadline'])

# Extract temporal features
df['launch_year'] = df['launched'].dt.year
df['launch_month'] = df['launched'].dt.month
df['launch_day'] = df['launched'].dt.day
df['launch_dayofweek'] = df['launched'].dt.dayofweek  # 0=Monday
df['launch_hour'] = df['launched'].dt.hour

# Campaign duration
df['campaign_duration_days'] = (df['deadline'] - df['launched']).dt.days

# Check data quality
print("Launch year range:", df['launch_year'].min(), "-", df['launch_year'].max())
print("Campaign duration stats:")
print(df['campaign_duration_days'].describe())
```

### B. Temporal Patterns
- [ ] Success rate by year (is it changing over time?)
- [ ] Success rate by month (seasonality?)
- [ ] Success rate by day of week (weekday vs weekend?)
- [ ] Success rate by campaign duration
- [ ] Plot time series of project launches

```python
# Success rate over time
temporal_features = ['launch_year', 'launch_month', 'launch_dayofweek', 'campaign_duration_days']

for col in temporal_features:
    success_rate_temporal = df.groupby(col)['state'].apply(
        lambda x: (x == 'successful').sum() / len(x) * 100
    )
    
    plt.figure(figsize=(12, 6))
    success_rate_temporal.plot(kind='bar', color='purple', alpha=0.7, edgecolor='black')
    plt.title(f'Success Rate by {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Success Rate (%)')
    plt.axhline(y=df['state'].value_counts(normalize=True)['successful']*100, 
                color='red', linestyle='--', label='Overall Success Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/07_{col}_temporal_success.png')
    plt.close()
```

**⚠️ INSIGHTS TO FIND:**
- Best month to launch?
- Optimal campaign duration?
- Is Kickstarter getting harder over time?

## Step 2.6: Text Analysis (Project Names)

- [ ] Length of project names: `df['name'].str.len()`
- [ ] Word count: `df['name'].str.split().str.len()`
- [ ] Common words (word cloud)
- [ ] Check for special characters
- [ ] Sentiment analysis (optional)

```python
# Text feature extraction
df['name_length'] = df['name'].str.len()
df['name_word_count'] = df['name'].str.split().str.len()

# Success rate by name length
df['name_length_bin'] = pd.cut(df['name_length'], bins=[0, 20, 40, 60, 80, 200], 
                                 labels=['very_short', 'short', 'medium', 'long', 'very_long'])

success_by_length = df.groupby('name_length_bin')['state'].apply(
    lambda x: (x == 'successful').sum() / len(x) * 100
)

print("Success rate by name length:")
print(success_by_length)
```

## Step 2.7: Key EDA Insights Summary

**Create a summary document with answers to:**
- [ ] What's the class distribution? (balanced/imbalanced)
- [ ] What features have most missing values?
- [ ] Which features are most correlated with success?
- [ ] Which categories perform best/worst?
- [ ] Are there temporal patterns?
- [ ] What's the typical goal amount for successful projects?
- [ ] What outliers exist and why?
- [ ] Which features need transformation?
- [ ] Which features might need encoding?
- [ ] What new features should we engineer?

---

# 🧹 PHASE 3: DATA CLEANING & PREPROCESSING

## Step 3.1: Filter Target Variable

**⚠️ CRITICAL DECISION POINT**

- [ ] Decide which states to keep
- [ ] Remove irrelevant states (live, undefined, suspended, canceled)
- [ ] Keep only: successful, failed
- [ ] Document reasoning

```python
# Filter to binary classification
print("Original state distribution:")
print(df['state'].value_counts())

# Keep only successful and failed
df_clean = df[df['state'].isin(['successful', 'failed'])].copy()

print("\nFiltered state distribution:")
print(df_clean['state'].value_counts())
print(f"\nRemoved {len(df) - len(df_clean)} rows")
```

## Step 3.2: Handle Missing Values

### Strategy A: Drop columns with >50% missing
```python
# Identify columns with excessive missing values
missing_threshold = 0.5
missing_pct = df_clean.isnull().sum() / len(df_clean)
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

print(f"Dropping columns with >{missing_threshold*100}% missing:")
print(cols_to_drop)

df_clean = df_clean.drop(columns=cols_to_drop)
```

### Strategy B: Impute remaining missing values

**For Numerical Features:**
- [ ] Median imputation (robust to outliers)
- [ ] Mean imputation (if normally distributed)
- [ ] Forward fill / backward fill (for time series)
- [ ] KNN imputation (more sophisticated)

```python
from sklearn.impute import SimpleImputer

# Numerical imputation - use MEDIAN (robust to outliers)
numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()

imputer_num = SimpleImputer(strategy='median')
df_clean[numerical_cols] = imputer_num.fit_transform(df_clean[numerical_cols])
```

**For Categorical Features:**
- [ ] Mode imputation (most frequent)
- [ ] Add 'Unknown' category
- [ ] Drop rows (if very few)

```python
# Categorical imputation - use MOST FREQUENT
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('state')  # Don't impute target!

imputer_cat = SimpleImputer(strategy='most_frequent')
df_clean[categorical_cols] = imputer_cat.fit_transform(df_clean[categorical_cols])
```

## Step 3.3: Handle Outliers

**⚠️ KICKSTARTER CONTEXT:** Some projects legitimately have very high goals!

### Strategy Options:
1. **Keep outliers** (if legitimate)
2. **Cap outliers** (Winsorization)
3. **Remove outliers** (if data errors)
4. **Log transformation** (reduce impact)

```python
# Option 1: Cap outliers (Winsorization)
def cap_outliers(df, column, lower_percentile=1, upper_percentile=99):
    """Cap outliers at specified percentiles"""
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# Apply to goal (many crazy high goals)
df_clean = cap_outliers(df_clean, 'usd_goal_real', lower_percentile=1, upper_percentile=99)

# Option 2: Log transformation (alternative)
df_clean['log_goal'] = np.log1p(df_clean['usd_goal_real'])  # log(1+x) to handle zeros
```

## Step 3.4: Handle Data Quality Issues

- [ ] Remove duplicates: `df_clean = df_clean.drop_duplicates()`
- [ ] Fix data types
- [ ] Standardize text (lowercase, strip whitespace)
- [ ] Handle impossible values (negative goal, negative backers)

```python
# Remove duplicates
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Removed {initial_rows - len(df_clean)} duplicate rows")

# Remove impossible values
df_clean = df_clean[df_clean['usd_goal_real'] > 0]  # Goal must be positive
df_clean = df_clean[df_clean['backers'] >= 0]  # Backers can't be negative
df_clean = df_clean[df_clean['campaign_duration_days'] > 0]  # Duration must be positive
df_clean = df_clean[df_clean['campaign_duration_days'] < 365]  # Max 1 year campaign

print(f"After removing impossible values: {len(df_clean)} rows remain")
```

## Step 3.5: Feature Selection (Drop Redundant Features)

**⚠️ CRITICAL:** Remove features that leak information or are redundant

### Features to DROP:
- [ ] **Target leakage:** `pledged`, `usd_pledged`, `usd_pledged_real`, `backers`
  - These are known AFTER campaign ends! Can't use for prediction.
- [ ] **Redundant:** Keep `usd_goal_real`, drop `goal` (currency-normalized version better)
- [ ] **Identifiers:** `ID`, `name` (unless doing NLP)
- [ ] **Redundant dates:** Keep `launched`, drop `deadline` (use `campaign_duration` instead)

```python
# ⚠️ PREVENT DATA LEAKAGE!
leakage_features = ['pledged', 'usd_pledged', 'usd_pledged_real', 'backers']
redundant_features = ['goal', 'deadline', 'ID', 'name']  # name: unless doing NLP

features_to_drop = leakage_features + redundant_features

print(f"Dropping {len(features_to_drop)} features to prevent leakage/redundancy:")
print(features_to_drop)

df_clean = df_clean.drop(columns=[col for col in features_to_drop if col in df_clean.columns])
```

## Step 3.6: Save Cleaned Data

```python
# Save cleaned dataset
df_clean.to_csv('data/processed/kickstarter_cleaned.csv', index=False)
print(f"Saved cleaned data: {df_clean.shape}")
print(f"Columns: {df_clean.columns.tolist()}")
```

---

# 🔧 PHASE 4: FEATURE ENGINEERING

## Step 4.1: Temporal Features

### A. Extract Date Components
```python
# Already done in EDA, but let's refine:
df_fe = df_clean.copy()

# Launch timing features
df_fe['launch_year'] = df_fe['launched'].dt.year
df_fe['launch_month'] = df_fe['launched'].dt.month
df_fe['launch_quarter'] = df_fe['launched'].dt.quarter
df_fe['launch_dayofweek'] = df_fe['launched'].dt.dayofweek
df_fe['launch_is_weekend'] = df_fe['launch_dayofweek'].isin([5, 6]).astype(int)
df_fe['launch_day'] = df_fe['launched'].dt.day
df_fe['launch_hour'] = df_fe['launched'].dt.hour

# Campaign duration
df_fe['campaign_duration_days'] = (df_fe['deadline'] - df_fe['launched']).dt.days
df_fe['campaign_duration_weeks'] = df_fe['campaign_duration_days'] / 7
df_fe['campaign_duration_months'] = df_fe['campaign_duration_days'] / 30
```

### B. Cyclical Encoding (for periodic features)

**⚠️ IMPORTANT:** Month and day of week are cyclical!
- December (12) is close to January (1)
- Sunday (6) is close to Monday (0)

```python
# Cyclical encoding for month
df_fe['launch_month_sin'] = np.sin(2 * np.pi * df_fe['launch_month'] / 12)
df_fe['launch_month_cos'] = np.cos(2 * np.pi * df_fe['launch_month'] / 12)

# Cyclical encoding for day of week
df_fe['launch_dayofweek_sin'] = np.sin(2 * np.pi * df_fe['launch_dayofweek'] / 7)
df_fe['launch_dayofweek_cos'] = np.cos(2 * np.pi * df_fe['launch_dayofweek'] / 7)

# Cyclical encoding for day of month
df_fe['launch_day_sin'] = np.sin(2 * np.pi * df_fe['launch_day'] / 31)
df_fe['launch_day_cos'] = np.cos(2 * np.pi * df_fe['launch_day'] / 31)
```

## Step 4.2: Goal-Based Features

```python
# Log-transform goal (reduce skewness)
df_fe['log_usd_goal_real'] = np.log1p(df_fe['usd_goal_real'])

# Goal categories (bins)
df_fe['goal_category'] = pd.cut(
    df_fe['usd_goal_real'],
    bins=[0, 1000, 5000, 10000, 25000, 50000, 100000, float('inf')],
    labels=['micro', 'small', 'medium', 'medium_large', 'large', 'very_large', 'mega']
)

# Goal per day (goal intensity)
df_fe['goal_per_day'] = df_fe['usd_goal_real'] / df_fe['campaign_duration_days']

# Ambitious goal flag (above median)
median_goal = df_fe['usd_goal_real'].median()
df_fe['is_ambitious_goal'] = (df_fe['usd_goal_real'] > median_goal).astype(int)
```

## Step 4.3: Category-Based Features

### A. Category Success Rate Encoding (Target Encoding)

**⚠️ IMPORTANT:** Must do this carefully to avoid leakage!

```python
from sklearn.model_selection import train_test_split

# Split data FIRST before target encoding
X = df_fe.drop(columns=['state'])
y = df_fe['state']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Now calculate success rates ONLY on training data
train_data = pd.concat([X_temp, y_temp], axis=1)

# Calculate success rate for each category
def calculate_success_rate(df, column):
    """Calculate success rate for each category"""
    success_rate = df.groupby(column)['state'].apply(
        lambda x: (x == 'successful').sum() / len(x)
    ).to_dict()
    return success_rate

# Main category success rate
main_cat_success_rate = calculate_success_rate(train_data, 'main_category')
X_temp['main_category_success_rate'] = X_temp['main_category'].map(main_cat_success_rate)
X_test['main_category_success_rate'] = X_test['main_category'].map(main_cat_success_rate)

# Subcategory success rate
cat_success_rate = calculate_success_rate(train_data, 'category')
X_temp['category_success_rate'] = X_temp['category'].map(cat_success_rate)
X_test['category_success_rate'] = X_test['category'].map(cat_success_rate)

# Country success rate
country_success_rate = calculate_success_rate(train_data, 'country')
X_temp['country_success_rate'] = X_temp['country'].map(country_success_rate)
X_test['country_success_rate'] = X_test['country'].map(country_success_rate)
```

### B. Category Frequency Encoding

```python
# How common is this category?
main_cat_freq = train_data['main_category'].value_counts(normalize=True).to_dict()
X_temp['main_category_frequency'] = X_temp['main_category'].map(main_cat_freq)
X_test['main_category_frequency'] = X_test['main_category'].map(main_cat_freq)
```

### C. Rare Category Flag

```python
# Is this a rare category? (less than 1% of projects)
rare_threshold = 0.01
category_counts = train_data['category'].value_counts(normalize=True)
rare_categories = category_counts[category_counts < rare_threshold].index

X_temp['is_rare_category'] = X_temp['category'].isin(rare_categories).astype(int)
X_test['is_rare_category'] = X_test['category'].isin(rare_categories).astype(int)
```

## Step 4.4: Text Features (Project Name)

```python
# Name length
X_temp['name_length'] = X_temp['name'].str.len()
X_test['name_length'] = X_test['name'].str.len()

# Word count
X_temp['name_word_count'] = X_temp['name'].str.split().str.len()
X_test['name_word_count'] = X_test['name'].str.split().str.len()

# Average word length
X_temp['avg_word_length'] = X_temp['name_length'] / X_temp['name_word_count']
X_test['avg_word_length'] = X_test['name_length'] / X_test['name_word_count']

# Has numbers in name
X_temp['name_has_numbers'] = X_temp['name'].str.contains(r'\d').astype(int)
X_test['name_has_numbers'] = X_test['name'].str.contains(r'\d').astype(int)

# Has special characters
X_temp['name_has_special'] = X_temp['name'].str.contains(r'[!?:;]').astype(int)
X_test['name_has_special'] = X_test['name'].str.contains(r'[!?:;]').astype(int)

# All caps (shouting?)
X_temp['name_all_caps'] = X_temp['name'].str.isupper().astype(int)
X_test['name_all_caps'] = X_test['name'].str.isupper().astype(int)
```

## Step 4.5: Interaction Features

```python
# Category × Country interaction
X_temp['category_country'] = X_temp['main_category'] + '_' + X_temp['country']
X_test['category_country'] = X_test['main_category'] + '_' + X_test['country']

# Goal × Duration interaction
X_temp['goal_duration_ratio'] = X_temp['usd_goal_real'] / (X_temp['campaign_duration_days'] + 1)
X_test['goal_duration_ratio'] = X_test['usd_goal_real'] / (X_test['campaign_duration_days'] + 1)

# Goal × Category Success Rate
X_temp['goal_times_cat_success'] = X_temp['usd_goal_real'] * X_temp['main_category_success_rate']
X_test['goal_times_cat_success'] = X_test['usd_goal_real'] * X_test['main_category_success_rate']
```

## Step 4.6: Domain-Specific Features (Kickstarter Logic)

```python
# Optimal duration flag (30-40 days is often considered optimal)
X_temp['is_optimal_duration'] = X_temp['campaign_duration_days'].between(30, 40).astype(int)
X_test['is_optimal_duration'] = X_test['campaign_duration_days'].between(30, 40).astype(int)

# Launched in peak month (some months are better)
peak_months = [10, 11]  # October, November (pre-holiday season)
X_temp['launched_in_peak_month'] = X_temp['launch_month'].isin(peak_months).astype(int)
X_test['launched_in_peak_month'] = X_test['launch_month'].isin(peak_months).astype(int)

# Realistic goal for category
# (Compare project goal to median goal in that category)
category_median_goal = train_data.groupby('main_category')['usd_goal_real'].median().to_dict()
X_temp['goal_vs_category_median'] = X_temp['usd_goal_real'] / X_temp['main_category'].map(category_median_goal)
X_test['goal_vs_category_median'] = X_test['usd_goal_real'] / X_test['main_category'].map(category_median_goal)
```

## Step 4.7: Save Engineered Features

```python
# Save feature-engineered data
X_temp.to_csv('data/processed/kickstarter_features_train.csv', index=False)
X_test.to_csv('data/processed/kickstarter_features_test.csv', index=False)
y_temp.to_csv('data/processed/kickstarter_target_train.csv', index=False)
y_test.to_csv('data/processed/kickstarter_target_test.csv', index=False)

print(f"Training data: {X_temp.shape}")
print(f"Test data: {X_test.shape}")
print(f"Total features: {X_temp.shape[1]}")
```

---

# 🔀 PHASE 5: TRAIN-TEST SPLIT & PREPROCESSING PIPELINE

## Step 5.1: Prepare Final Feature Set

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load engineered features
X_train = pd.read_csv('data/processed/kickstarter_features_train.csv')
X_test = pd.read_csv('data/processed/kickstarter_features_test.csv')
y_train = pd.read_csv('data/processed/kickstarter_target_train.csv').squeeze()
y_test = pd.read_csv('data/processed/kickstarter_target_test.csv').squeeze()

# Encode target variable (successful=1, failed=0)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print(f"Target encoding: {label_encoder.classes_}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")
```

## Step 5.2: Identify Feature Types

```python
# Identify different feature types
numerical_features = [
    'usd_goal_real', 'log_usd_goal_real', 'campaign_duration_days',
    'campaign_duration_weeks', 'launch_year', 'launch_month', 'launch_day',
    'launch_dayofweek', 'launch_hour', 'launch_month_sin', 'launch_month_cos',
    'launch_dayofweek_sin', 'launch_dayofweek_cos', 'goal_per_day',
    'main_category_success_rate', 'category_success_rate', 'country_success_rate',
    'main_category_frequency', 'name_length', 'name_word_count', 'avg_word_length',
    'goal_duration_ratio', 'goal_times_cat_success', 'goal_vs_category_median'
]

categorical_features = [
    'main_category', 'category', 'currency', 'country', 'goal_category'
]

binary_features = [
    'launch_is_weekend', 'is_ambitious_goal', 'is_rare_category',
    'name_has_numbers', 'name_has_special', 'name_all_caps',
    'is_optimal_duration', 'launched_in_peak_month'
]

# Filter features that actually exist in the data
numerical_features = [f for f in numerical_features if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]
binary_features = [f for f in binary_features if f in X_train.columns]

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Binary features: {len(binary_features)}")
```

## Step 5.3: Create Preprocessing Pipeline

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', 'passthrough', binary_features)
    ],
    remainder='drop'  # Drop any features not specified
)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Original feature count: {X_train.shape[1]}")
print(f"Processed feature count: {X_train_processed.shape[1]}")
print(f"Train shape: {X_train_processed.shape}")
print(f"Test shape: {X_test_processed.shape}")
```

## Step 5.4: Handle Class Imbalance (if needed)

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

print("Original class distribution:")
print(Counter(y_train))

# If imbalanced (e.g., 70/30 or worse), apply SMOTE
class_ratio = min(np.bincount(y_train)) / max(np.bincount(y_train))

if class_ratio < 0.7:  # Less than 70/30
    print(f"\n⚠️ Class imbalance detected: {class_ratio:.2%}")
    print("Applying SMOTE...")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    print("After SMOTE:")
    print(Counter(y_train_balanced))
else:
    print("\n✓ Classes are relatively balanced, no SMOTE needed")
    X_train_balanced = X_train_processed
    y_train_balanced = y_train
```

## Step 5.5: Final Validation Split

```python
# Create validation set from training data
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_balanced, y_train_balanced,
    test_size=0.2, random_state=42, stratify=y_train_balanced
)

print(f"\nFinal dataset sizes:")
print(f"Training set: {X_train_final.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test_processed.shape}")
```

---

# 🤖 PHASE 6: MODEL DEVELOPMENT

## Step 6.1: Baseline Model

**⚠️ ALWAYS START WITH A BASELINE!**

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Baseline: Always predict the majority class
baseline = DummyClassifier(strategy='most_frequent', random_state=42)
baseline.fit(X_train_final, y_train_final)

y_val_pred_baseline = baseline.predict(X_val)
y_val_proba_baseline = baseline.predict_proba(X_val)[:, 1]

print("="*50)
print("BASELINE MODEL (Majority Class)")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_baseline):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_baseline):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_baseline):.4f}")

# This is your benchmark - any model must beat this!
baseline_accuracy = accuracy_score(y_val, y_val_pred_baseline)
```

## Step 6.2: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_final, y_train_final)

y_val_pred_lr = lr.predict(X_val)
y_val_proba_lr = lr.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_lr):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_lr):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_lr):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_lr):.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_lr, target_names=['Failed', 'Successful']))

# Confusion matrix
cm_lr = confusion_matrix(y_val, y_val_pred_lr)
print("\nConfusion Matrix:")
print(cm_lr)
```

## Step 6.3: Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42, class_weight='balanced')
dt.fit(X_train_final, y_train_final)

y_val_pred_dt = dt.predict(X_val)
y_val_proba_dt = dt.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print("DECISION TREE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_dt):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_dt):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_dt):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_dt):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_dt):.4f}")
```

## Step 6.4: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                            random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train_final, y_train_final)

y_val_pred_rf = rf.predict(X_val)
y_val_proba_rf = rf.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_rf):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_rf):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_rf):.4f}")

# Feature importance
feature_names = (
    numerical_features + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
    binary_features
)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

## Step 6.5: Gradient Boosting (XGBoost)

```python
from xgboost import XGBClassifier

# XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1, eval_metric='logloss')

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = len(y_train_final[y_train_final == 0]) / len(y_train_final[y_train_final == 1])
xgb.set_params(scale_pos_weight=scale_pos_weight)

xgb.fit(X_train_final, y_train_final)

y_val_pred_xgb = xgb.predict(X_val)
y_val_proba_xgb = xgb.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print("XGBOOST")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_xgb):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_xgb):.4f}")
```

## Step 6.6: LightGBM

```python
from lightgbm import LGBMClassifier

# LightGBM
lgbm = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                      random_state=42, n_jobs=-1, class_weight='balanced')
lgbm.fit(X_train_final, y_train_final)

y_val_pred_lgbm = lgbm.predict(X_val)
y_val_proba_lgbm = lgbm.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print("LIGHTGBM")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_lgbm):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_lgbm):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_lgbm):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_lgbm):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_lgbm):.4f}")
```

## Step 6.7: Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

# SVM (might be slow on large datasets)
svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train_final, y_train_final)

y_val_pred_svm = svm.predict(X_val)
y_val_proba_svm = svm.predict_proba(X_val)[:, 1]

print("\n" + "="*50)
print("SUPPORT VECTOR MACHINE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_svm):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_svm):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_svm):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_svm):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_svm):.4f}")
```

---

# 📊 PHASE 7: MODEL COMPARISON & SELECTION

## Step 7.1: Compare All Models

```python
# Create comparison dataframe
results = pd.DataFrame({
    'Model': ['Baseline', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'SVM'],
    'Accuracy': [
        accuracy_score(y_val, y_val_pred_baseline),
        accuracy_score(y_val, y_val_pred_lr),
        accuracy_score(y_val, y_val_pred_dt),
        accuracy_score(y_val, y_val_pred_rf),
        accuracy_score(y_val, y_val_pred_xgb),
        accuracy_score(y_val, y_val_pred_lgbm),
        accuracy_score(y_val, y_val_pred_svm)
    ],
    'Precision': [
        precision_score(y_val, y_val_pred_baseline),
        precision_score(y_val, y_val_pred_lr),
        precision_score(y_val, y_val_pred_dt),
        precision_score(y_val, y_val_pred_rf),
        precision_score(y_val, y_val_pred_xgb),
        precision_score(y_val, y_val_pred_lgbm),
        precision_score(y_val, y_val_pred_svm)
    ],
    'Recall': [
        recall_score(y_val, y_val_pred_baseline),
        recall_score(y_val, y_val_pred_lr),
        recall_score(y_val, y_val_pred_dt),
        recall_score(y_val, y_val_pred_rf),
        recall_score(y_val, y_val_pred_xgb),
        recall_score(y_val, y_val_pred_lgbm),
        recall_score(y_val, y_val_pred_svm)
    ],
    'F1': [
        f1_score(y_val, y_val_pred_baseline),
        f1_score(y_val, y_val_pred_lr),
        f1_score(y_val, y_val_pred_dt),
        f1_score(y_val, y_val_pred_rf),
        f1_score(y_val, y_val_pred_xgb),
        f1_score(y_val, y_val_pred_lgbm),
        f1_score(y_val, y_val_pred_svm)
    ],
    'ROC-AUC': [
        0.5,  # Baseline has no predictive power
        roc_auc_score(y_val, y_val_proba_lr),
        roc_auc_score(y_val, y_val_proba_dt),
        roc_auc_score(y_val, y_val_proba_rf),
        roc_auc_score(y_val, y_val_proba_xgb),
        roc_auc_score(y_val, y_val_proba_lgbm),
        roc_auc_score(y_val, y_val_proba_svm)
    ]
}).round(4)

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(results.to_string(index=False))

# Save results
results.to_csv('reports/metrics/model_comparison.csv', index=False)
```

## Step 7.2: Visualize Model Comparison

```python
# Bar chart comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    results.plot(x='Model', y=metric, kind='bar', ax=ax, color='steelblue', legend=False)
    ax.set_title(f'{metric} Comparison', fontsize=14)
    ax.set_ylabel(metric)
    ax.set_xlabel('')
    ax.set_xticklabels(results['Model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)

# Remove extra subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('reports/figures/08_model_comparison.png', dpi=300)
plt.show()
```

## Step 7.3: ROC Curves Comparison

```python
from sklearn.metrics import roc_curve, auc

# Plot ROC curves for all models
plt.figure(figsize=(12, 8))

models_proba = {
    'Logistic Regression': y_val_proba_lr,
    'Decision Tree': y_val_proba_dt,
    'Random Forest': y_val_proba_rf,
    'XGBoost': y_val_proba_xgb,
    'LightGBM': y_val_proba_lgbm,
    'SVM': y_val_proba_svm
}

for name, y_proba in models_proba.items():
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/09_roc_curves_comparison.png', dpi=300)
plt.show()
```

## Step 7.4: Confusion Matrices Comparison

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

predictions = {
    'Logistic Regression': y_val_pred_lr,
    'Decision Tree': y_val_pred_dt,
    'Random Forest': y_val_pred_rf,
    'XGBoost': y_val_pred_xgb,
    'LightGBM': y_val_pred_lgbm,
    'SVM': y_val_pred_svm
}

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failed', 'Successful'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{name}', fontsize=12)

plt.tight_layout()
plt.savefig('reports/figures/10_confusion_matrices.png', dpi=300)
plt.show()
```

## Step 7.5: Select Best Model

```python
# Identify best model based on F1 score (balanced metric)
best_model_idx = results['F1'].idxmax()
best_model_name = results.loc[best_model_idx, 'Model']
best_f1_score = results.loc[best_model_idx, 'F1']

print("\n" + "="*70)
print("BEST MODEL SELECTION")
print("="*70)
print(f"Best Model: {best_model_name}")
print(f"F1 Score: {best_f1_score:.4f}")
print(f"\nFull metrics:")
print(results.loc[best_model_idx])

# Map model name to actual model object
model_mapping = {
    'Logistic Regression': lr,
    'Decision Tree': dt,
    'Random Forest': rf,
    'XGBoost': xgb,
    'LightGBM': lgbm,
    'SVM': svm
}

best_model = model_mapping.get(best_model_name)
```

---

# 🔧 PHASE 8: HYPERPARAMETER TUNING

## Step 8.1: Grid Search for Best Model

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid based on best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    model_for_tuning = RandomForestClassifier(random_state=42, n_jobs=-1)

elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    model_for_tuning = XGBClassifier(random_state=42, n_jobs=-1)

elif best_model_name == 'LightGBM':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'class_weight': ['balanced', None]
    }
    model_for_tuning = LGBMClassifier(random_state=42, n_jobs=-1)

else:
    param_grid = {}
    model_for_tuning = best_model

# Perform Grid Search
if param_grid:
    print(f"\nPerforming Grid Search for {best_model_name}...")
    print(f"Parameter grid: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=model_for_tuning,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_final, y_train_final)
    
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    print(f"\nBest Cross-Validation F1 Score: {grid_search.best_score_:.4f}")
    
    # Use best model
    best_model_tuned = grid_search.best_estimator_
else:
    best_model_tuned = best_model
```

## Step 8.2: Evaluate Tuned Model

```python
# Evaluate tuned model on validation set
y_val_pred_tuned = best_model_tuned.predict(X_val)
y_val_proba_tuned = best_model_tuned.predict_proba(X_val)[:, 1]

print("\n" + "="*70)
print(f"TUNED {best_model_name.upper()} - VALIDATION SET")
print("="*70)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred_tuned):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred_tuned):.4f}")
print(f"F1 Score: {f1_score(y_val, y_val_pred_tuned):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba_tuned):.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_tuned, target_names=['Failed', 'Successful']))
```

---

# 🎯 PHASE 9: FINAL EVALUATION ON TEST SET

## Step 9.1: Evaluate on Hold-Out Test Set

**⚠️ DO THIS ONLY ONCE AT THE VERY END!**

```python
# Final evaluation on test set
y_test_pred = best_model_tuned.predict(X_test_processed)
y_test_proba = best_model_tuned.predict_proba(X_test_processed)[:, 1]

print("\n" + "="*70)
print(f"FINAL MODEL: {best_model_name.upper()} - TEST SET RESULTS")
print("="*70)
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Failed', 'Successful']))

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm_test)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Failed', 'Successful'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Final Model Confusion Matrix - Test Set\n{best_model_name}', fontsize=14)
plt.tight_layout()
plt.savefig('reports/figures/11_final_confusion_matrix.png', dpi=300)
plt.show()
```

## Step 9.2: Feature Importance Analysis

```python
# Get feature importance (if model supports it)
if hasattr(best_model_tuned, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model_tuned.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*70)
    print(feature_importance.head(20).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Feature Importances', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/figures/12_feature_importance.png', dpi=300)
    plt.show()
```

## Step 9.3: Prediction Analysis

```python
# Analyze predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'probability_success': y_test_proba
})

# Correct predictions
predictions_df['correct'] = (predictions_df['actual'] == predictions_df['predicted'])

print("\n" + "="*70)
print("PREDICTION ANALYSIS")
print("="*70)
print(f"Total predictions: {len(predictions_df)}")
print(f"Correct predictions: {predictions_df['correct'].sum()} ({predictions_df['correct'].mean()*100:.2f}%)")
print(f"Incorrect predictions: {(~predictions_df['correct']).sum()} ({(~predictions_df['correct']).mean()*100:.2f}%)")

# Analyze false positives and false negatives
false_positives = predictions_df[(predictions_df['actual'] == 0) & (predictions_df['predicted'] == 1)]
false_negatives = predictions_df[(predictions_df['actual'] == 1) & (predictions_df['predicted'] == 0)]

print(f"\nFalse Positives (predicted success, actually failed): {len(false_positives)}")
print(f"False Negatives (predicted failure, actually successful): {len(false_negatives)}")

# Distribution of prediction probabilities
plt.figure(figsize=(12, 6))
plt.hist(predictions_df[predictions_df['actual'] == 0]['probability_success'], 
         bins=50, alpha=0.5, label='Actually Failed', color='red')
plt.hist(predictions_df[predictions_df['actual'] == 1]['probability_success'], 
         bins=50, alpha=0.5, label='Actually Successful', color='green')
plt.xlabel('Predicted Probability of Success', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Predicted Probabilities', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/13_probability_distribution.png', dpi=300)
plt.show()
```

## Step 9.4: Business Impact Analysis

```python
# Calculate business metrics
print("\n" + "="*70)
print("BUSINESS IMPACT ANALYSIS")
print("="*70)

# If we use this model to pre-screen projects:
# - True Positives: Correctly predicted successes → Fund these!
# - False Positives: Wrongly predicted successes → Lost opportunity
# - True Negatives: Correctly predicted failures → Don't fund
# - False Negatives: Wrongly predicted failures → Missed opportunity

tn, fp, fn, tp = cm_test.ravel()

print(f"\nTrue Positives (TP): {tp} - Correctly predicted successful projects")
print(f"False Positives (FP): {fp} - Incorrectly predicted successful (actually failed)")
print(f"True Negatives (TN): {tn} - Correctly predicted failed projects")
print(f"False Negatives (FN): {fn} - Incorrectly predicted failed (actually successful)")

# Assuming each project has a cost/benefit
# This is hypothetical - adjust based on business case
avg_project_value = 10000  # Average value of successful project
cost_of_failure = 5000     # Cost of backing failed project

# Calculate potential value
value_from_tp = tp * avg_project_value
loss_from_fp = fp * cost_of_failure
missed_value_from_fn = fn * avg_project_value

net_value = value_from_tp - loss_from_fp
theoretical_max_value = (tp + fn) * avg_project_value

print(f"\n💰 FINANCIAL IMPACT (Hypothetical):")
print(f"Value from True Positives: ${value_from_tp:,}")
print(f"Loss from False Positives: -${loss_from_fp:,}")
print(f"Net Value: ${net_value:,}")
print(f"Theoretical Max (Perfect Model): ${theoretical_max_value:,}")
print(f"Efficiency: {(net_value / theoretical_max_value * 100):.2f}%")
```

---

# 💾 PHASE 10: SAVE FINAL MODEL

## Step 10.1: Save Model and Preprocessing Pipeline

```python
import joblib
from datetime import datetime

# Create timestamp for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save preprocessing pipeline
joblib.dump(preprocessor, f'models/preprocessor_{timestamp}.pkl')
print(f"✓ Saved preprocessing pipeline: models/preprocessor_{timestamp}.pkl")

# Save label encoder
joblib.dump(label_encoder, f'models/label_encoder_{timestamp}.pkl')
print(f"✓ Saved label encoder: models/label_encoder_{timestamp}.pkl")

# Save final model
joblib.dump(best_model_tuned, f'models/final_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl')
print(f"✓ Saved final model: models/final_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'timestamp': timestamp,
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'test_f1': f1_score(y_test, y_test_pred),
    'test_roc_auc': roc_auc_score(y_test, y_test_proba),
    'hyperparameters': best_model_tuned.get_params() if hasattr(best_model_tuned, 'get_params') else None,
    'feature_count': X_train_processed.shape[1],
    'training_samples': len(X_train_final)
}

import json
with open(f'models/model_metadata_{timestamp}.json', 'w') as f:
    json.dump(model_metadata, f, indent=4, default=str)
print(f"✓ Saved model metadata: models/model_metadata_{timestamp}.json")
```

## Step 10.2: Create Prediction Function

```python
def predict_kickstarter_success(project_data, preprocessor, model, label_encoder):
    """
    Predict if a Kickstarter project will be successful
    
    Parameters:
    -----------
    project_data : dict or pd.DataFrame
        Project features (must match training data structure)
    preprocessor : sklearn ColumnTransformer
        Fitted preprocessing pipeline
    model : sklearn model
        Trained classification model
    label_encoder : sklearn LabelEncoder
        Fitted label encoder
    
    Returns:
    --------
    dict : Prediction result with probability
    """
    # Convert to DataFrame if dict
    if isinstance(project_data, dict):
        project_data = pd.DataFrame([project_data])
    
    # Preprocess
    X_processed = preprocessor.transform(project_data)
    
    # Predict
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0]
    
    # Decode prediction
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    return {
        'prediction': predicted_label,
        'probability_failed': probability[0],
        'probability_successful': probability[1],
        'confidence': max(probability)
    }

# Save prediction function
joblib.dump(predict_kickstarter_success, f'models/predict_function_{timestamp}.pkl')
print(f"✓ Saved prediction function: models/predict_function_{timestamp}.pkl")
```

## Step 10.3: Test Prediction Function

```python
# Example prediction
example_project = {
    'main_category': 'Technology',
    'category': 'Hardware',
    'currency': 'USD',
    'country': 'US',
    'usd_goal_real': 10000,
    'launched': pd.Timestamp('2024-01-15'),
    'deadline': pd.Timestamp('2024-02-29'),
    'campaign_duration_days': 45,
    'launch_month': 1,
    'launch_year': 2024,
    # ... other features
}

# Make prediction
# result = predict_kickstarter_success(example_project, preprocessor, best_model_tuned, label_encoder)
# print("\nExample Prediction:")
# print(f"Project will be: {result['prediction']}")
# print(f"Probability of success: {result['probability_successful']:.2%}")
# print(f"Confidence: {result['confidence']:.2%}")
```

---

# 📊 PHASE 11: CREATE PRESENTATION

## Step 11.1: Key Insights Summary

**Document your findings:**

1. **Dataset Overview**
   - Total projects analyzed
   - Success rate
   - Date range
   - Categories covered

2. **Key Findings from EDA**
   - Which categories perform best?
   - Optimal campaign duration?
   - Best time to launch?
   - Goal amount sweet spot?
   - Geographic patterns?

3. **Model Performance**
   - Best model and why
   - Final test metrics
   - Comparison to baseline
   - Feature importance insights

4. **Business Recommendations**
   - How to use this model?
   - What makes projects successful?
   - Red flags to avoid?
   - Confidence thresholds for decisions

## Step 11.2: Create Presentation Slides

**Suggested Structure:**

1. **Title Slide**
   - Project name
   - Team members
   - Date

2. **Problem Statement**
   - Goal: Predict Kickstarter success
   - Why it matters
   - Approach overview

3. **Dataset**
   - Source and size
   - Key features
   - Target variable

4. **Exploratory Data Analysis**
   - Key visualizations (3-5 plots)
   - Main insights
   - Challenges discovered

5. **Data Preprocessing**
   - Missing value strategy
   - Outlier handling
   - Feature engineering highlights

6. **Model Development**
   - Models tested
   - Comparison table
   - ROC curves

7. **Best Model**
   - Model choice and rationale
   - Hyperparameters
   - Performance metrics
   - Feature importance

8. **Results & Insights**
   - Test set performance
   - Confusion matrix
   - Key predictive features
   - Business impact

9. **Recommendations**
   - For project creators
   - For backers/investors
   - Model limitations
   - Future improvements

10. **Conclusion**
    - Summary of achievements
    - Lessons learned
    - Next steps

## Step 11.3: Key Visualizations for Presentation

**Must-have plots:**
1. Target variable distribution
2. Success rate by category
3. Success rate by goal amount
4. Temporal trends
5. Model comparison bar chart
6. ROC curves
7. Final confusion matrix
8. Feature importance
9. Prediction probability distribution

---

# 📋 COMPLETE CHECKLIST

## Data Understanding ✓
- [ ] Load data
- [ ] Check shape and structure
- [ ] Identify target variable
- [ ] Document feature meanings

## EDA ✓
- [ ] Missing values analysis
- [ ] Target distribution
- [ ] Numerical features analysis
- [ ] Categorical features analysis
- [ ] Correlation analysis
- [ ] Temporal analysis
- [ ] Outlier detection

## Data Cleaning ✓
- [ ] Handle missing values
- [ ] Remove outliers (if needed)
- [ ] Remove duplicates
- [ ] Fix data quality issues
- [ ] Drop leakage features
- [ ] Save cleaned data

## Feature Engineering ✓
- [ ] Temporal features
- [ ] Cyclical encoding
- [ ] Goal-based features
- [ ] Category encoding (target encoding)
- [ ] Text features
- [ ] Interaction features
- [ ] Domain-specific features

## Preprocessing ✓
- [ ] Train-test split (with stratification)
- [ ] Create preprocessing pipeline
- [ ] Scale numerical features
- [ ] Encode categorical features
- [ ] Handle class imbalance (if needed)
- [ ] Create validation set

## Modeling ✓
- [ ] Baseline model
- [ ] Logistic Regression
- [ ] Decision Tree
- [ ] Random Forest
- [ ] XGBoost
- [ ] LightGBM
- [ ] SVM (optional)

## Evaluation ✓
- [ ] Compare all models
- [ ] Visualize comparisons
- [ ] Select best model
- [ ] Hyperparameter tuning
- [ ] Final test set evaluation
- [ ] Feature importance analysis

## Deployment ✓
- [ ] Save final model
- [ ] Save preprocessing pipeline
- [ ] Create prediction function
- [ ] Document model usage
- [ ] Version control

## Presentation ✓
- [ ] Create slide deck
- [ ] Include key visualizations
- [ ] Document insights
- [ ] Business recommendations
- [ ] Practice presentation

---

# 🎓 LEARNING OUTCOMES

By completing this project, you will:

1. ✓ Master end-to-end ML workflow
2. ✓ Learn proper train-test splitting
3. ✓ Understand feature engineering importance
4. ✓ Practice multiple classification algorithms
5. ✓ Learn model evaluation techniques
6. ✓ Avoid common pitfalls (data leakage!)
7. ✓ Present ML results effectively
8. ✓ Work collaboratively on ML projects

---

# 📚 ADDITIONAL RESOURCES

## Python Libraries Documentation
- **Pandas:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **XGBoost:** https://xgboost.readthedocs.io/
- **LightGBM:** https://lightgbm.readthedocs.io/
- **Matplotlib/Seaborn:** https://matplotlib.org/, https://seaborn.pydata.org/

## Tutorials
- Kaggle Learn: https://www.kaggle.com/learn
- Scikit-learn tutorials: https://scikit-learn.org/stable/tutorial/index.html

## Best Practices
- Google ML Crash Course: https://developers.google.com/machine-learning/crash-course
- Machine Learning Mastery: https://machinelearningmastery.com/

---

# ⚠️ COMMON PITFALLS TO AVOID

1. **Data Leakage** - Using features known only after campaign ends
2. **Not splitting data properly** - Split BEFORE feature engineering
3. **Ignoring class imbalance** - Use stratified split and class weights
4. **Overfitting** - Don't tune on test set, use cross-validation
5. **Not handling missing values** - Always check for NaNs
6. **Forgetting to scale** - Normalize features for some algorithms
7. **Not validating assumptions** - Check data distributions
8. **Poor feature engineering** - This is where most gains come from!
9. **Testing on train data** - Always use hold-out test set
10. **Not documenting work** - Keep notes and comments

---

# 🚀 SUCCESS METRICS

**Your project is successful if:**
- ✓ Model beats baseline significantly (10%+ improvement)
- ✓ Test F1 score > 0.70
- ✓ ROC-AUC > 0.75
- ✓ No data leakage detected
- ✓ Code is well-documented
- ✓ Presentation is clear and insightful
- ✓ Team collaboration was effective

---

**Good luck with your Kickstarter prediction project! 🎯🚀**