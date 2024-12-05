# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Function to read and prepare data
def load_data(file_path):
    """
    Load data from SQL export and perform initial filtering
    """
    # Read data
    df = pd.read_csv(file_path)  # Adjust based on your export format

    # Filter for eligibility
    df = df[df['ELIGIBILITY_YN'] == 1].copy()

    print(f"Total records after eligibility filter: {len(df)}")
    print(f"NBA5 attempted distribution:\n{df['NBA5_ATTEMPTED'].value_counts(normalize=True)}")

    return df


# Function for initial data analysis
def analyze_data_quality(df):
    """
    Perform initial data quality checks
    """
    # Missing values analysis
    missing_stats = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('missing_percentage', ascending=False)

    # Data types analysis
    dtype_stats = pd.DataFrame(df.dtypes, columns=['dtype'])

    # Basic statistics for numeric columns
    numeric_stats = df.describe()

    return missing_stats, dtype_stats, numeric_stats


# Function to handle missing values
def handle_missing_values(df):
    """
    Handle missing values based on column type and business logic
    """
    df_processed = df.copy()

    # Identify numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns

    # Fill numeric columns with median
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    # Fill categorical columns with mode
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    return df_processed


# Function to encode categorical variables
def encode_categorical_features(df):
    """
    Encode categorical variables using Label Encoding
    """
    df_encoded = df.copy()
    encoders = {}

    categorical_cols = df_encoded.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if col != 'TARGET':  # Don't encode target variable if it's categorical
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            encoders[col] = encoder

    return df_encoded, encoders


# Function to scale numerical features
def scale_numerical_features(df):
    """
    Scale numerical features using StandardScaler
    """
    df_scaled = df.copy()
    scaler = StandardScaler()

    numerical_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'NBA5_ATTEMPTED']

    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    return df_scaled, scaler


# Function to prepare features and target
def prepare_features_target(df):
    """
    Separate features and target variable
    """
    # Remove any columns that shouldn't be features
    columns_to_drop = ['MESSAGE_ID', 'NBA5_ATTEMPTED', 'ELIGIBILITY_YN']

    X = df.drop(columns=columns_to_drop)
    y = df['NBA5_ATTEMPTED']

    return X, y


# Function to train and evaluate base models
def train_evaluate_base_models(X, y):
    """
    Train and evaluate multiple models for comparison
    """
    # Initialize models
    models = {
        'LightGBM': lgb.LGBMClassifier(random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE)
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        results[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'model': model,
            'feature_importance': None
        }

        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importance'] = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

    return results, X_train, X_test, y_train, y_test


# Function to analyze feature importance using SHAP
def analyze_shap_values(model, X_train, X_test):
    """
    Calculate and analyze SHAP values for model interpretability
    """
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification

    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    return explainer, shap_values


# Function to plot model performance metrics
def plot_model_performance(results):
    """
    Create visualizations for model performance comparison
    """
    # ROC AUC comparison
    roc_scores = {name: result['roc_auc'] for name, result in results.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(roc_scores.keys(), roc_scores.values())
    plt.title('ROC AUC Scores by Model')
    plt.ylabel('ROC AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt


# Main execution
if __name__ == "__main__":
    # Load data
    df = load_data('nba5_final.csv')

    # Analyze data quality
    missing_stats, dtype_stats, numeric_stats = analyze_data_quality(df)
    print("\nMissing Value Analysis:")
    print(missing_stats[missing_stats['missing_count'] > 0])

    # Handle missing values
    df_processed = handle_missing_values(df)

    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df_processed)

    # Scale numerical features
    df_scaled, scaler = scale_numerical_features(df_encoded)

    # Prepare features and target
    X, y = prepare_features_target(df_scaled)

    # Train and evaluate models
    results, X_train, X_test, y_train, y_test = train_evaluate_base_models(X, y)

    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        print("Classification Report:")
        print(result['classification_report'])
        print(f"ROC AUC Score: {result['roc_auc']:.4f}")

        if result['feature_importance'] is not None:
            print("\nTop 10 Important Features:")
            print(result['feature_importance'].head(10))

    # Plot model performance
    performance_plot = plot_model_performance(results)

    # Analyze best model with SHAP
    best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_model = results[best_model_name]['model']
    explainer, shap_values = analyze_shap_values(best_model, X_train, X_test)
