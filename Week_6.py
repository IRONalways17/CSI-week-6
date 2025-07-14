# =============================================================================
# Machine Learning Model Comparison and Hyperparameter Tuning
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== Machine Learning Model Comparison and Hyperparameter Tuning ===\n")

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data():
    """Load and prepare the wine dataset"""
    print("1. Loading Wine Dataset...")
    
    # Load the wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(X, columns=wine.feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Feature names: {wine.feature_names[:5]}... (showing first 5)")
    
    # Save dataset to CSV
    df.to_csv('wine.csv', index=False)
    print("Dataset saved as 'wine.csv'")
    
    return X, y, wine.feature_names, wine.target_names

# =============================================================================
# 2. BASELINE MODEL EVALUATION
# =============================================================================

def evaluate_baseline_models(X_train, X_test, y_train, y_test):
    """Train and evaluate baseline models"""
    print("\n2. Training Baseline Models...")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    baseline_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        baseline_results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Model': model
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    return baseline_results

# =============================================================================
# 3. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# =============================================================================

def grid_search_tuning(X_train, y_train):
    """Perform GridSearchCV for Random Forest"""
    print("\n3. Hyperparameter Tuning with GridSearchCV...")
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Perform Grid Search
    print("Performing GridSearchCV on Random Forest...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

# =============================================================================
# 4. HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV
# =============================================================================

def randomized_search_tuning(X_train, y_train):
    """Perform RandomizedSearchCV for SVM"""
    print("\n4. Hyperparameter Tuning with RandomizedSearchCV...")
    
    # Define parameter distribution for SVM
    param_dist = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.001, 1),
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    # Initialize SVM
    svm = SVC(random_state=42, probability=True)
    
    # Perform Randomized Search
    print("Performing RandomizedSearchCV on SVM...")
    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

# =============================================================================
# 5. EVALUATE TUNED MODELS
# =============================================================================

def evaluate_tuned_models(X_test, y_test, tuned_rf, tuned_svm):
    """Evaluate tuned models on test set"""
    print("\n5. Evaluating Tuned Models...")
    
    tuned_results = {}
    
    # Evaluate tuned Random Forest
    rf_pred = tuned_rf.predict(X_test)
    tuned_results['Tuned Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred, average='weighted'),
        'Recall': recall_score(y_test, rf_pred, average='weighted'),
        'F1-Score': f1_score(y_test, rf_pred, average='weighted'),
        'Model': tuned_rf
    }
    
    # Evaluate tuned SVM
    svm_pred = tuned_svm.predict(X_test)
    tuned_results['Tuned SVM'] = {
        'Accuracy': accuracy_score(y_test, svm_pred),
        'Precision': precision_score(y_test, svm_pred, average='weighted'),
        'Recall': recall_score(y_test, svm_pred, average='weighted'),
        'F1-Score': f1_score(y_test, svm_pred, average='weighted'),
        'Model': tuned_svm
    }
    
    for name, metrics in tuned_results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    
    return tuned_results

# =============================================================================
# 6. COMPARE ALL MODELS
# =============================================================================

def compare_all_models(baseline_results, tuned_results):
    """Compare all models and select the best one"""
    print("\n6. Model Comparison...")
    
    # Combine all results
    all_results = {**baseline_results, **tuned_results}
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score']
        }
        for name, metrics in all_results.items()
    }).T
    
    print("\nModel Comparison Results:")
    print(comparison_df.round(4))
    
    # Find best model based on F1-Score
    best_model_name = comparison_df['F1-Score'].idxmax()
    best_model = all_results[best_model_name]['Model']
    best_f1_score = comparison_df.loc[best_model_name, 'F1-Score']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best F1-Score: {best_f1_score:.4f}")
    
    # Save comparison results
    comparison_df.to_csv('model_comparison_results.csv')
    print("Comparison results saved as 'model_comparison_results.csv'")
    
    return best_model, best_model_name, comparison_df

# =============================================================================
# 7. GENERATE CLASSIFICATION REPORT AND CONFUSION MATRIX
# =============================================================================

def generate_detailed_report(X_test, y_test, best_model, best_model_name, target_names):
    """Generate detailed classification report and confusion matrix"""
    print(f"\n7. Detailed Analysis for {best_model_name}...")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred

# =============================================================================
# 8. SAVE BEST MODEL
# =============================================================================

def save_best_model(best_model, best_model_name):
    """Save the best model for future use"""
    print(f"\n8. Saving Best Model ({best_model_name})...")
    
    # Save model
    model_filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Model saved as '{model_filename}'")
    
    return model_filename

# =============================================================================
# 9. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    # Load and prepare data
    X, y, feature_names, target_names = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Evaluate baseline models
    baseline_results = evaluate_baseline_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Hyperparameter tuning
    tuned_rf, rf_params, rf_score = grid_search_tuning(X_train_scaled, y_train)
    tuned_svm, svm_params, svm_score = randomized_search_tuning(X_train_scaled, y_train)
    
    # Evaluate tuned models
    tuned_results = evaluate_tuned_models(X_test_scaled, y_test, tuned_rf, tuned_svm)
    
    # Compare all models
    best_model, best_model_name, comparison_df = compare_all_models(baseline_results, tuned_results)
    
    # Generate detailed report
    y_pred = generate_detailed_report(X_test_scaled, y_test, best_model, best_model_name, target_names)
    
    # Save best model
    model_filename = save_best_model(best_model, best_model_name)
    
    # Save scaler for future use
    joblib.dump(scaler, 'scaler.joblib')
    print("Feature scaler saved as 'scaler.joblib'")
    
    print("\n=== Analysis Complete ===")
    print(f"Best performing model: {best_model_name}")
    print(f"All results saved to files for future reference.")

if __name__ == "__main__":
    main()
