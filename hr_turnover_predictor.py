"""
Salifort Motors HR Analytics: Predicting Employee Turnover
==========================================================

A production-ready machine learning pipeline to predict employee churn
and identify key drivers of attrition.

This script implements the PACE framework:
- Plan: Define scope and stakeholders
- Analyze: Explore and understand data  
- Construct: Build and validate models
- Execute: Deliver insights and recommendations

Author: Kyle Ware
Date: Feb 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA PROCESSING
# =============================================================================

class HRDataProcessor:
    """Handles data loading, cleaning, and preprocessing for HR analytics."""
    
    # Column name standardization mapping
    COLUMN_RENAME_MAP = {
        'Work_accident': 'work_accident',
        'average_montly_hours': 'average_monthly_hours',  # Fix typo in original data
        'time_spend_company': 'tenure',
        'Department': 'department'
    }
    
    # Ordinal encoding for salary levels
    SALARY_ENCODING = {'low': 0, 'medium': 1, 'high': 2}
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the HR dataset CSV file.
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the raw dataset from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.raw_data):,} records with {len(self.raw_data.columns)} features")
        return self.raw_data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset: standardize column names and remove duplicates.
        
        Returns:
            Cleaned DataFrame.
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df = self.raw_data.copy()
        
        # Standardize column names to snake_case
        df = df.rename(columns=self.COLUMN_RENAME_MAP)
        logger.info("Standardized column names to snake_case")
        
        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates(keep='first')
        duplicates_removed = initial_count - len(df)
        logger.info(f"Removed {duplicates_removed:,} duplicate records")
        
        self.processed_data = df
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for modeling.
        
        Args:
            df: DataFrame to encode.
            
        Returns:
            Encoded DataFrame.
        """
        df_encoded = df.copy()
        
        # Ordinal encoding for salary (low=0, medium=1, high=2)
        df_encoded['salary'] = df_encoded['salary'].map(self.SALARY_ENCODING)
        
        # One-hot encoding for department
        df_encoded = pd.get_dummies(df_encoded, columns=['department'], drop_first=False)
        
        logger.info("Encoded categorical features")
        return df_encoded
    
    def get_feature_target_split(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'left'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).
        
        Args:
            df: Encoded DataFrame.
            target_col: Name of the target column.
            
        Returns:
            Tuple of (features, target).
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
        return X, y


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

class TurnoverPredictor:
    """Machine learning pipeline for employee turnover prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets with stratification.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        logger.info(f"Training set: {len(X_train):,} samples, Test set: {len(X_test):,} samples")
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> LogisticRegression:
        """Train a Logistic Regression baseline model."""
        logger.info("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=500, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_estimators: int = 100
    ) -> RandomForestClassifier:
        """Train a Random Forest classifier (champion model)."""
        logger.info(f"Training Random Forest model with {n_estimators} trees...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def evaluate_model(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a trained model and return metrics.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            model_name: Name for logging.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        
        self.results[model_name] = metrics
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_name.upper()} RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model, feature_names: list) -> pd.Series:
        """
        Extract feature importances from a tree-based model.
        
        Args:
            model: Trained model with feature_importances_ attribute.
            feature_names: List of feature names.
            
        Returns:
            Sorted Series of feature importances.
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importances = pd.Series(
            model.feature_importances_, 
            index=feature_names
        ).sort_values(ascending=False)
        
        return importances
    
    def save_model(self, model, filepath: str) -> None:
        """Save a trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model


# =============================================================================
# VISUALIZATIONS
# =============================================================================

class TurnoverVisualizer:
    """Visualization utilities for HR turnover analysis."""
    
    def __init__(self, style: str = 'whitegrid'):
        """Initialize with a Seaborn style."""
        sns.set_style(style)
        
    def plot_confusion_matrix(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.show()
        
    def plot_feature_importance(
        self, 
        importances: pd.Series, 
        top_n: int = 10,
        save_path: str = None
    ) -> None:
        """Plot top N feature importances."""
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importances.head(top_n)
        
        # Use hue parameter to avoid deprecation warning
        sns.barplot(
            x=top_features.values, 
            y=top_features.index, 
            hue=top_features.index,
            legend=False,
            palette='viridis',
            ax=ax
        )
        ax.set_xlabel('Importance (Mean Decrease in Impurity)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - Random Forest', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        plt.tight_layout()
        plt.show()
        
    def plot_burnout_cluster(
        self, 
        df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """Visualize the burnout cluster (hours vs satisfaction)."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.scatterplot(
            data=df, 
            x='average_monthly_hours', 
            y='satisfaction_level', 
            hue='left',
            alpha=0.4,
            palette={0: '#3498db', 1: '#e74c3c'},
            ax=ax
        )
        
        # Add reference line for standard 40-hour week (166.67 hrs/month)
        ax.axvline(x=166.67, color='red', linestyle='--', linewidth=2, label='40 hrs/week')
        
        ax.set_xlabel('Average Monthly Hours', fontsize=12)
        ax.set_ylabel('Satisfaction Level', fontsize=12)
        ax.set_title('Satisfaction vs. Monthly Hours (Burnout Analysis)', fontsize=14, fontweight='bold')
        ax.legend(title='Status', loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Burnout cluster plot saved to {save_path}")
        plt.show()
    
    def plot_correlation_heatmap(
        self, 
        df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """Plot correlation heatmap for numeric features."""
        fig, ax = plt.subplots(figsize=(12, 8))
        numeric_cols = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='vlag', center=0, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        plt.tight_layout()
        plt.show()
    
    def plot_turnover_by_department(
        self, 
        df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """Plot turnover distribution by department."""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='department', hue='left', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Turnover by Department', fontsize=14, fontweight='bold')
        ax.legend(title='Left', labels=['Stayed', 'Left'])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Department turnover plot saved to {save_path}")
        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def find_data_file(script_dir: str) -> str:
    """
    Search for the HR dataset in common locations.
    
    Args:
        script_dir: Directory where the script is located.
        
    Returns:
        Path to the data file if found, None otherwise.
    """
    possible_paths = [
        # Relative to script location
        os.path.join(script_dir, 'HR_capstone_dataset.csv'),
        os.path.join(script_dir, 'data', 'HR_capstone_dataset.csv'),
        os.path.join(script_dir, '..', 'data', 'HR_capstone_dataset.csv'),
        # Relative to current working directory
        'HR_capstone_dataset.csv',
        'data/HR_capstone_dataset.csv',
        # Common user locations
        os.path.expanduser('~/Downloads/HR_capstone_dataset.csv'),
        os.path.expanduser('~/Documents/HR_capstone_dataset.csv'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main(data_path: str = None, model_output_path: str = None):
    """
    Main execution pipeline following the PACE framework.
    
    Args:
        data_path: Path to the HR dataset CSV file. If None, searches common locations.
        model_output_path: Path to save the trained model. If None, saves to script directory.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find data file if not provided
    if data_path is None:
        data_path = find_data_file(script_dir)
        
        if data_path is None:
            logger.error("=" * 60)
            logger.error("DATA FILE NOT FOUND!")
            logger.error("=" * 60)
            logger.error("\nPlease do ONE of the following:")
            logger.error("  1. Place 'HR_capstone_dataset.csv' in the same folder as this script")
            logger.error("  2. Create a 'data/' subfolder and put the CSV there")
            logger.error("  3. Run with path argument: python hr_turnover_predictor.py path/to/file.csv")
            logger.error(f"\nScript location: {script_dir}")
            sys.exit(1)
    
    # Set default model output path
    if model_output_path is None:
        model_output_path = os.path.join(script_dir, 'random_forest_model.pkl')
    
    logger.info(f"Using data file: {os.path.abspath(data_path)}")
    
    # Initialize components
    processor = HRDataProcessor(data_path)
    predictor = TurnoverPredictor(random_state=42)
    visualizer = TurnoverVisualizer()
    
    # =========================================================================
    # STAGE 1: PLAN & DATA PROCESSING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PROCESSING (Plan & Analyze)")
    logger.info("=" * 60)
    
    # Load and clean data
    df = processor.load_data()
    df_clean = processor.clean_data()
    
    # Show turnover distribution
    turnover_counts = df_clean['left'].value_counts()
    turnover_pct = df_clean['left'].value_counts(normalize=True) * 100
    logger.info(f"\nTurnover Distribution:")
    logger.info(f"  Stayed: {turnover_counts[0]:,} ({turnover_pct[0]:.1f}%)")
    logger.info(f"  Left:   {turnover_counts[1]:,} ({turnover_pct[1]:.1f}%)")
    
    # =========================================================================
    # STAGE 2: ANALYZE (EDA)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    
    # Mean satisfaction by turnover status
    satisfaction_by_status = df_clean.groupby('left')['satisfaction_level'].agg(['mean', 'median'])
    logger.info(f"\nSatisfaction Level by Turnover Status:")
    logger.info(f"  Stayed - Mean: {satisfaction_by_status.loc[0, 'mean']:.3f}, Median: {satisfaction_by_status.loc[0, 'median']:.3f}")
    logger.info(f"  Left   - Mean: {satisfaction_by_status.loc[1, 'mean']:.3f}, Median: {satisfaction_by_status.loc[1, 'median']:.3f}")
    
    # =========================================================================
    # STAGE 3: CONSTRUCT (Model Building)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: MODEL CONSTRUCTION")
    logger.info("=" * 60)
    
    # Encode features
    df_encoded = processor.encode_features(df_clean)
    X, y = processor.get_feature_target_split(df_encoded)
    
    # Split data (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Train baseline model: Logistic Regression
    lr_model = predictor.train_logistic_regression(X_train, y_train)
    predictor.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    
    # Train champion model: Random Forest
    rf_model = predictor.train_random_forest(X_train, y_train, n_estimators=100)
    predictor.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    
    # =========================================================================
    # STAGE 4: EXECUTE (Results & Recommendations)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: EXECUTE (Feature Importance & Insights)")
    logger.info("=" * 60)
    
    # Feature importance analysis
    importances = predictor.get_feature_importance(rf_model, X.columns.tolist())
    logger.info("\nTop 5 Drivers of Employee Turnover:")
    for i, (feature, importance) in enumerate(importances.head(5).items(), 1):
        logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    # Create figures directory
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Saving figures to: {figures_dir}")
    
    y_pred_rf = rf_model.predict(X_test)
    
    # Generate and save plots
    visualizer.plot_confusion_matrix(
        y_test, y_pred_rf, 'Random Forest',
        save_path=os.path.join(figures_dir, 'confusion_matrix.png')
    )
    visualizer.plot_feature_importance(
        importances, top_n=10,
        save_path=os.path.join(figures_dir, 'feature_importance.png')
    )
    visualizer.plot_burnout_cluster(
        df_clean,
        save_path=os.path.join(figures_dir, 'burnout_cluster.png')
    )
    visualizer.plot_correlation_heatmap(
        df_clean,
        save_path=os.path.join(figures_dir, 'correlation_heatmap.png')
    )
    visualizer.plot_turnover_by_department(
        df_clean,
        save_path=os.path.join(figures_dir, 'turnover_by_department.png')
    )
    
    # Save champion model
    predictor.save_model(rf_model, model_output_path)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 60)
    
    rf_results = predictor.results['Random Forest']
    logger.info(f"\nChampion Model: Random Forest")
    logger.info(f"  Recall:    {rf_results['recall']*100:.1f}% (catches {rf_results['recall']*100:.1f}% of employees who will leave)")
    logger.info(f"  Precision: {rf_results['precision']*100:.1f}% (predictions are {rf_results['precision']*100:.1f}% accurate)")
    logger.info(f"  F1 Score:  {rf_results['f1_score']*100:.1f}%")
    
    logger.info("\nKey Recommendations:")
    logger.info("  1. Cap employee workloads at 5 concurrent projects")
    logger.info("  2. Flag high performers working 250+ hrs/month for intervention")
    logger.info("  3. Implement retention program for 3-5 year tenure employees")
    logger.info("  4. Review compensation in low/medium salary bands")
    
    logger.info("\nFiles Saved:")
    logger.info(f"  Model: {model_output_path}")
    logger.info(f"  Figures:")
    logger.info(f"    - {os.path.join(figures_dir, 'confusion_matrix.png')}")
    logger.info(f"    - {os.path.join(figures_dir, 'feature_importance.png')}")
    logger.info(f"    - {os.path.join(figures_dir, 'burnout_cluster.png')}")
    logger.info(f"    - {os.path.join(figures_dir, 'correlation_heatmap.png')}")
    logger.info(f"    - {os.path.join(figures_dir, 'turnover_by_department.png')}")
    
    return predictor.results


if __name__ == '__main__':
    # Allow passing data path as command-line argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        results = main(data_path=data_file)
    else:
        results = main()
