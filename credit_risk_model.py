"""
Credit Risk Model - Loan Default Prediction
Author: Amy Ke
Description: Logistic regression model to predict probability of loan default
using the Give Me Some Credit dataset (150,000 borrowers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:
    def __init__(self, filepath='cs-training.csv'):
        self.filepath = filepath
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_and_clean_data(self):
        """Load and clean the dataset"""
        print("="*60)
        print("CREDIT RISK MODEL - LOAN DEFAULT PREDICTION")
        print("="*60)
        print("\nLoading data...")
        
        self.df = pd.read_csv(self.filepath)
        
        # Drop unnamed index column if present
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Default rate: {self.df['SeriousDlqin2yrs'].mean():.2%}")
        
        # Handle missing values
        print("\nHandling missing values...")
        missing_before = self.df.isnull().sum().sum()
        
        # Fill missing values with median (robust to outliers)
        self.df['MonthlyIncome'].fillna(
            self.df['MonthlyIncome'].median(), inplace=True)
        self.df['NumberOfDependents'].fillna(
            self.df['NumberOfDependents'].median(), inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
        
        # Remove outliers (cap extreme values)
        self.df['RevolvingUtilizationOfUnsecuredLines'] = \
            self.df['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1)
        self.df['DebtRatio'] = self.df['DebtRatio'].clip(0, 10)
        self.df['age'] = self.df['age'].clip(18, 100)
        
        print("Data cleaning complete!")
        return self.df

    def explore_data(self):
        """Print key statistics"""
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        stats = {
            'Total Borrowers': len(self.df),
            'Defaulted (%)': f"{self.df['SeriousDlqin2yrs'].mean():.2%}",
            'Average Age': f"{self.df['age'].mean():.1f} years",
            'Avg Monthly Income': f"${self.df['MonthlyIncome'].median():,.0f}",
            'Avg Debt Ratio': f"{self.df['DebtRatio'].mean():.2%}",
            'Avg Credit Lines': f"{self.df['NumberOfOpenCreditLinesAndLoans'].mean():.1f}"
        }
        
        for key, value in stats.items():
            print(f"{key:.<40} {value}")

    def prepare_features(self):
        """Prepare features for modeling"""
        print("\nPreparing features...")
        
        self.feature_names = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ]
        
        X = self.df[self.feature_names]
        y = self.df['SeriousDlqin2yrs']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42,
                           stratify=y)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train):,} borrowers")
        print(f"Test set: {len(self.X_test):,} borrowers")

    def train_model(self):
        """Train logistic regression model"""
        print("\nTraining logistic regression model...")
        
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Model training complete!")

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_prob = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # AUC Score
        auc = roc_auc_score(self.y_test, y_prob)
        
        # Confusion matrix values
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'Test Set Size': f"{len(self.y_test):,} borrowers",
            'AUC Score': f"{auc:.4f}",
            'Accuracy': f"{accuracy:.2%}",
            'Precision': f"{precision:.2%}",
            'Recall (Default Detection)': f"{recall:.2%}",
            'True Positives (Defaults Caught)': f"{tp:,}",
            'False Positives (False Alarms)': f"{fp:,}",
            'True Negatives (Correct Safe)': f"{tn:,}",
            'False Negatives (Missed Defaults)': f"{fn:,}"
        }
        
        for key, value in metrics.items():
            print(f"{key:.<45} {value}")
        
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.auc = auc
        
        return metrics

    def plot_results(self):
        """Create visualizations"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Plot 1: ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        ax1.plot(fpr, tpr, color='blue', linewidth=2,
                label=f'ROC Curve (AUC = {self.auc:.4f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax1.fill_between(fpr, tpr, alpha=0.1, color='blue')
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Feature Importance
        ax2 = fig.add_subplot(gs[0, 1])
        coefficients = pd.Series(
            np.abs(self.model.coef_[0]),
            index=self.feature_names
        ).sort_values(ascending=True)
        
        colors = ['#2196F3' if c < coefficients.mean() 
                 else '#F44336' for c in coefficients]
        coefficients.plot(kind='barh', ax=ax2, color=colors)
        ax2.set_title('Feature Importance\n(Absolute Coefficients)',
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('Importance Score', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='x')

        # Plot 3: Confusion Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(self.y_test, self.y_pred)
        im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax3)
        ax3.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
        classes = ['No Default', 'Default']
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(classes)
        ax3.set_yticklabels(classes)
        ax3.set_ylabel('True Label', fontsize=11)
        ax3.set_xlabel('Predicted Label', fontsize=11)
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 
                        else "black", fontsize=12)

        # Plot 4: Default Probability Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(self.y_prob[self.y_test == 0], bins=50,
                alpha=0.6, color='green', label='No Default',
                density=True)
        ax4.hist(self.y_prob[self.y_test == 1], bins=50,
                alpha=0.6, color='red', label='Default',
                density=True)
        ax4.set_xlabel('Predicted Default Probability', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title('Default Probability Distribution',
                     fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0.5, color='black', linestyle='--',
                   alpha=0.7, label='Decision Threshold')

        plt.suptitle('Credit Risk Model - Performance Dashboard',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('credit_risk_performance.png', dpi=300,
                   bbox_inches='tight')
        print("\nChart saved as 'credit_risk_performance.png'")
        plt.show()

    def run(self):
        """Run complete pipeline"""
        self.load_and_clean_data()
        self.explore_data()
        self.prepare_features()
        self.train_model()
        metrics = self.evaluate_model()
        self.plot_results()
        return metrics

if __name__ == "__main__":
    model = CreditRiskModel(filepath='cs-training.csv')
    results = model.run()