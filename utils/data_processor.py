"""
Data processing utilities for the Logistic Regression Playground.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from .data_models import Dataset, SyntheticDataConfig, DataType

class DataValidator:
    """Validate data quality and suitability for logistic regression."""
    
    @staticmethod
    def validate_for_classification(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Validate dataset for classification tasks."""
        issues = []
        recommendations = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            recommendations.append("Consider imputing missing values or removing rows/columns with too many missing values")
        
        # Check target variable
        if target_col not in df.columns:
            issues.append(f"Target column '{target_col}' not found")
        else:
            unique_targets = df[target_col].nunique()
            if unique_targets < 2:
                issues.append("Target variable has less than 2 unique values")
            elif unique_targets > 10:
                recommendations.append("Consider grouping classes if you have too many categories")
        
        # Check for constant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        constant_features = []
        for col in numeric_cols:
            if df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"Constant features found: {constant_features}")
            recommendations.append("Remove constant features as they don't provide information")
        
        # Check data types
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 1:  # Excluding target if it's categorical
            recommendations.append("Consider encoding categorical variables for better model performance")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
    
    @staticmethod
    def suggest_target_column(df: pd.DataFrame) -> List[str]:
        """Suggest potential target columns."""
        suggestions = []
        
        for col in df.columns:
            # Check if column has reasonable number of unique values for classification
            unique_count = df[col].nunique()
            if 2 <= unique_count <= min(10, len(df) // 10):
                suggestions.append(col)
        
        return suggestions
    
    @staticmethod
    def suggest_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
        """Suggest feature columns excluding target."""
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Prioritize numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_cols if col != target_col]
        
        if numeric_features:
            return numeric_features
        
        return feature_cols

class PreprocessingPipeline:
    """Comprehensive preprocessing pipeline for classification data."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.preprocessing_steps = []
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_processed = df.copy()
        
        if strategy == 'mean':
            # Fill numeric columns with mean
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        elif strategy == 'median':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        elif strategy == 'drop':
            df_processed = df_processed.dropna()
        
        self.preprocessing_steps.append(f"Missing values handled using {strategy} strategy")
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Encode categorical variables."""
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns
        
        # Remove target column from encoding if specified
        if target_col and target_col in categorical_cols:
            categorical_cols = categorical_cols.drop(target_col)
        
        for col in categorical_cols:
            unique_values = df_processed[col].nunique()
            
            if unique_values == 2:
                # Binary encoding for binary categorical variables
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.encoders[col] = le
                self.preprocessing_steps.append(f"Binary encoded column: {col}")
            
            elif unique_values <= 10:
                # One-hot encoding for low cardinality categorical variables
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
                self.preprocessing_steps.append(f"One-hot encoded column: {col}")
            
            else:
                # Label encoding for high cardinality (not ideal, but simple)
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.encoders[col] = le
                self.preprocessing_steps.append(f"Label encoded high-cardinality column: {col}")
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', target_col: str = None) -> pd.DataFrame:
        """Scale numerical features."""
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Remove target column from scaling if specified
        if target_col and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        if method == 'standard':
            scaler = StandardScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            self.scalers['standard'] = scaler
            self.preprocessing_steps.append("Applied standard scaling to numeric features")
        
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            self.scalers['minmax'] = scaler
            self.preprocessing_steps.append("Applied min-max scaling to numeric features")
        
        return df_processed
    
    def get_preprocessing_summary(self) -> str:
        """Get summary of preprocessing steps applied."""
        if not self.preprocessing_steps:
            return "No preprocessing steps applied."
        
        summary = "Preprocessing steps applied:\n"
        for i, step in enumerate(self.preprocessing_steps, 1):
            summary += f"{i}. {step}\n"
        
        return summary

class DataProcessor:
    """Main data processing class."""
    
    def __init__(self):
        self.preprocessor = PreprocessingPipeline()
        self.validator = DataValidator()
    
    def load_dataset(self, source: str, **kwargs) -> Dataset:
        """Load dataset from various sources."""
        if source == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
            return Dataset(
                name="Iris",
                data=df,
                target_column="target",
                feature_columns=list(data.feature_names),
                data_type=DataType.BUILTIN,
                class_names=list(data.target_names),
                description="Classic iris flower classification dataset"
            )
        
        elif source == "wine":
            from sklearn.datasets import load_wine
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
            return Dataset(
                name="Wine",
                data=df,
                target_column="target",
                feature_columns=list(data.feature_names),
                data_type=DataType.BUILTIN,
                class_names=list(data.target_names),
                description="Wine recognition dataset"
            )
        
        elif source == "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
            return Dataset(
                name="Breast Cancer",
                data=df,
                target_column="target",
                feature_columns=list(data.feature_names),
                data_type=DataType.BUILTIN,
                class_names=list(data.target_names),
                description="Breast cancer diagnosis dataset"
            )
        
        else:
            raise ValueError(f"Unknown dataset source: {source}")
    
    def generate_synthetic_data(self, config: SyntheticDataConfig) -> Dataset:
        """Generate synthetic classification data."""
        if not config.validate():
            raise ValueError("Invalid synthetic data configuration")
        
        if config.n_classes == 2:
            # Binary classification
            X, y = make_classification(
                n_samples=config.n_samples,
                n_features=config.n_features,
                n_redundant=0,
                n_informative=config.n_features,
                n_clusters_per_class=1,
                class_sep=config.class_separation,
                random_state=config.random_state
            )
        else:
            # Multiclass classification using blobs
            X, y = make_blobs(
                n_samples=config.n_samples,
                centers=config.n_classes,
                n_features=config.n_features,
                cluster_std=config.cluster_std,
                random_state=config.random_state
            )
        
        # Add noise if specified
        if config.noise_level > 0:
            noise = np.random.normal(0, config.noise_level, X.shape)
            X += noise
        
        # Create DataFrame
        feature_names = [f"Feature_{i+1}" for i in range(config.n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        class_names = [f"Class_{i}" for i in range(config.n_classes)]
        
        return Dataset(
            name=f"Synthetic_{config.n_classes}Class",
            data=df,
            target_column="target",
            feature_columns=feature_names,
            data_type=DataType.SYNTHETIC,
            class_names=class_names,
            description=f"Synthetic {config.n_classes}-class dataset with {config.n_features} features"
        )
    
    def preprocess_data(self, dataset: Dataset, options: Dict[str, Any]) -> Dataset:
        """Apply preprocessing to dataset."""
        df_processed = dataset.data.copy()
        
        # Handle missing values
        if options.get('handle_missing', False):
            strategy = options.get('missing_strategy', 'mean')
            df_processed = self.preprocessor.handle_missing_values(df_processed, strategy)
        
        # Encode categorical variables
        if options.get('encode_categorical', False):
            df_processed = self.preprocessor.encode_categorical_variables(
                df_processed, dataset.target_column
            )
        
        # Scale features
        if options.get('scale_features', False):
            method = options.get('scaling_method', 'standard')
            df_processed = self.preprocessor.scale_features(
                df_processed, method, dataset.target_column
            )
        
        # Update feature columns after preprocessing
        new_feature_cols = [col for col in df_processed.columns if col != dataset.target_column]
        
        # Create new dataset with processed data
        processed_dataset = Dataset(
            name=f"{dataset.name}_processed",
            data=df_processed,
            target_column=dataset.target_column,
            feature_columns=new_feature_cols,
            data_type=dataset.data_type,
            class_names=dataset.class_names,
            description=f"{dataset.description} (preprocessed)",
            preprocessing_applied=options
        )
        
        return processed_dataset
    
    def validate_data_quality(self, dataset: Dataset) -> Dict[str, Any]:
        """Validate dataset quality."""
        return self.validator.validate_for_classification(dataset.data, dataset.target_column)
    
    def split_data(self, dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split dataset into train and test sets."""
        X, y = dataset.get_X_y()
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_data_summary(self, dataset: Dataset) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        df = dataset.data
        
        summary = {
            'name': dataset.name,
            'n_samples': dataset.n_samples,
            'n_features': dataset.n_features,
            'n_classes': dataset.n_classes,
            'class_distribution': dataset.get_class_distribution(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        # Add basic statistics for numeric features
        numeric_stats = df.select_dtypes(include=[np.number]).describe()
        if not numeric_stats.empty:
            summary['numeric_statistics'] = numeric_stats.to_dict()
        
        return summary