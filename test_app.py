#!/usr/bin/env python3
"""
Test script to verify the Logistic Regression Playground works correctly.
"""

import sys
import importlib.util

def test_imports():
    """Test that all required modules can be imported."""
    required_modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly.express',
        'plotly.graph_objects',
        'plotly.subplots',
        'sklearn.linear_model',
        'sklearn.datasets',
        'sklearn.metrics',
        'sklearn.preprocessing'
    ]
    
    print("Testing imports...")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

def test_utility_classes():
    """Test that utility classes can be imported and instantiated."""
    print("\nTesting utility classes...")
    
    try:
        from utils.data_models import Dataset, SyntheticDataConfig, DataType
        from utils.data_processor import DataProcessor
        from utils.visualization_engine import VisualizationEngine
        from utils.ai_client import GeminiClient
        
        # Test instantiation
        data_processor = DataProcessor()
        viz_engine = VisualizationEngine()
        
        print("✅ All utility classes imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Utility class error: {e}")
        return False

def test_data_loading():
    """Test that built-in datasets can be loaded."""
    print("\nTesting data loading...")
    
    try:
        from utils.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Test loading built-in datasets
        iris_dataset = processor.load_dataset("iris")
        wine_dataset = processor.load_dataset("wine")
        
        print(f"✅ Iris dataset: {iris_dataset.n_samples} samples, {iris_dataset.n_features} features")
        print(f"✅ Wine dataset: {wine_dataset.n_samples} samples, {wine_dataset.n_features} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_training():
    """Test that logistic regression models can be trained."""
    print("\nTesting model training...")
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        import numpy as np
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        # Test different penalty configurations
        models = [
            LogisticRegression(penalty=None, random_state=42),
            LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            LogisticRegression(penalty='l2', random_state=42)
        ]
        
        for i, model in enumerate(models):
            model.fit(X, y)
            accuracy = model.score(X, y)
            print(f"✅ Model {i+1} trained successfully, accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Testing Logistic Regression Playground")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_utility_classes,
        test_data_loading,
        test_model_training
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! The app should work correctly.")
        print("\nTo run the app:")
        print("streamlit run logistic_regression_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()