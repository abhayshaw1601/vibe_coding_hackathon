#!/usr/bin/env python3
"""
Quick test to verify the main functionality works.
"""

def test_basic_functionality():
    """Test basic functionality without running the full Streamlit app."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        print("✅ Basic imports successful")
        
        # Test data generation
        X, y = make_classification(n_samples=100, n_features=4, n_classes=3, 
                                 n_informative=4, n_redundant=0, n_clusters_per_class=1, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
        # Use only first 2 features for testing
        X_df = X_df[['Feature_1', 'Feature_2']]
        print("✅ Data generation successful")
        
        # Test boolean indexing (the issue we fixed)
        for class_idx in range(3):
            mask = y == class_idx
            if np.any(mask):
                # This was the problematic line - now using .loc instead of .iloc
                x_vals = X_df.loc[mask, X_df.columns[0]]
                y_vals = X_df.loc[mask, X_df.columns[1]]
                print(f"✅ Boolean indexing for class {class_idx}: {len(x_vals)} points")
        
        # Test model training with different penalties
        models = [
            LogisticRegression(penalty=None, random_state=42, max_iter=1000),
            LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            LogisticRegression(penalty='l2', random_state=42, max_iter=1000)
        ]
        
        for i, model in enumerate(models):
            model.fit(X, y)
            accuracy = model.score(X, y)
            print(f"✅ Model {i+1} (penalty={model.penalty}): accuracy={accuracy:.3f}")
        
        # Test utility imports
        from utils.data_processor import DataProcessor
        from utils.visualization_engine import VisualizationEngine
        
        processor = DataProcessor()
        viz_engine = VisualizationEngine()
        print("✅ Utility classes imported successfully")
        
        # Test dataset loading
        iris_dataset = processor.load_dataset("iris")
        print(f"✅ Iris dataset loaded: {iris_dataset.n_samples} samples, {iris_dataset.n_features} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🎯 Quick Test for Logistic Regression Playground")
    print("=" * 50)
    
    if test_basic_functionality():
        print("\n🎉 All tests passed!")
        print("The application should now work correctly.")
        print("\nTo run the app:")
        print("  streamlit run logistic_regression_app.py")
        print("\nOr use the startup script:")
        print("  python run_app.py")
    else:
        print("\n❌ Tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()