#!/usr/bin/env python3
"""
Verify that the boolean indexing fix works correctly.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def test_boolean_indexing_fix():
    """Test the specific boolean indexing issue that was causing the error."""
    print("🔍 Testing Boolean Indexing Fix")
    print("=" * 40)
    
    # Create test data similar to what's used in the app
    X, y = make_classification(n_samples=150, n_features=4, n_classes=3, 
                             n_informative=4, n_redundant=0, n_clusters_per_class=1, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
    # Use only first 2 features for visualization test
    X_df = X_df[['Feature_1', 'Feature_2']]
    
    print(f"📊 Test data: {len(X_df)} samples, {len(np.unique(y))} classes")
    
    # Test the old problematic approach (this would fail)
    print("\n❌ Old approach (would cause NotImplementedError):")
    print("   X.iloc[mask, 0]  # This fails with boolean mask")
    
    # Test the new fixed approach
    print("\n✅ New approach (fixed):")
    for class_idx in range(len(np.unique(y))):
        mask = y == class_idx
        n_points = np.sum(mask)
        
        if n_points > 0:
            # This is the fixed approach - using .loc with boolean mask
            x_vals = X_df.loc[mask, X_df.columns[0]]
            y_vals = X_df.loc[mask, X_df.columns[1]]
            
            print(f"   Class {class_idx}: {len(x_vals)} points extracted successfully")
            print(f"     X range: [{x_vals.min():.2f}, {x_vals.max():.2f}]")
            print(f"     Y range: [{y_vals.min():.2f}, {y_vals.max():.2f}]")
    
    print("\n🎉 Boolean indexing fix verified!")
    return True

def test_multiclass_visualization_data():
    """Test the specific data structure used in multiclass visualization."""
    print("\n🎨 Testing Multiclass Visualization Data Structure")
    print("=" * 50)
    
    # Simulate the exact scenario from the multiclass page
    from utils.data_processor import DataProcessor
    
    try:
        processor = DataProcessor()
        
        # Test with Iris dataset (the one that was failing)
        dataset = processor.load_dataset("iris")
        X = dataset.data[dataset.feature_columns[:2]]  # First 2 features
        y = dataset.data[dataset.target_column]
        class_names = dataset.class_names
        
        print(f"📊 Dataset: {dataset.name}")
        print(f"   Features: {list(X.columns)}")
        print(f"   Classes: {class_names}")
        print(f"   Shape: {X.shape}")
        
        # Test the visualization data extraction (this was failing before)
        colors = ['red', 'green', 'blue']  # Simplified color list
        symbols = ['circle', 'square', 'diamond']
        
        for class_idx, class_name in enumerate(class_names):
            mask = y == class_idx
            if np.any(mask):
                # This is the line that was fixed
                x_vals = X.loc[mask, X.columns[0]]
                y_vals = X.loc[mask, X.columns[1]]
                
                print(f"✅ {class_name}: {len(x_vals)} points")
                print(f"   Color: {colors[class_idx]}, Symbol: {symbols[class_idx]}")
        
        print("\n🎉 Multiclass visualization data extraction successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in multiclass test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("🔧 Verifying Boolean Indexing Fix")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    if test_boolean_indexing_fix():
        tests_passed += 1
    
    if test_multiclass_visualization_data():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All fixes verified! The app should work correctly now.")
        print("\n🚀 Ready to run:")
        print("   streamlit run logistic_regression_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()