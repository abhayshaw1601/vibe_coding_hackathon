#!/usr/bin/env python3
"""
Startup script for the Logistic Regression Playground.
This script checks dependencies and starts the Streamlit app.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy', 
        'scikit-learn',
        'plotly',
        'google-generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_app():
    """Start the Streamlit application."""
    print("🚀 Starting Logistic Regression Playground...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    print("💡 Tips:")
    print("- Configure your Gemini API key in the sidebar for AI features")
    print("- Try different pages to explore various concepts")
    print("- Upload your own CSV files in the Data Playground")
    print("- Press Ctrl+C to stop the app")
    print("="*50 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "logistic_regression_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using the Logistic Regression Playground!")
    except Exception as e:
        print(f"❌ Error starting the app: {e}")
        print("Try running manually: streamlit run logistic_regression_app.py")

def main():
    """Main function."""
    print("🎯 Logistic Regression Playground")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    print()
    run_app()

if __name__ == "__main__":
    main()