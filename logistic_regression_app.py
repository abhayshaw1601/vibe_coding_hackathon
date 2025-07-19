import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# Import utility classes
from utils.ai_client import get_ai_client, get_ai_explanation, get_ai_code_example
from utils.visualization_engine import VisualizationEngine
from utils.data_processor import DataProcessor
from utils.data_models import SyntheticDataConfig, DataType, Dataset

# Try to import Gemini API (optional for development)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Logistic Regression Playground",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced styling ---
# --- Custom CSS for enhanced styling ---
# --- Custom CSS for enhanced styling ---
# --- Custom CSS for enhanced styling ---
# --- Custom CSS for enhanced styling ---
# --- Custom CSS for enhanced styling ---
st.markdown("""
<style>
    /* Set the background for the main app and the sidebar */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e6e6e6;
    }

    /* --- THE DEFINITIVE FIX --- */
    /* 1. Force ALL text on the page to be dark by default */
    * {
        color: #2c3e50 !important;
    }

    /* 2. Create exceptions ONLY for elements that need white text */
    .stButton>button *, .ai-response * {
        color: white !important;
    }
    /* -------------------------- */

    /* Font sizes */
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.75rem; }
    p, div, li, label {
       font-size: 1.1rem;
    }

    /* Custom button styling */
    .stButton>button {
        font-size: 1rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 25px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-card h2[style] {
        color: #667eea !important; /* Keep the special color for the probability value */
    }

    /* AI Response styling */
    .ai-response {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    /* Concept box styling */
    .concept-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)
# --- Sidebar Navigation ---
st.sidebar.title("🎯 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio("Choose Your Learning Path:", [
    " 1. Core Concepts: Sigmoid & Probability",
    " 2. Binary Classification: Decision Boundaries", 
    " 3. Learning Process: Gradient Descent",
    " 4. Multiclass Classification",
    " 5. Model Diagnostics & Evaluation",
    " 6. Your Data Playground",
    " 7. AI Assistant & Help"
])

# --- AI Configuration Section ---
if GEMINI_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.subheader(" AI Assistant")
    
    # Check if API key is configured
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    api_key_input = st.sidebar.text_input(
        "Gemini API Key (Optional)", 
        value=st.session_state.gemini_api_key,
        type="password",
        help="Enter your Google Gemini API key to enable AI explanations"
    )
    
    if api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key_input
        if api_key_input:
            try:
                genai.configure(api_key=api_key_input)
                st.sidebar.success("✅ AI Assistant Ready!")
            except Exception as e:
                st.sidebar.error("❌ Invalid API Key")

# --- Helper Functions ---
@st.cache_data
def generate_binary_data(n_samples=200, noise=0.1, random_state=42):
    """Generate synthetic binary classification data."""
    np.random.seed(random_state)
    
    # Create two clusters
    cluster1_x = np.random.normal(2, 1, n_samples//2)
    cluster1_y = np.random.normal(2, 1, n_samples//2)
    
    cluster2_x = np.random.normal(6, 1, n_samples//2)
    cluster2_y = np.random.normal(6, 1, n_samples//2)
    
    X = np.column_stack([
        np.concatenate([cluster1_x, cluster2_x]),
        np.concatenate([cluster1_y, cluster2_y])
    ])
    
    y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add some noise to make it more realistic
    noise_indices = np.random.choice(len(y), int(len(y) * noise), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    return pd.DataFrame(X, columns=['Feature_1', 'Feature_2']), y

@st.cache_data
def load_dataset(dataset_name):
    """Load built-in datasets."""
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names
    elif dataset_name == "Digits":
        data = load_digits()
        df = pd.DataFrame(data.data)
        df['target'] = data.target
        return df, [str(i) for i in range(10)]

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow

def get_ai_explanation(prompt, context=""):
    """Get AI explanation using Gemini API."""
    if not GEMINI_AVAILABLE or not st.session_state.get('gemini_api_key'):
        return "AI explanations are not available. Please configure your Gemini API key in the sidebar."
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        full_prompt = f"""
        You are an expert machine learning educator explaining logistic regression concepts.
        Context: {context}
        
        Question: {prompt}
        
        Please provide a clear, educational explanation suitable for students learning machine learning.
        Use simple language and include practical insights where relevant.
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI service temporarily unavailable: {str(e)}"

# --- Initialize Session State ---
if 'user_data_points' not in st.session_state:
    st.session_state.user_data_points = {'x': [], 'y': [], 'class': []}

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize utility classes
if 'viz_engine' not in st.session_state:
    st.session_state.viz_engine = VisualizationEngine()

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

# --- Page Routing ---
if page == " 1. Core Concepts: Sigmoid & Probability":
    st.title(" Core Concepts: The Sigmoid Function & Probability")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎯 Why Logistic Regression?</h3>
    Linear regression predicts continuous values, but what if we want to predict categories? 
    That's where <strong>Logistic Regression</strong> comes in! It's perfect for classification problems.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 The Sigmoid Function")
        st.markdown("""
        The **sigmoid function** is the heart of logistic regression. It transforms any real number 
        into a value between 0 and 1, making it perfect for probabilities!
        
        **Formula:** σ(z) = 1 / (1 + e^(-z))
        """)
        
        # Interactive sigmoid parameters with enhanced controls
        col1a, col1b = st.columns(2)
        with col1a:
            slope = st.slider("Adjust Slope (steepness)", 0.1, 5.0, 1.0, 0.1, 
                            help="Higher values make the curve steeper")
            shift = st.slider("Adjust Horizontal Shift", -5.0, 5.0, 0.0, 0.1,
                            help="Moves the curve left or right")
        
        with col1b:
            show_comparison = st.checkbox("Compare with Linear Function", False)
            animate_changes = st.checkbox("Smooth Animations", True)
        
        # Sample point for highlighting
        sample_z = st.slider("Sample Input Value", -5.0, 5.0, 0.0, 0.1, key="sigmoid_sample_z")
        
        # Use visualization engine for enhanced sigmoid plot
        viz_params = {
            'slope': slope,
            'shift': shift,
            'highlight_point': sample_z
        }
        
        fig = st.session_state.viz_engine.create_sigmoid_plot(viz_params)
        
        # Add linear comparison if requested
        if show_comparison:
            z_linear = np.linspace(-10, 10, 1000)
            linear_values = np.clip(slope * (z_linear - shift) * 0.1 + 0.5, 0, 1)
            
            fig.add_trace(go.Scatter(
                x=z_linear, y=linear_values,
                mode='lines',
                name='Linear Approximation',
                line=dict(color='orange', width=2, dash='dot'),
                opacity=0.7
            ))
        
        # Enhanced layout with better styling
        fig.update_layout(
            title="Interactive Sigmoid Function",
            xaxis_title="Input (z)",
            yaxis_title="Probability",
            height=450,
            showlegend=True,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mathematical insights
        st.markdown("""
        <div class="concept-box">
        <h4>🔍 Mathematical Insights:</h4>
        <ul>
        <li><strong>S-Shape:</strong> The sigmoid creates a smooth S-shaped curve</li>
        <li><strong>Asymptotes:</strong> Approaches 0 and 1 but never reaches them</li>
        <li><strong>Derivative:</strong> σ'(z) = σ(z) × (1 - σ(z)) - easy to compute!</li>
        <li><strong>Symmetry:</strong> σ(-z) = 1 - σ(z)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎲 Probability Interpretation")
        
        # Sample point for probability calculation
        sample_z_prob = st.slider("Sample Input Value", -5.0, 5.0, 0.0, 0.1, key="prob_sample_z")
        sample_prob = sigmoid(slope * (sample_z_prob - shift))
        
        st.markdown(f"""
        <div class="metric-card">
        <h4>For input z = {sample_z:.1f}:</h4>
        <h2 style="color: #667eea;">P = {sample_prob:.3f}</h2>
        <p><strong>Interpretation:</strong></p>
        <p>• {sample_prob*100:.1f}% chance of Class 1</p>
        <p>• {(1-sample_prob)*100:.1f}% chance of Class 0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Explanation Button
        if st.button("🤖 Get AI Explanation"):
            with st.spinner("Getting AI explanation..."):
                explanation = get_ai_explanation(
                    "Explain how the sigmoid function works in logistic regression and why it's useful for classification",
                    f"Current sigmoid parameters: slope={slope}, shift={shift}, sample_z={sample_z}, probability={sample_prob:.3f}"
                )
                st.markdown(f"""
                <div class="ai-response">
                <h4>🤖 AI Explanation:</h4>
                {explanation}
                </div>
                """, unsafe_allow_html=True)

elif page == " 2. Binary Classification: Decision Boundaries":
    st.title("🎯 Binary Classification: Decision Boundaries")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎯 Interactive Classification Playground</h3>
    Explore how logistic regression creates decision boundaries to separate classes. 
    Generate different datasets and see how the model adapts!
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎮 Dataset Controls")
        
        # Dataset generation options
        dataset_type = st.selectbox("Choose Dataset Type:", [
            "Two Clusters", "Linearly Separable", "Overlapping Classes", 
            "XOR Pattern", "Custom Points"
        ])
        
        n_samples = st.slider("Number of Samples", 50, 500, 200, 50)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1, 0.05)
        
        if st.button("🎲 Generate New Dataset"):
            # Generate different types of datasets
            if dataset_type == "Two Clusters":
                X_base, y_base = generate_binary_data(n_samples=n_samples, noise=noise_level)
            elif dataset_type == "Linearly Separable":
                from sklearn.datasets import make_classification
                X_array, y_base = make_classification(
                    n_samples=n_samples, n_features=2, n_redundant=0, 
                    n_informative=2, n_clusters_per_class=1, class_sep=2.0,
                    random_state=42
                )
                X_base = pd.DataFrame(X_array, columns=['Feature_1', 'Feature_2'])
            elif dataset_type == "Overlapping Classes":
                X_array, y_base = make_classification(
                    n_samples=n_samples, n_features=2, n_redundant=0,
                    n_informative=2, n_clusters_per_class=1, class_sep=0.5,
                    random_state=42
                )
                X_base = pd.DataFrame(X_array, columns=['Feature_1', 'Feature_2'])
            elif dataset_type == "XOR Pattern":
                # Create XOR-like pattern
                np.random.seed(42)
                n_per_quad = n_samples // 4
                X_list = []
                y_list = []
                
                # Quadrant 1: (positive, positive) -> Class 0
                X_list.append(np.random.normal([2, 2], 0.5, (n_per_quad, 2)))
                y_list.extend([0] * n_per_quad)
                
                # Quadrant 2: (negative, positive) -> Class 1
                X_list.append(np.random.normal([-2, 2], 0.5, (n_per_quad, 2)))
                y_list.extend([1] * n_per_quad)
                
                # Quadrant 3: (negative, negative) -> Class 0
                X_list.append(np.random.normal([-2, -2], 0.5, (n_per_quad, 2)))
                y_list.extend([0] * n_per_quad)
                
                # Quadrant 4: (positive, negative) -> Class 1
                X_list.append(np.random.normal([2, -2], 0.5, (n_per_quad, 2)))
                y_list.extend([1] * n_per_quad)
                
                X_array = np.vstack(X_list)
                X_base = pd.DataFrame(X_array, columns=['Feature_1', 'Feature_2'])
                y_base = np.array(y_list)
            
            # Store in session state
            st.session_state.current_dataset = {'X': X_base, 'y': y_base}
            st.session_state.user_data_points = {'x': [], 'y': [], 'class': []}
            st.rerun()
        
        # Manual point addition for custom dataset
        if dataset_type == "Custom Points":
            st.markdown("---")
            st.subheader("➕ Add Points Manually")
            
            add_class = st.radio("Next point class:", ["Class 0 (Blue)", "Class 1 (Red)"])
            class_value = 0 if "Class 0" in add_class else 1
            
            col2a, col2b = st.columns(2)
            with col2a:
                x_coord = st.number_input("X coordinate", -5.0, 5.0, 0.0, 0.1)
            with col2b:
                y_coord = st.number_input("Y coordinate", -5.0, 5.0, 0.0, 0.1)
            
            if st.button("➕ Add Point"):
                st.session_state.user_data_points['x'].append(x_coord)
                st.session_state.user_data_points['y'].append(y_coord)
                st.session_state.user_data_points['class'].append(class_value)
                st.rerun()
            
            if st.button("🗑️ Clear All Points"):
                st.session_state.user_data_points = {'x': [], 'y': [], 'class': []}
                st.rerun()
        
        st.markdown("---")
        st.subheader("📊 Dataset Info")
        
        # Get current dataset
        if hasattr(st.session_state, 'current_dataset'):
            X_base = st.session_state.current_dataset['X']
            y_base = st.session_state.current_dataset['y']
        else:
            X_base, y_base = generate_binary_data(n_samples=200, noise=0.1)
            st.session_state.current_dataset = {'X': X_base, 'y': y_base}
        
        # Combine with user data
        if st.session_state.user_data_points['x']:
            user_df = pd.DataFrame({
                'Feature_1': st.session_state.user_data_points['x'],
                'Feature_2': st.session_state.user_data_points['y']
            })
            user_y = np.array(st.session_state.user_data_points['class'])
            
            X_combined = pd.concat([X_base, user_df], ignore_index=True)
            y_combined = np.concatenate([y_base, user_y])
        else:
            X_combined = X_base
            y_combined = y_base
        
        # Display dataset statistics
        n_class_0 = sum(1 for c in y_combined if c == 0)
        n_class_1 = sum(1 for c in y_combined if c == 1)
        st.write(f"🔵 Class 0: {n_class_0} points")
        st.write(f"🔴 Class 1: {n_class_1} points")
        st.write(f"📊 Total: {len(y_combined)} points")
        
        # Class balance indicator
        balance_ratio = min(n_class_0, n_class_1) / max(n_class_0, n_class_1) if max(n_class_0, n_class_1) > 0 else 0
        if balance_ratio > 0.8:
            st.success("✅ Well balanced classes")
        elif balance_ratio > 0.5:
            st.warning("⚠️ Moderately imbalanced")
        else:
            st.error("❌ Highly imbalanced classes")
    
    with col1:
        # Train logistic regression model
        if len(X_combined) > 0 and len(np.unique(y_combined)) > 1:
            # Model training with different regularization options
            st.subheader("🤖 Model Configuration")
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                regularization = st.selectbox("Regularization:", ["None", "L1 (Lasso)", "L2 (Ridge)"])
            with col1b:
                C_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01,
                                  help="Lower values = stronger regularization")
            with col1c:
                show_probabilities = st.checkbox("Show Probability Heatmap", True)
            
            # Configure model based on user selection
            if regularization == "L1 (Lasso)":
                model = LogisticRegression(penalty='l1', C=C_value, solver='liblinear', random_state=42)
            elif regularization == "L2 (Ridge)":
                model = LogisticRegression(penalty='l2', C=C_value, random_state=42)
            else:
                model = LogisticRegression(penalty=None, random_state=42)
            
            model.fit(X_combined, y_combined)
            
            # Use enhanced visualization engine
            fig = st.session_state.viz_engine.create_decision_boundary_plot(model, X_combined, y_combined)
            
            # Enhance the plot with additional features
            if not show_probabilities:
                # Remove probability heatmap if not requested
                fig.data = [trace for trace in fig.data if 'Probability' not in trace.name]
            
            # Add model coefficients as annotation
            coef_text = f"Model: {model.coef_[0][0]:.2f}×X₁ + {model.coef_[0][1]:.2f}×X₂ + {model.intercept_[0]:.2f}"
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=coef_text,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced model performance metrics
            st.subheader("📈 Model Performance")
            
            # Calculate comprehensive metrics
            y_pred = model.predict(X_combined)
            y_prob = model.predict_proba(X_combined)[:, 1]
            
            accuracy = accuracy_score(y_combined, y_pred)
            
            # Display metrics in columns
            col1a, col1b, col1c, col1d = st.columns(4)
            
            with col1a:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Accuracy</h4>
                <h2 style="color: #667eea;">{accuracy:.3f}</h2>
                <p>{accuracy*100:.1f}% correct</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col1b:
                # Calculate precision for class 1
                from sklearn.metrics import precision_score
                precision = precision_score(y_combined, y_pred, zero_division=0)
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Precision</h4>
                <h2 style="color: #e74c3c;">{precision:.3f}</h2>
                <p>True positives</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col1c:
                # Calculate recall for class 1
                from sklearn.metrics import recall_score
                recall = recall_score(y_combined, y_pred, zero_division=0)
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Recall</h4>
                <h2 style="color: #f39c12;">{recall:.3f}</h2>
                <p>Coverage</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col1d:
                # Calculate F1 score
                from sklearn.metrics import f1_score
                f1 = f1_score(y_combined, y_pred, zero_division=0)
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 F1-Score</h4>
                <h2 style="color: #2ecc71;">{f1:.3f}</h2>
                <p>Harmonic mean</p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI-powered analysis
            if st.button("🤖 Get AI Analysis"):
                with st.spinner("Analyzing model performance..."):
                    model_results = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'dataset_type': dataset_type,
                        'n_samples': len(y_combined),
                        'class_balance': f"{n_class_0}/{n_class_1}",
                        'regularization': regularization,
                        'C_value': C_value
                    }
                    
                    analysis = get_ai_explanation(
                        f"Analyze this binary classification model performance: {model_results}",
                        {'page': page, 'user_data': model_results}
                    )
                    
                    st.markdown(f"""
                    <div class="ai-response">
                    <h4>🤖 AI Performance Analysis:</h4>
                    {analysis}
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ Need at least 2 data points from different classes to train a model!")
            st.info("💡 Generate a dataset or add custom points to get started.")

elif page == " 3. Learning Process: Gradient Descent":
    st.title("📉 Learning Process: Gradient Descent")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎯 How Logistic Regression Learns</h3>
    Watch how gradient descent finds the optimal parameters by iteratively minimizing the cost function.
    Unlike linear regression, logistic regression uses the <strong>log-likelihood</strong> cost function.
    </div>
    """, unsafe_allow_html=True)
    
    # Educational content about gradient descent
    with st.expander("📚 Understanding Gradient Descent", expanded=False):
        st.markdown("""
        **Gradient Descent** is like finding the bottom of a valley while blindfolded:
        
        1. **Start anywhere** on the cost function surface
        2. **Feel the slope** (calculate the gradient) 
        3. **Take a step downhill** (opposite to gradient direction)
        4. **Repeat** until you reach the bottom (minimum cost)
        
        **Key Concepts:**
        - **Cost Function**: For logistic regression, we use log-likelihood (cross-entropy)
        - **Learning Rate**: How big steps to take (too big = overshoot, too small = slow)
        - **Convergence**: When the algorithm stops improving significantly
        """)
    
    # Controls for gradient descent
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🎮 Animation Controls")
        
        # Dataset selection
        gd_dataset = st.selectbox("Choose Dataset:", [
            "Simple 2D", "Challenging Separation", "Noisy Data"
        ])
        
        # Gradient descent parameters
        learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.1, 0.001, 
                                format="%.3f", help="Step size for each iteration")
        max_iterations = st.slider("Max Iterations", 10, 500, 100, 10)
        
        # Animation settings
        animation_speed = st.slider("Animation Speed", 0.01, 0.5, 0.1, 0.01,
                                  help="Seconds between frames")
        
        show_3d_surface = st.checkbox("Show 3D Cost Surface", False)
        show_convergence_plot = st.checkbox("Show Convergence Plot", True)
        
        # Start animation button
        start_animation = st.button("🚀 Start Gradient Descent Animation", type="primary")
        
        if st.button("🔄 Reset"):
            if 'gd_history' in st.session_state:
                del st.session_state.gd_history
            st.rerun()
    
    with col1:
        # Generate dataset based on selection
        if gd_dataset == "Simple 2D":
            X_gd, y_gd = generate_binary_data(n_samples=100, noise=0.05)
        elif gd_dataset == "Challenging Separation":
            from sklearn.datasets import make_classification
            X_array, y_gd = make_classification(
                n_samples=150, n_features=2, n_redundant=0,
                n_informative=2, n_clusters_per_class=1, class_sep=0.8,
                random_state=42
            )
            X_gd = pd.DataFrame(X_array, columns=['Feature_1', 'Feature_2'])
        else:  # Noisy Data
            X_gd, y_gd = generate_binary_data(n_samples=120, noise=0.25)
        
        # Standardize features for better gradient descent performance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_gd)
        X_gd_scaled = pd.DataFrame(X_scaled, columns=['Feature_1', 'Feature_2'])
        
        if start_animation:
            # Initialize gradient descent
            st.subheader("🎬 Gradient Descent Animation")
            
            # Placeholders for dynamic content
            plot_placeholder = st.empty()
            metrics_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Initialize parameters randomly
            np.random.seed(42)
            w1, w2, b = np.random.normal(0, 0.1, 3)
            
            # Store history for visualization
            history = {
                'weights': [],
                'bias': [],
                'cost': [],
                'accuracy': [],
                'iteration': []
            }
            
            # Gradient descent implementation
            n_samples = len(X_gd_scaled)
            
            for iteration in range(max_iterations):
                # Forward pass
                z = w1 * X_gd_scaled.iloc[:, 0] + w2 * X_gd_scaled.iloc[:, 1] + b
                predictions = sigmoid(z)
                
                # Calculate cost (log-likelihood)
                epsilon = 1e-15  # Prevent log(0)
                predictions = np.clip(predictions, epsilon, 1 - epsilon)
                cost = -np.mean(y_gd * np.log(predictions) + (1 - y_gd) * np.log(1 - predictions))
                
                # Calculate accuracy
                y_pred_binary = (predictions > 0.5).astype(int)
                accuracy = np.mean(y_pred_binary == y_gd)
                
                # Store history
                history['weights'].append([w1, w2])
                history['bias'].append(b)
                history['cost'].append(cost)
                history['accuracy'].append(accuracy)
                history['iteration'].append(iteration)
                
                # Calculate gradients
                dw1 = np.mean((predictions - y_gd) * X_gd_scaled.iloc[:, 0])
                dw2 = np.mean((predictions - y_gd) * X_gd_scaled.iloc[:, 1])
                db = np.mean(predictions - y_gd)
                
                # Update parameters
                w1 -= learning_rate * dw1
                w2 -= learning_rate * dw2
                b -= learning_rate * db
                
                # Update visualization every few iterations or at the end
                if iteration % max(1, max_iterations // 20) == 0 or iteration == max_iterations - 1:
                    # Create current decision boundary plot
                    fig = go.Figure()
                    
                    # Add data points
                    colors = ['#3498db' if c == 0 else '#e74c3c' for c in y_gd]
                    fig.add_trace(go.Scatter(
                        x=X_gd_scaled.iloc[:, 0],
                        y=X_gd_scaled.iloc[:, 1],
                        mode='markers',
                        marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
                        name="Data Points",
                        showlegend=False
                    ))
                    
                    # Add current decision boundary
                    x_range = np.linspace(X_gd_scaled.iloc[:, 0].min()-1, X_gd_scaled.iloc[:, 0].max()+1, 100)
                    y_range = -(w1 * x_range + b) / w2  # Decision boundary: w1*x1 + w2*x2 + b = 0
                    
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_range,
                        mode='lines',
                        line=dict(color='black', width=3),
                        name=f"Decision Boundary (Iter {iteration})",
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"Gradient Descent Progress - Iteration {iteration + 1}/{max_iterations}",
                        xaxis_title="Feature 1 (Standardized)",
                        yaxis_title="Feature 2 (Standardized)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Update metrics
                    metrics_placeholder.markdown(f"""
                    <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                        <div class="metric-card" style="flex: 1;">
                            <h4>📊 Current Metrics</h4>
                            <p><strong>Iteration:</strong> {iteration + 1}</p>
                            <p><strong>Cost:</strong> {cost:.4f}</p>
                            <p><strong>Accuracy:</strong> {accuracy:.3f}</p>
                        </div>
                        <div class="metric-card" style="flex: 1;">
                            <h4>⚙️ Parameters</h4>
                            <p><strong>w₁:</strong> {w1:.4f}</p>
                            <p><strong>w₂:</strong> {w2:.4f}</p>
                            <p><strong>b:</strong> {b:.4f}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update progress bar
                    progress_bar.progress((iteration + 1) / max_iterations)
                    
                    # Add delay for animation
                    time.sleep(animation_speed)
                
                # Early stopping if converged
                if len(history['cost']) > 10:
                    recent_costs = history['cost'][-10:]
                    if max(recent_costs) - min(recent_costs) < 1e-6:
                        st.success(f"✅ Converged early at iteration {iteration + 1}!")
                        break
            
            # Store history in session state
            st.session_state.gd_history = history
            
            st.success("🎉 Gradient Descent Animation Complete!")
        
        # Show additional visualizations if history exists
        if 'gd_history' in st.session_state and st.session_state.gd_history:
            history = st.session_state.gd_history
            
            if show_convergence_plot:
                st.subheader("📈 Convergence Analysis")
                
                # Create convergence plots
                fig_conv = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Cost Function Over Time', 'Accuracy Over Time')
                )
                
                # Cost plot
                fig_conv.add_trace(
                    go.Scatter(
                        x=history['iteration'],
                        y=history['cost'],
                        mode='lines+markers',
                        name='Cost',
                        line=dict(color='#e74c3c', width=2)
                    ),
                    row=1, col=1
                )
                
                # Accuracy plot
                fig_conv.add_trace(
                    go.Scatter(
                        x=history['iteration'],
                        y=history['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='#2ecc71', width=2)
                    ),
                    row=1, col=2
                )
                
                fig_conv.update_layout(
                    height=400,
                    showlegend=False,
                    template="plotly_white"
                )
                
                fig_conv.update_xaxes(title_text="Iteration")
                fig_conv.update_yaxes(title_text="Cost", row=1, col=1)
                fig_conv.update_yaxes(title_text="Accuracy", row=1, col=2)
                
                st.plotly_chart(fig_conv, use_container_width=True)
            
            if show_3d_surface:
                st.subheader("🏔️ 3D Cost Surface")
                st.info("💡 This shows how the cost function changes with different parameter values. The red line shows the path taken by gradient descent.")
                
                # Create 3D surface plot (simplified version)
                # Note: Full 3D surface would be computationally expensive, so we'll show a conceptual version
                weights_history = np.array(history['weights'])
                
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=weights_history[:, 0],
                    y=weights_history[:, 1],
                    z=history['cost'],
                    mode='markers+lines',
                    marker=dict(
                        size=5,
                        color=history['cost'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Cost")
                    ),
                    line=dict(color='red', width=4),
                    name='Optimization Path'
                )])
                
                fig_3d.update_layout(
                    title="Gradient Descent Path in Parameter Space",
                    scene=dict(
                        xaxis_title="Weight 1 (w₁)",
                        yaxis_title="Weight 2 (w₂)",
                        zaxis_title="Cost"
                    ),
                    height=600
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # AI explanation of the results
            if st.button("🤖 Explain Gradient Descent Results"):
                with st.spinner("Analyzing gradient descent performance..."):
                    final_cost = history['cost'][-1]
                    final_accuracy = history['accuracy'][-1]
                    iterations_used = len(history['cost'])
                    
                    gd_results = {
                        'final_cost': final_cost,
                        'final_accuracy': final_accuracy,
                        'iterations_used': iterations_used,
                        'learning_rate': learning_rate,
                        'converged': iterations_used < max_iterations
                    }
                    
                    explanation = get_ai_explanation(
                        f"Explain these gradient descent results for logistic regression: {gd_results}",
                        {'page': page, 'user_data': gd_results}
                    )
                    
                    st.markdown(f"""
                    <div class="ai-response">
                    <h4>🤖 Gradient Descent Analysis:</h4>
                    {explanation}
                    </div>
                    """, unsafe_allow_html=True)
        
        elif not start_animation:
            # Show static educational content
            st.subheader("🎯 Ready to Learn!")
            st.markdown("""
            <div class="concept-box">
            <h4>What you'll see in the animation:</h4>
            <ul>
            <li><strong>Decision Boundary Evolution:</strong> Watch how the boundary moves to better separate classes</li>
            <li><strong>Parameter Updates:</strong> See how weights and bias change each iteration</li>
            <li><strong>Cost Reduction:</strong> Observe the cost function decreasing over time</li>
            <li><strong>Convergence:</strong> Notice when the algorithm stops improving</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Show the current dataset
            fig_preview = go.Figure()
            colors = ['#3498db' if c == 0 else '#e74c3c' for c in y_gd]
            fig_preview.add_trace(go.Scatter(
                x=X_gd_scaled.iloc[:, 0],
                y=X_gd_scaled.iloc[:, 1],
                mode='markers',
                marker=dict(color=colors, size=8, line=dict(width=1, color='white')),
                name="Data Points"
            ))
            
            fig_preview.update_layout(
                title=f"Dataset Preview: {gd_dataset}",
                xaxis_title="Feature 1 (Standardized)",
                yaxis_title="Feature 2 (Standardized)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_preview, use_container_width=True)

elif page == " 4. Multiclass Classification":
    st.title(" Multiclass Classification")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎯 Beyond Binary: Multiple Classes</h3>
    Logistic regression can handle multiple classes using two main strategies:
    <strong>One-vs-Rest (OvR)</strong> and <strong>Multinomial</strong> approaches.
    </div>
    """, unsafe_allow_html=True)
    
    # Educational content
    with st.expander("📚 Understanding Multiclass Strategies", expanded=False):
        st.markdown("""
        **One-vs-Rest (OvR):**
        - Train one binary classifier for each class vs. all others
        - For 3 classes: "Class A vs Others", "Class B vs Others", "Class C vs Others"
        - Prediction: Choose class with highest probability
        
        **Multinomial (Softmax):**
        - Single model that directly handles all classes
        - Uses softmax function to ensure probabilities sum to 1
        - More efficient and often more accurate
        
        **When to use which:**
        - OvR: Good for imbalanced datasets, interpretable
        - Multinomial: Better for balanced datasets, more efficient
        """)
    
    # Dataset and model controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎮 Configuration")
        
        # Dataset selection
        dataset_choice = st.selectbox("Choose Dataset:", [
            "Iris (3 classes)", "Wine (3 classes)", "Synthetic 3-Class", 
            "Synthetic 4-Class", "Synthetic 5-Class"
        ])
        
        # Model strategy
        strategy = st.selectbox("Classification Strategy:", [
            "Multinomial (Softmax)", "One-vs-Rest (OvR)"
        ])
        
        # Feature selection for built-in datasets
        if "Iris" in dataset_choice or "Wine" in dataset_choice:
            st.subheader("🔧 Feature Selection")
            if "Iris" in dataset_choice:
                dataset = st.session_state.data_processor.load_dataset("iris")
                available_features = dataset.feature_columns
            else:
                dataset = st.session_state.data_processor.load_dataset("wine")
                available_features = dataset.feature_columns
            
            selected_features = st.multiselect(
                "Select 2 features for visualization:",
                available_features,
                default=available_features[:2]
            )
            
            if len(selected_features) != 2:
                st.warning("⚠️ Please select exactly 2 features for 2D visualization")
                selected_features = available_features[:2]
        
        # Synthetic data parameters
        elif "Synthetic" in dataset_choice:
            st.subheader("🎲 Synthetic Data")
            # Extract number from "Synthetic X-Class"
            if "3-Class" in dataset_choice:
                n_classes = 3
            elif "4-Class" in dataset_choice:
                n_classes = 4
            elif "5-Class" in dataset_choice:
                n_classes = 5
            else:
                n_classes = 3  # Default
            n_samples = st.slider("Samples per class", 50, 200, 100, 25)
            class_separation = st.slider("Class separation", 0.5, 3.0, 1.5, 0.1)
            noise_level = st.slider("Noise level", 0.0, 0.5, 0.1, 0.05)
        
        # Model parameters
        st.subheader("⚙️ Model Parameters")
        regularization_strength = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.slider("Max iterations", 100, 2000, 1000, 100)
        
        # Visualization options
        st.subheader("📊 Visualization")
        show_decision_regions = st.checkbox("Show decision regions", True)
        show_probabilities = st.checkbox("Show probability contours", False)
        show_confusion_matrix = st.checkbox("Show confusion matrix", True)
    
    with col1:
        # Load or generate dataset
        if "Iris" in dataset_choice:
            dataset = st.session_state.data_processor.load_dataset("iris")
            X = dataset.data[selected_features]
            y = dataset.data[dataset.target_column]
            class_names = dataset.class_names
            
        elif "Wine" in dataset_choice:
            dataset = st.session_state.data_processor.load_dataset("wine")
            X = dataset.data[selected_features]
            y = dataset.data[dataset.target_column]
            class_names = dataset.class_names
            
        else:  # Synthetic data
            # Extract number from "Synthetic X-Class"
            if "3-Class" in dataset_choice:
                n_classes = 3
            elif "4-Class" in dataset_choice:
                n_classes = 4
            elif "5-Class" in dataset_choice:
                n_classes = 5
            else:
                n_classes = 3  # Default
            config = SyntheticDataConfig(
                n_samples=n_samples * n_classes,
                n_features=2,
                n_classes=n_classes,
                noise_level=noise_level,
                class_separation=class_separation,
                random_state=42
            )
            dataset = st.session_state.data_processor.generate_synthetic_data(config)
            X = dataset.data[dataset.feature_columns]
            y = dataset.data[dataset.target_column]
            class_names = dataset.class_names
        
        # Train multiclass model
        if strategy == "Multinomial (Softmax)":
            model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                C=regularization_strength,
                max_iter=max_iter,
                random_state=42
            )
        else:  # One-vs-Rest
            model = LogisticRegression(
                multi_class='ovr',
                C=regularization_strength,
                max_iter=max_iter,
                random_state=42
            )
        
        # Fit the model
        model.fit(X, y)
        
        # Create visualization
        st.subheader(f"🎨 {strategy} Classification")
        
        # Create decision boundary plot
        h = 0.02  # Step size in the mesh
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Get predictions for the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Get probabilities if requested
        if show_probabilities:
            Z_proba = model.predict_proba(mesh_points)
            Z_proba_max = np.max(Z_proba, axis=1).reshape(xx.shape)
        
        # Create the plot
        fig = go.Figure()
        
        # Add decision regions
        if show_decision_regions:
            # Create a color map for classes
            colors = px.colors.qualitative.Set1[:len(class_names)]
            
            for class_idx, class_name in enumerate(class_names):
                mask = Z == class_idx
                if np.any(mask):
                    fig.add_trace(go.Contour(
                        x=np.arange(x_min, x_max, h),
                        y=np.arange(y_min, y_max, h),
                        z=(Z == class_idx).astype(int),
                        contours=dict(start=0.5, end=0.5, size=0),
                        line=dict(width=0),
                        fillcolor=colors[class_idx],
                        opacity=0.3,
                        showscale=False,
                        hoverinfo='skip',
                        name=f"{class_name} Region"
                    ))
        
        # Add probability contours
        if show_probabilities:
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z_proba_max,
                colorscale='Viridis',
                opacity=0.4,
                showscale=True,
                colorbar=dict(title="Max Probability"),
                name="Probability Contours"
            ))
        
        # Add data points
        colors = px.colors.qualitative.Set1[:len(class_names)]
        symbols = ['circle', 'square', 'diamond', 'cross', 'star']
        
        for class_idx, class_name in enumerate(class_names):
            mask = y == class_idx
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=X.loc[mask, X.columns[0]],
                    y=X.loc[mask, X.columns[1]],
                    mode='markers',
                    marker=dict(
                        color=colors[class_idx],
                        symbol=symbols[class_idx % len(symbols)],
                        size=8,
                        line=dict(width=2, color='white')
                    ),
                    name=class_name,
                    hovertemplate=f"<b>{class_name}</b><br>%{{xaxis.title.text}}: %{{x:.2f}}<br>%{{yaxis.title.text}}: %{{y:.2f}}<extra></extra>"
                ))
        
        fig.update_layout(
            title=f"Multiclass Classification: {strategy}",
            xaxis_title=X.columns[0],
            yaxis_title=X.columns[1],
            height=500,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.subheader("📈 Model Performance")
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y, y_pred)
        
        # Display metrics in columns
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            st.markdown(f"""
            <div class="metric-card">
            <h4>🎯 Overall Accuracy</h4>
            <h2 style="color: #667eea;">{accuracy:.3f}</h2>
            <p>{accuracy*100:.1f}% correct</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1b:
            # Calculate macro-averaged F1 score
            from sklearn.metrics import f1_score
            f1_macro = f1_score(y, y_pred, average='macro')
            st.markdown(f"""
            <div class="metric-card">
            <h4>🎯 Macro F1-Score</h4>
            <h2 style="color: #e74c3c;">{f1_macro:.3f}</h2>
            <p>Average across classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1c:
            # Number of classes
            n_classes = len(class_names)
            st.markdown(f"""
            <div class="metric-card">
            <h4>🌈 Classes</h4>
            <h2 style="color: #2ecc71;">{n_classes}</h2>
            <p>Total categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrix
        if show_confusion_matrix:
            st.subheader("🔍 Confusion Matrix")
            
            cm = confusion_matrix(y, y_pred)
            
            # Create interactive confusion matrix
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Class",
                yaxis_title="Actual Class",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Detailed classification report
        with st.expander("📊 Detailed Classification Report", expanded=False):
            report = classification_report(y, y_pred, target_names=class_names)
            st.text(report)
        
        # Model coefficients analysis (for interpretability)
        if strategy == "One-vs-Rest (OvR)":
            st.subheader("🔍 Model Coefficients (One-vs-Rest)")
            
            # Create coefficient visualization
            coef_data = []
            for i, class_name in enumerate(class_names):
                for j, feature_name in enumerate(X.columns):
                    coef_data.append({
                        'Class': f"{class_name} vs Rest",
                        'Feature': feature_name,
                        'Coefficient': model.coef_[i][j]
                    })
            
            coef_df = pd.DataFrame(coef_data)
            
            fig_coef = px.bar(
                coef_df, 
                x='Feature', 
                y='Coefficient', 
                color='Class',
                title="Feature Coefficients for Each Binary Classifier",
                barmode='group'
            )
            
            fig_coef.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_coef, use_container_width=True)
        
        else:  # Multinomial
            st.subheader("🔍 Model Coefficients (Multinomial)")
            
            # For multinomial, show coefficients as heatmap
            coef_matrix = model.coef_
            
            fig_coef = go.Figure(data=go.Heatmap(
                z=coef_matrix,
                x=X.columns,
                y=class_names,
                colorscale='RdBu',
                zmid=0,
                showscale=True,
                colorbar=dict(title="Coefficient Value"),
                hovertemplate="Class: %{y}<br>Feature: %{x}<br>Coefficient: %{z:.3f}<extra></extra>"
            ))
            
            fig_coef.update_layout(
                title="Feature Coefficients Heatmap",
                xaxis_title="Features",
                yaxis_title="Classes",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_coef, use_container_width=True)
        
        # AI Analysis
        if st.button("🤖 Get AI Analysis of Multiclass Results"):
            with st.spinner("Analyzing multiclass classification results..."):
                results = {
                    'strategy': strategy,
                    'dataset': dataset_choice,
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'n_classes': n_classes,
                    'n_samples': len(X),
                    'features_used': list(X.columns)
                }
                
                analysis = get_ai_explanation(
                    f"Analyze these multiclass logistic regression results: {results}",
                    {'page': page, 'user_data': results}
                )
                
                st.markdown(f"""
                <div class="ai-response">
                <h4>🤖 Multiclass Classification Analysis:</h4>
                {analysis}
                </div>
                """, unsafe_allow_html=True)

elif page == " 5. Model Diagnostics & Evaluation":
    st.title("📊 Model Diagnostics & Evaluation")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🔍 Deep Dive into Model Performance</h3>
    Understanding your model's performance goes beyond accuracy. Let's explore comprehensive 
    evaluation metrics and diagnostic tools for logistic regression.
    </div>
    """, unsafe_allow_html=True)
    
    # Educational content about evaluation metrics
    with st.expander("📚 Understanding Evaluation Metrics", expanded=False):
        st.markdown("""
        **Key Metrics for Classification:**
        
        - **Accuracy**: Overall correctness (TP+TN)/(TP+TN+FP+FN)
        - **Precision**: Of predicted positives, how many are actually positive? TP/(TP+FP)
        - **Recall (Sensitivity)**: Of actual positives, how many did we catch? TP/(TP+FN)
        - **Specificity**: Of actual negatives, how many did we correctly identify? TN/(TN+FP)
        - **F1-Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under ROC curve - measures separability
        - **AUC-PR**: Area under Precision-Recall curve - better for imbalanced data
        
        **When to use which:**
        - **Balanced data**: Focus on accuracy and F1-score
        - **Imbalanced data**: Focus on precision, recall, and AUC-PR
        - **Cost-sensitive**: Consider false positive vs false negative costs
        """)
    
    # Model setup and data selection
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎮 Model Setup")
        
        # Dataset selection
        eval_dataset = st.selectbox("Choose Dataset:", [
            "Breast Cancer (Binary)", "Iris (Multiclass)", "Wine (Multiclass)",
            "Imbalanced Synthetic", "Balanced Synthetic"
        ])
        
        # Model configuration
        st.subheader("⚙️ Model Configuration")
        eval_strategy = st.selectbox("Model Type:", [
            "Standard Logistic Regression", "L1 Regularized", "L2 Regularized"
        ])
        
        if "Regularized" in eval_strategy:
            C_param = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
        else:
            C_param = 1.0
        
        # Evaluation options
        st.subheader("📊 Evaluation Options")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        cross_validation = st.checkbox("Cross-Validation", True)
        cv_folds = st.slider("CV Folds", 3, 10, 5) if cross_validation else 5
        
        # Visualization options
        st.subheader("📈 Visualizations")
        show_roc = st.checkbox("ROC Curve", True)
        show_pr = st.checkbox("Precision-Recall Curve", True)
        show_confusion = st.checkbox("Confusion Matrix", True)
        show_calibration = st.checkbox("Calibration Plot", False)
        
        # Generate report button
        generate_report = st.button("📋 Generate Full Report", type="primary")
    
    with col1:
        # Load dataset based on selection
        if eval_dataset == "Breast Cancer (Binary)":
            dataset = st.session_state.data_processor.load_dataset("breast_cancer")
            X = dataset.data[dataset.feature_columns[:4]]  # Use first 4 features for simplicity
            y = dataset.data[dataset.target_column]
            class_names = dataset.class_names
            is_binary = True
            
        elif eval_dataset == "Iris (Multiclass)":
            dataset = st.session_state.data_processor.load_dataset("iris")
            X = dataset.data[dataset.feature_columns]
            y = dataset.data[dataset.target_column]
            class_names = dataset.class_names
            is_binary = False
            
        elif eval_dataset == "Wine (Multiclass)":
            dataset = st.session_state.data_processor.load_dataset("wine")
            X = dataset.data[dataset.feature_columns[:6]]  # Use first 6 features
            y = dataset.data[dataset.target_column]
            class_names = dataset.class_names
            is_binary = False
            
        elif eval_dataset == "Imbalanced Synthetic":
            # Create imbalanced binary dataset
            from sklearn.datasets import make_classification
            X_array, y = make_classification(
                n_samples=1000, n_features=4, n_redundant=0,
                n_informative=4, n_clusters_per_class=1,
                weights=[0.9, 0.1], random_state=42
            )
            X = pd.DataFrame(X_array, columns=[f'Feature_{i+1}' for i in range(4)])
            class_names = ['Majority Class', 'Minority Class']
            is_binary = True
            
        else:  # Balanced Synthetic
            from sklearn.datasets import make_classification
            X_array, y = make_classification(
                n_samples=800, n_features=4, n_redundant=0,
                n_informative=4, n_clusters_per_class=1,
                weights=[0.5, 0.5], random_state=42
            )
            X = pd.DataFrame(X_array, columns=[f'Feature_{i+1}' for i in range(4)])
            class_names = ['Class 0', 'Class 1']
            is_binary = True
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Configure and train model
        if eval_strategy == "L1 Regularized":
            model = LogisticRegression(penalty='l1', C=C_param, solver='liblinear', random_state=42)
        elif eval_strategy == "L2 Regularized":
            model = LogisticRegression(penalty='l2', C=C_param, random_state=42)
        else:
            model = LogisticRegression(random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Display basic performance
        st.subheader("🎯 Performance Overview")
        
        accuracy = accuracy_score(y_test, y_pred)
        
        if is_binary:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_proba[:, 1])
            
            # Display metrics in a grid
            col1a, col1b, col1c, col1d = st.columns(4)
            
            with col1a:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Accuracy</h4>
                <h2 style="color: #667eea;">{accuracy:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1b:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Precision</h4>
                <h2 style="color: #e74c3c;">{precision:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1c:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Recall</h4>
                <h2 style="color: #f39c12;">{recall:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1d:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 AUC-ROC</h4>
                <h2 style="color: #2ecc71;">{auc_roc:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Multiclass
            from sklearn.metrics import f1_score
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 Accuracy</h4>
                <h2 style="color: #667eea;">{accuracy:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1b:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 F1-Macro</h4>
                <h2 style="color: #e74c3c;">{f1_macro:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1c:
                st.markdown(f"""
                <div class="metric-card">
                <h4>🎯 F1-Weighted</h4>
                <h2 style="color: #f39c12;">{f1_weighted:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Visualization section
        st.subheader("📊 Diagnostic Visualizations")
        
        # Create subplot layout based on selected visualizations
        n_plots = sum([show_roc, show_pr, show_confusion, show_calibration])
        
        if n_plots > 0:
            if n_plots == 1:
                rows, cols = 1, 1
            elif n_plots == 2:
                rows, cols = 1, 2
            elif n_plots <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 2, 3
            
            subplot_titles = []
            if show_confusion: subplot_titles.append("Confusion Matrix")
            if show_roc and is_binary: subplot_titles.append("ROC Curve")
            if show_pr and is_binary: subplot_titles.append("Precision-Recall Curve")
            if show_calibration and is_binary: subplot_titles.append("Calibration Plot")
            
            fig_diag = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles[:n_plots]
            )
            
            plot_idx = 1
            
            # Confusion Matrix
            if show_confusion:
                cm = confusion_matrix(y_test, y_pred)
                
                fig_diag.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=class_names if len(class_names) <= 10 else [f"Class {i}" for i in range(len(cm))],
                        y=class_names if len(class_names) <= 10 else [f"Class {i}" for i in range(len(cm))],
                        colorscale='Blues',
                        showscale=False,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ),
                    row=(plot_idx-1)//cols + 1, col=(plot_idx-1)%cols + 1
                )
                plot_idx += 1
            
            # ROC Curve (binary only)
            if show_roc and is_binary:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                
                fig_diag.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC (AUC={auc_roc:.3f})',
                        line=dict(color='blue', width=2)
                    ),
                    row=(plot_idx-1)//cols + 1, col=(plot_idx-1)%cols + 1
                )
                
                # Add diagonal line
                fig_diag.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=(plot_idx-1)//cols + 1, col=(plot_idx-1)%cols + 1
                )
                plot_idx += 1
            
            # Precision-Recall Curve (binary only)
            if show_pr and is_binary:
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1])
                
                fig_diag.add_trace(
                    go.Scatter(
                        x=recall_curve, y=precision_curve,
                        mode='lines',
                        name='PR Curve',
                        line=dict(color='green', width=2)
                    ),
                    row=(plot_idx-1)//cols + 1, col=(plot_idx-1)%cols + 1
                )
                plot_idx += 1
            
            # Calibration Plot (binary only)
            if show_calibration and is_binary:
                from sklearn.calibration import calibration_curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_proba[:, 1], n_bins=10
                )
                
                fig_diag.add_trace(
                    go.Scatter(
                        x=mean_predicted_value, y=fraction_of_positives,
                        mode='lines+markers',
                        name='Calibration',
                        line=dict(color='purple', width=2)
                    ),
                    row=(plot_idx-1)//cols + 1, col=(plot_idx-1)%cols + 1
                )
                
                # Add perfect calibration line
                fig_diag.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=(plot_idx-1)//cols + 1, col=(plot_idx-1)%cols + 1
                )
            
            fig_diag.update_layout(
                height=400 * rows,
                showlegend=False,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_diag, use_container_width=True)
        
        # Cross-validation results
        if cross_validation:
            st.subheader("🔄 Cross-Validation Results")
            
            from sklearn.model_selection import cross_val_score
            
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                st.markdown(f"""
                <div class="metric-card">
                <h4>📊 CV Mean</h4>
                <h2 style="color: #667eea;">{cv_scores.mean():.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1b:
                st.markdown(f"""
                <div class="metric-card">
                <h4>📊 CV Std</h4>
                <h2 style="color: #e74c3c;">{cv_scores.std():.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1c:
                st.markdown(f"""
                <div class="metric-card">
                <h4>📊 CV Range</h4>
                <h2 style="color: #f39c12;">{cv_scores.max() - cv_scores.min():.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # CV scores distribution
            fig_cv = go.Figure(data=[
                go.Box(y=cv_scores, name="CV Scores", boxpoints='all')
            ])
            
            fig_cv.update_layout(
                title="Cross-Validation Score Distribution",
                yaxis_title="Accuracy",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_cv, use_container_width=True)
        
        # Generate comprehensive report
        if generate_report:
            st.subheader("📋 Comprehensive Model Report")
            
            with st.spinner("Generating comprehensive analysis..."):
                # Compile all metrics
                report_data = {
                    'dataset': eval_dataset,
                    'model_type': eval_strategy,
                    'test_accuracy': accuracy,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'n_features': X.shape[1],
                    'is_binary': is_binary
                }
                
                if is_binary:
                    report_data.update({
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc_roc': auc_roc
                    })
                
                if cross_validation:
                    report_data.update({
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    })
                
                # Get AI analysis
                analysis = get_ai_explanation(
                    f"Provide a comprehensive analysis of this logistic regression model evaluation: {report_data}",
                    {'page': page, 'user_data': report_data}
                )
                
                st.markdown(f"""
                <div class="ai-response">
                <h4>🤖 Comprehensive Model Analysis:</h4>
                {analysis}
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed classification report
                st.subheader("📊 Detailed Classification Report")
                report = classification_report(y_test, y_pred, target_names=class_names)
                st.text(report)

elif page == " 6. Your Data Playground":
    st.title("🚀 Your Data Playground")
    
    st.markdown("""
    <div class="concept-box">
    <h3>🎯 Bring Your Own Data!</h3>
    Upload your CSV files and build custom logistic regression models with advanced preprocessing 
    and AI-powered insights. Perfect for real-world data analysis!
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📁 Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with your data. Make sure it has a target column for classification."
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            df_user = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df_user.shape}")
            
            # Data preview
            st.subheader("👀 Data Preview")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df_user.head(10))
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                <h4>📊 Dataset Info</h4>
                <p><strong>Rows:</strong> {len(df_user)}</p>
                <p><strong>Columns:</strong> {len(df_user.columns)}</p>
                <p><strong>Memory:</strong> {df_user.memory_usage(deep=True).sum() / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Data quality check
                missing_pct = (df_user.isnull().sum().sum() / (len(df_user) * len(df_user.columns))) * 100
                if missing_pct == 0:
                    st.success("✅ No missing values")
                elif missing_pct < 5:
                    st.warning(f"⚠️ {missing_pct:.1f}% missing values")
                else:
                    st.error(f"❌ {missing_pct:.1f}% missing values")
            
            # Data validation and suggestions
            validator = st.session_state.data_processor.validator
            
            # Column selection
            st.subheader("🎯 Column Selection")
            col1, col2 = st.columns(2)
            
            with col1:
                # Target column selection
                suggested_targets = validator.suggest_target_column(df_user)
                
                if suggested_targets:
                    st.info(f"💡 Suggested target columns: {', '.join(suggested_targets)}")
                
                target_col = st.selectbox(
                    "Select Target Column (what you want to predict):",
                    df_user.columns.tolist(),
                    index=df_user.columns.tolist().index(suggested_targets[0]) if suggested_targets else 0
                )
            
            with col2:
                # Feature column selection
                suggested_features = validator.suggest_feature_columns(df_user, target_col)
                
                feature_cols = st.multiselect(
                    "Select Feature Columns:",
                    [col for col in df_user.columns if col != target_col],
                    default=suggested_features[:min(10, len(suggested_features))]  # Limit to 10 features
                )
            
            if not feature_cols:
                st.warning("⚠️ Please select at least one feature column.")
            else:
                # Data validation
                st.subheader("🔍 Data Quality Analysis")
                
                # Create dataset object for validation
                try:
                    dataset = Dataset(
                        name="User Upload",
                        data=df_user,
                        target_column=target_col,
                        feature_columns=feature_cols,
                        data_type=DataType.UPLOADED
                    )
                    
                    validation_results = validator.validate_for_classification(df_user, target_col)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if validation_results['is_valid']:
                            st.success("✅ Data looks good for classification!")
                        else:
                            st.error("❌ Data quality issues detected:")
                            for issue in validation_results['issues']:
                                st.write(f"• {issue}")
                    
                    with col2:
                        if validation_results['recommendations']:
                            st.info("💡 Recommendations:")
                            for rec in validation_results['recommendations']:
                                st.write(f"• {rec}")
                    
                    # Preprocessing options
                    st.subheader("🛠️ Data Preprocessing")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        handle_missing = st.checkbox("Handle Missing Values", True)
                        if handle_missing:
                            missing_strategy = st.selectbox(
                                "Missing Value Strategy:",
                                ["mean", "median", "drop"]
                            )
                    
                    with col2:
                        encode_categorical = st.checkbox("Encode Categorical Variables", True)
                        scale_features = st.checkbox("Scale Features", True)
                        if scale_features:
                            scaling_method = st.selectbox(
                                "Scaling Method:",
                                ["standard", "minmax"]
                            )
                    
                    with col3:
                        # Advanced options
                        remove_outliers = st.checkbox("Remove Outliers", False)
                        feature_selection = st.checkbox("Automatic Feature Selection", False)
                    
                    # Apply preprocessing
                    if st.button("🔄 Apply Preprocessing", type="primary"):
                        with st.spinner("Preprocessing data..."):
                            preprocessing_options = {
                                'handle_missing': handle_missing,
                                'missing_strategy': missing_strategy if handle_missing else 'mean',
                                'encode_categorical': encode_categorical,
                                'scale_features': scale_features,
                                'scaling_method': scaling_method if scale_features else 'standard'
                            }
                            
                            processed_dataset = st.session_state.data_processor.preprocess_data(
                                dataset, preprocessing_options
                            )
                            
                            st.session_state.processed_dataset = processed_dataset
                            st.success("✅ Preprocessing completed!")
                            st.rerun()
                    
                    # Model training section
                    if 'processed_dataset' in st.session_state:
                        processed_dataset = st.session_state.processed_dataset
                        
                        st.subheader("🤖 Model Training")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col2:
                            st.markdown("**Model Configuration:**")
                            
                            # Model parameters
                            model_type = st.selectbox("Model Type:", [
                                "Standard Logistic Regression",
                                "L1 Regularized (Lasso)",
                                "L2 Regularized (Ridge)"
                            ])
                            
                            if "Regularized" in model_type:
                                C_param = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
                            
                            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
                            
                            # Train model button
                            train_model = st.button("🚀 Train Model", type="primary")
                        
                        with col1:
                            if train_model:
                                with st.spinner("Training model..."):
                                    # Get processed data
                                    X, y = processed_dataset.get_X_y()
                                    
                                    # Split data
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=test_size, random_state=42, stratify=y
                                    )
                                    
                                    # Configure model
                                    if model_type == "L1 Regularized (Lasso)":
                                        model = LogisticRegression(penalty='l1', C=C_param, solver='liblinear', random_state=42)
                                    elif model_type == "L2 Regularized (Ridge)":
                                        model = LogisticRegression(penalty='l2', C=C_param, random_state=42)
                                    else:
                                        model = LogisticRegression(random_state=42)
                                    
                                    # Train model
                                    model.fit(X_train, y_train)
                                    
                                    # Make predictions
                                    y_pred = model.predict(X_test)
                                    y_proba = model.predict_proba(X_test)
                                    
                                    # Calculate metrics
                                    accuracy = accuracy_score(y_test, y_pred)
                                    
                                    # Store results
                                    st.session_state.model_results = {
                                        'model': model,
                                        'X_test': X_test,
                                        'y_test': y_test,
                                        'y_pred': y_pred,
                                        'y_proba': y_proba,
                                        'accuracy': accuracy,
                                        'feature_names': list(X.columns),
                                        'target_name': target_col
                                    }
                                    
                                    st.success(f"✅ Model trained! Accuracy: {accuracy:.3f}")
                                    st.rerun()
                            
                            # Display results if model is trained
                            if 'model_results' in st.session_state:
                                results = st.session_state.model_results
                                
                                st.markdown("### 📊 Model Performance")
                                
                                # Performance metrics
                                col1a, col1b, col1c = st.columns(3)
                                
                                with col1a:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                    <h4>🎯 Accuracy</h4>
                                    <h2 style="color: #667eea;">{results['accuracy']:.3f}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Additional metrics for binary classification
                                if len(np.unique(results['y_test'])) == 2:
                                    precision = precision_score(results['y_test'], results['y_pred'])
                                    recall = recall_score(results['y_test'], results['y_pred'])
                                    
                                    with col1b:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                        <h4>🎯 Precision</h4>
                                        <h2 style="color: #e74c3c;">{precision:.3f}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col1c:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                        <h4>🎯 Recall</h4>
                                        <h2 style="color: #f39c12;">{recall:.3f}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Feature importance
                                st.markdown("### 🔍 Feature Importance")
                                
                                feature_importance = pd.DataFrame({
                                    'Feature': results['feature_names'],
                                    'Coefficient': results['model'].coef_[0] if len(results['model'].coef_) == 1 else results['model'].coef_[0]
                                })
                                feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
                                feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
                                
                                fig_importance = px.bar(
                                    feature_importance.head(10),
                                    x='Abs_Coefficient',
                                    y='Feature',
                                    orientation='h',
                                    title="Top 10 Most Important Features",
                                    color='Coefficient',
                                    color_continuous_scale='RdBu'
                                )
                                
                                fig_importance.update_layout(height=400, template="plotly_white")
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Confusion matrix for multiclass
                                if len(np.unique(results['y_test'])) > 2:
                                    st.markdown("### 🔍 Confusion Matrix")
                                    
                                    cm = confusion_matrix(results['y_test'], results['y_pred'])
                                    
                                    fig_cm = go.Figure(data=go.Heatmap(
                                        z=cm,
                                        colorscale='Blues',
                                        showscale=True,
                                        text=cm,
                                        texttemplate="%{text}",
                                        textfont={"size": 12}
                                    ))
                                    
                                    fig_cm.update_layout(
                                        title="Confusion Matrix",
                                        xaxis_title="Predicted",
                                        yaxis_title="Actual",
                                        height=400,
                                        template="plotly_white"
                                    )
                                    
                                    st.plotly_chart(fig_cm, use_container_width=True)
                                
                                # AI Analysis
                                if st.button("🤖 Get AI Analysis of Your Model"):
                                    with st.spinner("Analyzing your custom model..."):
                                        model_summary = {
                                            'dataset_name': uploaded_file.name,
                                            'n_samples': len(processed_dataset.data),
                                            'n_features': len(results['feature_names']),
                                            'accuracy': results['accuracy'],
                                            'model_type': model_type,
                                            'preprocessing_applied': processed_dataset.preprocessing_applied
                                        }
                                        
                                        analysis = get_ai_explanation(
                                            f"Analyze this custom logistic regression model: {model_summary}",
                                            {'page': page, 'user_data': model_summary}
                                        )
                                        
                                        st.markdown(f"""
                                        <div class="ai-response">
                                        <h4>🤖 Custom Model Analysis:</h4>
                                        {analysis}
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Code generation
                                if st.button("💻 Generate Python Code"):
                                    with st.spinner("Generating code for your model..."):
                                        code_params = {
                                            'model_type': model_type,
                                            'features': results['feature_names'],
                                            'target': results['target_name'],
                                            'preprocessing': processed_dataset.preprocessing_applied
                                        }
                                        
                                        code = get_ai_code_example(
                                            "logistic regression with preprocessing",
                                            code_params
                                        )
                                        
                                        st.markdown("### 💻 Generated Python Code")
                                        st.code(code, language='python')
                
                except Exception as e:
                    st.error(f"❌ Error creating dataset: {str(e)}")
                    st.info("💡 Please check your column selections and data format.")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Please make sure your file is a valid CSV format.")
    
    else:
        # Show example and instructions when no file is uploaded
        st.subheader("📋 How to Use the Data Playground")
        
        st.markdown("""
        <div class="concept-box">
        <h4>🎯 Getting Started:</h4>
        <ol>
        <li><strong>Upload your CSV file</strong> - Make sure it has a target column for classification</li>
        <li><strong>Select columns</strong> - Choose your target and feature columns</li>
        <li><strong>Preprocess data</strong> - Handle missing values, encode categories, scale features</li>
        <li><strong>Train model</strong> - Configure and train your logistic regression model</li>
        <li><strong>Analyze results</strong> - Get AI-powered insights and generated code</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data format
        st.subheader("📊 Expected Data Format")
        
        sample_data = pd.DataFrame({
            'feature_1': [1.2, 2.3, 3.1, 4.5, 2.8],
            'feature_2': [0.8, 1.5, 2.1, 3.2, 1.9],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        st.dataframe(sample_data)
        
        st.markdown("""
        **Requirements:**
        - CSV format with headers
        - At least one target column (what you want to predict)
        - One or more feature columns (predictors)
        - Numeric or categorical data (we'll handle the preprocessing!)
        """)
        
        # Download sample dataset
        if st.button("📥 Download Sample Dataset"):
            # Create a more comprehensive sample dataset
            np.random.seed(42)
            n_samples = 200
            
            sample_df = pd.DataFrame({
                'age': np.random.randint(18, 80, n_samples),
                'income': np.random.normal(50000, 15000, n_samples),
                'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
                'experience': np.random.randint(0, 40, n_samples),
                'satisfaction': np.random.randint(1, 6, n_samples),
                'purchased': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
            })
            
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download sample_data.csv",
                data=csv,
                file_name="sample_data.csv",
                mime="text/csv"
            )

elif page == " 7. AI Assistant & Help":
    st.title("🤖 AI Assistant & Help")
    
    if not GEMINI_AVAILABLE or not st.session_state.get('gemini_api_key'):
        st.warning("Please configure your Gemini API key in the sidebar to use the AI assistant.")
        st.markdown("""
        ### How to get a Gemini API Key:
        1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Create a new API key
        4. Copy and paste it in the sidebar
        """)
    else:
        st.success("AI Assistant is ready to help!")
        
        # Chat interface
        st.subheader("💬 Ask me anything about logistic regression!")
        
        user_question = st.text_input("Your question:", placeholder="e.g., How does logistic regression differ from linear regression?")
        
        if st.button("Ask AI") and user_question:
            with st.spinner("Thinking..."):
                response = get_ai_explanation(user_question, "General logistic regression help")
                st.session_state.conversation_history.append({"question": user_question, "answer": response})
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("💭 Conversation History")
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
                with st.expander(f"Q: {conv['question'][:50]}..."):
                    st.markdown(f"**Question:** {conv['question']}")
                    st.markdown(f"**Answer:** {conv['answer']}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
🎯 <strong>Logistic Regression Playground</strong> | Built with ❤️ using Streamlit & AI
</div>
""", unsafe_allow_html=True)