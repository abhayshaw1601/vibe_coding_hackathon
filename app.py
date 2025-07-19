import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import time
import statsmodels.api as sm

# --- Page Configuration ---
st.set_page_config(
    page_title="Linear Regression Explained",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 5px;
        border: 1px solid #4F8BF9;
    }
    .stButton>button:hover {
        color: #ffffff;
        background-color: #4F8BF9;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. The Core Idea: Line of Best Fit",
    "2. Simple Linear Regression: Residuals",
    "3. How a Model Learns: Gradient Descent",
    "4. Multiple Linear Regression",
    "5. Model Assumptions & Diagnostics",
    "6. Your Data Playground"
])

# --- Helper Functions ---
@st.cache_data
def generate_data(n_samples=100, noise=20):
    """Generates sample data for demonstration."""
    X = np.linspace(0, 10, n_samples)
    y = 3 * X + 10 + np.random.normal(0, noise, n_samples)
    return pd.DataFrame({'X': X, 'y': y})

@st.cache_data
def get_housing_data():
    """Loads the California Housing dataset."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    return df

# ==============================================================================
# PAGE 1: THE CORE IDEA
# ==============================================================================
if page == "1. The Core Idea: Line of Best Fit":
    st.title("📊 1. The Core Idea: The Line of Best Fit")
    st.markdown("""
    Linear Regression is all about finding a straight line that best represents the relationship between two variables.
    
    *   **Independent Variable (X):** The variable we use to make a prediction.
    *   **Dependent Variable (y):** The variable we are trying to predict.
    
    The line is defined by the equation: **`y = mX + c`**
    *   **`m`** is the **slope** or **coefficient**: How much `y` changes for a one-unit change in `X`.
    *   **`c`** is the **intercept**: The value of `y` when `X` is 0.
    
    **Your Goal:** Adjust the sliders for `m` and `c` below to find the "best fit" for the data points. The best line is the one that minimizes the total error. We measure this error using **Mean Squared Error (MSE)**.
    """)

    data = generate_data()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Plot")
        slope_m = st.slider("Adjust Slope (m)", -5.0, 10.0, 3.0, 0.1)
        intercept_c = st.slider("Adjust Intercept (c)", -20.0, 40.0, 10.0, 0.5)

        # Create plot
        fig = px.scatter(data, x='X', y='y', title="Find the Best Fit Line")

        # Add the user's adjustable line
        x_range = np.array([data['X'].min(), data['X'].max()])
        y_range = slope_m * x_range + intercept_c
        fig.add_traces(go.Scatter(x=x_range, y=y_range, name="Your Line", mode='lines', line=dict(color='orange', width=4)))
        
        # Add the true "best fit" line for comparison
        model = LinearRegression()
        model.fit(data[['X']], data['y'])
        true_y_range = model.predict(x_range.reshape(-1, 1))
        fig.add_traces(go.Scatter(x=x_range, y=true_y_range, name="Best Fit Line (OLS)", mode='lines', line=dict(color='green', dash='dash')))

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Measuring Error")
        st.markdown("How good is your line? We calculate the **Mean Squared Error (MSE)**.")
        st.latex(r'''
        MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        ''')
        st.markdown("""
        Where:
        - `n`: number of data points
        - `y_i`: the actual value of a data point
        - `ŷ_i`: the predicted value from your line
        
        **Lower MSE is better!**
        """)
        
        y_pred = slope_m * data['X'] + intercept_c
        mse = mean_squared_error(data['y'], y_pred)
        
        best_mse = mean_squared_error(data['y'], model.predict(data[['X']]))
        
        st.metric("Your Line's MSE", f"{mse:.2f}")
        st.metric("Best Possible MSE", f"{best_mse:.2f}")

# ==============================================================================
# PAGE 2: SIMPLE LINEAR REGRESSION (SLR) & RESIDUALS
# ==============================================================================
elif page == "2. Simple Linear Regression: Residuals":
    st.title("📏 2. Simple Linear Regression & Residuals")
    st.markdown("""
    How does a computer find the "best fit" line automatically? It uses a method called **Ordinary Least Squares (OLS)**.
    
    The core idea is to minimize the **Sum of Squared Residuals**.
    
    A **residual** is the vertical distance between an actual data point (`y`) and the predicted point on the regression line (`ŷ`). It's the error for a single point.
    
    `Residual = y_actual - y_predicted`
    
    The visualization below shows the residuals for each point. OLS finds the line where the sum of the squares of these red lines is as small as possible.
    """)

    data = generate_data(n_samples=50, noise=15)
    X = data[['X']]
    y = data['y']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    fig = px.scatter(data, x='X', y='y', title="Visualizing Residuals")
    fig.add_traces(go.Scatter(x=data['X'], y=y_pred, name="Regression Line", mode='lines', line=dict(color='green')))

    # Add residual lines
    for i in range(len(data)):
        fig.add_shape(
            type="line",
            x0=data['X'][i], y0=data['y'][i],
            x1=data['X'][i], y1=y_pred[i],
            line=dict(color="Red", width=2, dash='dot'),
            name="Residual"
        )
    
    # To prevent legend spam
    fig.data[-1].showlegend=True
    fig.data[-1].name="Residuals"

    st.plotly_chart(fig, use_container_width=True)
    
    residuals = y - y_pred.flatten()
    rss = np.sum(residuals**2)
    
    st.subheader("Results from OLS")
    st.write(f"**Intercept (c):** {model.intercept_:.4f}")
    st.write(f"**Slope (m):** {model.coef_[0]:.4f}")
    st.write(f"**Sum of Squared Residuals (minimized):** {rss:.2f}")

# ==============================================================================
# PAGE 3: GRADIENT DESCENT
# ==============================================================================
elif page == "3. How a Model Learns: Gradient Descent":
    st.title("📉 3. How a Model Learns: Gradient Descent")
    st.markdown("""
    **Gradient Descent** is an alternative, iterative optimization algorithm used to find the best-fit line. It's especially useful for more complex models where a direct solution like OLS is not feasible.

    Imagine the **Mean Squared Error (MSE)** as a valley. Our goal is to find the lowest point in this valley.
    1.  **Start Anywhere:** We begin with random values for slope (`m`) and intercept (`c`).
    2.  **Take a Step:** We calculate the "gradient" (the direction of the steepest slope) of the error at our current position.
    3.  **Go Downhill:** We take a small step in the opposite direction of the gradient.
    4.  **Repeat:** We repeat this process until we reach the bottom of the valley (the minimum MSE).

    The size of the step is controlled by the **Learning Rate**.
    - **Too small:** Takes a very long time to converge.
    - **Too large:** Might overshoot the minimum and never find it.
    
    Press the button below to watch the algorithm learn!
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controls")
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        n_iterations = st.slider("Number of Iterations", 10, 1000, 100, 10)
        start_btn = st.button("Start Gradient Descent Animation")

    data = generate_data(n_samples=75, noise=10)
    X_gd = data['X'].values
    y_gd = data['y'].values
    n = len(X_gd)

    # Placeholders for the plot and metrics
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Initial plot
    fig_gd = go.Figure()
    fig_gd.add_trace(go.Scatter(x=X_gd, y=y_gd, mode='markers', name='Data'))
    fig_gd.update_layout(title="Gradient Descent in Action", xaxis_title="X", yaxis_title="y")
    plot_placeholder.plotly_chart(fig_gd, use_container_width=True)
    
    if start_btn:
        m, c = -2.0, 20.0  # Start with a "bad" line
        
        for i in range(n_iterations):
            y_pred = m * X_gd + c
            
            # Calculate gradients
            D_m = (-2/n) * sum(X_gd * (y_gd - y_pred))
            D_c = (-2/n) * sum(y_gd - y_pred)
            
            # Update weights
            m = m - learning_rate * D_m
            c = c - learning_rate * D_c
            
            mse = mean_squared_error(y_gd, y_pred)

            # Update Plot
            fig_gd = go.Figure()
            fig_gd.add_trace(go.Scatter(x=X_gd, y=y_gd, mode='markers', name='Data'))
            x_range = np.array([X_gd.min(), X_gd.max()])
            y_range = m * x_range + c
            fig_gd.add_trace(go.Scatter(x=x_range, y=y_range, name=f'Iteration {i+1}', mode='lines', line=dict(color='orange')))
            fig_gd.update_layout(title=f"Gradient Descent in Action (Iteration {i+1}/{n_iterations})", xaxis_title="X", yaxis_title="y")
            
            plot_placeholder.plotly_chart(fig_gd, use_container_width=True)
            
            # Update Metrics
            metrics_placeholder.markdown(f"""
            **Iteration:** {i+1}  
            **Slope (m):** {m:.4f}  
            **Intercept (c):** {c:.4f}  
            **MSE:** {mse:.4f}
            """)
            
            time.sleep(0.05) # Control animation speed
        
        st.success("Gradient Descent complete!")

# ==============================================================================
# PAGE 4: MULTIPLE LINEAR REGRESSION (MLR)
# ==============================================================================
elif page == "4. Multiple Linear Regression":
    st.title("📈 4. Multiple Linear Regression (MLR)")
    st.markdown("""
    Life is rarely simple! Often, we want to predict a value using **multiple** independent variables. This is called **Multiple Linear Regression**.

    The equation expands: **`y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ`**
    - **`β₀`** is the intercept.
    - **`β₁`, `β₂`, ...** are the coefficients for each feature (`X₁`, `X₂`, ...).

    Each coefficient (`βₙ`) represents the change in `y` for a one-unit change in that feature (`Xₙ`), **holding all other features constant.**

    We'll use the classic California Housing dataset to predict house prices.
    """)

    df = get_housing_data()
    
    st.subheader("1. Explore the Data")
    st.dataframe(df.head())
    
    st.subheader("2. Select Features (X) and Target (y)")
    all_features = df.columns.tolist()
    target_variable = 'PRICE'
    all_features.remove(target_variable)

    selected_features = st.multiselect("Select features to include in the model:", all_features, default=['MedInc', 'HouseAge', 'AveRooms', 'Population'])

    if not selected_features:
        st.warning("Please select at least one feature.")
    else:
        X = df[selected_features]
        y = df[target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("3. Train the Model & View Results")
        model_mlr = LinearRegression()
        model_mlr.fit(X_train, y_train)
        y_pred_mlr = model_mlr.predict(X_test)
        
        # --- Results ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Coefficients")
            st.markdown("This tells you how each feature influences the price.")
            coeffs = pd.DataFrame(
                model_mlr.coef_,
                index=selected_features,
                columns=['Coefficient']
            )
            st.dataframe(coeffs)
            st.write(f"**Intercept (β₀):** {model_mlr.intercept_:.4f}")

        with col2:
            st.markdown("#### Model Performance")
            mse_mlr = mean_squared_error(y_test, y_pred_mlr)
            r2_mlr = r2_score(y_test, y_pred_mlr)

            st.metric("Mean Squared Error (MSE)", f"{mse_mlr:.4f}")
            st.metric("R-squared (R²)", f"{r2_mlr:.4f}")
            st.info("""
            **R-squared (R²)**: The proportion of the variance in the dependent variable that is predictable from the independent variables.
            - Ranges from 0 to 1.
            - Higher is generally better.
            - An R² of 0.61 means that 61% of the variation in house prices can be explained by the selected features.
            """)


# ==============================================================================
# PAGE 5: ASSUMPTIONS & DIAGNOSTICS
# ==============================================================================
elif page == "5. Model Assumptions & Diagnostics":
    st.title("🔬 5. Model Assumptions & Diagnostics")
    st.markdown("""
    Linear regression is powerful, but it relies on several key assumptions. If these assumptions are violated, your model's predictions might be unreliable. A **robust** analysis always checks them!

    We'll build a model on the California Housing data and then check its diagnostic plots.
    """)

    # --- Build a model for diagnostics ---
    df_diag = get_housing_data()
    features_diag = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
    X_diag = df_diag[features_diag]
    y_diag = df_diag['PRICE']
    
    # Using statsmodels for more detailed diagnostics
    X_diag_sm = sm.add_constant(X_diag)
    model_sm = sm.OLS(y_diag, X_diag_sm).fit()
    y_pred_diag = model_sm.predict(X_diag_sm)
    residuals_diag = model_sm.resid

    st.subheader("1. Linearity Assumption")
    st.markdown("""
    **Assumption:** The relationship between the independent variables and the dependent variable is linear.
    **How to Check:** Plot actual values vs. predicted values. The points should be scattered randomly around a 45-degree line.
    """)
    fig_linearity = px.scatter(x=y_diag, y=y_pred_diag, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                               title="Actual vs. Predicted Values")
    fig_linearity.add_shape(type='line', x0=y_diag.min(), y0=y_diag.min(), x1=y_diag.max(), y1=y_diag.max(),
                            line=dict(color='Red', dash='dash'))
    st.plotly_chart(fig_linearity, use_container_width=True)

    st.subheader("2. Homoscedasticity Assumption")
    st.markdown("""
    **Assumption:** The residuals have constant variance at every level of x. (The spread of errors is consistent).
    **How to Check:** Plot residuals vs. predicted values. The plot should show a random, horizontal band of points with no clear pattern (like a funnel or a curve).
    **Violation is called Heteroscedasticity.**
    """)
    fig_homo = px.scatter(x=y_pred_diag, y=residuals_diag, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                          title="Residuals vs. Predicted Values")
    fig_homo.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_homo, use_container_width=True)
    st.markdown("*In this chart, we see the variance of residuals tends to increase as the predicted value increases (a funnel shape), suggesting a violation (heteroscedasticity).*")


    st.subheader("3. Normality of Residuals Assumption")
    st.markdown("""
    **Assumption:** The residuals of the model are normally distributed.
    **How to Check:** A histogram or a Q-Q (Quantile-Quantile) plot of the residuals. The histogram should look like a bell curve. In the Q-Q plot, points should fall closely along the red diagonal line.
    """)
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(residuals_diag, nbins=50, title="Histogram of Residuals")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        # Using statsmodels for Q-Q plot is easier
        fig_qq = sm.qqplot(residuals_diag, line='45', fit=True)
        st.pyplot(fig_qq)

    st.subheader("Full Model Summary (from `statsmodels`)")
    st.markdown("`statsmodels` provides a comprehensive summary that includes coefficients, p-values, R-squared, and warnings about potential assumption violations.")
    st.code(str(model_sm.summary()), language='text')

# ==============================================================================
# PAGE 6: YOUR DATA PLAYGROUND
# ==============================================================================
elif page == "6. Your Data Playground":
    st.title("🚀 6. Your Data Playground")
    st.markdown("Upload your own CSV file and build a linear regression model!")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df_user.head())

        st.subheader("1. Select Target and Feature Columns")
        # Filter for numeric columns only, as LR can only handle numeric data
        numeric_cols = df_user.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("Your dataset contains no numeric columns. Linear Regression requires numeric data for features and target.")
        else:
            target_col_user = st.selectbox("Select your Target Variable (y):", numeric_cols)
            
            feature_cols_user = st.multiselect("Select your Feature Variables (X):", 
                                                [col for col in numeric_cols if col != target_col_user],
                                                default=[col for col in numeric_cols if col != target_col_user][:3])

            if st.button("Train Model"):
                if not feature_cols_user:
                    st.warning("Please select at least one feature.")
                else:
                    X_user = df_user[feature_cols_user]
                    y_user = df_user[target_col_user]

                    # Drop rows with NaN values for simplicity
                    data_user = pd.concat([X_user, y_user], axis=1).dropna()
                    X_user = data_user[feature_cols_user]
                    y_user = data_user[target_col_user]

                    if len(data_user) < 10:
                        st.error("Not enough data to train a model after removing missing values. Please provide more data.")
                    else:
                        st.subheader("2. Model Results")
                        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_user, y_user, test_size=0.2, random_state=42)

                        model_user = LinearRegression()
                        model_user.fit(X_train_u, y_train_u)
                        y_pred_user = model_user.predict(X_test_u)

                        # --- Display Results ---
                        col1_user, col2_user = st.columns(2)
                        with col1_user:
                            st.markdown("#### Coefficients")
                            coeffs_user = pd.DataFrame(model_user.coef_, index=feature_cols_user, columns=['Coefficient'])
                            st.dataframe(coeffs_user)
                            st.write(f"**Intercept:** {model_user.intercept_:.4f}")
                        
                        with col2_user:
                            st.markdown("#### Performance")
                            mse_user = mean_squared_error(y_test_u, y_pred_user)
                            r2_user = r2_score(y_test_u, y_pred_user)
                            st.metric("Mean Squared Error (MSE)", f"{mse_user:.4f}")
                            st.metric("R-squared (R²)", f"{r2_user:.4f}")

                        st.subheader("3. Diagnostic Plots")
                        residuals_user = y_test_u - y_pred_user
                        
                        # Linearity
                        fig_lin_user = px.scatter(x=y_test_u, y=y_pred_user, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                                title="Actual vs. Predicted Values")
                        fig_lin_user.add_shape(type='line', x0=y_test_u.min(), y0=y_test_u.min(), x1=y_test_u.max(), y1=y_test_u.max(),
                                               line=dict(color='Red', dash='dash'))
                        st.plotly_chart(fig_lin_user, use_container_width=True)

                        # Homoscedasticity
                        fig_homo_user = px.scatter(x=y_pred_user, y=residuals_user, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                                 title="Residuals vs. Predicted Values")
                        fig_homo_user.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_homo_user, use_container_width=True)
    else:
        st.info("Upload a CSV to get started!")