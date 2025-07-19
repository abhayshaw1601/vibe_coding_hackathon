"""
Visualization engine for the Logistic Regression Playground.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

from .data_models import Dataset, AnimationSequence, AnimationFrame

class PlotFactory:
    """Factory for creating different types of plots."""
    
    @staticmethod
    def create_sigmoid_plot(slope: float = 1.0, shift: float = 0.0, 
                          highlight_point: Optional[float] = None) -> go.Figure:
        """Create interactive sigmoid function plot."""
        z = np.linspace(-10, 10, 1000)
        sigmoid_values = 1 / (1 + np.exp(-slope * (z - shift)))
        
        fig = go.Figure()
        
        # Main sigmoid curve
        fig.add_trace(go.Scatter(
            x=z, y=sigmoid_values,
            mode='lines',
            name='Sigmoid Function',
            line=dict(color='#667eea', width=3),
            hovertemplate="Input: %{x:.2f}<br>Probability: %{y:.3f}<extra></extra>"
        ))
        
        # Decision threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Decision Threshold (0.5)")
        
        # Highlight specific point if provided
        if highlight_point is not None:
            highlight_prob = 1 / (1 + np.exp(-slope * (highlight_point - shift)))
            fig.add_trace(go.Scatter(
                x=[highlight_point], y=[highlight_prob],
                mode='markers',
                marker=dict(size=12, color='red', symbol='diamond'),
                name=f'Sample Point (z={highlight_point:.1f})',
                hovertemplate=f"Input: {highlight_point:.2f}<br>Probability: {highlight_prob:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Interactive Sigmoid Function",
            xaxis_title="Input (z)",
            yaxis_title="Probability",
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_decision_boundary_plot(model, X: pd.DataFrame, y: np.ndarray, 
                                    resolution: float = 0.1) -> go.Figure:
        """Create decision boundary plot with probability heatmap."""
        # Create mesh grid
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))
        
        # Get probability predictions
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        
        # Add probability heatmap
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, resolution),
            y=np.arange(y_min, y_max, resolution),
            z=Z,
            colorscale='RdYlBu_r',
            opacity=0.6,
            showscale=True,
            colorbar=dict(title="P(Class 1)"),
            contours=dict(start=0, end=1, size=0.1),
            name="Probability"
        ))
        
        # Add decision boundary
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, resolution),
            y=np.arange(y_min, y_max, resolution),
            z=Z,
            contours=dict(start=0.5, end=0.5, size=0),
            line=dict(color='black', width=3),
            showscale=False,
            name="Decision Boundary"
        ))
        
        # Add data points
        colors = ['#3498db' if c == 0 else '#e74c3c' for c in y]
        symbols = ['circle' if c == 0 else 'diamond' for c in y]
        
        fig.add_trace(go.Scatter(
            x=X.iloc[:, 0],
            y=X.iloc[:, 1],
            mode='markers',
            marker=dict(
                color=colors,
                symbol=symbols,
                size=8,
                line=dict(width=1, color='white')
            ),
            name="Data Points",
            text=[f"Class {int(c)}" for c in y],
            hovertemplate="<b>%{text}</b><br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Decision Boundary with Probability Heatmap",
            xaxis_title=X.columns[0],
            yaxis_title=X.columns[1],
            height=500,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_3d_cost_surface(cost_history: List[Dict[str, float]], 
                             parameter_names: List[str]) -> go.Figure:
        """Create 3D cost function surface."""
        if len(parameter_names) != 2:
            raise ValueError("3D surface requires exactly 2 parameters")
        
        # Extract parameter values and costs
        param1_values = [point[parameter_names[0]] for point in cost_history]
        param2_values = [point[parameter_names[1]] for point in cost_history]
        costs = [point['cost'] for point in cost_history]
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Scatter3d(
            x=param1_values,
            y=param2_values,
            z=costs,
            mode='markers+lines',
            marker=dict(
                size=5,
                color=costs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cost")
            ),
            line=dict(color='red', width=3),
            name='Optimization Path'
        )])
        
        fig.update_layout(
            title="3D Cost Function Surface",
            scene=dict(
                xaxis_title=parameter_names[0],
                yaxis_title=parameter_names[1],
                zaxis_title="Cost"
            ),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                   class_names: Optional[List[str]] = None) -> go.Figure:
        """Create interactive confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            text=[[f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)" 
                   for j in range(len(cm[0]))] for i in range(len(cm))],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_roc_curve_plot(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
        """Create ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = np.trapz(tpr, fpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='#667eea', width=3),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_precision_recall_curve_plot(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
        """Create Precision-Recall curve plot."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auc_score = np.trapz(precision, recall)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {auc_score:.3f})',
            line=dict(color='#e74c3c', width=3),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
            template="plotly_white"
        )
        
        return fig

class AnimationController:
    """Control animations and smooth transitions."""
    
    def __init__(self):
        self.animation_speed = 0.1
        self.frame_buffer = []
        self.current_sequence = None
    
    def create_smooth_transition(self, start_state: Dict[str, float], 
                               end_state: Dict[str, float], steps: int = 50) -> List[Dict[str, float]]:
        """Create smooth transition between two states."""
        transitions = []
        
        for i in range(steps + 1):
            t = i / steps
            # Use easing function for smoother animation
            eased_t = self._ease_in_out_cubic(t)
            
            current_state = {}
            for key in start_state:
                if key in end_state:
                    start_val = start_state[key]
                    end_val = end_state[key]
                    current_state[key] = start_val + (end_val - start_val) * eased_t
                else:
                    current_state[key] = start_state[key]
            
            transitions.append(current_state)
        
        return transitions
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth animations."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def animate_parameter_change(self, parameter_name: str, values: List[float]) -> AnimationSequence:
        """Create animation sequence for parameter changes."""
        sequence = AnimationSequence(name=f"{parameter_name}_animation")
        
        for i, value in enumerate(values):
            timestamp = i * self.animation_speed
            frame_data = {parameter_name: value, 'frame_id': i}
            sequence.add_frame(frame_data, timestamp)
        
        return sequence
    
    def create_gradient_descent_animation(self, optimization_path: List[Dict[str, Any]]) -> AnimationSequence:
        """Create gradient descent animation sequence."""
        sequence = AnimationSequence(name="gradient_descent", duration=len(optimization_path) * 0.1)
        
        for i, step in enumerate(optimization_path):
            timestamp = i * 0.1
            sequence.add_frame(step, timestamp)
        
        return sequence

class VisualizationEngine:
    """Main visualization engine combining all plotting capabilities."""
    
    def __init__(self):
        self.plot_factory = PlotFactory()
        self.animation_controller = AnimationController()
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#3498db'
        }
    
    def create_sigmoid_plot(self, params: Dict[str, float]) -> go.Figure:
        """Create sigmoid plot with given parameters."""
        return self.plot_factory.create_sigmoid_plot(
            slope=params.get('slope', 1.0),
            shift=params.get('shift', 0.0),
            highlight_point=params.get('highlight_point')
        )
    
    def create_decision_boundary_plot(self, model, X: pd.DataFrame, y: np.ndarray) -> go.Figure:
        """Create decision boundary plot."""
        return self.plot_factory.create_decision_boundary_plot(model, X, y)
    
    def create_3d_cost_surface(self, cost_history: List[Dict[str, float]]) -> go.Figure:
        """Create 3D cost surface plot."""
        parameter_names = [key for key in cost_history[0].keys() if key != 'cost'][:2]
        return self.plot_factory.create_3d_cost_surface(cost_history, parameter_names)
    
    def create_model_evaluation_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_prob: np.ndarray, class_names: Optional[List[str]] = None) -> go.Figure:
        """Create comprehensive model evaluation dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Class Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig.add_trace(
            go.Heatmap(z=cm, x=class_names, y=class_names, colorscale='Blues'),
            row=1, col=1
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = np.trapz(tpr, fpr)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_score:.3f})'),
            row=1, col=2
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'),
            row=2, col=1
        )
        
        # Class Distribution
        unique, counts = np.unique(y_true, return_counts=True)
        fig.add_trace(
            go.Bar(x=[class_names[i] for i in unique], y=counts, name='True Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Model Evaluation Dashboard")
        
        return fig
    
    def animate_gradient_descent(self, steps: List[Dict[str, Any]]) -> None:
        """Create animated gradient descent visualization."""
        sequence = self.animation_controller.create_gradient_descent_animation(steps)
        self.animation_controller.current_sequence = sequence
        return sequence
    
    def get_color_palette(self) -> Dict[str, str]:
        """Get the current color palette."""
        return self.color_palette.copy()
    
    def set_theme(self, theme: str = 'default') -> None:
        """Set visualization theme."""
        if theme == 'dark':
            self.color_palette.update({
                'primary': '#bb86fc',
                'secondary': '#03dac6',
                'background': '#121212'
            })
        elif theme == 'light':
            self.color_palette.update({
                'primary': '#667eea',
                'secondary': '#764ba2',
                'background': '#ffffff'
            })