"""
Data models and classes for the Logistic Regression Playground.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from enum import Enum

class ModelType(Enum):
    """Enumeration for different model types."""
    BINARY = "binary"
    MULTICLASS = "multiclass"
    CUSTOM = "custom"

class DataType(Enum):
    """Enumeration for different data types."""
    SYNTHETIC = "synthetic"
    BUILTIN = "builtin"
    UPLOADED = "uploaded"

class QueryType(Enum):
    """Enumeration for AI query types."""
    EXPLANATION = "explanation"
    CODE = "code"
    ANALYSIS = "analysis"
    SUGGESTION = "suggestion"

@dataclass
class AppState:
    """Application state management."""
    current_page: str = "🧠 1. Core Concepts: Sigmoid & Probability"
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    model_cache: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    dataset_cache: Dict[str, Any] = field(default_factory=dict)
    gemini_api_key: str = ""
    
    def update_preference(self, key: str, value: Any) -> None:
        """Update user preference."""
        self.user_preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference with default."""
        return self.user_preferences.get(key, default)

@dataclass
class ModelState:
    """State of a trained model."""
    model_type: ModelType
    parameters: Dict[str, Any]
    training_data: Optional[pd.DataFrame] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_trained: bool = False
    model_object: Any = None
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics."""
        self.performance_metrics.update(metrics)
    
    def is_valid(self) -> bool:
        """Check if model state is valid."""
        return (self.is_trained and 
                self.model_object is not None and 
                self.training_data is not None)

@dataclass
class Dataset:
    """Dataset container with metadata."""
    name: str
    data: pd.DataFrame
    target_column: str
    feature_columns: List[str]
    data_type: DataType
    preprocessing_applied: Dict[str, Any] = field(default_factory=dict)
    class_names: Optional[List[str]] = None
    description: str = ""
    
    def __post_init__(self):
        """Validate dataset after initialization."""
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        missing_features = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
    
    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return len(self.data)
    
    @property
    def n_features(self) -> int:
        """Number of features in dataset."""
        return len(self.feature_columns)
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return self.data[self.target_column].nunique()
    
    def get_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get features and target as separate objects."""
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        return X, y
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes."""
        return self.data[self.target_column].value_counts().to_dict()

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 200
    n_features: int = 2
    n_classes: int = 2
    noise_level: float = 0.1
    class_separation: float = 1.0
    random_state: int = 42
    cluster_std: float = 1.0
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.n_samples > 0 and 
                self.n_features > 0 and 
                self.n_classes >= 2 and 
                0 <= self.noise_level <= 1 and 
                self.class_separation > 0)

@dataclass
class AIQuery:
    """AI query container."""
    query_type: QueryType
    context: Dict[str, Any]
    user_input: str
    page_context: str
    timestamp: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert query to formatted prompt."""
        base_prompt = f"""
        You are an expert machine learning educator explaining logistic regression concepts.
        
        Query Type: {self.query_type.value}
        Page Context: {self.page_context}
        User Input: {self.user_input}
        
        Additional Context: {self.context}
        
        Please provide a clear, educational response suitable for students learning machine learning.
        """
        
        if self.query_type == QueryType.CODE:
            base_prompt += "\nPlease include relevant Python code examples with explanations."
        elif self.query_type == QueryType.ANALYSIS:
            base_prompt += "\nPlease analyze the provided data/results and give insights."
        elif self.query_type == QueryType.SUGGESTION:
            base_prompt += "\nPlease provide actionable suggestions for improvement."
        
        return base_prompt

@dataclass
class AIResponse:
    """AI response container."""
    response_text: str
    code_examples: List[str] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    
    def format_for_display(self) -> str:
        """Format response for Streamlit display."""
        formatted = self.response_text
        
        if self.code_examples:
            formatted += "\n\n### Code Examples:\n"
            for i, code in enumerate(self.code_examples, 1):
                formatted += f"\n**Example {i}:**\n```python\n{code}\n```\n"
        
        if self.follow_up_questions:
            formatted += "\n\n### Related Questions:\n"
            for question in self.follow_up_questions:
                formatted += f"- {question}\n"
        
        return formatted

@dataclass
class AnimationFrame:
    """Single frame in an animation sequence."""
    frame_id: int
    data: Dict[str, Any]
    timestamp: float
    
@dataclass
class AnimationSequence:
    """Sequence of animation frames."""
    name: str
    frames: List[AnimationFrame] = field(default_factory=list)
    duration: float = 1.0
    loop: bool = False
    
    def add_frame(self, data: Dict[str, Any], timestamp: float) -> None:
        """Add a frame to the sequence."""
        frame_id = len(self.frames)
        frame = AnimationFrame(frame_id, data, timestamp)
        self.frames.append(frame)
    
    def get_frame_at_time(self, time: float) -> Optional[AnimationFrame]:
        """Get frame at specific time."""
        for frame in self.frames:
            if abs(frame.timestamp - time) < 0.01:  # Small tolerance
                return frame
        return None

@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_score': self.auc_score,
            'classification_report': self.classification_report
        }
    
    def get_summary(self) -> str:
        """Get formatted summary of performance."""
        return f"""
        Accuracy: {self.accuracy:.3f}
        Precision: {self.precision:.3f}
        Recall: {self.recall:.3f}
        F1-Score: {self.f1_score:.3f}
        AUC Score: {self.auc_score:.3f}
        """

@dataclass
class UserInteraction:
    """Track user interactions for analytics."""
    page: str
    action: str
    timestamp: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
class SessionManager:
    """Manage session state and user interactions."""
    
    def __init__(self):
        self.interactions: List[UserInteraction] = []
        self.session_start: Optional[str] = None
    
    def log_interaction(self, page: str, action: str, parameters: Dict[str, Any] = None) -> None:
        """Log user interaction."""
        from datetime import datetime
        
        interaction = UserInteraction(
            page=page,
            action=action,
            timestamp=datetime.now().isoformat(),
            parameters=parameters or {}
        )
        self.interactions.append(interaction)
    
    def get_user_journey(self) -> List[str]:
        """Get sequence of pages visited."""
        return [interaction.page for interaction in self.interactions]
    
    def get_most_used_features(self) -> Dict[str, int]:
        """Get most frequently used features."""
        feature_counts = {}
        for interaction in self.interactions:
            action = interaction.action
            feature_counts[action] = feature_counts.get(action, 0) + 1
        return dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))