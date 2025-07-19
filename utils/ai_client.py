"""
AI integration layer using Google Gemini API for the Logistic Regression Playground.
"""

import streamlit as st
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime
import hashlib

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .data_models import AIQuery, AIResponse, QueryType

class ContextManager:
    """Manage context for AI conversations."""
    
    def __init__(self):
        self.conversation_context = []
        self.page_contexts = {
            "🧠 1. Core Concepts: Sigmoid & Probability": "sigmoid function, probability interpretation, logistic vs linear regression",
            "🎯 2. Binary Classification: Decision Boundaries": "binary classification, decision boundaries, interactive data points",
            "📉 3. Learning Process: Gradient Descent": "gradient descent optimization, cost function, learning rate",
            "🌈 4. Multiclass Classification": "multiclass logistic regression, one-vs-rest, multinomial",
            "📊 5. Model Diagnostics & Evaluation": "ROC curves, confusion matrix, precision, recall, model evaluation",
            "🚀 6. Your Data Playground": "custom datasets, preprocessing, feature engineering",
            "🤖 7. AI Assistant & Help": "general help, explanations, code examples"
        }
    
    def build_context(self, page: str, user_data: Dict[str, Any]) -> str:
        """Build comprehensive context for AI queries."""
        context_parts = []
        
        # Add page-specific context
        if page in self.page_contexts:
            context_parts.append(f"Current page context: {self.page_contexts[page]}")
        
        # Add user data context
        if user_data:
            context_parts.append("Current user state:")
            for key, value in user_data.items():
                if isinstance(value, (int, float, str, bool)):
                    context_parts.append(f"- {key}: {value}")
                elif isinstance(value, (list, dict)) and len(str(value)) < 200:
                    context_parts.append(f"- {key}: {value}")
        
        # Add recent conversation context
        if self.conversation_context:
            context_parts.append("Recent conversation:")
            for conv in self.conversation_context[-3:]:  # Last 3 exchanges
                context_parts.append(f"Q: {conv['question'][:100]}...")
                context_parts.append(f"A: {conv['answer'][:100]}...")
        
        return "\n".join(context_parts)
    
    def maintain_conversation_history(self, messages: List[Dict[str, str]]) -> None:
        """Maintain conversation history with size limits."""
        self.conversation_context.extend(messages)
        
        # Keep only last 10 conversations to manage context size
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def extract_relevant_context(self, query: str) -> Dict[str, Any]:
        """Extract relevant context based on query content."""
        relevant_context = {}
        
        # Keywords for different topics
        topic_keywords = {
            'sigmoid': ['sigmoid', 'probability', 'logistic function', 'curve'],
            'classification': ['classify', 'class', 'boundary', 'decision'],
            'gradient_descent': ['gradient', 'descent', 'optimization', 'learning rate'],
            'evaluation': ['accuracy', 'precision', 'recall', 'roc', 'auc', 'confusion'],
            'preprocessing': ['preprocess', 'scale', 'encode', 'missing', 'clean']
        }
        
        query_lower = query.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_context[f'{topic}_related'] = True
        
        return relevant_context

class ResponseCache:
    """Cache AI responses to reduce API calls and improve performance."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def _generate_key(self, query: str, context: str) -> str:
        """Generate cache key from query and context."""
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context: str) -> Optional[str]:
        """Get cached response if available."""
        key = self._generate_key(query, context)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, query: str, context: str, response: str) -> None:
        """Cache response with LRU eviction."""
        key = self._generate_key(query, context)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = response
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        self.access_times.clear()

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 60, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_make_call(self) -> bool:
        """Check if a call can be made within rate limits."""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self) -> None:
        """Record a new API call."""
        self.calls.append(time.time())
    
    def get_wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        if not self.calls:
            return 0
        
        oldest_call = min(self.calls)
        wait_time = self.time_window - (time.time() - oldest_call)
        return max(0, wait_time)

class GeminiClient:
    """Client for Google Gemini API with error handling and optimization."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.context_manager = ContextManager()
        self.response_cache = ResponseCache()
        self.rate_limiter = RateLimiter()
        self.model = None
        
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
            except Exception as e:
                st.error(f"Failed to initialize Gemini client: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Gemini API is available and configured."""
        return GEMINI_AVAILABLE and self.model is not None
    
    def get_explanation(self, concept: str, context: Dict[str, Any]) -> str:
        """Get AI explanation for a concept."""
        if not self.is_available():
            return self._get_fallback_explanation(concept)
        
        # Build context
        context_str = self.context_manager.build_context(
            context.get('page', ''), 
            context.get('user_data', {})
        )
        
        # Check cache first
        cached_response = self.response_cache.get(concept, context_str)
        if cached_response:
            return cached_response
        
        # Check rate limits
        if not self.rate_limiter.can_make_call():
            wait_time = self.rate_limiter.get_wait_time()
            return f"Rate limit reached. Please wait {wait_time:.0f} seconds before asking again."
        
        try:
            query = AIQuery(
                query_type=QueryType.EXPLANATION,
                context=context,
                user_input=concept,
                page_context=context.get('page', '')
            )
            
            prompt = self._build_explanation_prompt(query, context_str)
            
            response = self.model.generate_content(prompt)
            self.rate_limiter.record_call()
            
            # Cache the response
            self.response_cache.set(concept, context_str, response.text)
            
            return response.text
            
        except Exception as e:
            return f"AI service temporarily unavailable: {str(e)}"
    
    def generate_code_example(self, topic: str, parameters: Dict[str, Any]) -> str:
        """Generate Python code example for a topic."""
        if not self.is_available():
            return self._get_fallback_code(topic)
        
        context_str = json.dumps(parameters, indent=2)
        
        # Check cache
        cached_response = self.response_cache.get(f"code_{topic}", context_str)
        if cached_response:
            return cached_response
        
        if not self.rate_limiter.can_make_call():
            return "Rate limit reached. Please try again later."
        
        try:
            prompt = f"""
            Generate a complete, runnable Python code example for: {topic}
            
            Parameters: {context_str}
            
            Requirements:
            - Use scikit-learn for logistic regression
            - Include necessary imports
            - Add comments explaining each step
            - Make it educational and easy to understand
            - Include data visualization if relevant
            
            Provide only the code with minimal explanation.
            """
            
            response = self.model.generate_content(prompt)
            self.rate_limiter.record_call()
            
            # Cache the response
            self.response_cache.set(f"code_{topic}", context_str, response.text)
            
            return response.text
            
        except Exception as e:
            return f"Code generation failed: {str(e)}"
    
    def analyze_results(self, model_results: Dict[str, Any]) -> str:
        """Analyze model results and provide insights."""
        if not self.is_available():
            return self._get_fallback_analysis(model_results)
        
        results_str = json.dumps(model_results, indent=2, default=str)
        
        # Check cache
        cached_response = self.response_cache.get("analysis", results_str)
        if cached_response:
            return cached_response
        
        if not self.rate_limiter.can_make_call():
            return "Rate limit reached. Please try again later."
        
        try:
            prompt = f"""
            Analyze these logistic regression model results and provide insights:
            
            {results_str}
            
            Please provide:
            1. Interpretation of the performance metrics
            2. What the results tell us about the model
            3. Potential areas for improvement
            4. Any concerns or red flags
            
            Keep the analysis educational and actionable.
            """
            
            response = self.model.generate_content(prompt)
            self.rate_limiter.record_call()
            
            # Cache the response
            self.response_cache.set("analysis", results_str, response.text)
            
            return response.text
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def suggest_improvements(self, model_performance: Dict[str, Any]) -> str:
        """Suggest improvements based on model performance."""
        if not self.is_available():
            return self._get_fallback_suggestions(model_performance)
        
        performance_str = json.dumps(model_performance, indent=2, default=str)
        
        if not self.rate_limiter.can_make_call():
            return "Rate limit reached. Please try again later."
        
        try:
            prompt = f"""
            Based on these logistic regression model performance metrics, suggest specific improvements:
            
            {performance_str}
            
            Provide actionable suggestions for:
            1. Data preprocessing improvements
            2. Feature engineering ideas
            3. Model hyperparameter tuning
            4. Alternative approaches to consider
            
            Make suggestions practical and educational.
            """
            
            response = self.model.generate_content(prompt)
            self.rate_limiter.record_call()
            
            return response.text
            
        except Exception as e:
            return f"Suggestion generation failed: {str(e)}"
    
    def _build_explanation_prompt(self, query: AIQuery, context: str) -> str:
        """Build comprehensive prompt for explanations."""
        return f"""
        You are an expert machine learning educator specializing in logistic regression.
        
        Context: {context}
        
        Question: {query.user_input}
        
        Please provide a clear, educational explanation that:
        1. Uses simple, accessible language
        2. Includes practical examples when helpful
        3. Connects to the broader context of machine learning
        4. Helps the student build intuition
        
        Keep the response concise but comprehensive.
        """
    
    def _get_fallback_explanation(self, concept: str) -> str:
        """Provide fallback explanations when AI is unavailable."""
        fallbacks = {
            'sigmoid': """
            The sigmoid function is a mathematical function that maps any real number to a value between 0 and 1, 
            making it perfect for representing probabilities. Its S-shaped curve is smooth and differentiable, 
            which makes it ideal for gradient-based optimization in logistic regression.
            """,
            'decision boundary': """
            A decision boundary is the line (or surface in higher dimensions) that separates different classes 
            in a classification problem. In logistic regression, this boundary represents where the predicted 
            probability equals 0.5 (50% chance for each class).
            """,
            'gradient descent': """
            Gradient descent is an optimization algorithm that finds the minimum of a function by iteratively 
            moving in the direction of steepest descent. In logistic regression, it's used to find the best 
            parameters that minimize the cost function.
            """
        }
        
        for key, explanation in fallbacks.items():
            if key.lower() in concept.lower():
                return explanation.strip()
        
        return "AI explanations are currently unavailable. Please check your API configuration."
    
    def _get_fallback_code(self, topic: str) -> str:
        """Provide fallback code examples."""
        if 'sigmoid' in topic.lower():
            return """
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Create data points
z = np.linspace(-10, 10, 100)
y = sigmoid(z)

# Plot sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(z, y, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision threshold')
plt.xlabel('Input (z)')
plt.ylabel('Probability')
plt.title('Sigmoid Function')
plt.grid(True)
plt.legend()
plt.show()
            """
        
        return "# Code examples are currently unavailable. Please check your AI configuration."
    
    def _get_fallback_analysis(self, results: Dict[str, Any]) -> str:
        """Provide basic fallback analysis."""
        accuracy = results.get('accuracy', 0)
        
        if accuracy > 0.9:
            return "Excellent model performance! The high accuracy suggests the model is working well."
        elif accuracy > 0.8:
            return "Good model performance. There might be room for some improvements."
        elif accuracy > 0.7:
            return "Moderate performance. Consider feature engineering or hyperparameter tuning."
        else:
            return "Model performance could be improved. Consider data quality, feature selection, or different algorithms."
    
    def _get_fallback_suggestions(self, performance: Dict[str, Any]) -> str:
        """Provide basic fallback suggestions."""
        return """
        General suggestions for improving logistic regression:
        1. Check data quality and handle missing values
        2. Scale/normalize features if they have different ranges
        3. Consider feature engineering or polynomial features
        4. Try regularization (L1 or L2) to prevent overfitting
        5. Collect more data if possible
        6. Ensure classes are balanced or use appropriate techniques for imbalanced data
        """

# Convenience functions for Streamlit integration
def get_ai_client() -> Optional[GeminiClient]:
    """Get AI client from session state."""
    if 'ai_client' not in st.session_state:
        api_key = st.session_state.get('gemini_api_key', '')
        if api_key:
            st.session_state.ai_client = GeminiClient(api_key)
        else:
            st.session_state.ai_client = None
    
    return st.session_state.ai_client

def get_ai_explanation(concept: str, context: Dict[str, Any] = None) -> str:
    """Convenience function to get AI explanation."""
    client = get_ai_client()
    if client and client.is_available():
        return client.get_explanation(concept, context or {})
    else:
        return "AI explanations are not available. Please configure your Gemini API key."

def get_ai_code_example(topic: str, parameters: Dict[str, Any] = None) -> str:
    """Convenience function to get AI code example."""
    client = get_ai_client()
    if client and client.is_available():
        return client.generate_code_example(topic, parameters or {})
    else:
        return "# AI code generation is not available. Please configure your Gemini API key."

def analyze_model_results(results: Dict[str, Any]) -> str:
    """Convenience function to analyze model results."""
    client = get_ai_client()
    if client and client.is_available():
        return client.analyze_results(results)
    else:
        return "AI analysis is not available. Please configure your Gemini API key."