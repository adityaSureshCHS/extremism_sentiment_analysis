"""
Hate Sentiment Analysis using HateBERT
This module extracts hate sentiment scores from text using HateBERT model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Union, Dict
import logging

class HateSentimentAnalyzer:
    def __init__(self, model_name: str = 'unitary/toxic-bert'):
        """
        Initialize HateBERT model for hate sentiment analysis.
        Alternative models:
        - 'unitary/toxic-bert'
        - 'martin-ha/toxic-comment-model'
        - 'unitary/unbiased-toxic-roberta'
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def predict_hate_score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Predict hate sentiment score for single text or batch of texts.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Float score (0-1) for single text or list of scores for batch
        """
        if isinstance(text, str):
            return self._predict_single(text)
        else:
            return [self._predict_single(t) for t in text]
    
    def _predict_single(self, text: str) -> float:
        """Predict hate score for a single text."""
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Assuming binary classification (0: non-hate, 1: hate)
            # Return probability of hate class
            hate_prob = probabilities[0][1].item()
            
        return hate_prob
    
    def get_detailed_scores(self, text: str) -> Dict[str, float]:
        """
        Get detailed hate sentiment analysis with multiple metrics.
        
        Returns:
            Dictionary with various hate-related scores
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Get raw logits for additional analysis
            logits = outputs.logits[0]
            
            scores = {
                'hate_probability': probabilities[0][1].item(),
                'non_hate_probability': probabilities[0][0].item(),
                'hate_confidence': torch.max(probabilities[0]).item(),
                'hate_intensity': logits[1].item(),  # Raw logit score
                'prediction_uncertainty': 1 - torch.max(probabilities[0]).item()
            }
            
        return scores

# Usage example and testing
if __name__ == "__main__":
    # Initialize analyzer
    hate_analyzer = HateSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "I love everyone regardless of their background",
        "This group of people should not exist",
        "Let's work together for a better future",
        "Those people are destroying our country"
    ]
    
    # Get hate scores
    for text in test_texts:
        score = hate_analyzer.predict_hate_score(text)
        detailed = hate_analyzer.get_detailed_scores(text)
        print(f"Text: {text}")
        print(f"Hate Score: {score:.3f}")
        print(f"Detailed: {detailed}")
        print("-" * 50)