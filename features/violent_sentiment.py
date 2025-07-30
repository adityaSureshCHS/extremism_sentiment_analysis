"""
Violence Sentiment Detection Module
This module provides multiple approaches to detect violent content in text.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import re
from typing import Dict, List, Union, Tuple
import logging

class ViolenceSentimentAnalyzer:
    def __init__(self):
        """Initialize violence sentiment analyzer."""
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        # Violence lexicons
        self.explicit_violence_words = {
            'kill', 'murder', 'shoot', 'stab', 'bomb', 'explode', 'attack',
            'assault', 'beat', 'hit', 'punch', 'kick', 'slap', 'torture',
            'execute', 'assassinate', 'slaughter', 'massacre', 'genocide',
            'war', 'battle', 'fight', 'combat', 'destroy', 'annihilate',
            'eliminate', 'terminate', 'exterminate', 'crush', 'smash'
        }
        
        self.implicit_violence_words = {
            'threat', 'threaten', 'intimidate', 'menace', 'warning',
            'revenge', 'retaliation', 'payback', 'consequences',
            'force', 'pressure', 'coerce', 'compel', 'dominate',
            'overpower', 'subdue', 'suppress', 'oppress', 'control'
        }
        
        self.weapons_words = {
            'gun', 'rifle', 'pistol', 'weapon', 'knife', 'blade', 'sword',
            'bomb', 'explosive', 'grenade', 'missile', 'bullet', 'ammunition',
            'machete', 'axe', 'hammer', 'bat', 'club', 'stick'
        }
        
        self.violence_targets = {
            'person', 'people', 'individual', 'group', 'crowd', 'family',
            'children', 'kids', 'women', 'men', 'civilians', 'innocents',
            'enemy', 'opponent', 'rival', 'target', 'victim'
        }
        
        # Violence intensity modifiers
        self.intensity_modifiers = {
            'extremely': 2.0, 'very': 1.5, 'really': 1.3, 'quite': 1.2,
            'brutally': 2.5, 'violently': 2.0, 'aggressively': 1.8,
            'mercilessly': 2.3, 'ruthlessly': 2.2, 'savagely': 2.4
        }
    
    def load_violence_dataset(self, dataset_path: str) -> Tuple[List[str], List[int]]:
        """
        Load violence dataset from file.
        
        Args:
            dataset_path: Path to violence dataset (CSV or TDF)
            
        Returns:
            Tuple of texts and labels
        """
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.tdf'):
                df = pd.read_csv(dataset_path, sep='\t')
            else:
                raise ValueError("Dataset must be CSV or TDF format")
            
            # Assume columns are 'text' and 'violence' (0 for non-violence, 1 for violence)
            texts = df['text'].astype(str).tolist()
            labels = df['violence'].astype(int).tolist()
            
            print(f"Loaded {len(texts)} samples from dataset")
            print(f"Violence distribution: {np.bincount(labels)}")
            
            return texts, labels
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Create sample data for demonstration
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> Tuple[List[str], List[int]]:
        """Create sample violence dataset for demonstration."""
        sample_data = [
            ("I love spending time with my family", 0),
            ("We should have a peaceful discussion", 0),
            ("I'm going to kill you if you don't stop", 1),
            ("The movie had great action scenes", 0),
            ("He threatened to beat me up", 1),
            ("Let's fight against injustice peacefully", 0),
            ("I will destroy anyone who gets in my way", 1),
            ("The game was intense but fun", 0),
            ("We need to eliminate our enemies violently", 1),
            ("Community building is important", 0)
        ]
        
        texts, labels = zip(*sample_data)
        return list(texts), list(labels)
    
    def train_model(self, texts: List[str], labels: List[int], 
                   model_type: str = 'logistic_regression') -> Dict[str, float]:
        """
        Train violence detection model.
        
        Args:
            texts: List of text samples
            labels: List of violence labels (0/1)
            model_type: Type of model ('logistic_regression' or 'random_forest')
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Handle class imbalance
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Train model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                class_weight=class_weight_dict,
                random_state=42,
                max_iter=1000
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weight_dict,
                random_state=42
            )
        
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        y_prob = self.model.predict_proba(X_test_vec)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        self.is_trained = True
        print(f"Model trained successfully:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
        
        return metrics
    
    def lexicon_based_violence_score(self, text: str) -> Dict[str, float]:
        """
        Calculate violence score using lexicon-based approach.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with various violence scores
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {
                'explicit_violence': 0.0,
                'implicit_violence': 0.0,
                'weapons_score': 0.0,
                'targets_score': 0.0,
                'intensity_score': 1.0,
                'lexicon_violence_score': 0.0
            }
        
        # Count different types of violence indicators
        explicit_count = sum(1 for word in words if word in self.explicit_violence_words)
        implicit_count = sum(1 for word in words if word in self.implicit_violence_words)
        weapons_count = sum(1 for word in words if word in self.weapons_words)
        targets_count = sum(1 for word in words if word in self.violence_targets)
        
        # Calculate intensity multiplier
        intensity_multiplier = 1.0
        for word in words:
            if word in self.intensity_modifiers:
                intensity_multiplier = max(intensity_multiplier, self.intensity_modifiers[word])
        
        # Normalize scores
        text_length = len(words)
        scores = {
            'explicit_violence': explicit_count / text_length,
            'implicit_violence': implicit_count / text_length,
            'weapons_score': weapons_count / text_length,
            'targets_score': targets_count / text_length,
            'intensity_score': intensity_multiplier
        }
        
        # Calculate composite lexicon score
        base_score = (scores['explicit_violence'] * 2.0 + 
                     scores['implicit_violence'] * 1.0 + 
                     scores['weapons_score'] * 1.5 + 
                     scores['targets_score'] * 0.5) / 5.0
        
        scores['lexicon_violence_score'] = min(base_score * intensity_multiplier, 1.0)
        
        return scores
    
    def predict_violence_probability(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """
        Predict violence probability using trained model.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Violence probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if isinstance(text, str):
            text = [text]
        
        # Vectorize text
        text_vec = self.vectorizer.transform(text)
        
        # Get probabilities
        probabilities = self.model.predict_proba(text_vec)[:, 1]
        
        return probabilities[0] if len(text) == 1 else probabilities.tolist()
    
    def get_comprehensive_violence_score(self, text: str) -> Dict[str, float]:
        """
        Get comprehensive violence analysis combining multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with comprehensive violence scores
        """
        results = {}
        
        # Lexicon-based scores
        lexicon_scores = self.lexicon_based_violence_score(text)
        results.update(lexicon_scores)
        
        # Model-based score (if available)
        if self.is_trained:
            results['model_violence_probability'] = self.predict_violence_probability(text)
        else:
            results['model_violence_probability'] = 0.0
        
        # Pattern-based analysis
        results['violence_patterns'] = self._analyze_violence_patterns(text)
        
        # Calculate composite score
        if self.is_trained:
            composite_score = (
                results['lexicon_violence_score'] * 0.4 +
                results['model_violence_probability'] * 0.4 +
                results['violence_patterns'] * 0.2
            )
        else:
            composite_score = (
                results['lexicon_violence_score'] * 0.7 +
                results['violence_patterns'] * 0.3
            )
        
        results['composite_violence_score'] = composite_score
        
        return results
    
    def _analyze_violence_patterns(self, text: str) -> float:
        """
        Analyze violence-indicating patterns in text.
        
        Args:
            text: Input text
            
        Returns:
            Pattern-based violence score
        """
        pattern_score = 0.0
        
        # Threat patterns
        threat_patterns = [
            r'i will .* you',
            r'going to .* you',
            r'i\'ll .* you',
            r'watch out',
            r'you\'re dead',
            r'i\'m coming for you'
        ]
        
        for pattern in threat_patterns:
            if re.search(pattern, text.lower()):
                pattern_score += 0.3
        
        # Violence escalation patterns
        escalation_words = ['if you don\'t', 'unless you', 'or else', 'final warning']
        for phrase in escalation_words:
            if phrase in text.lower():
                pattern_score += 0.2
        
        # Time-bound threats
        time_patterns = [r'by \w+day', r'in \d+ \w+', r'before \w+']
        for pattern in time_patterns:
            if re.search(pattern, text.lower()):
                pattern_score += 0.1
        
        return min(pattern_score, 1.0)
    
    def save_model(self, model_path: str):
        """Save trained model and vectorizer."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and vectorizer."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    violence_analyzer = ViolenceSentimentAnalyzer()
    
    # Load and train model (using sample data)
    texts, labels = violence_analyzer.load_violence_dataset('data/violence_dataset.tdf')
    metrics = violence_analyzer.train_model(texts, labels)
    
    # Test texts
    test_texts = [
        "I love spending time with my family",
        "I'm going to kill you if you don't stop",
        "We should fight against injustice peacefully",
        "I will destroy anyone who gets in my way"
    ]
    
    # Analyze violence
    for text in test_texts:
        analysis = violence_analyzer.get_comprehensive_violence_score(text)
        print(f"Text: {text}")
        print(f"Composite Violence Score: {analysis['composite_violence_score']:.3f}")
        print(f"Model Probability: {analysis['model_violence_probability']:.3f}")
        print(f"Lexicon Score: {analysis['lexicon_violence_score']:.3f}")
        print("-" * 60)