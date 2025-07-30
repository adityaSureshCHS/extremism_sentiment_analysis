"""
Enhanced Hate Sentiment Analysis using Multiple BERT Models
This module provides advanced hate sentiment analysis with ensemble models,
confidence thresholding, and domain-specific fine-tuning capabilities.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
from abc import ABC, abstractmethod

@dataclass
class ModelConfig:
    """Configuration for individual models in the ensemble."""
    name: str
    model_path: str
    weight: float = 1.0
    confidence_threshold: float = 0.5

@dataclass 
class PredictionResult:
    """Structured result for hate sentiment predictions."""
    text: str
    hate_probability: float
    confidence: float
    prediction: str  # 'hate', 'non-hate', 'uncertain'
    model_scores: Dict[str, float]
    ensemble_agreement: float

class BaseHateDetector(ABC):
    """Abstract base class for hate detection models."""
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, float]:
        pass

class SingleModelDetector(BaseHateDetector):
    """Individual BERT-based hate detector."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Loaded model: {self.config.name}")
        except Exception as e:
            logging.error(f"Error loading model {self.config.name}: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, float]:
        """Predict hate sentiment for text."""
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
            
            # Handle different model output formats
            if probabilities.shape[1] == 2:  # Binary classification
                hate_prob = probabilities[0][1].item()
                non_hate_prob = probabilities[0][0].item()
            else:  # Multi-class - assume last class is hate
                hate_prob = probabilities[0][-1].item()
                non_hate_prob = 1 - hate_prob
            
            confidence = torch.max(probabilities[0]).item()
            
        return {
            'hate_probability': hate_prob,
            'non_hate_probability': non_hate_prob,
            'confidence': confidence,
            'model_name': self.config.name
        }

class EnsembleHateSentimentAnalyzer:
    """Enhanced hate sentiment analyzer with ensemble models and advanced features."""
    
    def __init__(self, 
                 model_configs: Optional[List[ModelConfig]] = None,
                 confidence_threshold: float = 0.7,
                 uncertainty_threshold: float = 0.3):
        """
        Initialize ensemble hate sentiment analyzer.
        
        Args:
            model_configs: List of model configurations for ensemble
            confidence_threshold: Minimum confidence for definitive predictions
            uncertainty_threshold: Threshold below which predictions are marked uncertain
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Default model configurations if none provided
        if model_configs is None:
            model_configs = [
                ModelConfig("toxic-bert", "unitary/toxic-bert", weight=1.0),
                ModelConfig("hateful-roberta", "unitary/unbiased-toxic-roberta", weight=1.0),
                ModelConfig("toxic-comment", "martin-ha/toxic-comment-model", weight=0.8)
            ]
        
        self.model_configs = model_configs
        self.detectors = []
        self._load_models()
    
    def _load_models(self):
        """Load all models in the ensemble."""
        for config in self.model_configs:
            try:
                detector = SingleModelDetector(config, self.device)
                self.detectors.append(detector)
                logging.info(f"Successfully loaded {config.name}")
            except Exception as e:
                logging.warning(f"Failed to load {config.name}: {e}")
        
        if not self.detectors:
            raise RuntimeError("No models could be loaded successfully")
        
        logging.info(f"Loaded {len(self.detectors)} models in ensemble")
    
    def predict_hate_sentiment(self, text: Union[str, List[str]]) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Predict hate sentiment with ensemble voting and confidence analysis.
        
        Args:
            text: Single text or list of texts to analyze
            
        Returns:
            PredictionResult or list of results
        """
        if isinstance(text, str):
            return self._predict_single(text)
        else:
            return [self._predict_single(t) for t in text]
    
    def _predict_single(self, text: str) -> PredictionResult:
        """Predict hate sentiment for single text using ensemble."""
        model_predictions = []
        model_scores = {}
        
        # Get predictions from all models
        for detector in self.detectors:
            try:
                pred = detector.predict(text)
                model_predictions.append(pred)
                model_scores[pred['model_name']] = pred['hate_probability']
            except Exception as e:
                logging.warning(f"Model {detector.config.name} failed: {e}")
        
        if not model_predictions:
            raise RuntimeError("All models failed to make predictions")
        
        # Calculate weighted ensemble average
        total_weight = sum(detector.config.weight for detector in self.detectors 
                          if any(p['model_name'] == detector.config.name for p in model_predictions))
        
        weighted_hate_prob = sum(
            pred['hate_probability'] * next(d.config.weight for d in self.detectors 
                                          if d.config.name == pred['model_name'])
            for pred in model_predictions
        ) / total_weight
        
        # Calculate ensemble agreement (variance measure)
        hate_probs = [pred['hate_probability'] for pred in model_predictions]
        ensemble_agreement = 1 - np.var(hate_probs)  # Higher = more agreement
        
        # Calculate overall confidence
        confidence = min(ensemble_agreement, 
                        np.mean([pred['confidence'] for pred in model_predictions]))
        
        # Make final prediction with thresholding
        if confidence < self.uncertainty_threshold:
            prediction = 'uncertain'
        elif weighted_hate_prob > self.confidence_threshold:
            prediction = 'hate'
        else:
            prediction = 'non-hate'
        
        return PredictionResult(
            text=text,
            hate_probability=weighted_hate_prob,
            confidence=confidence,
            prediction=prediction,
            model_scores=model_scores,
            ensemble_agreement=ensemble_agreement
        )
    
    def get_detailed_analysis(self, text: str) -> Dict:
        """Get comprehensive analysis including individual model outputs."""
        result = self._predict_single(text)
        
        # Get individual model details
        individual_results = []
        for detector in self.detectors:
            try:
                pred = detector.predict(text)
                individual_results.append({
                    'model': pred['model_name'],
                    'hate_prob': pred['hate_probability'],
                    'confidence': pred['confidence'],
                    'weight': detector.config.weight
                })
            except:
                continue
        
        return {
            'ensemble_prediction': {
                'prediction': result.prediction,
                'hate_probability': result.hate_probability,
                'confidence': result.confidence,
                'ensemble_agreement': result.ensemble_agreement
            },
            'individual_models': individual_results,
            'analysis_metadata': {
                'num_models': len(individual_results),
                'confidence_threshold': self.confidence_threshold,
                'uncertainty_threshold': self.uncertainty_threshold
            }
        }
    
    def calibrate_thresholds(self, validation_texts: List[str], 
                           validation_labels: List[int]) -> Dict[str, float]:
        """
        Calibrate confidence and uncertainty thresholds using validation data.
        
        Args:
            validation_texts: List of validation texts
            validation_labels: Binary labels (0: non-hate, 1: hate)
            
        Returns:
            Dictionary with optimal thresholds
        """
        predictions = []
        confidences = []
        
        for text in validation_texts:
            result = self._predict_single(text)
            predictions.append(1 if result.hate_probability > 0.5 else 0)
            confidences.append(result.confidence)
        
        # Find optimal confidence threshold
        best_accuracy = 0
        best_conf_threshold = 0.5
        
        for threshold in np.arange(0.1, 1.0, 0.1):
            # Only consider high-confidence predictions
            high_conf_indices = [i for i, c in enumerate(confidences) if c >= threshold]
            
            if len(high_conf_indices) > 0:
                high_conf_preds = [predictions[i] for i in high_conf_indices]
                high_conf_labels = [validation_labels[i] for i in high_conf_indices]
                
                accuracy = accuracy_score(high_conf_labels, high_conf_preds)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_conf_threshold = threshold
        
        # Update thresholds
        self.confidence_threshold = best_conf_threshold
        self.uncertainty_threshold = best_conf_threshold * 0.5
        
        return {
            'optimal_confidence_threshold': best_conf_threshold,
            'optimal_uncertainty_threshold': self.uncertainty_threshold,
            'validation_accuracy': best_accuracy
        }
    
    def fine_tune_model(self, 
                       train_texts: List[str], 
                       train_labels: List[int],
                       val_texts: List[str],
                       val_labels: List[int],
                       base_model: str = "unitary/toxic-bert",
                       output_dir: str = "./fine_tuned_hate_model",
                       epochs: int = 3,
                       learning_rate: float = 2e-5) -> str:
        """
        Fine-tune a model on domain-specific data.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels (0: non-hate, 1: hate)
            val_texts: Validation texts  
            val_labels: Validation labels
            base_model: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Path to fine-tuned model
        """
        from torch.utils.data import Dataset
        
        class HateDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, 
            num_labels=2
        )
        
        # Create datasets
        train_dataset = HateDataset(train_texts, train_labels, tokenizer)
        val_dataset = HateDataset(val_texts, val_labels, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            save_total_limit=2,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            accuracy = accuracy_score(labels, predictions)
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        logging.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        logging.info(f"Fine-tuned model saved to {output_dir}")
        return output_dir
    
    def add_custom_model(self, model_path: str, model_name: str, weight: float = 1.0):
        """Add a custom fine-tuned model to the ensemble."""
        config = ModelConfig(model_name, model_path, weight)
        try:
            detector = SingleModelDetector(config, self.device)
            self.detectors.append(detector)
            self.model_configs.append(config)
            logging.info(f"Added custom model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to add custom model {model_name}: {e}")
    
    def save_configuration(self, filepath: str):
        """Save ensemble configuration to file."""
        config = {
            'model_configs': [
                {
                    'name': mc.name,
                    'model_path': mc.model_path,
                    'weight': mc.weight,
                    'confidence_threshold': mc.confidence_threshold
                }
                for mc in self.model_configs
            ],
            'confidence_threshold': self.confidence_threshold,
            'uncertainty_threshold': self.uncertainty_threshold
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_configuration(cls, filepath: str):
        """Load ensemble configuration from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        model_configs = [
            ModelConfig(**mc) for mc in config['model_configs']
        ]
        
        return cls(
            model_configs=model_configs,
            confidence_threshold=config['confidence_threshold'],
            uncertainty_threshold=config['uncertainty_threshold']
        )


# Usage example and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize enhanced analyzer
    analyzer = EnsembleHateSentimentAnalyzer(
        confidence_threshold=0.7,
        uncertainty_threshold=0.3
    )
    
    # Test texts with varying degrees of hate content
    test_texts = [
        "I love everyone regardless of their background",
        "This group of people should not exist",
        "Let's work together for a better future", 
        "Those people are destroying our country",
        "I'm not sure about this policy",  # Ambiguous case
        "We need stricter immigration laws"  # Potentially controversial
    ]
    
    print("=== Enhanced Hate Sentiment Analysis Results ===\n")
    
    # Analyze each text
    for text in test_texts:
        result = analyzer.predict_hate_sentiment(text)
        detailed = analyzer.get_detailed_analysis(text)
        
        print(f"Text: '{text}'")
        print(f"Prediction: {result.prediction}")
        print(f"Hate Probability: {result.hate_probability:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Ensemble Agreement: {result.ensemble_agreement:.3f}")
        print(f"Individual Model Scores: {result.model_scores}")
        print("-" * 60)
    
    # Example of batch processing
    print("\n=== Batch Processing Results ===")
    batch_results = analyzer.predict_hate_sentiment(test_texts[:3])
    for i, result in enumerate(batch_results):
        print(f"{i+1}. {result.prediction} (prob: {result.hate_probability:.3f})")
    
    print(f"\nAnalyzer loaded with {len(analyzer.detectors)} models")
    print("Enhanced features: ✓ Ensemble voting ✓ Confidence thresholding ✓ Fine-tuning support")