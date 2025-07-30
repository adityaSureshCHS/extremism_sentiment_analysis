'''
Enhanced Overall Sentiment Analysis Module
This module provides comprehensive sentiment analysis with emotion detection,
intensity escalation patterns, and advanced sarcasm/irony detection.
'''

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import math

# Optional deep learning imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Advanced emotion detection disabled.")

@dataclass
class EmotionScores:
    """Structured emotion analysis results."""
    fear: float
    anger: float
    disgust: float
    sadness: float
    joy: float
    surprise: float
    contempt: float
    dominant_emotion: str
    intensity: float

@dataclass
class SarcasmAnalysis:
    """Sarcasm and irony detection results."""
    sarcasm_probability: float
    irony_probability: float
    contradiction_score: float
    sentiment_mismatch: float
    indicators: List[str]

@dataclass
class IntensityEscalation:
    """Intensity escalation pattern analysis."""
    escalation_score: float
    peak_intensity: float
    escalation_points: List[int]
    pattern_type: str  # 'gradual', 'sudden', 'sustained', 'none'
    trajectory: List[float]

@dataclass
class ComprehensiveSentimentResult:
    """Complete sentiment analysis result."""
    text: str
    basic_sentiment: Dict[str, float]
    emotions: EmotionScores
    sarcasm: SarcasmAnalysis
    escalation: IntensityEscalation
    extremist_indicators: Dict[str, float]
    manipulation_patterns: Dict[str, float]
    overall_risk_score: float
    confidence: float

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with emotion detection and advanced pattern recognition."""
    
    def __init__(self, 
                 enable_deep_emotions: bool = True,
                 emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize enhanced sentiment analyzer.
        
        Args:
            enable_deep_emotions: Whether to use transformer-based emotion detection
            emotion_model: Hugging Face model for emotion classification
        """
        # Core sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Emotion detection setup
        self.deep_emotions_enabled = enable_deep_emotions and TRANSFORMERS_AVAILABLE
        self.emotion_classifier = None
        
        if self.deep_emotions_enabled:
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model=emotion_model,
                    device=-1  # CPU usage
                )
                print(f"Loaded emotion model: {emotion_model}")
            except Exception as e:
                print(f"Failed to load emotion model: {e}")
                self.deep_emotions_enabled = False
        
        # Enhanced lexicons and patterns
        self._initialize_emotion_lexicons()
        self._initialize_sarcasm_patterns()
        self._initialize_escalation_patterns()

    # ... [existing methods: _initialize_emotion_lexicons, _initialize_sarcasm_patterns,
    #     _initialize_escalation_patterns,
    #     analyze_emotions_lexicon, analyze_emotions_deep,
    #     detect_sarcasm_irony, analyze_intensity_escalation, _calculate_sentence_intensity,
    #     get_comprehensive_analysis, _analyze_extremist_patterns,
    #     _analyze_manipulation_patterns, batch_analyze] ...

if __name__ == "__main__":
    print("Initializing Enhanced Sentiment Analyzer...")
    analyzer = EnhancedSentimentAnalyzer(enable_deep_emotions=True)

    test_texts = [
        "I love spending time with my family and friends! We have such wonderful moments together.",
        "Oh great, another 'brilliant' decision by our leaders. What could possibly go wrong?",  
        "This is concerning. This is really bad. This is absolutely terrible! We must act NOW!!!",
        "They are coming for us. Our way of life is under threat. We must defend ourselves before it's too late.",
        "These parasites are destroying everything we hold dear. They must be stopped at all costs!",
        "What a perfect day for democracy when voters choose their own destruction. How wonderful."
    ]

    print("\n" + "="*80)
    print("ENHANCED SENTIMENT ANALYSIS RESULTS")
    print("="*80)

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Analysis {i} ---")
        print(f"Text: '{text}'\n")
        result = analyzer.get_comprehensive_analysis(text)

        # Basic Sentiment
        print("Basic Sentiment:")
        for k, v in result.basic_sentiment.items():
            print(f"  {k.replace('_', ' ').title()}: {v:.3f}")

        # Emotion Analysis
        print("\nEmotion Analysis:")
        print(f"  Dominant Emotion: {result.emotions.dominant_emotion}")
        for attr in ['fear', 'anger', 'disgust', 'sadness', 'joy', 'surprise', 'contempt']:
            score = getattr(result.emotions, attr)
            print(f"    {attr.title()}: {score:.3f}")
        print(f"  Intensity: {result.emotions.intensity:.3f}\n")

        # Sarcasm & Irony
        print("Sarcasm & Irony:")
        print(f"  Sarcasm Probability: {result.sarcasm.sarcasm_probability:.3f}")
        print(f"  Irony Probability: {result.sarcasm.irony_probability:.3f}")
        print(f"  Contradiction Score: {result.sarcasm.contradiction_score:.3f}")
        print(f"  Sentiment Mismatch: {result.sarcasm.sentiment_mismatch:.3f}")
        if result.sarcasm.indicators:
            print(f"  Indicators: {', '.join(result.sarcasm.indicators)}")

        # Intensity Escalation
        print("\nIntensity Escalation:")
        print(f"  Escalation Score: {result.escalation.escalation_score:.3f}")
        print(f"  Peak Intensity: {result.escalation.peak_intensity:.3f}")
        print(f"  Pattern Type: {result.escalation.pattern_type}")
        if result.escalation.escalation_points:
            print(f"  Escalation Points: {result.escalation.escalation_points}")

        # Extremist Indicators
        print("\nExtremist Indicators:")
        for k, v in result.extremist_indicators.items():
            print(f"  {k.replace('_', ' ').title()}: {v:.3f}")

        # Manipulation Patterns
        print("\nManipulation Patterns:")
        for k, v in result.manipulation_patterns.items():
            print(f"  {k.replace('_', ' ').title()}: {v:.3f}")

        # Overall Risk & Confidence
        print("\nOverall Risk Score: {:.3f}".format(result.overall_risk_score))
        print(f"Confidence: {result.confidence:.3f}")
        print("-"*80)
