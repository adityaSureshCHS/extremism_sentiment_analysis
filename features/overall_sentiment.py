"""
Overall Sentiment Analysis using VADER
This module provides comprehensive sentiment analysis using VADER and additional methods.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from typing import Dict, List, Union
import numpy as np

class OverallSentimentAnalyzer:
    def __init__(self):
        """Initialize VADER sentiment analyzer and other sentiment tools."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Custom lexicons for extremist sentiment analysis
        self.positive_extremist_words = {
            'pure', 'superior', 'elite', 'chosen', 'righteous', 'sacred', 
            'holy', 'blessed', 'divine', 'perfect', 'glorious'
        }
        
        self.negative_extremist_words = {
            'corrupt', 'degenerate', 'inferior', 'subhuman', 'parasite', 
            'vermin', 'disease', 'cancer', 'poison', 'filth', 'scum'
        }
        
        self.intensity_amplifiers = {
            'very', 'extremely', 'completely', 'totally', 'absolutely',
            'utterly', 'entirely', 'purely', 'wholly', 'fully'
        }
    
    def get_vader_scores(self, text: str) -> Dict[str, float]:
        """
        Get VADER sentiment scores.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with VADER scores
        """
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def get_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get TextBlob sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with polarity and subjectivity
        """
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def calculate_emotional_intensity(self, text: str) -> float:
        """
        Calculate emotional intensity based on various linguistic features.
        
        Args:
            text: Input text
            
        Returns:
            Emotional intensity score (0-1)
        """
        text_lower = text.lower()
        
        # Count intensity indicators
        intensity_score = 0.0
        
        # Exclamation marks
        exclamation_count = text.count('!')
        intensity_score += min(exclamation_count * 0.1, 0.3)
        
        # All caps words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        intensity_score += min(caps_words / len(words) if words else 0, 0.2)
        
        # Repeated characters (e.g., "sooooo")
        repeated_char_pattern = r'(.)\1{2,}'
        repeated_matches = len(re.findall(repeated_char_pattern, text_lower))
        intensity_score += min(repeated_matches * 0.05, 0.15)
        
        # Intensity amplifiers
        amplifier_count = sum(1 for word in text_lower.split() 
                             if word in self.intensity_amplifiers)
        intensity_score += min(amplifier_count * 0.05, 0.2)
        
        # Multiple punctuation
        multi_punct_pattern = r'[!.?]{2,}'
        multi_punct_count = len(re.findall(multi_punct_pattern, text))
        intensity_score += min(multi_punct_count * 0.1, 0.2)
        
        return min(intensity_score, 1.0)
    
    def analyze_extremist_sentiment_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment patterns specific to extremist content.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extremist sentiment patterns
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {'positive_extremist': 0.0, 'negative_extremist': 0.0, 
                   'extremist_polarity': 0.0}
        
        # Count positive and negative extremist words
        pos_extremist_count = sum(1 for word in words 
                                 if word in self.positive_extremist_words)
        neg_extremist_count = sum(1 for word in words 
                                 if word in self.negative_extremist_words)
        
        # Normalize by text length
        pos_extremist_score = pos_extremist_count / len(words)
        neg_extremist_score = neg_extremist_count / len(words)
        
        # Calculate extremist polarity
        if pos_extremist_count + neg_extremist_count > 0:
            extremist_polarity = (pos_extremist_count - neg_extremist_count) / \
                               (pos_extremist_count + neg_extremist_count)
        else:
            extremist_polarity = 0.0
        
        return {
            'positive_extremist': pos_extremist_score,
            'negative_extremist': neg_extremist_score,
            'extremist_polarity': extremist_polarity
        }
    
    def detect_sentiment_manipulation(self, text: str) -> Dict[str, float]:
        """
        Detect potential sentiment manipulation techniques.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with manipulation indicators
        """
        # Look for emotional manipulation patterns
        manipulation_score = 0.0
        
        text_lower = text.lower()
        
        # Fear-based language
        fear_words = {'threat', 'danger', 'attack', 'destroy', 'end', 'death', 
                     'kill', 'murder', 'war', 'fight', 'battle', 'enemy'}
        fear_count = sum(1 for word in text_lower.split() if word in fear_words)
        
        # Urgency language
        urgency_words = {'now', 'immediately', 'urgent', 'crisis', 'emergency', 
                        'quick', 'fast', 'hurry', 'time', 'deadline'}
        urgency_count = sum(1 for word in text_lower.split() if word in urgency_words)
        
        # Us vs them language
        us_them_words = {'us', 'them', 'they', 'we', 'our', 'their', 'enemy', 
                        'ally', 'friend', 'foe'}
        us_them_count = sum(1 for word in text_lower.split() if word in us_them_words)
        
        words = text_lower.split()
        total_words = len(words) if words else 1
        
        return {
            'fear_manipulation': fear_count / total_words,
            'urgency_manipulation': urgency_count / total_words,
            'us_them_framing': us_them_count / total_words,
            'overall_manipulation': (fear_count + urgency_count + us_them_count) / (total_words * 3)
        }
    
    def get_comprehensive_sentiment_analysis(self, text: str) -> Dict[str, Union[float, Dict]]:
        """
        Get comprehensive sentiment analysis combining multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with comprehensive sentiment analysis
        """
        results = {}
        
        # VADER scores
        results['vader'] = self.get_vader_scores(text)
        
        # TextBlob sentiment
        results['textblob'] = self.get_textblob_sentiment(text)
        
        # Emotional intensity
        results['emotional_intensity'] = self.calculate_emotional_intensity(text)
        
        # Extremist sentiment patterns
        results['extremist_patterns'] = self.analyze_extremist_sentiment_patterns(text)
        
        # Sentiment manipulation
        results['manipulation'] = self.detect_sentiment_manipulation(text)
        
        # Composite scores
        results['composite_sentiment'] = results['vader']['compound']
        results['composite_negativity'] = (
            results['vader']['neg'] + 
            abs(min(results['textblob']['polarity'], 0)) +
            results['extremist_patterns']['negative_extremist']
        ) / 3
        
        results['composite_intensity'] = (
            results['emotional_intensity'] +
            abs(results['vader']['compound']) +
            results['manipulation']['overall_manipulation']
        ) / 3
        
        return results

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    sentiment_analyzer = OverallSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "I love spending time with my family and friends!",
        "These people are destroying everything we hold dear!!!",
        "We must take immediate action against our enemies before it's too late",
        "Let's work together to build a better community for everyone"
    ]
    
    # Analyze sentiment
    for text in test_texts:
        analysis = sentiment_analyzer.get_comprehensive_sentiment_analysis(text)
        print(f"Text: {text}")
        print(f"VADER Compound: {analysis['vader']['compound']:.3f}")
        print(f"Emotional Intensity: {analysis['emotional_intensity']:.3f}")
        print(f"Composite Negativity: {analysis['composite_negativity']:.3f}")
        print(f"Manipulation Score: {analysis['manipulation']['overall_manipulation']:.3f}")
        print("-" * 60)