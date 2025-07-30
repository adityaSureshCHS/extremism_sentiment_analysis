"""
Extremist Lexicon Scoring Module
This module creates various scoring methods for extremist terminology using a lexicon.
"""

import re
import math
from typing import List, Dict, Set, Union, Tuple
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ExtremistLexiconScorer:
    def __init__(self, lexicon_path: str = None):
        """
        Initialize extremist lexicon scorer.
        
        Args:
            lexicon_path: Path to extremist lexicon file (one term per line)
        """
        self.extremist_terms = set()
        self.stop_words = set(stopwords.words('english'))
        self.weighted_terms = {}  # For term-specific weights
        
        if lexicon_path:
            self.load_lexicon(lexicon_path)
    
    def load_lexicon(self, lexicon_path: str, weighted: bool = False):
        """
        Load extremist lexicon from file.
        
        Args:
            lexicon_path: Path to lexicon file
            weighted: If True, expects format "term:weight", else just terms
        """
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().lower()
                    if not line or line.startswith('#'):
                        continue
                    
                    if weighted and ':' in line:
                        term, weight = line.split(':', 1)
                        self.extremist_terms.add(term.strip())
                        self.weighted_terms[term.strip()] = float(weight)
                    else:
                        self.extremist_terms.add(line)
                        
            print(f"Loaded {len(self.extremist_terms)} extremist terms from lexicon")
            
        except FileNotFoundError:
            print(f"Lexicon file not found: {lexicon_path}")
            # Create sample extremist terms for demonstration
            self._create_sample_lexicon()
    
    def _create_sample_lexicon(self):
        """Create a sample extremist lexicon for demonstration purposes."""
        # This is a sanitized sample - real lexicons should be carefully curated
        sample_terms = {
            'supremacist', 'nationalist', 'extremist', 'radical', 'militant',
            'overthrow', 'revolution', 'uprising', 'resistance', 'rebellion',
            'enemy', 'traitor', 'invasion', 'replacement', 'inferior',
            'pure', 'cleanse', 'eliminate', 'destroy', 'annihilate'
        }
        self.extremist_terms = sample_terms
        print(f"Created sample lexicon with {len(sample_terms)} terms")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for lexicon matching.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def simple_match_score(self, text: str) -> float:
        """
        Simple lexicon matching - count of extremist terms.
        
        Args:
            text: Input text
            
        Returns:
            Normalized score (0-1)
        """
        tokens = self.preprocess_text(text)
        matches = sum(1 for token in tokens if token in self.extremist_terms)
        
        # Normalize by text length
        if len(tokens) == 0:
            return 0.0
        
        return min(matches / len(tokens), 1.0)
    
    def weighted_match_score(self, text: str) -> float:
        """
        Weighted lexicon matching using term-specific weights.
        
        Args:
            text: Input text
            
        Returns:
            Weighted score
        """
        tokens = self.preprocess_text(text)
        if len(tokens) == 0:
            return 0.0
        
        total_weight = 0.0
        for token in tokens:
            if token in self.extremist_terms:
                weight = self.weighted_terms.get(token, 1.0)
                total_weight += weight
        
        # Normalize by text length
        return min(total_weight / len(tokens), 1.0)
    
    def tfidf_score(self, text: str, corpus_stats: Dict = None) -> float:
        """
        TF-IDF based scoring for extremist terms.
        
        Args:
            text: Input text
            corpus_stats: Dictionary with document frequencies for terms
            
        Returns:
            TF-IDF based extremist score
        """
        tokens = self.preprocess_text(text)
        if len(tokens) == 0:
            return 0.0
        
        # Calculate term frequencies
        tf_counts = Counter(tokens)
        text_length = len(tokens)
        
        tfidf_score = 0.0
        extremist_terms_found = 0
        
        for term in self.extremist_terms:
            if term in tf_counts:
                # Term frequency
                tf = tf_counts[term] / text_length
                
                # Inverse document frequency (use default if not provided)
                if corpus_stats and term in corpus_stats:
                    idf = math.log(corpus_stats['total_docs'] / corpus_stats[term])
                else:
                    idf = 1.0  # Default IDF
                
                tfidf_score += tf * idf
                extremist_terms_found += 1
        
        # Normalize by number of extremist terms found
        if extremist_terms_found > 0:
            tfidf_score /= extremist_terms_found
        
        return min(tfidf_score, 1.0)
    
    def density_score(self, text: str) -> float:
        """
        Calculate density of extremist terms in text.
        
        Args:
            text: Input text
            
        Returns:
            Density score (extremist terms / total terms)
        """
        tokens = self.preprocess_text(text)
        if len(tokens) == 0:
            return 0.0
        
        extremist_count = sum(1 for token in tokens if token in self.extremist_terms)
        return extremist_count / len(tokens)
    
    def proximity_score(self, text: str, window_size: int = 5) -> float:
        """
        Score based on proximity of extremist terms to each other.
        
        Args:
            text: Input text
            window_size: Window size for proximity calculation
            
        Returns:
            Proximity-based score
        """
        tokens = self.preprocess_text(text)
        if len(tokens) < 2:
            return 0.0
        
        extremist_positions = []
        for i, token in enumerate(tokens):
            if token in self.extremist_terms:
                extremist_positions.append(i)
        
        if len(extremist_positions) < 2:
            return self.simple_match_score(text)
        
        # Calculate proximity score
        proximity_score = 0.0
        for i in range(len(extremist_positions) - 1):
            distance = extremist_positions[i + 1] - extremist_positions[i]
            if distance <= window_size:
                proximity_score += 1.0 / distance
        
        # Normalize by text length
        return min(proximity_score / len(tokens), 1.0)
    
    def get_comprehensive_score(self, text: str) -> Dict[str, float]:
        """
        Get comprehensive extremist lexicon scores using multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with various scoring methods
        """
        scores = {
            'simple_match': self.simple_match_score(text),
            'weighted_match': self.weighted_match_score(text),
            'density': self.density_score(text),
            'proximity': self.proximity_score(text),
            'tfidf': self.tfidf_score(text)
        }
        
        # Calculate composite score
        scores['composite'] = np.mean(list(scores.values()))
        
        # Get matched terms for analysis
        tokens = self.preprocess_text(text)
        matched_terms = [token for token in tokens if token in self.extremist_terms]
        scores['matched_terms'] = matched_terms
        scores['unique_matches'] = len(set(matched_terms))
        
        return scores

# Usage example
if __name__ == "__main__":
    # Initialize scorer
    lexicon_scorer = ExtremistLexiconScorer('data/extremism_lexicon.txt')
    
    # Test texts
    test_texts = [
        "This is a normal peaceful message about cooperation",
        "The radical extremist groups are planning an uprising",
        "We need to overthrow the current system and establish supremacist rule",
        "Community building and dialogue are important for society"
    ]
    
    # Get scores
    for text in test_texts:
        scores = lexicon_scorer.get_comprehensive_score(text)
        print(f"Text: {text}")
        print(f"Scores: {scores}")
        print("-" * 50)