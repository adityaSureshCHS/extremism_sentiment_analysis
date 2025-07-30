"""
Enhanced Extremist Lexicon Scoring Module
This module provides advanced scoring methods for extremist terminology using comprehensive lexicons,
with support for n-grams, phrases, semantic similarity, and context-aware matching.
"""

import re
import math
import json
from typing import List, Dict, Set, Union, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass
import pickle
import os

# NLP libraries
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk

# Semantic similarity (optional - falls back gracefully if not available)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity disabled.")

# Download required NLTK data
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

@dataclass
class MatchResult:
    """Structured result for lexicon matches."""
    term: str
    positions: List[int]
    context: str
    confidence: float
    match_type: str  # 'exact', 'partial', 'semantic', 'phrase'

@dataclass
class ExtremistScore:
    """Comprehensive extremist scoring result."""
    text: str
    scores: Dict[str, float]
    matches: List[MatchResult]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    context_analysis: Dict[str, any]

class EnhancedExtremistLexiconScorer:
    """Enhanced extremist lexicon scorer with advanced matching capabilities."""
    
    def __init__(self, 
                 lexicon_path: str = "data/extremism_lexicon.txt",
                 enable_semantic_matching: bool = True,
                 semantic_model: str = "all-MiniLM-L6-v2",
                 semantic_threshold: float = 0.7):
        """
        Initialize enhanced extremist lexicon scorer.
        
        Args:
            lexicon_path: Path to comprehensive extremist lexicon file
            enable_semantic_matching: Whether to enable semantic similarity matching
            semantic_model: Sentence transformer model for semantic matching
            semantic_threshold: Minimum similarity score for semantic matches
        """
        self.lexicon_path = lexicon_path
        self.semantic_threshold = semantic_threshold
        
        # Core lexicon storage
        self.single_terms = set()
        self.phrase_terms = set()
        self.weighted_terms = {}
        self.category_terms = defaultdict(set)
        
        # N-gram storage
        self.bigrams = set()
        self.trigrams = set()
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Semantic matching components
        self.semantic_model = None
        self.term_embeddings = None
        
        if enable_semantic_matching and SEMANTIC_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer(semantic_model)
                print(f"Loaded semantic model: {semantic_model}")
            except Exception as e:
                print(f"Failed to load semantic model: {e}")
                enable_semantic_matching = False
        
        self.semantic_enabled = enable_semantic_matching and self.semantic_model is not None
        
        # Load lexicon
        self.load_comprehensive_lexicon()
        
        # Build semantic embeddings if enabled
        if self.semantic_enabled:
            self._build_semantic_embeddings()
    
    def load_comprehensive_lexicon(self):
        """Load comprehensive extremist lexicon with enhanced parsing."""
        if not os.path.exists(self.lexicon_path):
            print(f"Lexicon file not found: {self.lexicon_path}")
            print("Creating minimal fallback lexicon...")
            self._create_fallback_lexicon()
            return
        
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"Loading lexicon from {self.lexicon_path} ({len(lines)} entries)")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse different formats
                self._parse_lexicon_entry(line, line_num)
            
            print(f"Loaded comprehensive lexicon:")
            print(f"  - Single terms: {len(self.single_terms)}")
            print(f"  - Phrase terms: {len(self.phrase_terms)}")
            print(f"  - Bigrams: {len(self.bigrams)}")
            print(f"  - Trigrams: {len(self.trigrams)}")
            print(f"  - Categories: {len(self.category_terms)}")
            
        except Exception as e:
            print(f"Error loading lexicon: {e}")
            self._create_fallback_lexicon()
    
    def _parse_lexicon_entry(self, entry: str, line_num: int):
        """Parse individual lexicon entry with support for various formats."""
        original_entry = entry
        entry = entry.lower().strip()
        
        # Handle weighted entries (term:weight:category)
        if ':' in entry:
            parts = entry.split(':')
            term = parts[0].strip()
            weight = float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '').isdigit() else 1.0
            category = parts[2].strip() if len(parts) > 2 else 'general'
            
            self.weighted_terms[term] = weight
            self.category_terms[category].add(term)
        else:
            term = entry
            weight = 1.0
            category = 'general'
        
        # Clean term
        term = re.sub(r'[^\w\s\-]', '', term)
        
        if not term:
            return
        
        # Categorize by length and structure
        tokens = term.split()
        
        if len(tokens) == 1:
            # Single word
            self.single_terms.add(term)
        elif len(tokens) == 2:
            # Bigram
            self.bigrams.add(term)
            self.phrase_terms.add(term)
        elif len(tokens) == 3:
            # Trigram
            self.trigrams.add(term)
            self.phrase_terms.add(term)
        else:
            # Longer phrase
            self.phrase_terms.add(term)
        
        # Add to category
        self.category_terms[category].add(term)
    
    def _create_fallback_lexicon(self):
        """Create a minimal fallback lexicon if main file not found."""
        fallback_terms = [
            "white supremacy:2.0:hate", "racial superiority:2.0:hate",
            "ethnic cleansing:3.0:violence", "race war:2.5:violence",
            "great replacement:2.0:conspiracy", "deep state:1.5:conspiracy",
            "storm front:2.0:platform", "fourteen words:2.5:symbol",
            "blood and soil:2.0:ideology", "day of the rope:3.0:violence",
            "accelerationism:2.0:ideology", "boogaloo:1.5:movement",
            "patriot front:1.5:group", "proud boys:1.5:group"
        ]
        
        for entry in fallback_terms:
            self._parse_lexicon_entry(entry, 0)
        
        print(f"Created fallback lexicon with {len(fallback_terms)} entries")
    
    def _build_semantic_embeddings(self):
        """Build semantic embeddings for lexicon terms."""
        if not self.semantic_enabled:
            return
        
        print("Building semantic embeddings for lexicon terms...")
        all_terms = list(self.single_terms.union(self.phrase_terms))
        
        if not all_terms:
            return
        
        try:
            self.term_embeddings = self.semantic_model.encode(all_terms)
            print(f"Built embeddings for {len(all_terms)} terms")
        except Exception as e:
            print(f"Failed to build semantic embeddings: {e}")
            self.semantic_enabled = False
    
    def preprocess_text(self, text: str, preserve_phrases: bool = True) -> Dict[str, any]:
        """
        Enhanced text preprocessing with phrase preservation.
        
        Args:
            text: Input text
            preserve_phrases: Whether to preserve phrase boundaries
            
        Returns:
            Dictionary with various text representations
        """
        # Original text for context
        original = text
        
        # Normalize text
        text_lower = text.lower()
        
        # Clean text while preserving structure
        text_clean = re.sub(r'[^\w\s\-]', ' ', text_lower)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        # Tokenization
        tokens = word_tokenize(text_clean)
        sentences = sent_tokenize(original)
        
        # Remove stopwords for some analyses
        content_tokens = [token for token in tokens 
                         if token not in self.stop_words and len(token) > 2]
        
        # Generate n-grams
        token_bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        token_trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
        
        return {
            'original': original,
            'cleaned': text_clean,
            'tokens': tokens,
            'content_tokens': content_tokens,
            'sentences': sentences,
            'bigrams': token_bigrams,
            'trigrams': token_trigrams,
            'word_count': len(tokens),
            'sentence_count': len(sentences)
        }
    
    def exact_match_score(self, text_data: Dict) -> Tuple[float, List[MatchResult]]:
        """Exact matching for single terms and phrases."""
        matches = []
        text_clean = text_data['cleaned']
        tokens = text_data['tokens']
        
        # Single term matches
        for i, token in enumerate(tokens):
            if token in self.single_terms:
                weight = self.weighted_terms.get(token, 1.0)
                context = self._get_context(text_data['original'], token, i)
                
                matches.append(MatchResult(
                    term=token,
                    positions=[i],
                    context=context,
                    confidence=1.0,
                    match_type='exact'
                ))
        
        # Phrase matches (bigrams and trigrams)
        for bigram in text_data['bigrams']:
            if bigram in self.bigrams:
                weight = self.weighted_terms.get(bigram, 1.0)
                context = self._get_phrase_context(text_data['original'], bigram)
                
                matches.append(MatchResult(
                    term=bigram,
                    positions=[],
                    context=context,
                    confidence=1.0,
                    match_type='phrase'
                ))
        
        for trigram in text_data['trigrams']:
            if trigram in self.trigrams:
                weight = self.weighted_terms.get(trigram, 1.0)
                context = self._get_phrase_context(text_data['original'], trigram)
                
                matches.append(MatchResult(
                    term=trigram,
                    positions=[],
                    context=context,
                    confidence=1.0,
                    match_type='phrase'
                ))
        
        # Calculate score
        if text_data['word_count'] == 0:
            score = 0.0
        else:
            total_weight = sum(self.weighted_terms.get(match.term, 1.0) for match in matches)
            score = min(total_weight / text_data['word_count'], 1.0)
        
        return score, matches
    
    def partial_match_score(self, text_data: Dict) -> Tuple[float, List[MatchResult]]:
        """Partial/fuzzy matching for terms."""
        matches = []
        text_clean = text_data['cleaned']
        
        # Partial matches for single terms
        for term in self.single_terms:
            if len(term) > 4:  # Only for longer terms
                # Check for partial matches
                pattern = self._create_fuzzy_pattern(term)
                regex_matches = re.finditer(pattern, text_clean)
                
                for match in regex_matches:
                    context = self._get_substring_context(text_data['original'], 
                                                        match.start(), match.end())
                    confidence = self._calculate_partial_confidence(term, match.group())
                    
                    if confidence > 0.6:  # Minimum confidence threshold
                        matches.append(MatchResult(
                            term=term,
                            positions=[],
                            context=context,
                            confidence=confidence,
                            match_type='partial'
                        ))
        
        # Calculate weighted score
        if text_data['word_count'] == 0:
            score = 0.0
        else:
            total_weight = sum(
                self.weighted_terms.get(match.term, 1.0) * match.confidence 
                for match in matches
            )
            score = min(total_weight / text_data['word_count'], 1.0)
        
        return score, matches
    
    def semantic_match_score(self, text_data: Dict) -> Tuple[float, List[MatchResult]]:
        """Semantic similarity matching."""
        if not self.semantic_enabled:
            return 0.0, []
        
        matches = []
        sentences = text_data['sentences']
        
        try:
            # Encode input sentences
            sentence_embeddings = self.semantic_model.encode(sentences)
            
            # Compare with lexicon embeddings
            similarities = np.dot(sentence_embeddings, self.term_embeddings.T)
            
            all_terms = list(self.single_terms.union(self.phrase_terms))
            
            for sent_idx, sentence in enumerate(sentences):
                for term_idx, term in enumerate(all_terms):
                    similarity = similarities[sent_idx][term_idx]
                    
                    if similarity > self.semantic_threshold:
                        matches.append(MatchResult(
                            term=term,
                            positions=[],
                            context=sentence,
                            confidence=float(similarity),
                            match_type='semantic'
                        ))
            
            # Calculate score
            if len(sentences) == 0:
                score = 0.0
            else:
                total_weight = sum(
                    self.weighted_terms.get(match.term, 1.0) * match.confidence 
                    for match in matches
                )
                score = min(total_weight / len(sentences), 1.0)
            
        except Exception as e:
            print(f"Semantic matching error: {e}")
            score = 0.0
            matches = []
        
        return score, matches
    
    def context_aware_score(self, text_data: Dict, matches: List[MatchResult]) -> Dict[str, float]:
        """Context-aware scoring considering surrounding words and sentence structure."""
        context_scores = {}
        
        # Analyze sentence-level context
        sentence_scores = []
        for sentence in text_data['sentences']:
            sentence_matches = [m for m in matches if sentence.lower() in m.context.lower()]
            if sentence_matches:
                # Score based on match density and weights
                match_weight = sum(self.weighted_terms.get(m.term, 1.0) for m in sentence_matches)
                sentence_length = len(sentence.split())
                sentence_score = min(match_weight / max(sentence_length, 1), 1.0)
                sentence_scores.append(sentence_score)
        
        context_scores['sentence_max'] = max(sentence_scores) if sentence_scores else 0.0
        context_scores['sentence_mean'] = np.mean(sentence_scores) if sentence_scores else 0.0
        
        # Category-based analysis
        category_matches = defaultdict(list)
        for match in matches:
            for category, terms in self.category_terms.items():
                if match.term in terms:
                    category_matches[category].append(match)
        
        context_scores['category_diversity'] = len(category_matches)
        context_scores['dominant_category'] = max(
            [(cat, len(matches)) for cat, matches in category_matches.items()],
            key=lambda x: x[1], default=('none', 0)
        )[0]
        
        return context_scores
    
    def get_comprehensive_score(self, text: str) -> ExtremistScore:
        """
        Get comprehensive extremist lexicon scores using all methods.
        
        Args:
            text: Input text
            
        Returns:
            ExtremistScore object with detailed analysis
        """
        # Preprocess text
        text_data = self.preprocess_text(text)
        
        # Get scores from different methods
        exact_score, exact_matches = self.exact_match_score(text_data)
        partial_score, partial_matches = self.partial_match_score(text_data)
        semantic_score, semantic_matches = self.semantic_match_score(text_data)
        
        # Combine all matches
        all_matches = exact_matches + partial_matches + semantic_matches
        
        # Context analysis
        context_analysis = self.context_aware_score(text_data, all_matches)
        
        # Calculate various scores
        scores = {
            'exact_match': exact_score,
            'partial_match': partial_score,
            'semantic_match': semantic_score,
            'density': len(all_matches) / max(text_data['word_count'], 1),
            'weighted_density': sum(
                self.weighted_terms.get(m.term, 1.0) * m.confidence 
                for m in all_matches
            ) / max(text_data['word_count'], 1),
            'context_sentence_max': context_analysis['sentence_max'],
            'category_diversity': context_analysis['category_diversity'] / max(len(self.category_terms), 1)
        }
        
        # Composite score with weighted combination
        weights = {
            'exact_match': 0.3,
            'partial_match': 0.2,
            'semantic_match': 0.2,
            'weighted_density': 0.2,
            'context_sentence_max': 0.1
        }
        
        scores['composite'] = sum(
            scores[key] * weight for key, weight in weights.items()
        )
        
        # Risk level assessment
        risk_level = self._assess_risk_level(scores['composite'], len(all_matches), 
                                           context_analysis)
        
        return ExtremistScore(
            text=text,
            scores=scores,
            matches=all_matches,
            risk_level=risk_level,
            context_analysis=context_analysis
        )
    
    def _create_fuzzy_pattern(self, term: str) -> str:
        """Create regex pattern for fuzzy matching."""
        # Allow for small character variations
        chars = list(term)
        pattern_parts = []
        
        for char in chars:
            if char.isalpha():
                pattern_parts.append(f"[{char}{char.upper()}]{{1,2}}")
            else:
                pattern_parts.append(re.escape(char))
        
        return ''.join(pattern_parts)
    
    def _calculate_partial_confidence(self, original: str, matched: str) -> float:
        """Calculate confidence for partial matches."""
        # Simple Levenshtein-based confidence
        import difflib
        similarity = difflib.SequenceMatcher(None, original, matched).ratio()
        return similarity
    
    def _get_context(self, text: str, term: str, position: int, window: int = 30) -> str:
        """Get context around a matched term."""
        tokens = text.split()
        start = max(0, position - window)
        end = min(len(tokens), position + window + 1)
        context_tokens = tokens[start:end]
        return ' '.join(context_tokens)
    
    def _get_phrase_context(self, text: str, phrase: str, window: int = 30) -> str:
        """Get context around a matched phrase."""
        text_lower = text.lower()
        phrase_pos = text_lower.find(phrase.lower())
        
        if phrase_pos == -1:
            return phrase
        
        start = max(0, phrase_pos - window)
        end = min(len(text), phrase_pos + len(phrase) + window)
        return text[start:end]
    
    def _get_substring_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Get context around a substring match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _assess_risk_level(self, composite_score: float, match_count: int, 
                          context_analysis: Dict) -> str:
        """Assess overall risk level based on multiple factors."""
        if composite_score >= 0.7 or match_count >= 10:
            return 'critical'
        elif composite_score >= 0.5 or match_count >= 5:
            return 'high'
        elif composite_score >= 0.2 or match_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def batch_analyze(self, texts: List[str]) -> List[ExtremistScore]:
        """Analyze multiple texts efficiently."""
        return [self.get_comprehensive_score(text) for text in texts]
    
    def get_lexicon_stats(self) -> Dict[str, any]:
        """Get statistics about the loaded lexicon."""
        return {
            'total_terms': len(self.single_terms) + len(self.phrase_terms),
            'single_terms': len(self.single_terms),
            'phrase_terms': len(self.phrase_terms),
            'bigrams': len(self.bigrams),
            'trigrams': len(self.trigrams),
            'categories': list(self.category_terms.keys()),
            'weighted_terms': len(self.weighted_terms),
            'semantic_enabled': self.semantic_enabled
        }


# Usage example and testing
if __name__ == "__main__":
    # Initialize enhanced scorer
    print("Initializing Enhanced Extremist Lexicon Scorer...")
    scorer = EnhancedExtremistLexiconScorer(
        lexicon_path='data/extremism_lexicon.txt',
        enable_semantic_matching=True,
        semantic_threshold=0.7
    )
    
    # Display lexicon statistics
    print(f"\nLexicon Statistics: {scorer.get_lexicon_stats()}")
    
    # Test texts with varying levels of extremist content
    test_texts = [
        "This is a normal message about community building and cooperation.",
        
        "The great replacement theory suggests demographic changes threaten society.",
        
        "We must secure a future for our people and defend against cultural marxism.",
        
        "Accelerationism and race war concepts are dangerous extremist ideologies.",
        
        "Building bridges between communities promotes understanding and peace.",
        
        "The deep state conspiracy involves government manipulation and control."
    ]
    
    print("\n" + "="*80)
    print("ENHANCED EXTREMIST LEXICON ANALYSIS RESULTS")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Input: '{text}'")
        
        result = scorer.get_comprehensive_score(text)
        
        print(f"Risk Level: {result.risk_level.upper()}")
        print(f"Composite Score: {result.scores['composite']:.3f}")
        print(f"Matches Found: {len(result.matches)}")
        
        # Show key scores
        key_scores = ['exact_match', 'partial_match', 'semantic_match', 'weighted_density']
        score_str = ", ".join([f"{k}: {result.scores[k]:.3f}" for k in key_scores])
        print(f"Key Scores: {score_str}")
        
        # Show matches if any
        if result.matches:
            print("Matched Terms:")
            for match in result.matches[:5]:  # Show first 5 matches
                print(f"  - '{match.term}' ({match.match_type}, conf: {match.confidence:.2f})")
        
        print("-" * 60)
    
    print(f"\nAnalysis complete. Processed {len(test_texts)} texts.")
    print("Enhanced features: ✓ Comprehensive lexicon ✓ N-gram matching ✓ Semantic similarity ✓ Context awareness")