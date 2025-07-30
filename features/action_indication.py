"""
Action Indication and Call-to-Action Detection Module
This module detects calls to action and action-oriented language in text.
"""

import re
from typing import Dict, List, Set, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class ActionIndicationAnalyzer:
    def __init__(self):
        """Initialize action indication analyzer with predefined lexicons."""
        
        # Direct action verbs (imperative mood indicators)
        self.action_verbs = {
            'attack', 'fight', 'destroy', 'eliminate', 'remove', 'stop', 'end',
            'join', 'unite', 'organize', 'mobilize', 'gather', 'assemble',
            'march', 'protest', 'demonstrate', 'boycott', 'strike',
            'resist', 'oppose', 'confront', 'challenge', 'defeat',
            'build', 'create', 'establish', 'form', 'start', 'begin',
            'share', 'spread', 'tell', 'inform', 'alert', 'warn',
            'vote', 'elect', 'support', 'defend', 'protect', 'save',
            'act', 'move', 'go', 'come', 'take', 'make', 'do'
        }
        
        # Urgency indicators
        self.urgency_words = {
            'now', 'immediately', 'urgent', 'quickly', 'asap', 'today',
            'tonight', 'tomorrow', 'soon', 'before', 'deadline', 'time',
            'emergency', 'crisis', 'critical', 'important', 'essential'
        }
        
        # Collective action indicators
        self.collective_pronouns = {
            'we', 'us', 'our', 'everyone', 'all', 'together', 'united',
            'people', 'citizens', 'community', 'nation', 'patriots',
            'brothers', 'sisters', 'comrades', 'allies', 'friends'
        }
        
        # Modal verbs indicating necessity/obligation
        self.modal_necessity = {
            'must', 'should', 'need', 'have', 'ought', 'required',
            'necessary', 'essential', 'vital', 'crucial', 'important'
        }
        
        # Direct address indicators
        self.direct_address = {
            'you', 'your', 'yourself', 'listen', 'hear', 'remember',
            'understand', 'realize', 'know', 'see', 'look', 'watch'
        }
        
        # Platform-specific action words
        self.social_media_actions = {
            'share', 'retweet', 'like', 'follow', 'subscribe', 'comment',
            'post', 'upload', 'tag', 'mention', 'dm', 'message',
            'viral', 'trending', 'hashtag', 'thread', 'story'
        }
        
        # Stop words for filtering if needed
        self.stop_words = set(stopwords.words('english'))

    def detect_imperative_sentences(self, text: str) -> Dict[str, float]:
        """
        Detect imperative sentences (commands) in text.

        Args:
            text: Input text

        Returns:
            Dictionary with:
              - total_sentences: number of non-empty sentences
              - imperative_count: how many were flagged imperative
              - imperative_ratio: imperative_count / total_sentences
        """
        # Split into sentences and filter out blanks
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        total_sentences = len(sentences)
        if total_sentences == 0:
            return {'total_sentences': 0, 'imperative_count': 0, 'imperative_ratio': 0.0}

        imperative_count = 0

        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            pos_tags = pos_tag(tokens)
            is_imperative = False

            # Pattern 1: Sentence starts with a base-form verb
            if pos_tags and pos_tags[0][1] in ['VB', 'VBP']:
                is_imperative = True

            # Pattern 2: Contains both an action verb and an urgency indicator
            action_verb_present = any(tok in self.action_verbs for tok in tokens)
            urgency_present     = any(tok in self.urgency_words for tok in tokens)
            if action_verb_present and urgency_present:
                is_imperative = True

            # Pattern 3: Modal necessity + action verb
            modal_present = any(tok in self.modal_necessity for tok in tokens)
            if modal_present and action_verb_present:
                is_imperative = True

            # Pattern 4: Direct address + action verb
            direct_address_present = any(tok in self.direct_address for tok in tokens)
            if direct_address_present and action_verb_present:
                is_imperative = True

            if is_imperative:
                imperative_count += 1

        return {
            'total_sentences': total_sentences,
            'imperative_count': imperative_count,
            'imperative_ratio': imperative_count / total_sentences
        }