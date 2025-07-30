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
        
        self.stop_words = set(stopwords.words('english'))
    
    def detect_imperative_sentences(self, text: str) -> Dict[str, float]:
        """
        Detect imperative sentences (commands) in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with imperative detection scores
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return {'imperative_ratio': 0.0, 'imperative_count': 0}
        
        imperative_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Tokenize and get POS tags
            tokens = word_tokenize(sentence.lower())
            pos_tags = pos_tag(tokens)
            
            # Check for imperative patterns
            is_imperative = False
            
            # Pattern 1: Starts with verb (base form)
            if pos_tags and pos_tags[0][1] in ['VB', 'VBP']:
                is_imperative = True
            
            # Pattern 2: Contains action verbs with urgency
            action_verb_present = any(token in self.action_verbs for token in tokens)
            urgency_present = any(token in self.urgency_words for token in tokens)
            
            if action_verb_present and urgency_present:
                is_imperative = True
            
            # Pattern 3: Modal + action verb
            modal_present = any(token in self.modal_necessity for token in tokens)
            if modal_present and action_verb_present:
                is_imperative = True
            
            # Pattern 4: Direct address + action
            direct_address_present = any(token in self.direct_address for token in tokens)
            if direct_address_present and action_verb_present:
                is_imperative = True
            
            if is_imperative:
                imperative_count += 1
        
        return {
            'imperative_ratio': imperative_count / len(sentences),
            'imperative_count': imperative_count,
            'total_sentences': len(sentences)
        }
    
    def detect_call_to_action_patterns(self, text: str) -> Dict[str, float]:
        """
        Detect call-to-action patterns in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with call-to-action scores
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {
                'action_verb_density': 0.0,
                'urgency_score': 0.0,
                'collective_score': 0.0,
                'necessity_score': 0.0,
                'social_media_cta': 0.0
            }
        
        # Count different types of CTA indicators
        action_verb_count = sum(1 for word in words if word in self.action_verbs)
        urgency_count = sum(1 for word in words if word in self.urgency_words)
        collective_count = sum(1 for word in words if word in self.collective_pronouns)
        necessity_count = sum(1 for word in words if word in self.modal_necessity)
        social_media_count = sum(1 for word in words if word in self.social_media_actions)
        
        text_length = len(words)
        
        return {
            'action_verb_density': action_verb_count / text_length,
            'urgency_score': urgency_count / text_length,
            'collective_score': collective_count / text_length,
            'necessity_score': necessity_count / text_length,
            'social_media_cta': social_media_count / text_length
        }
    
    def analyze_persuasion_techniques(self, text: str) -> Dict[str, float]:
        """
        Analyze persuasion techniques that might indicate calls to action.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with persuasion technique scores
        """
        persuasion_score = 0.0
        
        # Question patterns (rhetorical questions)
        question_patterns = [
            r'what are you waiting for\?',
            r'why not\?',
            r'isn\'t it time\?',
            r'don\'t you think\?',
            r'how long will\?'
        ]
        
        question_score = 0.0
        for pattern in question_patterns:
            if re.search(pattern, text.lower()):
                question_score += 0.2
        
        # Emotional appeal patterns
        emotional_patterns = [
            r'imagine if',
            r'think about',
            r'remember when',
            r'how would you feel',
            r'what if'
        ]
        
        emotional_score = 0.0
        for pattern in emotional_patterns:
            if re.search(pattern, text.lower()):
                emotional_score += 0.15
        
        # Authority/credibility indicators
        authority_words = ['expert', 'study', 'research', 'proven', 'fact', 'truth', 'evidence']
        authority_count = sum(1 for word in text.lower().split() if word in authority_words)
        authority_score = min(authority_count * 0.1, 0.3)
        
        # Social proof indicators
        social_proof_words = ['everyone', 'thousands', 'millions', 'people', 'join', 'popular']
        social_proof_count = sum(1 for word in text.lower().split() if word in social_proof_words)
        social_proof_score = min(social_proof_count * 0.05, 0.2)
        
        return {
            'rhetorical_questions': min(question_score, 1.0),
            'emotional_appeal': min(emotional_score, 1.0),
            'authority_indicators': authority_score,
            'social_proof': social_proof_score,
            'overall_persuasion': min((question_score + emotional_score + authority_score + social_proof_score) / 4, 1.0)
        }
    
    def detect_mobilization_language(self, text: str) -> Dict[str, float]:
        """
        Detect language patterns that indicate mobilization or organizing.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with mobilization scores
        """
        mobilization_words = {
            'organize', 'mobilize', 'unite', 'gather', 'assemble', 'rally',
            'march', 'protest', 'demonstrate', 'movement', 'revolution',
            'uprising', 'resistance', 'rebellion', 'fight', 'struggle'
        }
        
        coordination_words = {
            'meet', 'plan', 'coordinate', 'schedule', 'arrange', 'prepare',
            'when', 'where', 'how', 'contact', 'reach', 'connect'
        }
        
        recruitment_words = {
            'join', 'recruit', 'invite', 'welcome', 'need', 'want',
            'volunteer', 'help', 'support', 'participate', 'contribute'
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {'mobilization_score': 0.0, 'coordination_score': 0.0, 'recruitment_score': 0.0}
        
        mobilization_count = sum(1 for word in words if word in mobilization_words)
        coordination_count = sum(1 for word in words if word in coordination_words)
        recruitment_count = sum(1 for word in words if word in recruitment_words)
        
        text_length = len(words)
        
        return {
            'mobilization_score': mobilization_count / text_length,
            'coordination_score': coordination_count / text_length,
            'recruitment_score': recruitment_count / text_length,
            'overall_mobilization': (mobilization_count + coordination_count + recruitment_count) / (text_length * 3)
        }
    
    def get_comprehensive_action_score(self, text: str) -> Dict[str, Union[float, Dict]]:
        """
        Get comprehensive call-to-action analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with comprehensive action analysis
        """
        results = {}
        
        # Get all component scores
        results['imperative'] = self.detect_imperative_sentences(text)
        results['cta_patterns'] = self.detect_call_to_action_patterns(text)
        results['persuasion'] = self.analyze_persuasion_techniques(text)
        results['mobilization'] = self.detect_mobilization_language(text)
        
        # Calculate composite scores
        results['action_intensity'] = (
            results['imperative']['imperative_ratio'] * 0.3 +
            results['cta_patterns']['action_verb_density'] * 0.2 +
            results['cta_patterns']['urgency_score'] * 0.2 +
            results['persuasion']['overall_persuasion'] * 0.15 +
            results['mobilization']['overall_mobilization'] * 0.15
        )
        
        results['collective_action_score'] = (
            results['cta_patterns']['collective_score'] * 0.4 +
            results['mobilization']['mobilization_score'] * 0.3 +
            results['mobilization']['recruitment_score'] * 0.3
        )
        
        results['urgency_intensity'] = (
            results['cta_patterns']['urgency_score'] * 0.6 +
            results['imperative']['imperative_ratio'] * 0.4
        )
        
        # Overall call-to-action score
        results['overall_cta_score'] = (
            results['action_intensity'] * 0.4 +
            results['collective_action_score'] * 0.3 +
            results['urgency_intensity'] * 0.3
        )
        
        return results

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    action_analyzer = ActionIndicationAnalyzer()
    
    # Test texts
    test_texts = [
        "Please consider joining our peaceful community event next week.",
        "We must act now! Unite and fight against this injustice immediately!",
        "Everyone should organize and mobilize before it's too late. Join us!",
        "I think we should have a discussion about this issue sometime.",
        "URGENT: Share this message! We need to stop them NOW! Act before tomorrow!"
    ]
    
    # Analyze action indication
    for text in test_texts:
        analysis = action_analyzer.get_comprehensive_action_score(text)
        print(f"Text: {text}")
        print(f"Overall CTA Score: {analysis['overall_cta_score']:.3f}")
        print(f"Action Intensity: {analysis['action_intensity']:.3f}")
        print(f"Collective Action: {analysis['collective_action_score']:.3f}")
        print(f"Urgency: {analysis['urgency_intensity']:.3f}")
        print("-" * 60)