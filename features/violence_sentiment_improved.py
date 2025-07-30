'''
Enhanced Violence Sentiment Detection Module
This module provides multiple approaches to detect violent content in text
with advanced ML+lexicon hybrid approach.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import re
from typing import Dict, List, Union, Tuple, Optional
import logging
import spacy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EnhancedViolenceSentimentAnalyzer:
    def __init__(self, use_advanced_nlp: bool = True):
        """Initialize enhanced violence sentiment analyzer."""
        self.model = None
        self.ensemble_model = None
        self.vectorizers = {}
        self.is_trained = False
        self.use_advanced_nlp = use_advanced_nlp
        
        # Try to load spaCy model for advanced NLP
        self.nlp = None
        if use_advanced_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_advanced_nlp = False
        
        # Enhanced violence lexicons with severity weights
        self.explicit_violence_words = {
            'kill': 3.0, 'murder': 3.0, 'assassinate': 3.0, 'execute': 3.0,
            'slaughter': 3.0, 'massacre': 3.0, 'genocide': 3.0, 'torture': 3.0,
            'mutilate': 3.0, 'dismember': 3.0, 'behead': 3.0, 'lynch': 3.0,
            'shoot': 2.5, 'stab': 2.5, 'bomb': 2.5, 'explode': 2.5,
            'attack': 2.5, 'assault': 2.5, 'rape': 2.5, 'molest': 2.5,
            'strangle': 2.5, 'suffocate': 2.5, 'drown': 2.5, 'poison': 2.5,
            'beat': 2.0, 'hit': 2.0, 'punch': 2.0, 'kick': 2.0, 'slap': 2.0,
            'crush': 2.0, 'smash': 2.0, 'destroy': 2.0, 'annihilate': 2.0,
            'eliminate': 2.0, 'terminate': 2.0, 'exterminate': 2.0,
            'fight': 1.5, 'combat': 1.5, 'battle': 1.5, 'war': 1.5,
            'strike': 1.5, 'wound': 1.5, 'injure': 1.5, 'harm': 1.5
        }
        
        self.implicit_violence_words = {
            'threaten': 2.5, 'intimidate': 2.5, 'menace': 2.5, 'terrorize': 2.5,
            'blackmail': 2.5, 'extort': 2.5, 'coerce': 2.5, 'bully': 2.5,
            'revenge': 2.0, 'retaliation': 2.0, 'payback': 2.0, 'vengeance': 2.0,
            'consequences': 2.0, 'warning': 2.0, 'ultimatum': 2.0,
            'force': 1.5, 'pressure': 1.5, 'compel': 1.5, 'dominate': 1.5,
            'overpower': 1.5, 'subdue': 1.5, 'suppress': 1.5, 'oppress': 1.5,
            'control': 1.5, 'manipulate': 1.5, 'exploit': 1.5
        }
        
        # Enhanced weapons categorization with specificity scores
        self.weapons_categories = {
            'firearms': {'gun':2.5,'rifle':2.5,'pistol':2.5,'shotgun':2.5,'revolver':2.5,'ak47':3.0,'ar15':3.0,'sniper':3.0,'machine gun':3.0,'assault rifle':3.0},
            'bladed_weapons': {'knife':2.0,'blade':2.0,'sword':2.0,'machete':2.5,'dagger':2.0,'razor':2.0,'scalpel':2.0,'katana':2.0},
            'explosives': {'bomb':3.0,'explosive':3.0,'grenade':3.0,'dynamite':3.0,'c4':3.0,'ied':3.0,'missile':3.0,'rocket':3.0,'landmine':3.0,'tnt':3.0},
            'blunt_weapons': {'bat':1.5,'club':1.5,'hammer':1.5,'axe':2.0,'crowbar':1.5,'pipe':1.5,'brick':1.5,'rock':1.0},
            'chemical_biological': {'poison':2.5,'acid':2.5,'anthrax':3.0,'ricin':3.0,'sarin':3.0,'cyanide':2.5,'arsenic':2.5}
        }
        
        self.violence_targets = {
            'specific_groups': {'children':3.0,'kids':3.0,'baby':3.0,'infant':3.0,'elderly':2.5,'disabled':2.5,'women':2.0,'minorities':2.5},
            'general_targets': {'person':1.5,'people':1.5,'individual':1.5,'group':1.5,'crowd':2.0,'family':2.0,'civilians':2.5,'innocents':2.5,'enemy':2.0,'opponent':1.5,'rival':1.5,'target':2.0,'victim':2.0,'hostage':2.5}
        }
        
        self.intensity_modifiers = {'extremely':2.5,'very':1.8,'really':1.5,'quite':1.3,'brutally':3.0,'violently':2.5,'aggressively':2.0,'mercilessly':2.8,'ruthlessly':2.5,'savagely':2.8,'viciously':2.5,'cruelly':2.3,'horribly':2.0,'painfully':2.0,'slowly':1.8,'quickly':1.3}
        
        self.threat_patterns = {
            'direct_threat': [r"i will .* you", r"going to .* you", r"i'll .* you", r"i'm gonna .* you", r"you're .* dead", r"you will .* die", r"i'm coming for you"],
            'conditional_threat': [r"if you don't .*, i will", r"unless you .*, you'll", r"or else i'll", r"better .* or", r"do .* or i'll"],
            'temporal_threat': [r"by \w+day .* or", r"in \d+ \w+ .* will", r"before \w+ .* die", r"24 hours", r"final warning", r"last chance"],
            'location_specific': [r"when you get home", r"at your house", r"where you live", r"find you", r"track you down", r"hunt you down"]
        }
        
        self.context_reducers = {
            'fictional': ['movie','game','book','story','fiction','novel','film'],
            'sports': ['boxing','mma','wrestling','football','hockey'],
            'metaphorical': ['kill it','crushed it','destroyed the competition'],
            'historical': ['war','battle','historical','documentary']
        }

    def load_violence_dataset(self, dataset_path: str) -> Tuple[List[str], List[int]]:
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.tdf'):
                df = pd.read_csv(dataset_path, sep='\t')
            else:
                raise ValueError("Dataset must be CSV or TDF format")
            text_cols = ['text','content','message','tweet','comment']
            label_cols = ['violence','violent','label','target','class']
            text_col = next((c for c in text_cols if c in df.columns), None)
            label_col = next((c for c in label_cols if c in df.columns), None)
            if not text_col or not label_col:
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError("Could not find text and label columns")
            texts = df[text_col].astype(str).tolist()
            labels = df[label_col].astype(int).tolist()
            print(f"Loaded {len(texts)} samples from dataset")
            print(f"Violence distribution: {np.bincount(labels)}")
            return texts, labels
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self._create_enhanced_sample_dataset()

    def _create_enhanced_sample_dataset(self) -> Tuple[List[str], List[int]]:
        sample_data = [
            ("I love spending time with my family",0), ("We should have a peaceful discussion",0),
            ("The movie had great action scenes with explosions",0), ("Let's fight against injustice peacefully",0),
            ("The game was intense but fun",0), ("Community building is important",0),
            ("I'm going to kill it at this presentation",0), ("The boxing match was brutal but fair",0),
            ("I'm going to kill you if you don't stop",1), ("He threatened to beat me up",1),
            ("I will destroy anyone who gets in my way",1), ("We need to eliminate our enemies violently",1),
            ("I'll shoot you if you come near me",1), ("You're dead meat when I find you",1),
            ("I'm going to torture you slowly",1), ("You better watch your back",1),
            ("Something bad might happen to you",1), ("I know where you live and work",1),
            ("You'll regret crossing me",1), ("Final warning - back off or else",1),
            ("I have a gun and I'm not afraid to use it",1), ("The bomb will go off at midnight",1),
            ("I'll stab you with this knife",1), ("Poison in your drink tonight",1)
        ]
        texts, labels = zip(*sample_data)
        return list(texts), list(labels)

    def train_enhanced_model(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
        tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=5000)
        tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=10000, stop_words='english', lowercase=True)
        count_vec = CountVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')
        X_train_tfidf_char = tfidf_char.fit_transform(X_train)
        X_train_tfidf_word = tfidf_word.fit_transform(X_train)
        X_train_count = count_vec.fit_transform(X_train)
        X_test_tfidf_char = tfidf_char.transform(X_test)
        X_test_tfidf_word = tfidf_word.transform(X_test)
        X_test_count = count_vec.transform(X_test)
        X_train_lexicon = np.array([self._extract_lexicon_features(t) for t in X_train])
        X_test_lexicon = np.array([self._extract_lexicon_features(t) for t in X_test])
        from scipy.sparse import hstack
        X_train_combined = hstack([X_train_tfidf_word, X_train_tfidf_char, X_train_count, X_train_lexicon])
        X_test_combined = hstack([X_test_tfidf_word, X_test_tfidf_char, X_test_count, X_test_lexicon])
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw = {i: class_weights[i] for i in range(len(class_weights))}
        models = [
            ('lr', LogisticRegression(class_weight=cw, random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42)),
            ('nb', MultinomialNB()),
            ('svm', SVC(class_weight=cw, probability=True, random_state=42))
        ]
        self.ensemble_model = VotingClassifier(estimators=models, voting='soft')
        self.ensemble_model.fit(X_train_combined, y_train)
        self.vectorizers = {'tfidf_word': tfidf_word,'tfidf_char': tfidf_char,'count': count_vec}
        y_pred = self.ensemble_model.predict(X_test_combined)
        y_prob = self.ensemble_model.predict_proba(X_test_combined)[:,1]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        self.is_trained = True
        print("Enhanced ensemble model trained successfully:")
        for k,v in metrics.items(): print(f"{k}: {v:.3f}")
        return metrics

    def _extract_lexicon_features(self, text: str) -> np.ndarray:
        text_lower, words = text.lower(), text.lower().split()
        if not words: return np.zeros(15)
        explicit_score = sum(self.explicit_violence_words.get(w,0) for w in words)/len(words)
        implicit_score = sum(self.implicit_violence_words.get(w,0) for w in words)/len(words)
        weapon_scores = [sum(cat.get(w,0) for w in words)/len(words) for cat in self.weapons_categories.values()]
        target_scores = [sum(cat.get(w,0) for w in words)/len(words) for cat in self.violence_targets.values()]
        pattern_scores = [sum(1 for pat in pats if re.search(pat, text_lower))/len(pats) for pats in self.threat_patterns.values()]
        intensity = max(self.intensity_modifiers.get(w,1.0) for w in words)
        context_reduction = sum(0.2 for reducers in self.context_reducers.values() if any(r in text_lower for r in reducers))
        feats = [explicit_score, implicit_score, intensity, context_reduction] + weapon_scores + target_scores + pattern_scores
        return np.array(feats[:15])

    def detect_implicit_violence(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        indicators = {'veiled_threats':0.0,'power_dynamics':0.0,'emotional_manipulation':0.0,'intimidation_tactics':0.0}
        for pat in [r'it would be a shame if',r'accidents happen',r'be careful',r'watch yourself',r'you never know',r'things could go wrong']:
            if re.search(pat, text_lower): indicators['veiled_threats']+=0.3
        pw_words=['control','power','dominant','submit','obey','comply']
        indicators['power_dynamics']=min(sum(1 for w in pw_words if w in text_lower)*0.2,1.0)
        mp=['you made me','look what you did','this is your fault']
        indicators['emotional_manipulation']=min(sum(1 for p in mp if p in text_lower)*0.3,1.0)
        iw=['reputation','consequences','regret','sorry']
        indicators['intimidation_tactics']=min(sum(1 for w in iw if w in text_lower)*0.25,1.0)
        return indicators

    def analyze_weapon_method_specificity(self, text: str) -> Dict[str, float]:
        text_lower, words = text.lower(), text.lower().split()
        spec={'weapon_specificity':0.0,'method_detail':0.0,'planning_indicators':0.0,'capability_claims':0.0}
        max_ws=0.0
        for weapons in self.weapons_categories.values():
            for w,score in weapons.items():
                if w in text_lower: max_ws=max(max_ws, score/3.0)
        spec['weapon_specificity']=max_ws
        md=['step by step','first i will','then i will','finally','slowly','carefully','precisely','exactly how']
        spec['method_detail']=min(sum(1 for i in md if i in text_lower)*0.2,1.0)
        pw=['plan','prepare','ready','when','where','how']
        spec['planning_indicators']=min(sum(1 for w in pw if w in words)*0.15,1.0)
        cp=['i have','i own','i can get','i know how']
        spec['capability_claims']=min(sum(1 for p in cp if p in text_lower)*0.25,1.0)
        return spec

    def enhanced_context_analysis(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        ca={'temporal_urgency':0.0,'spatial_context':0.0,'emotional_state':0.0,'social_context':0.0,'intent_clarity':0.0}
        ca['temporal_urgency']=min(sum(1 for w in ['now','immediately','tonight','today','asap'] if w in text_lower)*0.3,1.0)
        ca['spatial_context']=min(sum(1 for w in ['here','there','home','work','school','address'] if w in text_lower)*0.2,1.0)
        ca['emotional_state']=min(sum(1 for w in ['angry','furious','rage','mad','pissed','hate'] if w in text_lower)*0.25,1.0)
        ca['social_context']=min(sum(1 for w in ['everyone','nobody','friends','family','public'] if w in text_lower)*0.2,1.0)
        ca['intent_clarity']=min(sum(1 for w in ['will','going to','definitely','certainly'] if w in text_lower)*0.3,1.0)
        return ca

    def predict_violence_probability(self, text: Union[str,List[str]]) -> Union[float,List[float]]:
        if not self.is_trained: raise ValueError("Model not trained. Call train_enhanced_model() first.")
        texts = [text] if isinstance(text,str) else text
        feats = []
        from scipy.sparse import hstack
        for t in texts:
            f1=self.vectorizers['tfidf_word'].transform([t])
            f2=self.vectorizers['tfidf_char'].transform([t])
            f3=self.vectorizers['count'].transform([t])
            lex=self._extract_lexicon_features(t).reshape(1,-1)
            feats.append(hstack([f1,f2,f3,lex]))
        from scipy.sparse import vstack
        all_feats=vstack(feats)
        probs=self.ensemble_model.predict_proba(all_feats)[:,1]
        return probs[0] if isinstance(text,str) else probs.tolist()

    def get_comprehensive_violence_analysis(self, text: str) -> Dict[str, Union[float, Dict]]:
        results = {}
        results.update(self.lexicon_based_violence_score(text))
        results['implicit_violence']=self.detect_implicit_violence(text)
        results['weapon_specificity']=self.analyze_weapon_method_specificity(text)
        results['context_analysis']=self.enhanced_context_analysis(text)
        results['threat_patterns']=self._analyze_enhanced_patterns(text)
        results['model_violence_probability']=self.predict_violence_probability(text) if self.is_trained else 0.0
        composite=self._calculate_enhanced_composite_score(results)
        results['enhanced_composite_score']=composite
        results['risk_level']=self._classify_risk_level(composite)
        return results

    def _analyze_enhanced_patterns(self, text: str) -> Dict[str, float]:
        return {cat: min(sum(1 for pat in pats if re.search(pat, text.lower()))/len(pats),1.0)
                for cat,pats in self.threat_patterns.items()}

    def _calculate_enhanced_composite_score(self, results: Dict) -> float:
        lex_w,mdl_w,imp_w,wpn_w,ctx_w,pat_w = (0.25,0.3 if self.is_trained else 0.0,0.15,0.15,0.1,0.05)
        if not self.is_trained: lex_w,imp_w,wpn_w,pat_w = (0.4,0.25,0.2,0.15)
        scores=[results.get('lexicon_violence_score',0),results.get('model_violence_probability',0),
                np.mean(list(results.get('implicit_violence',{}).values())),
                np.mean(list(results.get('weapon_specificity',{}).values())),
                np.mean(list(results.get('context_analysis',{}).values())),
                np.mean(list(results.get('threat_patterns',{}).values()))]
        weights=[lex_w,mdl_w,imp_w,wpn_w,ctx_w,pat_w]
        return min(sum(s*w for s,w in zip(scores,weights)),1.0)

    def lexicon_based_violence_score(self, text: str) -> Dict[str, float]:
        text_lower, words = text.lower(), text.lower().split()
        if not words: return {'lexicon_violence_score': 0.0}
        explicit = sum(self.explicit_violence_words.get(w,0.0) for w in words)/len(words)
        implicit = sum(self.implicit_violence_words.get(w,0.0) for w in words)/len(words)
        return {'explicit_lexicon_score': explicit, 'implicit_lexicon_score': implicit,
                'lexicon_violence_score': (explicit+implicit)/2}

    def _classify_risk_level(self, composite_score: float) -> str:
        if composite_score < 0.30: return 'low'
        if composite_score < 0.60: return 'medium'
        if composite_score < 0.80: return 'high'
        return 'critical'

# End of module