"""
Main Feature Extraction Orchestrator for Extremism Sentiment Analysis
This module combines all feature extractors and prepares data for model training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging
from datetime import datetime
import json
import os

# Import all feature extractors
from features.hate_sentiment import HateSentimentAnalyzer
from features.extremist_reference import ExtremistLexiconScorer
from features.overall_sentiment import OverallSentimentAnalyzer
from features.action_indication import ActionIndicationAnalyzer
from features.violent_sentiment import ViolenceSentimentAnalyzer

class ExtremismFeatureExtractor:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the main feature extractor.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize all feature extractors
        self.hate_analyzer = None
        self.extremist_lexicon = None
        self.sentiment_analyzer = None
        self.action_analyzer = None
        self.violence_analyzer = None
        
        self._initialize_analyzers()
        
        # Feature metadata
        self.feature_names = []
        self.feature_descriptions = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration settings."""
        default_config = {
            'hate_model': 'unitary/toxic-bert',
            'lexicon_path': 'data/extremism_lexicon.txt',
            'violence_dataset_path': 'data/violence_dataset.tdf',
            'output_dir': 'output',
            'log_level': 'INFO',
            'feature_weights': {
                'hate_sentiment': 0.25,
                'extremist_lexicon': 0.20,
                'overall_sentiment': 0.15,
                'action_indication': 0.20,
                'violence_sentiment': 0.20
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_analyzers(self):
        """Initialize all feature analyzers."""
        try:
            self.logger.info("Initializing feature analyzers...")
            
            # Initialize hate sentiment analyzer
            self.hate_analyzer = HateSentimentAnalyzer(self.config['hate_model'])
            self.logger.info("✓ Hate sentiment analyzer initialized")
            
            # Initialize extremist lexicon scorer
            self.extremist_lexicon = ExtremistLexiconScorer(self.config['lexicon_path'])
            self.logger.info("✓ Extremist lexicon scorer initialized")
            
            # Initialize overall sentiment analyzer
            self.sentiment_analyzer = OverallSentimentAnalyzer()
            self.logger.info("✓ Overall sentiment analyzer initialized")
            
            # Initialize action indication analyzer
            self.action_analyzer = ActionIndicationAnalyzer()
            self.logger.info("✓ Action indication analyzer initialized")
            
            # Initialize violence sentiment analyzer
            self.violence_analyzer = ViolenceSentimentAnalyzer()
            self.logger.info("✓ Violence sentiment analyzer initialized")
            
            # Train violence model if dataset exists
            if os.path.exists(self.config['violence_dataset_path']):
                texts, labels = self.violence_analyzer.load_violence_dataset(
                    self.config['violence_dataset_path']
                )
                self.violence_analyzer.train_model(texts, labels)
                self.logger.info("✓ Violence model trained")
            
            self.logger.info("All analyzers initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Error initializing analyzers: {e}")
            raise
    
    def extract_hate_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract hate sentiment features."""
        try:
            detailed_scores = self.hate_analyzer.get_detailed_scores(text)
            
            features = {
                'hate_probability': detailed_scores['hate_probability'],
                'hate_confidence': detailed_scores['hate_confidence'],
                'hate_intensity': detailed_scores['hate_intensity'],
                'prediction_uncertainty': detailed_scores['prediction_uncertainty']
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting hate sentiment features: {e}")
            return {'hate_probability': 0.0, 'hate_confidence': 0.0, 
                   'hate_intensity': 0.0, 'prediction_uncertainty': 1.0}
    
    def extract_extremist_lexicon_features(self, text: str) -> Dict[str, float]:
        """Extract extremist lexicon features."""
        try:
            scores = self.extremist_lexicon.get_comprehensive_score(text)
            
            features = {
                'lexicon_simple_match': scores['simple_match'],
                'lexicon_weighted_match': scores['weighted_match'],
                'lexicon_density': scores['density'],
                'lexicon_proximity': scores['proximity'],
                'lexicon_tfidf': scores['tfidf'],
                'lexicon_composite': scores['composite'],
                'lexicon_unique_matches': scores['unique_matches']
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting extremist lexicon features: {e}")
            return {'lexicon_simple_match': 0.0, 'lexicon_weighted_match': 0.0,
                   'lexicon_density': 0.0, 'lexicon_proximity': 0.0,
                   'lexicon_tfidf': 0.0, 'lexicon_composite': 0.0,
                   'lexicon_unique_matches': 0}
    
    def extract_overall_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract overall sentiment features."""
        try:
            analysis = self.sentiment_analyzer.get_comprehensive_sentiment_analysis(text)
            
            features = {
                'vader_compound': analysis['vader']['compound'],
                'vader_positive': analysis['vader']['pos'],
                'vader_negative': analysis['vader']['neg'],
                'vader_neutral': analysis['vader']['neu'],
                'textblob_polarity': analysis['textblob']['polarity'],
                'textblob_subjectivity': analysis['textblob']['subjectivity'],
                'emotional_intensity': analysis['emotional_intensity'],
                'extremist_positive': analysis['extremist_patterns']['positive_extremist'],
                'extremist_negative': analysis['extremist_patterns']['negative_extremist'],
                'extremist_polarity': analysis['extremist_patterns']['extremist_polarity'],
                'fear_manipulation': analysis['manipulation']['fear_manipulation'],
                'urgency_manipulation': analysis['manipulation']['urgency_manipulation'],
                'us_them_framing': analysis['manipulation']['us_them_framing'],
                'composite_sentiment': analysis['composite_sentiment'],
                'composite_negativity': analysis['composite_negativity'],
                'composite_intensity': analysis['composite_intensity']
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting sentiment features: {e}")
            return {f'vader_{k}': 0.0 for k in ['compound', 'positive', 'negative', 'neutral']}
    
    def extract_action_indication_features(self, text: str) -> Dict[str, float]:
        """Extract action indication features."""
        try:
            analysis = self.action_analyzer.get_comprehensive_action_score(text)
            
            features = {
                'imperative_ratio': analysis['imperative']['imperative_ratio'],
                'imperative_count': analysis['imperative']['imperative_count'],
                'action_verb_density': analysis['cta_patterns']['action_verb_density'],
                'urgency_score': analysis['cta_patterns']['urgency_score'],
                'collective_score': analysis['cta_patterns']['collective_score'],
                'necessity_score': analysis['cta_patterns']['necessity_score'],
                'social_media_cta': analysis['cta_patterns']['social_media_cta'],
                'rhetorical_questions': analysis['persuasion']['rhetorical_questions'],
                'emotional_appeal': analysis['persuasion']['emotional_appeal'],
                'authority_indicators': analysis['persuasion']['authority_indicators'],
                'social_proof': analysis['persuasion']['social_proof'],
                'mobilization_score': analysis['mobilization']['mobilization_score'],
                'coordination_score': analysis['mobilization']['coordination_score'],
                'recruitment_score': analysis['mobilization']['recruitment_score'],
                'action_intensity': analysis['action_intensity'],
                'collective_action_score': analysis['collective_action_score'],
                'urgency_intensity': analysis['urgency_intensity'],
                'overall_cta_score': analysis['overall_cta_score']
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting action indication features: {e}")
            return {'overall_cta_score': 0.0, 'action_intensity': 0.0}
    
    def extract_violence_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract violence sentiment features."""
        try:
            analysis = self.violence_analyzer.get_comprehensive_violence_score(text)
            
            features = {
                'explicit_violence': analysis['explicit_violence'],
                'implicit_violence': analysis['implicit_violence'],
                'weapons_score': analysis['weapons_score'],
                'targets_score': analysis['targets_score'],
                'violence_intensity': analysis['intensity_score'],
                'lexicon_violence_score': analysis['lexicon_violence_score'],
                'model_violence_probability': analysis['model_violence_probability'],
                'violence_patterns': analysis['violence_patterns'],
                'composite_violence_score': analysis['composite_violence_score']
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting violence features: {e}")
            return {'composite_violence_score': 0.0, 'explicit_violence': 0.0}
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all extracted features
        """
        all_features = {}
        
        # Extract features from each analyzer
        hate_features = self.extract_hate_sentiment_features(text)
        lexicon_features = self.extract_extremist_lexicon_features(text)
        sentiment_features = self.extract_overall_sentiment_features(text)
        action_features = self.extract_action_indication_features(text)
        violence_features = self.extract_violence_sentiment_features(text)
        
        # Combine all features
        all_features.update(hate_features)
        all_features.update(lexicon_features)
        all_features.update(sentiment_features)
        all_features.update(action_features)
        all_features.update(violence_features)
        
        # Add meta features
        all_features['text_length'] = len(text)
        all_features['word_count'] = len(text.split())
        all_features['sentence_count'] = len(text.split('.'))
        all_features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Calculate composite extremism score
        all_features['extremism_composite_score'] = self._calculate_composite_extremism_score(all_features)
        
        return all_features
    
    def _calculate_composite_extremism_score(self, features: Dict[str, float]) -> float:
        """
        Calculate composite extremism score using weighted features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Composite extremism score (0-1)
        """
        weights = self.config['feature_weights']
        
        # Main component scores
        hate_score = features.get('hate_probability', 0.0)
        lexicon_score = features.get('lexicon_composite', 0.0)
        sentiment_score = abs(features.get('composite_negativity', 0.0))
        action_score = features.get('overall_cta_score', 0.0)
        violence_score = features.get('composite_violence_score', 0.0)
        
        # Calculate weighted composite
        composite_score = (
            hate_score * weights['hate_sentiment'] +
            lexicon_score * weights['extremist_lexicon'] +
            sentiment_score * weights['overall_sentiment'] +
            action_score * weights['action_indication'] +
            violence_score * weights['violence_sentiment']
        )
        
        return min(composite_score, 1.0)
    
    def process_batch(self, texts: List[str], batch_size: int = 100) -> pd.DataFrame:
        """
        Process a batch of texts and extract features.
        
        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with extracted features
        """
        self.logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_features = []
            
            for j, text in enumerate(batch_texts):
                try:
                    features = self.extract_all_features(text)
                    features['text_id'] = i + j
                    features['original_text'] = text
                    batch_features.append(features)
                    
                    if (j + 1) % 10 == 0:
                        self.logger.info(f"Processed {j + 1}/{len(batch_texts)} texts in current batch")
                        
                except Exception as e:
                    self.logger.error(f"Error processing text {i + j}: {e}")
                    # Add empty feature set for failed texts
                    empty_features = {f: 0.0 for f in self.get_feature_names()}
                    empty_features['text_id'] = i + j
                    empty_features['original_text'] = text
                    empty_features['processing_error'] = True
                    batch_features.append(empty_features)
            
            all_features.extend(batch_features)
            self.logger.info(f"Completed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        if not self.feature_names:
            # Generate feature names by extracting from a sample text
            sample_features = self.extract_all_features("sample text")
            self.feature_names = [k for k in sample_features.keys() 
                                if k not in ['text_id', 'original_text']]
        
        return self.feature_names
    
    def save_features(self, df: pd.DataFrame, output_path: str):
        """
        Save extracted features to file.
        
        Args:
            df: DataFrame with features
            output_path: Output file path
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save features
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            df.to_csv(f"{output_path}.csv", index=False)
        
        self.logger.info(f"Features saved to {output_path}")
        
        # Save feature metadata
        metadata = {
            'feature_names': self.get_feature_names(),
            'feature_count': len(self.get_feature_names()),
            'sample_count': len(df),
            'extraction_timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.json').replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def generate_feature_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive feature extraction report.
        
        Args:
            df: DataFrame with extracted features
            
        Returns:
            Dictionary with report statistics
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features 
                          if col not in ['text_id', 'text_length', 'word_count', 'sentence_count']]
        
        report = {
            'summary': {
                'total_samples': len(df),
                'total_features': len(numeric_features),
                'processing_errors': df.get('processing_error', pd.Series([False]*len(df))).sum(),
                'avg_text_length': df['text_length'].mean(),
                'avg_word_count': df['word_count'].mean()
            },
            'feature_statistics': {},
            'top_extremism_samples': [],
            'feature_correlations': {}
        }
        
        # Feature statistics
        for feature in numeric_features:
            if feature in df.columns:
                report['feature_statistics'][feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'median': float(df[feature].median())
                }
        
        # Top extremism samples
        if 'extremism_composite_score' in df.columns:
            top_samples = df.nlargest(5, 'extremism_composite_score')[
                ['text_id', 'original_text', 'extremism_composite_score']
            ].to_dict('records')
            report['top_extremism_samples'] = top_samples
        
        # Feature correlations with composite score
        if 'extremism_composite_score' in df.columns:
            correlations = df[numeric_features].corrwith(df['extremism_composite_score']).sort_values(ascending=False)
            report['feature_correlations'] = correlations.to_dict()
        
        return report

# Usage example and testing
if __name__ == "__main__":
    # Initialize feature extractor
    extractor = ExtremismFeatureExtractor()
    
    # Sample texts for testing
    sample_texts = [
        "I love spending time with my family and friends.",
        "We must unite and fight against the corrupt system immediately!",
        "These people are destroying our country and we need to stop them now!",
        "Let's organize a peaceful community meeting next week.",
        "I will eliminate anyone who gets in my way - join our movement!",
        "Community building and dialogue are important for society.",
        "URGENT: Share this now! We need to take action before it's too late!",
        "The weather is nice today, perfect for a walk in the park."
    ]
    
    # Process texts
    print("Processing sample texts...")
    results_df = extractor.process_batch(sample_texts)
    
    # Save results
    output_path = "output/sample_features.csv"
    extractor.save_features(results_df, output_path)
    
    # Generate report
    report = extractor.generate_feature_report(results_df)
    
    # Print summary
    print("\n" + "="*50)
    print("FEATURE EXTRACTION REPORT")
    print("="*50)
    print(f"Total samples processed: {report['summary']['total_samples']}")
    print(f"Total features extracted: {report['summary']['total_features']}")
    print(f"Average text length: {report['summary']['avg_text_length']:.1f} characters")
    print(f"Average word count: {report['summary']['avg_word_count']:.1f} words")
    
    print("\nTop 5 Extremism Scores:")
    for i, sample in enumerate(report['top_extremism_samples'], 1):
        print(f"{i}. Score: {sample['extremism_composite_score']:.3f}")
        print(f"   Text: {sample['original_text'][:100]}...")
        print()
    
    print("Top 10 Most Correlated Features:")
    for feature, correlation in list(report['feature_correlations'].items())[:10]:
        print(f"{feature}: {correlation:.3f}")
    
    # Save full report
    report_path = "output/feature_extraction_report.json"
    os.makedirs("output", exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to: {report_path}")
    print(f"Features saved to: {output_path}")