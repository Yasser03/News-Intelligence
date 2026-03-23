"""
News Binary Classifier Module
==============================

This module provides optimized binary classification for news articles at scale.
It determines whether articles are relevant (e.g., incidents) or not.

PERFORMANCE OPTIMIZATIONS:
- No tqdm progress bars (adds overhead when processing 10,000+ articles)
- Vectorized operations wherever possible
- Batch processing support
- Minimal dependencies

USAGE EXAMPLES:
---------------

1. Basic usage - Classify new articles:
    ```python
    from news_tk.news_binary_classifier import NewsBinaryClassifier
    import pandas as pd
    
    # Initialize classifier
    classifier = NewsBinaryClassifier(model_path='path/to/models')
    
    # Load your data
    df = pd.read_csv('news_articles.csv')
    
    # Classify using full text (title + description + article)
    results = classifier.predict(df)
    
    # Or classify using only headlines (faster)
    results = classifier.predict_from_headlines(df)
    ```

2. Batch processing with custom columns:
    ```python
    # If your DataFrame has different column names
    results = classifier.predict(
        df,
        title_col='headline',
        description_col='summary',
        article_col='full_text'
    )
    ```

3. Get predictions with probabilities:
    ```python
    # Get both predictions and confidence scores
    results = classifier.predict_with_probability(df)
    # Results will have 'prediction' and 'probability' columns
    ```

REQUIRED DATAFRAME COLUMNS:
---------------------------
- 'title': Article headline/title
- 'description': Article description/summary (optional for headline-only mode)
- 'article': Full article text (optional for headline-only mode)

OUTPUT:
-------
Returns DataFrame with original data plus:
- 'prediction': Binary classification result (e.g., 'incident' or 'non-incident')
- 'probability': Confidence score (0-1) if using predict_with_probability()

TEAM NOTES:
-----------
- This module is optimized for processing large batches (10,000+ articles)
- Text normalization is done in bulk without progress bars for speed
- Models are loaded once and reused for efficiency
- Error handling is built-in for missing or malformed data
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from langdetect import detect
import cleantext
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


class NewsBinaryClassifier:
    """
    Optimized binary classifier for news articles.
    
    This class provides fast, scalable binary classification without overhead
    from progress tracking (tqdm removed for performance at scale).
    
    Attributes:
        model_path (str): Path to directory containing trained model files
        model: Loaded classification model
        label_encoder: Loaded label encoder for transforming predictions
        vectorizer: Loaded word vectorizer for text feature extraction
    """
    
    def __init__(self, model_path='../../TALOS/best_model_params/'):
        """
        Initialize the binary classifier.
        
        Args:
            model_path (str): Directory path where model files are stored.
                              Should contain:
                              - 'Logistic Regression_62k.pkl'
                              - 'label_encoder_transformer.pkl'
                              - 'word_vectorizer.pkl'
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.vectorizer = None
        self._load_models()
    
    def _load_models(self):
        """
        Load pre-trained model, label encoder, and vectorizer from disk.
        
        This method is called automatically during initialization.
        Models are loaded once and cached for reuse.
        
        Raises:
            FileNotFoundError: If model files are not found at specified path
            Exception: If models cannot be loaded (corrupted files, version mismatch)
        """
        try:
            # Load classification model (typically Logistic Regression)
            with open(self.model_path + 'Logistic Regression_62k.pkl', 'rb') as file:
                self.model = pickle.load(file)
            
            # Load label encoder (converts numeric predictions back to labels)
            with open(self.model_path + 'label_encoder_transformer.pkl', 'rb') as file:
                self.label_encoder = pickle.load(file)
            
            # Load word vectorizer (converts text to numeric features)
            with open(self.model_path + 'word_vectorizer.pkl', 'rb') as file:
                self.vectorizer = pickle.load(file)
                
            print(f"✓ Models loaded successfully from {self.model_path}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model files not found at {self.model_path}. "
                f"Please ensure the following files exist:\n"
                f"  - Logistic Regression_62k.pkl\n"
                f"  - label_encoder_transformer.pkl\n"
                f"  - word_vectorizer.pkl"
            ) from e
        except Exception as e:
            raise Exception(
                f"Failed to load models: {e}\n"
                f"This may be due to version incompatibility or corrupted files."
            ) from e
    
    def _detect_english(self, text):
        """
        Detect if text is in English.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is English, False otherwise
        """
        try:
            return detect(str(text)) == 'en'
        except:
            return False
    
    def _clean_text(self, text):
        """
        Clean and normalize a single text string.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned and normalized text (lowercase, standardized)
        """
        return cleantext.clean(str(text).lower())
    
    def _normalize_text_batch(self, texts):
        """
        Normalize a batch of texts efficiently (no progress bars).
        
        Args:
            texts (pd.Series or list): Collection of texts to normalize
            
        Returns:
            list: Cleaned and normalized texts
        """
        # Use list comprehension for speed (no tqdm overhead)
        return [self._clean_text(text) for text in texts]
    
    def _prepare_dataframe(self, df, title_col='title', description_col='description', 
                          article_col='article', filter_english=True):
        """
        Prepare DataFrame by normalizing text columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame with news articles
            title_col (str): Name of title column
            description_col (str): Name of description column
            article_col (str): Name of article column
            filter_english (bool): If True, remove non-English articles
            
        Returns:
            pd.DataFrame: DataFrame with normalized text columns added
        """
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Filter English articles if requested
        if filter_english and title_col in df_processed.columns:
            print("Filtering non-English articles...")
            initial_count = len(df_processed)
            df_processed = df_processed[df_processed[title_col].apply(self._detect_english)]
            removed_count = initial_count - len(df_processed)
            if removed_count > 0:
                print(f"  Removed {removed_count} non-English articles")
        
        # Normalize title (required)
        if title_col in df_processed.columns:
            print(f"Normalizing {len(df_processed)} titles...")
            df_processed['title_norm'] = self._normalize_text_batch(df_processed[title_col])
        else:
            raise ValueError(f"Required column '{title_col}' not found in DataFrame")
        
        # Normalize description (optional)
        if description_col in df_processed.columns:
            print(f"Normalizing {len(df_processed)} descriptions...")
            df_processed['description_norm'] = self._normalize_text_batch(df_processed[description_col])
            # Combine title and description for richer feature set
            df_processed['text'] = df_processed['title_norm'] + ' ' + df_processed['description_norm']
        else:
            # Use only title if description not available
            df_processed['text'] = df_processed['title_norm']
            print("Warning: Description column not found, using title only")
        
        # Normalize article (optional, for full-text analysis)
        if article_col in df_processed.columns:
            print(f"Normalizing {len(df_processed)} articles...")
            df_processed['article_norm'] = self._normalize_text_batch(df_processed[article_col])
        
        return df_processed
    
    def predict(self, df, title_col='title', description_col='description', 
                article_col='article', filter_english=True):
        """
        Classify news articles using full text (title + description).
        
        This is the main method for binary classification of news articles.
        Optimized for processing large batches (10,000+ articles) without overhead.
        
        Args:
            df (pd.DataFrame): DataFrame containing news articles
            title_col (str): Column name for article titles
            description_col (str): Column name for article descriptions
            article_col (str): Column name for full article text
            filter_english (bool): If True, filter out non-English articles
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'prediction' column
            
        Example:
            >>> df = pd.read_csv('news.csv')
            >>> classifier = NewsBinaryClassifier()
            >>> results = classifier.predict(df)
            >>> print(results[['title', 'prediction']].head())
        """
        print(f"\n{'='*60}")
        print(f"BINARY CLASSIFICATION - Processing {len(df)} articles")
        print(f"{'='*60}\n")
        
        # Prepare and normalize text
        df_processed = self._prepare_dataframe(
            df, title_col, description_col, article_col, filter_english
        )
        
        # Vectorize the combined text
        print("Vectorizing text features...")
        X = self.vectorizer.transform(df_processed['text'])
        
        # Make predictions
        print("Generating predictions...")
        predictions_numeric = self.model.predict(X)
        
        # Convert numeric predictions to labels
        predictions_labels = self.label_encoder.inverse_transform(predictions_numeric)
        
        # Add predictions to original DataFrame
        df_processed['prediction'] = predictions_labels
        
        print(f"\n✓ Classification complete!")
        print(f"  Total processed: {len(df_processed)}")
        
        # Show prediction distribution
        print("\nPrediction Distribution:")
        print(df_processed['prediction'].value_counts())
        print(f"\n{'='*60}\n")
        
        return df_processed
    
    def predict_from_headlines(self, df, title_col='title', filter_english=True):
        """
        Fast classification using only article headlines/titles.
        
        This method is ~2-3x faster than full-text classification since it
        only processes titles. Use when speed is critical and titles are
        sufficiently descriptive.
        
        Args:
            df (pd.DataFrame): DataFrame containing news articles
            title_col (str): Column name for article titles
            filter_english (bool): If True, filter out non-English articles
            
        Returns:
            pd.DataFrame: DataFrame with 'prediction' column added
            
        Example:
            >>> # Fast classification for quick filtering
            >>> results = classifier.predict_from_headlines(df)
            >>> incidents = results[results['prediction'] == 'incident']
        """
        print(f"\n{'='*60}")
        print(f"HEADLINE-ONLY CLASSIFICATION - Processing {len(df)} articles")
        print(f"{'='*60}\n")
        
        df_processed = df.copy()
        
        # Filter English if requested
        if filter_english and title_col in df_processed.columns:
            print("Filtering non-English articles...")
            initial_count = len(df_processed)
            df_processed = df_processed[df_processed[title_col].apply(self._detect_english)]
            removed_count = initial_count - len(df_processed)
            if removed_count > 0:
                print(f"  Removed {removed_count} non-English articles")
        
        # Normalize titles only
        print(f"Normalizing {len(df_processed)} titles...")
        df_processed['title_norm'] = self._normalize_text_batch(df_processed[title_col])
        
        # Vectorize and predict
        print("Vectorizing and predicting...")
        X = self.vectorizer.transform(df_processed['title_norm'])
        predictions_numeric = self.model.predict(X)
        predictions_labels = self.label_encoder.inverse_transform(predictions_numeric)
        
        df_processed['prediction'] = predictions_labels
        
        print(f"\n✓ Classification complete!")
        print(f"  Total processed: {len(df_processed)}")
        print("\nPrediction Distribution:")
        print(df_processed['prediction'].value_counts())
        print(f"\n{'='*60}\n")
        
        return df_processed
    
    def predict_with_probability(self, df, title_col='title', description_col='description',
                                 article_col='article', filter_english=True):
        """
        Classify articles and return confidence probabilities.
        
        Use this when you need to know how confident the model is in its predictions.
        Helpful for filtering by confidence threshold or identifying edge cases.
        
        Args:
            df (pd.DataFrame): DataFrame containing news articles
            title_col (str): Column name for article titles
            description_col (str): Column name for article descriptions
            article_col (str): Column name for full article text
            filter_english (bool): If True, filter out non-English articles
            
        Returns:
            pd.DataFrame: DataFrame with 'prediction' and 'probability' columns
            
        Example:
            >>> results = classifier.predict_with_probability(df)
            >>> # Filter for high-confidence predictions only
            >>> high_confidence = results[results['probability'] > 0.8]
        """
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION WITH PROBABILITIES - Processing {len(df)} articles")
        print(f"{'='*60}\n")
        
        # Prepare and normalize text
        df_processed = self._prepare_dataframe(
            df, title_col, description_col, article_col, filter_english
        )
        
        # Vectorize the combined text
        print("Vectorizing text features...")
        X = self.vectorizer.transform(df_processed['text'])
        
        # Make predictions with probabilities
        print("Generating predictions with probabilities...")
        predictions_numeric = self.model.predict(X)
        predictions_proba = self.model.predict_proba(X)
        
        # Get the probability of the predicted class
        max_probabilities = np.max(predictions_proba, axis=1)
        
        # Convert numeric predictions to labels
        predictions_labels = self.label_encoder.inverse_transform(predictions_numeric)
        
        # Add predictions and probabilities
        df_processed['prediction'] = predictions_labels
        df_processed['probability'] = max_probabilities
        
        print(f"\n✓ Classification complete!")
        print(f"  Total processed: {len(df_processed)}")
        print("\nPrediction Distribution:")
        print(df_processed['prediction'].value_counts())
        print("\nConfidence Statistics:")
        print(df_processed['probability'].describe())
        print(f"\n{'='*60}\n")
        
        return df_processed
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model metadata including classes and feature count
        """
        return {
            'model_type': type(self.model).__name__,
            'classes': list(self.label_encoder.classes_),
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'model_path': self.model_path
        }


# ============================================================================
# UTILITY FUNCTIONS FOR COMMON USE CASES
# ============================================================================

def classify_news_batch(df, model_path='../../TALOS/best_model_params/', 
                       use_headlines_only=False):
    """
    Convenience function for quick batch classification.
    
    Args:
        df (pd.DataFrame): DataFrame with news articles
        model_path (str): Path to model directory
        use_headlines_only (bool): If True, use faster headline-only mode
        
    Returns:
        pd.DataFrame: DataFrame with predictions
        
    Example:
        >>> from news_tk.news_binary_classifier import classify_news_batch
        >>> results = classify_news_batch(df)
    """
    classifier = NewsBinaryClassifier(model_path=model_path)
    
    if use_headlines_only:
        return classifier.predict_from_headlines(df)
    else:
        return classifier.predict(df)


def filter_incidents(df, model_path='../../TALOS/best_model_params/', 
                    incident_label='incident'):
    """
    Classify and filter to return only incident articles.
    
    Args:
        df (pd.DataFrame): DataFrame with news articles
        model_path (str): Path to model directory
        incident_label (str): Label value representing incidents
        
    Returns:
        pd.DataFrame: DataFrame containing only incident articles
        
    Example:
        >>> from news_tk.news_binary_classifier import filter_incidents
        >>> incidents_only = filter_incidents(df)
    """
    classifier = NewsBinaryClassifier(model_path=model_path)
    df_classified = classifier.predict(df)
    return df_classified[df_classified['prediction'] == incident_label]


# ============================================================================
# EXAMPLE USAGE (uncomment to test)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating the classifier.
    Run this file directly to test: python news_binary_classifier.py
    """
    
    # Sample data for testing
    sample_data = {
        'title': [
            'Major fire breaks out in downtown building',
            'New restaurant opens on Main Street',
            'Car accident on Highway 101 causes delays',
            'Local team wins championship game'
        ],
        'description': [
            'Firefighters battling blaze in commercial district',
            'Award-winning chef brings new cuisine to town',
            'Two vehicles collided causing traffic backup',
            'Celebrations continue after historic victory'
        ],
        'article': [
            'Full article text about the fire incident...',
            'Full article text about the restaurant...',
            'Full article text about the accident...',
            'Full article text about the game...'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("EXAMPLE: Testing Binary Classifier\n")
    
    # Initialize classifier (update path as needed)
    try:
        classifier = NewsBinaryClassifier(model_path='../../TALOS/best_model_params/')
        
        # Method 1: Full text classification
        results = classifier.predict(df, filter_english=False)
        print("\nResults (first 2 rows):")
        print(results[['title', 'prediction']].head(2))
        
        # Method 2: Headline-only (faster)
        results_fast = classifier.predict_from_headlines(df, filter_english=False)
        print("\nFast results (headline-only):")
        print(results_fast[['title', 'prediction']].head(2))
        
        # Method 3: With probabilities
        results_proba = classifier.predict_with_probability(df, filter_english=False)
        print("\nResults with confidence:")
        print(results_proba[['title', 'prediction', 'probability']].head(2))
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Update model_path to point to your trained models directory")
