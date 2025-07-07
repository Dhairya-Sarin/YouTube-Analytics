import pandas as pd
import numpy as np
import cv2
import requests
from PIL import Image
import textstat
import re
from typing import List, Dict, Any
from collections import Counter
import io
import string
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from langdetect import detect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import emoji

    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False


class TitleFeatureExtractor:
    def __init__(self):
        # Initialize sentiment analyzers
        self.sentiment_analyzer = None
        self.vader_analyzer = None

        if NLTK_AVAILABLE:
            try:
                nltk.data.find('vader_lexicon')
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                except:
                    pass

        if VADER_AVAILABLE:
            self.vader_analyzer = VaderAnalyzer()

        # Define keyword dictionaries
        self.clickbait_words = {
            'curiosity': ['secret', 'hidden', 'revealed', 'exposed', 'truth', 'mystery', 'shocking', 'unbelievable'],
            'urgency': ['now', 'today', 'urgent', 'quick', 'fast', 'immediately', 'breaking', 'live', 'new', 'latest'],
            'superlatives': ['best', 'worst', 'amazing', 'incredible', 'ultimate', 'perfect', 'epic', 'genius',
                             'insane'],
            'emotional': ['love', 'hate', 'angry', 'sad', 'happy', 'excited', 'scared', 'surprised', 'shocked'],
            'numbers': ['top', 'reasons', 'ways', 'things', 'facts', 'tips', 'secrets', 'tricks']
        }

        self.content_type_keywords = {
            'tutorial': ['how', 'tutorial', 'guide', 'learn', 'teach', 'explain', 'show', 'demo', 'walkthrough',
                         'step'],
            'review': ['review', 'unboxing', 'test', 'vs', 'comparison', 'opinion', 'thoughts', 'first', 'impression'],
            'entertainment': ['funny', 'comedy', 'prank', 'challenge', 'reaction', 'fails', 'compilation', 'meme'],
            'educational': ['facts', 'science', 'history', 'documentary', 'explained', 'analysis', 'study', 'research'],
            'gaming': ['gameplay', 'gaming', 'game', 'play', 'stream', 'speedrun', 'walkthrough', 'let\'s play'],
            'vlog': ['vlog', 'day', 'life', 'routine', 'daily', 'morning', 'night', 'week', 'month']
        }

        self.power_words = [
            'free', 'guaranteed', 'proven', 'results', 'discover', 'reveal', 'unlock', 'master',
            'transform', 'breakthrough', 'exclusive', 'limited', 'bonus', 'instant', 'simple', 'easy'
        ]

        self.question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who', 'can', 'will', 'should', 'would',
                               'could']

    def extract_features(self, title: str) -> Dict[str, Any]:
        """Extract comprehensive features from a single title"""
        features = {}
        title_lower = title.lower()
        words = title.split()

        # === BASIC TEXT METRICS ===
        features.update(self._extract_basic_metrics(title, words))

        # === LINGUISTIC FEATURES ===
        features.update(self._extract_linguistic_features(title, title_lower))

        # === PUNCTUATION AND SYMBOLS ===
        features.update(self._extract_punctuation_features(title))

        # === CAPITALIZATION PATTERNS ===
        features.update(self._extract_capitalization_features(title, words))

        # === SENTIMENT ANALYSIS ===
        features.update(self._extract_sentiment_features(title))

        # === CLICKBAIT INDICATORS ===
        features.update(self._extract_clickbait_features(title_lower, words))

        # === CONTENT TYPE CLASSIFICATION ===
        features.update(self._extract_content_type_features(title_lower))

        # === NUMBERS AND DIGITS ===
        features.update(self._extract_number_features(title))

        # === READABILITY METRICS ===
        features.update(self._extract_readability_features(title))

        # === KEYWORD DENSITY ===
        features.update(self._extract_keyword_features(title_lower, words))

        # === TEMPORAL REFERENCES ===
        features.update(self._extract_temporal_features(title_lower))

        # === STRUCTURAL FEATURES ===
        features.update(self._extract_structural_features(title))

        # === LANGUAGE FEATURES ===
        features.update(self._extract_language_features(title))

        # === EMOJI AND SPECIAL CHARACTERS ===
        features.update(self._extract_emoji_features(title))

        return features

    def _extract_basic_metrics(self, title: str, words: List[str]) -> Dict[str, Any]:
        """Basic text statistics"""
        return {
            'char_count': len(title),
            'word_count': len(words),
            'sentence_count': len([s for s in title.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'max_word_length': max([len(word) for word in words]) if words else 0,
            'min_word_length': min([len(word) for word in words]) if words else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'syllable_count': sum([self._count_syllables(word) for word in words]),
            'avg_syllables_per_word': np.mean([self._count_syllables(word) for word in words]) if words else 0
        }

    def _extract_linguistic_features(self, title: str, title_lower: str) -> Dict[str, Any]:
        """Linguistic and grammatical features"""
        features = {}

        # Vowel/consonant ratios
        vowels = 'aeiou'
        vowel_count = sum(1 for char in title_lower if char in vowels)
        consonant_count = sum(1 for char in title_lower if char.isalpha() and char not in vowels)

        features.update({
            'vowel_count': vowel_count,
            'consonant_count': consonant_count,
            'vowel_ratio': vowel_count / len(title) if title else 0,
            'consonant_ratio': consonant_count / len(title) if title else 0
        })

        # Question structure
        is_question = title.strip().endswith('?') or any(title_lower.startswith(q) for q in self.question_words)
        features['is_question'] = int(is_question)
        features['question_words_count'] = sum(1 for word in title_lower.split() if word in self.question_words)

        # Language detection
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(title)
                features['is_english'] = int(detected_lang == 'en')
                features['detected_language'] = detected_lang
            except:
                features['is_english'] = 1
                features['detected_language'] = 'en'
        else:
            features['is_english'] = 1
            features['detected_language'] = 'en'

        return features

    def _extract_punctuation_features(self, title: str) -> Dict[str, Any]:
        """Punctuation and symbol analysis"""
        return {
            'exclamation_count': title.count('!'),
            'question_count': title.count('?'),
            'comma_count': title.count(','),
            'period_count': title.count('.'),
            'colon_count': title.count(':'),
            'semicolon_count': title.count(';'),
            'hyphen_count': title.count('-'),
            'underscore_count': title.count('_'),
            'parentheses_count': title.count('(') + title.count(')'),
            'brackets_count': title.count('[') + title.count(']'),
            'quotation_count': title.count('"') + title.count("'"),
            'ampersand_count': title.count('&'),
            'at_symbol_count': title.count('@'),
            'hash_count': title.count('#'),
            'dollar_count': title.count('$'),
            'percent_count': title.count('%'),
            'total_punctuation': sum(1 for char in title if char in string.punctuation),
            'punctuation_ratio': sum(1 for char in title if char in string.punctuation) / len(title) if title else 0
        }

    def _extract_capitalization_features(self, title: str, words: List[str]) -> Dict[str, Any]:
        """Capitalization pattern analysis"""
        if not words:
            return {f'caps_{key}': 0 for key in ['uppercase_words', 'lowercase_words', 'capitalized_words',
                                                 'mixed_case_words', 'all_caps_ratio', 'title_case', 'camel_case']}

        uppercase_words = sum(1 for word in words if word.isupper() and word.isalpha())
        lowercase_words = sum(1 for word in words if word.islower() and word.isalpha())
        capitalized_words = sum(1 for word in words if word[0].isupper() and len(word) > 1 and word[1:].islower())
        mixed_case_words = sum(
            1 for word in words if any(c.isupper() for c in word[1:]) and any(c.islower() for c in word))

        return {
            'caps_uppercase_words': uppercase_words,
            'caps_lowercase_words': lowercase_words,
            'caps_capitalized_words': capitalized_words,
            'caps_mixed_case_words': mixed_case_words,
            'caps_all_caps_ratio': uppercase_words / len(words),
            'caps_title_case': int(title.istitle()),
            'caps_camel_case': int(
                any(len(word) > 1 and word[0].islower() and any(c.isupper() for c in word[1:]) for word in words))
        }

    def _extract_sentiment_features(self, title: str) -> Dict[str, Any]:
        """Sentiment analysis using multiple methods"""
        features = {}

        # VADER sentiment (if available)
        if self.sentiment_analyzer or self.vader_analyzer:
            analyzer = self.sentiment_analyzer or self.vader_analyzer
            try:
                sentiment = analyzer.polarity_scores(title)
                features.update({
                    'sentiment_positive': sentiment['pos'],
                    'sentiment_negative': sentiment['neg'],
                    'sentiment_neutral': sentiment['neu'],
                    'sentiment_compound': sentiment['compound'],
                    'sentiment_intensity': abs(sentiment['compound'])
                })
            except:
                features.update({
                    'sentiment_positive': 0.5,
                    'sentiment_negative': 0.5,
                    'sentiment_neutral': 0.5,
                    'sentiment_compound': 0,
                    'sentiment_intensity': 0
                })

        # TextBlob sentiment (if available)
        if NLTK_AVAILABLE:
            try:
                blob = TextBlob(title)
                features.update({
                    'textblob_polarity': blob.sentiment.polarity,
                    'textblob_subjectivity': blob.sentiment.subjectivity
                })
            except:
                features.update({
                    'textblob_polarity': 0,
                    'textblob_subjectivity': 0.5
                })

        # Manual emotion detection
        positive_words = ['amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great', 'incredible',
                          'outstanding', 'perfect', 'wonderful']
        negative_words = ['awful', 'terrible', 'horrible', 'worst', 'bad', 'disappointing', 'failed', 'disaster',
                          'nightmare', 'pathetic']

        title_lower = title.lower()
        features.update({
            'positive_words_count': sum(1 for word in positive_words if word in title_lower),
            'negative_words_count': sum(1 for word in negative_words if word in title_lower)
        })

        return features

    def _extract_clickbait_features(self, title_lower: str, words: List[str]) -> Dict[str, Any]:
        """Clickbait and engagement indicators"""
        features = {}

        # Count different types of clickbait words
        for category, word_list in self.clickbait_words.items():
            count = sum(1 for word in word_list if word in title_lower)
            features[f'clickbait_{category}_count'] = count

        # Power words
        power_count = sum(1 for word in self.power_words if word in title_lower)
        features['power_words_count'] = power_count

        # Listicle indicators
        listicle_patterns = [r'\d+\s+(reasons|ways|things|tips|secrets|facts|tricks|methods|steps)',
                             r'top\s+\d+', r'\d+\s+best', r'\d+\s+worst']
        features['listicle_indicators'] = sum(1 for pattern in listicle_patterns if re.search(pattern, title_lower))

        # Curiosity gap indicators
        curiosity_phrases = ['you won\'t believe', 'this will', 'wait until', 'what happens next',
                             'the reason why', 'here\'s why', 'this is what', 'find out']
        features['curiosity_gap_count'] = sum(1 for phrase in curiosity_phrases if phrase in title_lower)

        # Controversy indicators
        controversy_words = ['vs', 'versus', 'against', 'debate', 'controversial', 'banned', 'censored', 'exposed']
        features['controversy_count'] = sum(1 for word in controversy_words if word in title_lower)

        return features

    def _extract_content_type_features(self, title_lower: str) -> Dict[str, Any]:
        """Content type classification"""
        features = {}

        for content_type, keywords in self.content_type_keywords.items():
            count = sum(1 for keyword in keywords if keyword in title_lower)
            features[f'content_{content_type}_score'] = count

        # Determine primary content type
        content_scores = {ct: features[f'content_{ct}_score'] for ct in self.content_type_keywords.keys()}
        primary_type = max(content_scores.items(), key=lambda x: x[1])
        features['primary_content_type'] = primary_type[0] if primary_type[1] > 0 else 'other'

        return features

    def _extract_number_features(self, title: str) -> Dict[str, Any]:
        """Number and digit analysis"""
        # Find all numbers
        numbers = re.findall(r'\d+', title)

        features = {
            'has_numbers': int(bool(numbers)),
            'number_count': len(numbers),
            'total_digits': sum(len(num) for num in numbers),
            'max_number': max([int(num) for num in numbers]) if numbers else 0,
            'min_number': min([int(num) for num in numbers]) if numbers else 0,
            'avg_number': np.mean([int(num) for num in numbers]) if numbers else 0
        }

        # Year detection
        current_year = datetime.now().year
        years = [int(num) for num in numbers if 1900 <= int(num) <= current_year + 10]
        features.update({
            'has_year': int(bool(years)),
            'year_count': len(years),
            'recent_year': int(any(year >= current_year - 2 for year in years))
        })

        # Special number patterns
        features.update({
            'has_percentages': int('%' in title and any(char.isdigit() for char in title)),
            'has_money': int(
                any(symbol in title for symbol in ['$', '€', '£', '¥']) and any(char.isdigit() for char in title)),
            'starts_with_number': int(title.strip() and title.strip()[0].isdigit())
        })

        return features

    def _extract_readability_features(self, title: str) -> Dict[str, Any]:
        """Readability and complexity metrics"""
        features = {}

        try:
            features.update({
                'flesch_reading_ease': textstat.flesch_reading_ease(title),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(title),
                'gunning_fog': textstat.gunning_fog(title),
                'smog_index': textstat.smog_index(title),
                'automated_readability_index': textstat.automated_readability_index(title),
                'coleman_liau_index': textstat.coleman_liau_index(title),
                'linsear_write_formula': textstat.linsear_write_formula(title),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(title)
            })
        except:
            # Fallback values if textstat fails
            features.update({
                'flesch_reading_ease': 50,
                'flesch_kincaid_grade': 8,
                'gunning_fog': 8,
                'smog_index': 8,
                'automated_readability_index': 8,
                'coleman_liau_index': 8,
                'linsear_write_formula': 8,
                'dale_chall_readability_score': 8
            })

        return features

    def _extract_keyword_features(self, title_lower: str, words: List[str]) -> Dict[str, Any]:
        """Keyword density and SEO features"""
        if not words:
            return {
                'keyword_density': 0,
                'repeated_words': 0,
                'brand_mentions': 0,
                'trending_terms': 0
            }

        word_freq = Counter(words)
        most_common = word_freq.most_common(1)
        max_freq = most_common[0][1] if most_common else 0

        # Brand/platform mentions
        brands = ['youtube', 'instagram', 'tiktok', 'twitter', 'facebook', 'netflix', 'apple', 'google', 'amazon']
        brand_mentions = sum(1 for brand in brands if brand in title_lower)

        # Trending/viral terms
        trending_terms = ['viral', 'trending', 'popular', 'famous', 'celebrity', 'influencer', 'meme', 'challenge']
        trending_count = sum(1 for term in trending_terms if term in title_lower)

        return {
            'keyword_density': max_freq / len(words),
            'repeated_words': sum(1 for count in word_freq.values() if count > 1),
            'brand_mentions': brand_mentions,
            'trending_terms': trending_count
        }

    def _extract_temporal_features(self, title_lower: str) -> Dict[str, Any]:
        """Time-related references"""
        time_words = {
            'immediate': ['now', 'today', 'right now', 'immediately', 'instant'],
            'recent': ['new', 'latest', 'recent', 'fresh', 'just', 'breaking'],
            'future': ['coming', 'soon', 'upcoming', 'next', 'future', 'will'],
            'past': ['old', 'vintage', 'classic', 'retro', 'throwback', 'nostalgia'],
            'duration': ['minute', 'hour', 'day', 'week', 'month', 'year', 'forever']
        }

        features = {}
        for category, words in time_words.items():
            count = sum(1 for word in words if word in title_lower)
            features[f'temporal_{category}_count'] = count

        return features

    def _extract_structural_features(self, title: str) -> Dict[str, Any]:
        """Title structure analysis"""
        return {
            'starts_with_how': int(title.lower().startswith('how')),
            'starts_with_why': int(title.lower().startswith('why')),
            'starts_with_what': int(title.lower().startswith('what')),
            'has_colon': int(':' in title),
            'has_pipe': int('|' in title),
            'has_brackets': int(any(char in title for char in '()[]{}')),
            'has_quotes': int(any(char in title for char in '"\'')),
            'title_format_list': int(bool(re.search(r'\d+.*:', title))),
            'title_format_how_to': int('how to' in title.lower()),
            'ends_with_punctuation': int(title.strip() and title.strip()[-1] in '!?.'),
        }

    def _extract_language_features(self, title: str) -> Dict[str, Any]:
        """Language complexity features"""
        words = title.split()

        # Word length distribution
        if words:
            word_lengths = [len(word) for word in words]
            features = {
                'short_words_ratio': sum(1 for length in word_lengths if length <= 3) / len(words),
                'medium_words_ratio': sum(1 for length in word_lengths if 4 <= length <= 6) / len(words),
                'long_words_ratio': sum(1 for length in word_lengths if length >= 7) / len(words),
                'very_long_words_ratio': sum(1 for length in word_lengths if length >= 10) / len(words)
            }
        else:
            features = {
                'short_words_ratio': 0,
                'medium_words_ratio': 0,
                'long_words_ratio': 0,
                'very_long_words_ratio': 0
            }

        return features

    def _extract_emoji_features(self, title: str) -> Dict[str, Any]:
        """Emoji and special character analysis"""
        features = {
            'emoji_count': 0,
            'has_emojis': 0,
            'face_emoji_count': 0,
            'heart_emoji_count': 0,
            'fire_emoji_count': 0
        }

        if EMOJI_AVAILABLE:
            try:
                emojis = [char for char in title if char in emoji.EMOJI_DATA]
                features['emoji_count'] = len(emojis)
                features['has_emojis'] = int(len(emojis) > 0)

                # Specific emoji types
                features['face_emoji_count'] = sum(
                    1 for e in emojis if any(face in emoji.demojize(e) for face in ['face', 'smile', 'laugh', 'cry']))
                features['heart_emoji_count'] = title.count('❤️') + title.count('💕') + title.count('💖') + title.count(
                    '💗')
                features['fire_emoji_count'] = title.count('🔥')
            except:
                pass

        return features

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def extract_batch(self, titles: List[str]) -> pd.DataFrame:
        """Extract features from multiple titles"""
        features_list = []
        for i, title in enumerate(titles):
            try:
                features = self.extract_features(title)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing title '{title}': {e}")
                # Add empty features dict to maintain alignment
                features_list.append({})

        df = pd.DataFrame(features_list)
        return df.fillna(0)


class ThumbnailFeatureExtractor:
    def __init__(self):
        # Initialize computer vision tools
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None

        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except:
            self.eye_cascade = None

    def extract_features(self, thumbnail_url: str) -> Dict[str, Any]:
        """Extract comprehensive features from a single thumbnail"""
        features = {}

        try:
            # Download and process image
            response = requests.get(thumbnail_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            image = Image.open(io.BytesIO(response.content))

            # Convert to different formats for analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image = np.array(image)

            # === BASIC IMAGE PROPERTIES ===
            features.update(self._extract_basic_image_features(image, cv_image))

            # === COLOR ANALYSIS ===
            features.update(self._extract_color_features(cv_image, rgb_image))

            # === FACE AND HUMAN DETECTION ===
            features.update(self._extract_face_features(cv_image))

            # === COMPOSITION AND LAYOUT ===
            features.update(self._extract_composition_features(cv_image))

            # === TEXT DETECTION ===
            features.update(self._extract_text_features(cv_image))

            # === VISUAL COMPLEXITY ===
            features.update(self._extract_complexity_features(cv_image))

            # === AESTHETIC FEATURES ===
            features.update(self._extract_aesthetic_features(cv_image, rgb_image))

            # === OBJECT AND SHAPE DETECTION ===
            features.update(self._extract_shape_features(cv_image))

        except Exception as e:
            print(f"Error processing thumbnail {thumbnail_url}: {e}")
            # Return default values
            features = self._get_default_features()

        return features

    def _extract_basic_image_features(self, image: Image.Image, cv_image: np.ndarray) -> Dict[str, Any]:
        """Basic image properties"""
        height, width = cv_image.shape[:2]

        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'total_pixels': width * height,
            'is_landscape': int(width > height),
            'is_portrait': int(height > width),
            'is_square': int(abs(width - height) < 10),
            'resolution_category': self._categorize_resolution(width, height)
        }

    def _extract_color_features(self, cv_image: np.ndarray, rgb_image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive color analysis"""
        features = {}

        # Convert to different color spaces
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)

        # Basic color statistics
        bgr_mean = np.mean(cv_image, axis=(0, 1))
        bgr_std = np.std(cv_image, axis=(0, 1))

        features.update({
            'mean_blue': bgr_mean[0],
            'mean_green': bgr_mean[1],
            'mean_red': bgr_mean[2],
            'std_blue': bgr_std[0],
            'std_green': bgr_std[1],
            'std_red': bgr_std[2]
        })

        # HSV analysis
        h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
        h_std, s_std, v_std = np.std(hsv, axis=(0, 1))

        features.update({
            'hue_mean': h_mean,
            'saturation_mean': s_mean,
            'brightness_mean': v_mean,
            'hue_std': h_std,
            'saturation_std': s_std,
            'brightness_std': v_std,
            'overall_brightness': np.mean(v_mean),
            'color_saturation': np.mean(s_mean),
            'color_vibrancy': np.mean(s_mean) * np.mean(v_mean) / 255
        })

        # Color diversity and dominant colors
        pixels = cv_image.reshape(-1, 3)
        unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))

        features.update({
            'unique_color_count': unique_colors,
            'color_diversity': unique_colors / len(pixels) if len(pixels) > 0 else 0,
            'color_contrast': np.std(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))
        })

        # Dominant color analysis
        try:
            # K-means clustering for dominant colors
            from sklearn.cluster import KMeans
            pixels_sample = pixels[::10]  # Sample for performance
            if len(pixels_sample) > 5:
                kmeans = KMeans(n_clusters=min(5, len(pixels_sample)), random_state=42, n_init=10)
                kmeans.fit(pixels_sample)

                # Get dominant colors
                dominant_colors = kmeans.cluster_centers_
                features.update({
                    'dominant_color_1_b': dominant_colors[0][0] if len(dominant_colors) > 0 else 0,
                    'dominant_color_1_g': dominant_colors[0][1] if len(dominant_colors) > 0 else 0,
                    'dominant_color_1_r': dominant_colors[0][2] if len(dominant_colors) > 0 else 0,
                    'dominant_color_2_b': dominant_colors[1][0] if len(dominant_colors) > 1 else 0,
                    'dominant_color_2_g': dominant_colors[1][1] if len(dominant_colors) > 1 else 0,
                    'dominant_color_2_r': dominant_colors[1][2] if len(dominant_colors) > 1 else 0
                })
        except:
            # Fallback: simple mean as dominant color
            features.update({
                'dominant_color_1_b': bgr_mean[0],
                'dominant_color_1_g': bgr_mean[1],
                'dominant_color_1_r': bgr_mean[2],
                'dominant_color_2_b': bgr_mean[0],
                'dominant_color_2_g': bgr_mean[1],
                'dominant_color_2_r': bgr_mean[2]
            })

        # Color temperature (warm vs cool)
        red_blue_ratio = np.mean(bgr_mean[2]) / (np.mean(bgr_mean[0]) + 1)
        features['color_temperature'] = red_blue_ratio
        features['is_warm_toned'] = int(red_blue_ratio > 1.1)
        features['is_cool_toned'] = int(red_blue_ratio < 0.9)

        return features

    def _extract_face_features(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Face and human detection features"""
        features = {
            'num_faces': 0,
            'largest_face_area': 0,
            'total_face_area_ratio': 0,
            'faces_in_center': 0,
            'num_eyes': 0,
            'face_density': 0,
            'has_large_face': 0,
            'multiple_faces': 0
        }

        if self.face_cascade is None:
            return features

        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                image_area = cv_image.shape[0] * cv_image.shape[1]
                face_areas = [w * h for (x, y, w, h) in faces]
                total_face_area = sum(face_areas)

                features.update({
                    'num_faces': len(faces),
                    'largest_face_area': max(face_areas),
                    'total_face_area_ratio': total_face_area / image_area,
                    'face_density': len(faces) / image_area * 10000,  # faces per 10k pixels
                    'has_large_face': int(max(face_areas) > image_area * 0.1),
                    'multiple_faces': int(len(faces) > 1)
                })

                # Check if faces are in center
                img_center_x, img_center_y = cv_image.shape[1] // 2, cv_image.shape[0] // 2
                center_faces = 0

                for (x, y, w, h) in faces:
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    # Check if face center is in middle third of image
                    if (cv_image.shape[1] // 3 < face_center_x < 2 * cv_image.shape[1] // 3 and
                            cv_image.shape[0] // 3 < face_center_y < 2 * cv_image.shape[0] // 3):
                        center_faces += 1

                features['faces_in_center'] = center_faces

                # Eye detection within face regions
                if self.eye_cascade is not None:
                    total_eyes = 0
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y + h, x:x + w]
                        eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
                        total_eyes += len(eyes)
                    features['num_eyes'] = total_eyes

        except Exception as e:
            print(f"Error in face detection: {e}")

        return features

    def _extract_composition_features(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Image composition and layout features"""
        features = {}

        # Edge detection for composition analysis
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Edge density
        features['edge_density'] = np.sum(edges > 0) / edges.size
        features['total_edges'] = np.sum(edges > 0)

        # Rule of thirds analysis
        height, width = cv_image.shape[:2]
        third_h, third_w = height // 3, width // 3

        # Divide image into 9 sections for rule of thirds
        sections = []
        for i in range(3):
            for j in range(3):
                section = edges[i * third_h:(i + 1) * third_h, j * third_w:(j + 1) * third_w]
                sections.append(np.sum(section > 0))

        # Rule of thirds hotspots (intersections)
        hotspot_activity = (sections[0] + sections[2] + sections[6] + sections[8]) / sum(sections) if sum(
            sections) > 0 else 0
        features['rule_of_thirds_score'] = hotspot_activity

        # Center vs periphery activity
        center_activity = sections[4] / sum(sections) if sum(sections) > 0 else 0
        features['center_focus'] = center_activity

        # Symmetry analysis
        left_half = edges[:, :width // 2]
        right_half = edges[:, width // 2:]
        right_half_flipped = np.fliplr(right_half)

        # Resize to match if odd width
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]

        symmetry_score = np.mean(left_half == right_half_flipped)
        features['horizontal_symmetry'] = symmetry_score

        # Vertical symmetry
        top_half = edges[:height // 2, :]
        bottom_half = edges[height // 2:, :]
        bottom_half_flipped = np.flipud(bottom_half)

        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half_flipped = bottom_half_flipped[:min_height, :]

        v_symmetry_score = np.mean(top_half == bottom_half_flipped)
        features['vertical_symmetry'] = v_symmetry_score

        # Overall composition balance
        features['composition_balance'] = (symmetry_score + v_symmetry_score) / 2

        return features

    def _extract_text_features(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Text detection and analysis in thumbnails"""
        features = {
            'has_text': 0,
            'text_area_ratio': 0,
            'text_regions': 0,
            'text_density': 0,
            'large_text_presence': 0,
            'text_contrast': 0
        }

        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Use MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)

            if len(regions) > 0:
                # Create mask for text regions
                mask = np.zeros_like(gray)
                for region in regions:
                    # Filter regions by size (likely text)
                    if 20 < len(region) < 2000:  # Reasonable text region size
                        for point in region:
                            mask[point[1], point[0]] = 255

                text_pixels = np.sum(mask > 0)
                total_pixels = mask.size

                features.update({
                    'has_text': int(text_pixels > 100),
                    'text_area_ratio': text_pixels / total_pixels,
                    'text_regions': len([r for r in regions if 20 < len(r) < 2000]),
                    'text_density': text_pixels / total_pixels,
                    'large_text_presence': int(any(len(r) > 500 for r in regions))
                })

                # Text contrast (difference between text regions and background)
                if text_pixels > 0:
                    text_intensity = np.mean(gray[mask > 0])
                    bg_intensity = np.mean(gray[mask == 0])
                    features['text_contrast'] = abs(text_intensity - bg_intensity) / 255

        except Exception as e:
            print(f"Error in text detection: {e}")

        return features

    def _extract_complexity_features(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Visual complexity and information density"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Entropy (information content)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # Gradient magnitude (how much change in intensity)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Texture analysis using Local Binary Patterns concept
        # Simplified version
        texture_variance = np.var(gray)

        # Corner detection (Harris corners)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corner_response = np.sum(corners > 0.01 * corners.max())

        return {
            'visual_entropy': entropy,
            'gradient_magnitude_mean': np.mean(gradient_magnitude),
            'gradient_magnitude_std': np.std(gradient_magnitude),
            'texture_variance': texture_variance,
            'corner_count': corner_response,
            'visual_complexity': entropy * np.mean(gradient_magnitude) / 1000,  # Combined complexity score
            'detail_level': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        }

    def _extract_aesthetic_features(self, cv_image: np.ndarray, rgb_image: np.ndarray) -> Dict[str, Any]:
        """Aesthetic and design quality features"""
        features = {}

        # Color harmony (simplified)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hue_values = hsv[:, :, 0].flatten()
        hue_hist = np.histogram(hue_values, bins=36, range=(0, 180))[0]

        # Find dominant hue ranges
        dominant_hues = np.argsort(hue_hist)[-3:]  # Top 3 hue ranges
        hue_harmony = np.std(dominant_hues) / 36  # Normalized hue spread

        features['color_harmony'] = 1 - hue_harmony  # Lower spread = better harmony

        # Brightness distribution
        brightness = hsv[:, :, 2]
        features.update({
            'brightness_uniformity': 1 - (np.std(brightness) / 255),
            'has_good_contrast': int(np.std(brightness) > 30),
            'overexposed_ratio': np.sum(brightness > 240) / brightness.size,
            'underexposed_ratio': np.sum(brightness < 15) / brightness.size
        })

        # Golden ratio approximation in composition
        height, width = cv_image.shape[:2]
        golden_ratio = 1.618
        actual_ratio = width / height if height > 0 else 0
        features['golden_ratio_deviation'] = abs(actual_ratio - golden_ratio) / golden_ratio

        # Image sharpness (using Laplacian variance)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['sharpness'] = laplacian_var
        features['is_sharp'] = int(laplacian_var > 100)

        return features

    def _extract_shape_features(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Geometric shapes and object detection"""
        features = {
            'circle_count': 0,
            'rectangle_count': 0,
            'triangle_count': 0,
            'line_count': 0,
            'arrow_indicators': 0,
            'geometric_elements': 0
        }

        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Circle detection
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                       param1=50, param2=30, minRadius=10, maxRadius=100)
            if circles is not None:
                features['circle_count'] = len(circles[0])

            # Line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
            if lines is not None:
                features['line_count'] = len(lines)

            # Contour analysis for rectangles and other shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangles = 0
            triangles = 0

            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Filter by area
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    if len(approx) == 3:
                        triangles += 1
                    elif len(approx) == 4:
                        rectangles += 1

            features['rectangle_count'] = rectangles
            features['triangle_count'] = triangles
            features['geometric_elements'] = circles is not None and len(circles[0]) + rectangles + triangles

            # Arrow detection (simplified - look for triangular shapes near lines)
            if triangles > 0 and lines is not None and len(lines) > 0:
                features['arrow_indicators'] = min(triangles, len(lines) // 2)

        except Exception as e:
            print(f"Error in shape detection: {e}")

        return features

    def _categorize_resolution(self, width: int, height: int) -> int:
        """Categorize image resolution"""
        total_pixels = width * height
        if total_pixels >= 2000000:  # 2MP+
            return 3  # High
        elif total_pixels >= 500000:  # 0.5-2MP
            return 2  # Medium
        else:
            return 1  # Low

    def _get_default_features(self) -> Dict[str, Any]:
        """Return default feature values when image processing fails"""
        return {
            # Basic features
            'width': 0, 'height': 0, 'aspect_ratio': 0, 'total_pixels': 0,
            'is_landscape': 0, 'is_portrait': 0, 'is_square': 0, 'resolution_category': 1,

            # Color features
            'mean_blue': 0, 'mean_green': 0, 'mean_red': 0,
            'std_blue': 0, 'std_green': 0, 'std_red': 0,
            'hue_mean': 0, 'saturation_mean': 0, 'brightness_mean': 0,
            'hue_std': 0, 'saturation_std': 0, 'brightness_std': 0,
            'overall_brightness': 0, 'color_saturation': 0, 'color_vibrancy': 0,
            'unique_color_count': 0, 'color_diversity': 0, 'color_contrast': 0,
            'dominant_color_1_b': 0, 'dominant_color_1_g': 0, 'dominant_color_1_r': 0,
            'dominant_color_2_b': 0, 'dominant_color_2_g': 0, 'dominant_color_2_r': 0,
            'color_temperature': 1, 'is_warm_toned': 0, 'is_cool_toned': 0,

            # Face features
            'num_faces': 0, 'largest_face_area': 0, 'total_face_area_ratio': 0,
            'faces_in_center': 0, 'num_eyes': 0, 'face_density': 0,
            'has_large_face': 0, 'multiple_faces': 0,

            # Composition features
            'edge_density': 0, 'total_edges': 0, 'rule_of_thirds_score': 0,
            'center_focus': 0, 'horizontal_symmetry': 0, 'vertical_symmetry': 0,
            'composition_balance': 0,

            # Text features
            'has_text': 0, 'text_area_ratio': 0, 'text_regions': 0,
            'text_density': 0, 'large_text_presence': 0, 'text_contrast': 0,

            # Complexity features
            'visual_entropy': 0, 'gradient_magnitude_mean': 0, 'gradient_magnitude_std': 0,
            'texture_variance': 0, 'corner_count': 0, 'visual_complexity': 0, 'detail_level': 0,

            # Aesthetic features
            'color_harmony': 0, 'brightness_uniformity': 0, 'has_good_contrast': 0,
            'overexposed_ratio': 0, 'underexposed_ratio': 0, 'golden_ratio_deviation': 1,
            'sharpness': 0, 'is_sharp': 0,

            # Shape features
            'circle_count': 0, 'rectangle_count': 0, 'triangle_count': 0,
            'line_count': 0, 'arrow_indicators': 0, 'geometric_elements': 0
        }

    def extract_batch(self, thumbnail_urls: List[str]) -> pd.DataFrame:
        """Extract features from multiple thumbnails"""
        features_list = []

        for i, url in enumerate(thumbnail_urls):
            try:
                print(f"Processing thumbnail {i + 1}/{len(thumbnail_urls)}")
                features = self.extract_features(url)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing thumbnail {i + 1}: {e}")
                features_list.append(self._get_default_features())

        df = pd.DataFrame(features_list)
        return df.fillna(0)