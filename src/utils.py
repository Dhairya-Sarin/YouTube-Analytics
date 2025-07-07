import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, Any, Optional, List
import hashlib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Try to import YAML, but don't fail if not available
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        if YAML_AVAILABLE and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            # Return default configuration
            return {
                'api': {
                    'timeout': 10,
                    'max_retries': 3
                },
                'analysis': {
                    'correlation_threshold': 0.3,
                    'max_videos': 500,
                    'cache_duration_hours': 24
                },
                'features': {
                    'enable_sentiment_analysis': True,
                    'enable_face_detection': True,
                    'enable_text_detection': True
                }
            }
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def cache_data(data: Any, cache_key: str, cache_dir: str = "data/cache") -> bool:
    """Cache data to disk"""
    try:
        os.makedirs(cache_dir, exist_ok=True)

        # Create filename from cache key
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{safe_key}.pkl")

        # Add timestamp to data
        cache_data_dict = {
            'data': data,
            'timestamp': datetime.now(),
            'cache_key': cache_key
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data_dict, f)

        return True
    except Exception as e:
        print(f"Error caching data: {e}")
        return False


def load_cached_data(cache_key: str, cache_dir: str = "data/cache",
                     max_age_hours: int = 24) -> Optional[Any]:
    """Load cached data if it exists and is not expired"""
    try:
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{safe_key}.pkl")

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'rb') as f:
            cache_data_dict = pickle.load(f)

        # Check if cache is expired
        cache_age = datetime.now() - cache_data_dict['timestamp']
        if cache_age > timedelta(hours=max_age_hours):
            return None

        return cache_data_dict['data']
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None


def export_data(data: Dict[str, pd.DataFrame], format_type: str = 'csv',
                filename: str = None) -> str:
    """Export analysis results to various formats"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"youtube_analysis_{timestamp}"

    try:
        if format_type.lower() == 'csv':
            # Export each dataframe as separate CSV
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df.to_csv(f"{filename}_{name}.csv", index=False)
            return f"Exported {len(data)} CSV files with prefix: {filename}"

        elif format_type.lower() == 'excel':
            # Export all dataframes to different sheets in one Excel file
            with pd.ExcelWriter(f"{filename}.xlsx", engine='openpyxl') as writer:
                for name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        # Truncate sheet name to 31 characters (Excel limit)
                        sheet_name = name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            return f"Exported to: {filename}.xlsx"

        elif format_type.lower() == 'json':
            # Convert dataframes to JSON
            json_data = {}
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    json_data[name] = df.to_dict('records')
                else:
                    json_data[name] = df

            with open(f"{filename}.json", 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            return f"Exported to: {filename}.json"

        else:
            raise ValueError(f"Unsupported format: {format_type}")

    except Exception as e:
        return f"Error exporting data: {e}"


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize feature names"""
    df = df.copy()

    # Replace problematic characters
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    df.columns = df.columns.str.replace('__+', '_', regex=True)
    df.columns = df.columns.str.strip('_')
    df.columns = df.columns.str.lower()

    return df


def validate_youtube_url(url: str) -> bool:
    """Validate if the provided URL is a valid YouTube channel URL"""
    youtube_patterns = [
        r'youtube\.com/channel/',
        r'youtube\.com/c/',
        r'youtube\.com/user/',
        r'youtube\.com/@',
        r'youtube\.com/channel/UC[a-zA-Z0-9_-]{22}'
    ]

    import re
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def calculate_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate comprehensive statistics for all numeric features"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    stats = {}
    for col in numeric_columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'skewness': float(df[col].skew()),
            'kurtosis': float(df[col].kurtosis()),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': float((df[col].isnull().sum() / len(df)) * 100)
        }

    return stats


def detect_feature_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
    """Detect outliers in all numeric features"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}

    for col in numeric_columns:
        col_outliers = []

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()

        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = df[z_scores > 3].index.tolist()

        outliers[col] = col_outliers

    return outliers


def create_feature_groups() -> Dict[str, List[str]]:
    """Define feature groups for organized analysis"""
    return {
        'title_basic': [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'unique_words', 'lexical_diversity'
        ],
        'title_sentiment': [
            'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'sentiment_compound', 'textblob_polarity', 'textblob_subjectivity'
        ],
        'title_clickbait': [
            'clickbait_curiosity_count', 'clickbait_urgency_count', 'clickbait_superlatives_count',
            'power_words_count', 'listicle_indicators', 'curiosity_gap_count'
        ],
        'title_structure': [
            'starts_with_how', 'starts_with_why', 'starts_with_what',
            'has_colon', 'has_brackets', 'is_question'
        ],
        'title_content_type': [
            'content_tutorial_score', 'content_review_score', 'content_entertainment_score',
            'content_educational_score', 'content_gaming_score', 'content_vlog_score'
        ],
        'title_temporal': [
            'temporal_immediate_count', 'temporal_recent_count', 'temporal_future_count',
            'has_year', 'recent_year'
        ],
        'thumbnail_color': [
            'mean_red', 'mean_green', 'mean_blue', 'hue_mean', 'saturation_mean',
            'brightness_mean', 'color_contrast', 'color_diversity', 'color_vibrancy'
        ],
        'thumbnail_composition': [
            'edge_density', 'rule_of_thirds_score', 'center_focus', 'horizontal_symmetry',
            'vertical_symmetry', 'composition_balance'
        ],
        'thumbnail_faces': [
            'num_faces', 'largest_face_area', 'total_face_area_ratio', 'faces_in_center',
            'num_eyes', 'has_large_face', 'multiple_faces'
        ],
        'thumbnail_aesthetic': [
            'color_harmony', 'brightness_uniformity', 'has_good_contrast',
            'golden_ratio_deviation', 'sharpness', 'is_sharp'
        ],
        'thumbnail_complexity': [
            'visual_entropy', 'gradient_magnitude_mean', 'texture_variance',
            'corner_count', 'visual_complexity', 'detail_level'
        ],
        'thumbnail_objects': [
            'circle_count', 'rectangle_count', 'triangle_count', 'line_count',
            'arrow_indicators', 'geometric_elements'
        ],
        'thumbnail_text': [
            'has_text', 'text_area_ratio', 'text_regions', 'text_density',
            'large_text_presence', 'text_contrast'
        ]
    }


def format_correlation_for_display(correlation_value: float) -> str:
    """Format correlation values for display"""
    if pd.isna(correlation_value):
        return "N/A"

    abs_corr = abs(correlation_value)
    sign = "+" if correlation_value >= 0 else "-"

    if abs_corr >= 0.7:
        strength = "Very Strong"
    elif abs_corr >= 0.5:
        strength = "Strong"
    elif abs_corr >= 0.3:
        strength = "Moderate"
    elif abs_corr >= 0.1:
        strength = "Weak"
    else:
        strength = "Very Weak"

    return f"{sign}{abs_corr:.3f} ({strength})"


def generate_feature_descriptions() -> Dict[str, str]:
    """Generate human-readable descriptions for features"""
    descriptions = {
        # Title features
        'char_count': 'Number of characters in the title',
        'word_count': 'Number of words in the title',
        'sentiment_positive': 'Positive sentiment score (0-1)',
        'sentiment_negative': 'Negative sentiment score (0-1)',
        'sentiment_compound': 'Overall sentiment (-1 to 1)',
        'clickbait_curiosity_count': 'Number of curiosity-inducing words',
        'clickbait_urgency_count': 'Number of urgency words',
        'power_words_count': 'Number of persuasive power words',
        'has_numbers': 'Contains numbers (0 or 1)',
        'is_question': 'Title is formatted as a question',
        'exclamation_count': 'Number of exclamation marks',
        'caps_uppercase_words': 'Number of fully capitalized words',
        'flesch_reading_ease': 'Readability score (higher = easier)',
        'content_tutorial_score': 'How-to/tutorial content indicators',
        'content_review_score': 'Review/comparison content indicators',
        'listicle_indicators': 'List-format content indicators',

        # Thumbnail features
        'num_faces': 'Number of faces detected',
        'total_face_area_ratio': 'Proportion of image occupied by faces',
        'brightness_mean': 'Average brightness (0-255)',
        'saturation_mean': 'Average color saturation (0-255)',
        'color_contrast': 'Color contrast level',
        'edge_density': 'Density of edges/details in image',
        'has_text': 'Contains text overlay (0 or 1)',
        'text_area_ratio': 'Proportion of image with text',
        'color_harmony': 'Color harmony score (0-1)',
        'rule_of_thirds_score': 'Composition following rule of thirds',
        'visual_complexity': 'Overall visual complexity score',
        'sharpness': 'Image sharpness measure',
        'color_vibrancy': 'Color vibrancy/saturation level',
        'circle_count': 'Number of circular elements detected',
        'arrow_indicators': 'Number of arrow-like elements'
    }

    return descriptions


def create_analysis_summary(correlations: pd.DataFrame,
                            engagement_data: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive analysis summary"""
    summary = {
        'overview': {
            'total_features_analyzed': len(correlations),
            'total_videos': len(engagement_data),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'engagement_stats': {
            'avg_views': float(engagement_data['views'].mean()),
            'median_views': float(engagement_data['views'].median()),
            'max_views': float(engagement_data['views'].max()),
            'avg_engagement_rate': float(
                engagement_data['engagement_rate'].mean()) if 'engagement_rate' in engagement_data.columns else 0.0
        },
        'correlation_insights': {},
        'recommendations': []
    }

    # Analyze correlations for each metric
    for metric in correlations.columns:
        metric_corr = correlations[metric].dropna()

        strong_positive = metric_corr[metric_corr > 0.5]
        strong_negative = metric_corr[metric_corr < -0.5]
        moderate_positive = metric_corr[(metric_corr > 0.3) & (metric_corr <= 0.5)]
        moderate_negative = metric_corr[(metric_corr < -0.3) & (metric_corr >= -0.5)]

        summary['correlation_insights'][metric] = {
            'strong_positive_count': len(strong_positive),
            'strong_negative_count': len(strong_negative),
            'moderate_positive_count': len(moderate_positive),
            'moderate_negative_count': len(moderate_negative),
            'top_positive_feature': strong_positive.idxmax() if len(strong_positive) > 0 else None,
            'top_negative_feature': strong_negative.idxmin() if len(strong_negative) > 0 else None,
            'max_correlation': float(metric_corr.max()),
            'min_correlation': float(metric_corr.min())
        }

    return summary


def validate_data_quality(title_features: Optional[pd.DataFrame] = None,
                          thumbnail_features: Optional[pd.DataFrame] = None,
                          engagement_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Validate the quality of extracted data"""
    quality_report = {
        'title_features': {},
        'thumbnail_features': {},
        'engagement_data': {},
        'overall_score': 0,
        'issues': [],
        'recommendations': []
    }

    # Validate title features
    if title_features is not None:
        title_quality = {
            'total_features': len(title_features.columns),
            'missing_data_percentage': float((title_features.isnull().sum().sum() / title_features.size) * 100),
            'zero_variance_features': len([col for col in title_features.columns if title_features[col].var() == 0]),
            'duplicate_rows': int(title_features.duplicated().sum())
        }
        quality_report['title_features'] = title_quality

        if title_quality['missing_data_percentage'] > 20:
            quality_report['issues'].append("High missing data in title features")
        if title_quality['zero_variance_features'] > 5:
            quality_report['issues'].append("Many zero-variance title features detected")

    # Validate thumbnail features
    if thumbnail_features is not None:
        thumbnail_quality = {
            'total_features': len(thumbnail_features.columns),
            'missing_data_percentage': float((thumbnail_features.isnull().sum().sum() / thumbnail_features.size) * 100),
            'zero_variance_features': len(
                [col for col in thumbnail_features.columns if thumbnail_features[col].var() == 0]),
            'processing_failures': int((thumbnail_features == 0).all(axis=1).sum())
        }
        quality_report['thumbnail_features'] = thumbnail_quality

        if thumbnail_quality['processing_failures'] > len(thumbnail_features) * 0.1:
            quality_report['issues'].append("High thumbnail processing failure rate")

    # Validate engagement data
    if engagement_data is not None:
        engagement_quality = {
            'total_videos': len(engagement_data),
            'zero_view_videos': int((engagement_data['views'] == 0).sum()),
            'missing_engagement_data': int(engagement_data[['views', 'likes', 'comments']].isnull().sum().sum()),
            'extreme_outliers': len([col for col in ['views', 'likes', 'comments']
                                     if col in engagement_data.columns and
                                     (engagement_data[col] > engagement_data[col].quantile(0.99) * 10).any()])
        }
        quality_report['engagement_data'] = engagement_quality

        if engagement_quality['zero_view_videos'] > 0:
            quality_report['issues'].append(f"{engagement_quality['zero_view_videos']} videos with zero views")

    # Calculate overall quality score
    issues_count = len(quality_report['issues'])
    quality_report['overall_score'] = max(0, 100 - (issues_count * 15))

    # Generate recommendations
    if quality_report['overall_score'] < 70:
        quality_report['recommendations'].append("Consider increasing sample size or checking data collection process")
    if issues_count > 0:
        quality_report['recommendations'].append("Review and clean identified data quality issues")

    return quality_report


def create_downloadable_report(correlations: pd.DataFrame,
                               insights: Dict[str, List[str]],
                               summary: Dict[str, Any]) -> str:
    """Create a formatted text report for download"""
    report_lines = [
        "YOUTUBE CHANNEL ANALYSIS REPORT",
        "=" * 50,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 20,
        f"Total videos analyzed: {summary['overview']['total_videos']}",
        f"Total features extracted: {summary['overview']['total_features_analyzed']}",
        f"Average views: {summary['engagement_stats']['avg_views']:,.0f}",
        f"Median views: {summary['engagement_stats']['median_views']:,.0f}",
        "",
        "KEY FINDINGS",
        "-" * 15
    ]

    # Add insights for each metric
    for metric, metric_insights in insights.items():
        if metric_insights:
            report_lines.append(f"\n{metric.upper()} INSIGHTS:")
            for insight in metric_insights:
                report_lines.append(f"  • {insight}")

    # Add top correlations
    report_lines.extend([
        "",
        "TOP CORRELATIONS BY METRIC",
        "-" * 30
    ])

    for metric in correlations.columns:
        metric_corr = correlations[metric].abs().sort_values(ascending=False).head(5)
        report_lines.append(f"\n{metric.upper()}:")
        for feature, corr in metric_corr.items():
            original_corr = correlations.loc[feature, metric]
            direction = "positive" if original_corr > 0 else "negative"
            report_lines.append(f"  • {feature}: {original_corr:.3f} ({direction})")

    # Add recommendations
    report_lines.extend([
        "",
        "RECOMMENDATIONS",
        "-" * 15,
        "Based on the correlation analysis, consider the following optimizations:",
        ""
    ])

    # Generate specific recommendations based on top correlations
    for metric in ['views', 'engagement_rate']:
        if metric in correlations.columns:
            top_positive = correlations[correlations[metric] > 0.3][metric].sort_values(ascending=False)
            if not top_positive.empty:
                top_feature = top_positive.index[0]
                corr_value = top_positive.iloc[0]
                report_lines.append(
                    f"• To improve {metric}, focus on optimizing '{top_feature}' (correlation: {corr_value:.3f})")

    report_lines.extend([
        "",
        "METHODOLOGY",
        "-" * 12,
        "This analysis used Pearson correlation coefficients to measure linear relationships",
        "between extracted features and engagement metrics. Features include:",
        "- Title analysis: sentiment, readability, structure, content type indicators",
        "- Thumbnail analysis: visual composition, color analysis, face detection, text detection",
        "",
        "Correlation strength interpretation:",
        "- Very Strong: |r| ≥ 0.7",
        "- Strong: 0.5 ≤ |r| < 0.7",
        "- Moderate: 0.3 ≤ |r| < 0.5",
        "- Weak: 0.1 ≤ |r| < 0.3",
        "- Very Weak: |r| < 0.1",
        "",
        "Note: Correlation does not imply causation. These insights should be tested",
        "through controlled experiments when possible."
    ])

    return "\n".join(report_lines)


def save_analysis_session(session_data: Dict[str, Any],
                          session_name: Optional[str] = None) -> str:
    """Save the entire analysis session for later loading"""
    if session_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"youtube_analysis_session_{timestamp}"

    session_file = f"{session_name}.pkl"

    try:
        with open(session_file, 'wb') as f:
            pickle.dump(session_data, f)
        return f"Session saved as: {session_file}"
    except Exception as e:
        return f"Error saving session: {e}"


def load_analysis_session(session_file: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved analysis session"""
    try:
        with open(session_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading session: {e}")
        return None