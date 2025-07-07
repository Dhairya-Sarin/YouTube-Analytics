import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    def __init__(self):
        self.correlation_threshold = 0.3
        self.high_correlation_threshold = 0.5

    def calculate_correlations(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix between features and target metrics"""
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Remove any infinite or NaN values
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
        numeric_data = numeric_data.fillna(0)

        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()

        # Extract correlations with target columns
        target_correlations = correlation_matrix[target_columns]

        # Remove target columns from rows (to avoid self-correlation)
        feature_correlations = target_correlations.drop(target_columns, axis=0, errors='ignore')

        return feature_correlations

    def plot_correlation_heatmap(self, correlations: pd.DataFrame, title: str) -> go.Figure:
        """Create an interactive correlation heatmap"""
        # Sort by absolute correlation values
        abs_corr_sum = correlations.abs().sum(axis=1).sort_values(ascending=False)
        correlations_sorted = correlations.loc[abs_corr_sum.index]

        # Limit to top 30 features for readability
        if len(correlations_sorted) > 30:
            correlations_sorted = correlations_sorted.head(30)

        fig = go.Figure(data=go.Heatmap(
            z=correlations_sorted.values,
            x=correlations_sorted.columns,
            y=correlations_sorted.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlations_sorted.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 9},
            hoverongaps=False,
            colorbar=dict(
                title="Correlation",
                titleside="right"
            )
        ))

        fig.update_layout(
            title=f"{title} - Correlation with Engagement Metrics",
            xaxis_title="Engagement Metrics",
            yaxis_title="Features",
            height=max(600, len(correlations_sorted) * 20),
            width=900,
            font=dict(size=10),
            margin=dict(l=200, r=100, t=80, b=50)
        )

        return fig

    def get_top_correlations(self, correlations: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """Get top correlations for each metric"""
        top_correlations = []

        for metric in correlations.columns:
            metric_corr = correlations[metric].abs().sort_values(ascending=False)

            for feature, abs_corr_value in metric_corr.head(top_n).items():
                original_corr = correlations.loc[feature, metric]
                top_correlations.append({
                    'Feature': feature,
                    'Metric': metric,
                    'Correlation': original_corr,
                    'Abs_Correlation': abs_corr_value,
                    'Direction': 'Positive' if original_corr > 0 else 'Negative',
                    'Strength': self._get_correlation_strength(abs_corr_value)
                })

        df = pd.DataFrame(top_correlations)
        return df.sort_values('Abs_Correlation', ascending=False).drop_duplicates(subset=['Feature', 'Metric'])

    def _get_correlation_strength(self, abs_corr: float) -> str:
        """Categorize correlation strength"""
        if abs_corr >= 0.7:
            return 'Very Strong'
        elif abs_corr >= 0.5:
            return 'Strong'
        elif abs_corr >= 0.3:
            return 'Moderate'
        elif abs_corr >= 0.1:
            return 'Weak'
        else:
            return 'Very Weak'

    def plot_feature_importance(self, correlations: pd.DataFrame, metric: str, top_n: int = 15) -> go.Figure:
        """Plot top features for a specific metric"""
        if metric not in correlations.columns:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(text=f"Metric '{metric}' not found",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        metric_corr = correlations[metric].abs().sort_values(ascending=False).head(top_n)
        original_values = [correlations.loc[feature, metric] for feature in metric_corr.index]

        colors = ['red' if val < 0 else 'blue' for val in original_values]

        fig = go.Figure(data=go.Bar(
            x=metric_corr.values,
            y=metric_corr.index,
            orientation='h',
            marker_color=colors,
            text=[f"{val:.3f}" for val in original_values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Correlation: %{text}<br>Absolute: %{x:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Top {top_n} Features Correlated with {metric.title()}",
            xaxis_title="Absolute Correlation",
            yaxis_title="Features",
            height=max(500, top_n * 25),
            width=800,
            showlegend=False,
            margin=dict(l=200, r=50, t=80, b=50)
        )

        return fig

    def generate_insights(self, correlations: pd.DataFrame, threshold: float = 0.3) -> Dict[str, List[str]]:
        """Generate textual insights from correlation analysis"""
        insights = {}

        for metric in correlations.columns:
            metric_insights = []

            # Strong positive correlations
            strong_positive = correlations[correlations[metric] > threshold][metric].sort_values(ascending=False)
            if not strong_positive.empty:
                top_positive = strong_positive.head(5)
                features_text = ", ".join([f"{feature} ({corr:.3f})" for feature, corr in top_positive.items()])
                metric_insights.append(f"✅ Positive drivers: {features_text}")

            # Strong negative correlations
            strong_negative = correlations[correlations[metric] < -threshold][metric].sort_values()
            if not strong_negative.empty:
                top_negative = strong_negative.head(3)
                features_text = ", ".join([f"{feature} ({corr:.3f})" for feature, corr in top_negative.items()])
                metric_insights.append(f"❌ Negative factors: {features_text}")

            # Overall insights
            if strong_positive.empty and strong_negative.empty:
                metric_insights.append(f"ℹ️ No strong correlations found above {threshold} threshold")

            insights[metric] = metric_insights

        return insights

    def compare_feature_groups(self, correlations: pd.DataFrame,
                               title_prefix: str = "title_",
                               thumbnail_prefix: str = "thumbnail_") -> Dict[str, pd.DataFrame]:
        """Compare correlations between different feature groups"""
        # This method would be used if we had prefixed features
        # For now, we'll identify features by common patterns

        title_features = correlations[
            correlations.index.str.contains('word|char|sentiment|clickbait|content_', na=False)]
        thumbnail_features = correlations[correlations.index.str.contains('color|face|edge|bright|contrast', na=False)]

        return {
            'title_features': title_features,
            'thumbnail_features': thumbnail_features
        }

    def create_correlation_summary(self, correlations: pd.DataFrame) -> pd.DataFrame:
        """Create a summary of correlation statistics"""
        summary_stats = []

        for metric in correlations.columns:
            metric_corr = correlations[metric].dropna()

            summary_stats.append({
                'Metric': metric,
                'Total_Features': len(metric_corr),
                'Strong_Positive': len(metric_corr[metric_corr > 0.5]),
                'Moderate_Positive': len(metric_corr[(metric_corr > 0.3) & (metric_corr <= 0.5)]),
                'Weak_Positive': len(metric_corr[(metric_corr > 0.1) & (metric_corr <= 0.3)]),
                'Weak_Negative': len(metric_corr[(metric_corr < -0.1) & (metric_corr >= -0.3)]),
                'Moderate_Negative': len(metric_corr[(metric_corr < -0.3) & (metric_corr >= -0.5)]),
                'Strong_Negative': len(metric_corr[metric_corr < -0.5]),
                'Max_Positive': metric_corr.max(),
                'Max_Negative': metric_corr.min(),
                'Mean_Abs_Correlation': metric_corr.abs().mean()
            })

        return pd.DataFrame(summary_stats)


class AdvancedAnalytics:
    def __init__(self):
        self.scaler = StandardScaler()

    def detect_outliers(self, data: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
        """Detect outliers in engagement metrics"""
        if column not in data.columns:
            return pd.DataFrame()

        values = data[column]

        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (values < lower_bound) | (values > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers_mask = z_scores > 3

        else:
            # Percentile method
            lower_percentile = values.quantile(0.05)
            upper_percentile = values.quantile(0.95)
            outliers_mask = (values < lower_percentile) | (values > upper_percentile)

        outliers = data[outliers_mask].copy()
        outliers['outlier_type'] = 'high' if method == 'iqr' else 'extreme'
        outliers['outlier_score'] = values[outliers_mask]

        return outliers.sort_values(column, ascending=False)

    def perform_clustering(self, features: pd.DataFrame, n_clusters: int = 3) -> List[List[int]]:
        """Perform K-means clustering on features"""
        if len(features) < n_clusters:
            return [list(range(len(features)))]

        # Prepare data
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0)

        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)

        # Group indices by cluster
        clusters = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0].tolist()
            clusters.append(cluster_indices)

        return clusters

    def analyze_temporal_patterns(self, data: pd.DataFrame, date_column: str) -> Dict[str, any]:
        """Analyze temporal patterns in the data"""
        if date_column not in data.columns:
            return {}

        # Convert to datetime if not already
        data[date_column] = pd.to_datetime(data[date_column])

        # Extract temporal features
        data['day_of_week'] = data[date_column].dt.day_name()
        data['month'] = data[date_column].dt.month_name()
        data['hour'] = data[date_column].dt.hour

        temporal_analysis = {
            'day_of_week_performance': data.groupby('day_of_week')['views'].mean().to_dict(),
            'monthly_performance': data.groupby('month')['views'].mean().to_dict(),
            'hourly_performance': data.groupby('hour')['views'].mean().to_dict() if 'hour' in data.columns else {},
            'upload_frequency': data.groupby(data[date_column].dt.date).size().describe().to_dict()
        }

        return temporal_analysis

    def calculate_feature_importance_pca(self, features: pd.DataFrame,
                                         engagement_metric: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using PCA"""
        # Prepare data
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0)

        if len(numeric_features.columns) < 2:
            return pd.DataFrame()

        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)

        # Apply PCA
        pca = PCA(n_components=min(5, len(numeric_features.columns)))
        pca_features = pca.fit_transform(scaled_features)

        # Calculate correlations with engagement metric
        pca_correlations = []
        for i in range(pca.n_components_):
            corr = np.corrcoef(pca_features[:, i], engagement_metric)[0, 1]
            pca_correlations.append(corr)

        # Get feature loadings
        feature_importance = []
        for i, feature_name in enumerate(numeric_features.columns):
            # Weighted sum of loadings across components
            importance = sum(abs(pca.components_[j, i]) * abs(pca_correlations[j])
                             for j in range(pca.n_components_))
            feature_importance.append({
                'feature': feature_name,
                'importance': importance,
                'explained_variance': sum(pca.explained_variance_ratio_)
            })

        importance_df = pd.DataFrame(feature_importance)
        return importance_df.sort_values('importance', ascending=False)

    def generate_performance_segments(self, data: pd.DataFrame,
                                      metric: str = 'views') -> Dict[str, pd.DataFrame]:
        """Segment videos into performance categories"""
        if metric not in data.columns:
            return {}

        # Define percentile-based segments
        percentiles = data[metric].quantile([0.25, 0.5, 0.75, 0.9])

        segments = {
            'top_performers': data[data[metric] >= percentiles[0.9]],
            'high_performers': data[(data[metric] >= percentiles[0.75]) & (data[metric] < percentiles[0.9])],
            'average_performers': data[(data[metric] >= percentiles[0.25]) & (data[metric] < percentiles[0.75])],
            'low_performers': data[data[metric] < percentiles[0.25]]
        }

        return segments

    def calculate_engagement_velocity(self, data: pd.DataFrame,
                                      date_column: str = 'published_at') -> pd.DataFrame:
        """Calculate how quickly videos gain engagement"""
        if date_column not in data.columns:
            return data

        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])

        # Calculate days since publication
        current_date = pd.Timestamp.now()
        data['days_since_publish'] = (current_date - data[date_column]).dt.days

        # Calculate velocity metrics (engagement per day)
        data['views_per_day'] = data['views'] / (data['days_since_publish'] + 1)
        data['likes_per_day'] = data['likes'] / (data['days_since_publish'] + 1)
        data['comments_per_day'] = data['comments'] / (data['days_since_publish'] + 1)

        return data

    def identify_content_patterns(self, title_features: pd.DataFrame,
                                  engagement_data: pd.DataFrame) -> Dict[str, any]:
        """Identify patterns in successful content"""
        if title_features is None or len(title_features) == 0:
            return {}

        # Combine data
        combined_data = pd.concat([title_features, engagement_data], axis=1)

        # Define success criteria (top 25% by views)
        success_threshold = engagement_data['views'].quantile(0.75)
        successful_videos = combined_data[combined_data['views'] >= success_threshold]
        unsuccessful_videos = combined_data[combined_data['views'] < success_threshold]

        patterns = {}

        # Compare feature means between successful and unsuccessful videos
        for column in title_features.columns:
            if pd.api.types.is_numeric_dtype(title_features[column]):
                successful_mean = successful_videos[column].mean()
                unsuccessful_mean = unsuccessful_videos[column].mean()

                if not np.isnan(successful_mean) and not np.isnan(unsuccessful_mean):
                    difference = successful_mean - unsuccessful_mean
                    relative_difference = difference / (unsuccessful_mean + 1e-10)

                    if abs(relative_difference) > 0.2:  # 20% difference threshold
                        patterns[column] = {
                            'successful_mean': successful_mean,
                            'unsuccessful_mean': unsuccessful_mean,
                            'difference': difference,
                            'relative_difference': relative_difference,
                            'pattern_type': 'higher' if difference > 0 else 'lower'
                        }

        return patterns

    def create_performance_dashboard_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Prepare data for performance dashboard"""
        dashboard_data = {}

        # Basic statistics
        dashboard_data['basic_stats'] = {
            'total_videos': len(data),
            'total_views': data['views'].sum(),
            'avg_views': data['views'].mean(),
            'median_views': data['views'].median(),
            'total_likes': data['likes'].sum(),
            'total_comments': data['comments'].sum(),
            'avg_engagement_rate': data['engagement_rate'].mean() if 'engagement_rate' in data.columns else 0
        }

        # Performance distribution
        dashboard_data['performance_distribution'] = {
            'views_quartiles': data['views'].quantile([0.25, 0.5, 0.75]).to_dict(),
            'engagement_quartiles': data['engagement_rate'].quantile(
                [0.25, 0.5, 0.75]).to_dict() if 'engagement_rate' in data.columns else {}
        }

        # Top performers
        dashboard_data['top_performers'] = data.nlargest(5, 'views')[['views', 'likes', 'comments']].to_dict('records')

        return dashboard_data