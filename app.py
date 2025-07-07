import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Import custom modules
from src.youtube_api import YouTubeAPI
from src.feature_extraction import TitleFeatureExtractor, ThumbnailFeatureExtractor
from src.analysis import CorrelationAnalyzer, AdvancedAnalytics
from src.utils import load_config, cache_data, export_data

# Page configuration
st.set_page_config(
    page_title="YouTube Channel Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better layout
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF0000;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    /* Make correlation heatmap more visible */
    .js-plotly-plot {
        width: 100% !important;
    }
    /* Expand the main content area */
    .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Better table display */
    .dataframe {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">📊 YouTube Channel Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Analytics for YouTube Content Optimization")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Configuration
        st.subheader("🔑 API Settings")
        api_key = st.text_input("YouTube API Key", type="password", help="Get from Google Cloud Console")

        # Channel Configuration
        st.subheader("📺 Channel Settings")
        channel_input = st.text_input("Channel Name/URL", placeholder="Enter channel name or URL")
        max_videos = st.slider("Max Videos to Analyze", 10, 500, 50, step=10)

        # Analysis Configuration
        st.subheader("🔍 Analysis Settings")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Complete Analysis", "Title Features Only", "Thumbnail Features Only", "Temporal Analysis",
             "Competitive Analysis"]
        )

        feature_groups = st.multiselect(
            "Feature Groups to Include",
            ["Basic Metrics", "NLP Features", "Visual Features", "Engagement Patterns", "Temporal Features",
             "Advanced Analytics"],
            default=["Basic Metrics", "NLP Features", "Visual Features"]
        )

        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            correlation_threshold = st.slider("Correlation Threshold", 0.1, 0.8, 0.3, 0.05)
            include_outliers = st.checkbox("Include Outlier Analysis", True)
            enable_clustering = st.checkbox("Enable Content Clustering", False)
            export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])

    # Main content area
    if not api_key:
        display_welcome_screen()
        return

    if not channel_input:
        st.info("👈 Please enter a channel name or URL in the sidebar to begin analysis")
        return

    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Comprehensive Analysis", type="primary", use_container_width=True):
            run_analysis(api_key, channel_input, max_videos, analysis_type, feature_groups,
                         correlation_threshold, include_outliers, enable_clustering, export_format)


def display_welcome_screen():
    """Display welcome screen with instructions"""
    st.markdown("""
    ## Welcome to YouTube Channel Analyzer Pro! 🎉

    This advanced tool helps you understand what makes YouTube videos successful by analyzing:

    ### 📝 Title Features (40+ metrics)
    - **Linguistic Analysis**: Readability, sentiment, complexity
    - **SEO Optimization**: Keywords, trending terms, hashtags
    - **Emotional Triggers**: Urgency, curiosity, power words
    - **Structure Analysis**: Length, punctuation, capitalization

    ### 🖼️ Thumbnail Features (35+ metrics)
    - **Visual Composition**: Colors, contrast, brightness
    - **Face Analysis**: Detection, emotions, eye contact
    - **Text Analysis**: OCR, readability, positioning
    - **Design Elements**: Symmetry, rule of thirds, visual weight

    ### 📊 Advanced Analytics
    - **Correlation Analysis**: Find what drives engagement
    - **Clustering**: Identify content patterns
    - **Temporal Analysis**: Track performance over time
    - **Competitive Analysis**: Compare with similar channels

    ### 🔑 Getting Started
    1. Get your YouTube API key from [Google Cloud Console](https://console.cloud.google.com/)
    2. Enable YouTube Data API v3
    3. Enter your API key in the sidebar
    4. Choose a channel to analyze
    5. Select your analysis preferences
    6. Click "Start Analysis"
    """)

    # API Key Instructions
    with st.expander("📋 How to get YouTube API Key"):
        st.markdown("""
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing
        3. Navigate to "APIs & Services" > "Library"
        4. Search for "YouTube Data API v3" and enable it
        5. Go to "APIs & Services" > "Credentials"
        6. Click "Create Credentials" > "API Key"
        7. Copy the API key and paste it in the sidebar
        """)


def run_analysis(api_key, channel_input, max_videos, analysis_type, feature_groups,
                 correlation_threshold, include_outliers, enable_clustering, export_format):
    """Main analysis workflow"""

    # Initialize components
    youtube_api = YouTubeAPI(api_key)
    title_extractor = TitleFeatureExtractor()
    thumbnail_extractor = ThumbnailFeatureExtractor()
    analyzer = CorrelationAnalyzer()
    advanced_analytics = AdvancedAnalytics()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Get channel data
        status_text.text("🔍 Fetching channel data...")
        progress_bar.progress(0.1)

        channel_data = youtube_api.get_channel_videos(channel_input, max_videos)
        if not channel_data:
            st.error("No videos found or error occurred")
            return

        st.success(f"✅ Found {len(channel_data)} videos from the channel")
        progress_bar.progress(0.2)

        # Step 2: Extract features based on selected groups
        status_text.text("🔄 Extracting features...")

        title_features = None
        thumbnail_features = None

        if "NLP Features" in feature_groups or analysis_type in ["Complete Analysis", "Title Features Only"]:
            status_text.text("📝 Analyzing titles...")
            title_features = title_extractor.extract_batch([video['title'] for video in channel_data])
            progress_bar.progress(0.4)

        if "Visual Features" in feature_groups or analysis_type in ["Complete Analysis", "Thumbnail Features Only"]:
            status_text.text("🖼️ Analyzing thumbnails...")
            thumbnail_features = thumbnail_extractor.extract_batch([video['thumbnail_url'] for video in channel_data])
            progress_bar.progress(0.6)

        # Step 3: Prepare engagement data with video metadata
        status_text.text("📊 Preparing engagement metrics...")
        engagement_data, video_metadata = prepare_engagement_data(channel_data)
        progress_bar.progress(0.8)

        # Step 4: Run analysis
        status_text.text("🧮 Running correlation analysis...")

        if analysis_type == "Complete Analysis":
            display_complete_analysis(title_features, thumbnail_features, engagement_data,
                                      video_metadata, analyzer, advanced_analytics,
                                      correlation_threshold, include_outliers, enable_clustering)
        elif analysis_type == "Title Features Only":
            display_title_analysis(title_features, engagement_data, video_metadata,
                                   analyzer, correlation_threshold)
        elif analysis_type == "Thumbnail Features Only":
            display_thumbnail_analysis(thumbnail_features, engagement_data, video_metadata,
                                       analyzer, correlation_threshold)
        elif analysis_type == "Temporal Analysis":
            display_temporal_analysis(channel_data, title_features, thumbnail_features,
                                      engagement_data, video_metadata, advanced_analytics)
        elif analysis_type == "Competitive Analysis":
            display_competitive_analysis(channel_data, engagement_data, advanced_analytics)

        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")

        # Export options
        display_export_options(title_features, thumbnail_features, engagement_data,
                               video_metadata, export_format)

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)


def prepare_engagement_data(channel_data):
    """Prepare engagement metrics dataframe and video metadata"""
    engagement_metrics = []
    video_metadata = pd.DataFrame()

    # Store video titles and other metadata
    video_titles = []
    video_ids = []
    published_dates = []

    for video in channel_data:
        # Store metadata
        video_ids.append(video['video_id'])
        video_titles.append(video['title'])
        published_dates.append(video['published_at'])

        # Calculate additional metrics
        views = max(video['view_count'], 1)
        likes = video['like_count']
        comments = video['comment_count']

        # Engagement rates
        like_rate = likes / views
        comment_rate = comments / views
        engagement_rate = (likes + comments) / views

        # Performance scores
        total_engagement = likes + comments
        viral_score = np.log10(views + 1) * engagement_rate

        # Temporal features - Fixed timestamp handling
        try:
            published = pd.to_datetime(video['published_at'])
            current_time = pd.Timestamp.now()
            days_since_publish = (current_time - published).days
        except Exception as e:
            print(f"Error parsing date for video {video.get('video_id', 'unknown')}: {e}")
            days_since_publish = 30

        engagement_metrics.append({
            'video_id': video['video_id'],
            'views': views,
            'likes': likes,
            'comments': comments,
            'like_rate': like_rate,
            'comment_rate': comment_rate,
            'engagement_rate': engagement_rate,
            'total_engagement': total_engagement,
            'viral_score': viral_score,
            'duration_minutes': video.get('duration_seconds', 0) / 60,
            'days_since_publish': days_since_publish,
            'views_per_day': views / max(days_since_publish, 1),
            'subscriber_count': video.get('channel_subscriber_count', 0)
        })

    # Create video metadata dataframe
    video_metadata = pd.DataFrame({
        'video_id': video_ids,
        'title': video_titles,
        'published_at': published_dates
    })

    return pd.DataFrame(engagement_metrics), video_metadata


def display_complete_analysis(title_features, thumbnail_features, engagement_data,
                              video_metadata, analyzer, advanced_analytics,
                              correlation_threshold, include_outliers, enable_clustering):
    """Display comprehensive analysis"""

    st.header("🔍 Complete Channel Analysis")

    # Overview metrics
    display_overview_metrics(engagement_data)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Correlation Analysis",
        "📝 Title Insights",
        "🖼️ Thumbnail Insights",
        "📹 Video Performance",
        "🔬 Advanced Analytics",
        "💡 Recommendations"
    ])

    with tab1:
        display_correlation_overview(title_features, thumbnail_features, engagement_data,
                                     analyzer, correlation_threshold)

    with tab2:
        if title_features is not None:
            display_detailed_title_analysis(title_features, engagement_data, video_metadata, analyzer)
        else:
            st.info("Title analysis not included in current configuration")

    with tab3:
        if thumbnail_features is not None:
            display_detailed_thumbnail_analysis(thumbnail_features, engagement_data, video_metadata, analyzer)
        else:
            st.info("Thumbnail analysis not included in current configuration")

    with tab4:
        display_video_performance_analysis(engagement_data, video_metadata)

    with tab5:
        display_advanced_analytics_tab(title_features, thumbnail_features, engagement_data,
                                       advanced_analytics, include_outliers, enable_clustering)

    with tab6:
        display_recommendations(title_features, thumbnail_features, engagement_data, analyzer)


def display_overview_metrics(engagement_data):
    """Display key channel metrics"""
    st.subheader("📈 Channel Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        avg_views = engagement_data['views'].mean()
        st.metric("Average Views", f"{avg_views:,.0f}")

    with col2:
        avg_engagement = engagement_data['engagement_rate'].mean() * 100
        st.metric("Avg Engagement Rate", f"{avg_engagement:.2f}%")

    with col3:
        total_views = engagement_data['views'].sum()
        st.metric("Total Views", f"{total_views:,.0f}")

    with col4:
        top_video_views = engagement_data['views'].max()
        st.metric("Best Video Views", f"{top_video_views:,.0f}")

    with col5:
        consistency = 1 - (engagement_data['views'].std() / engagement_data['views'].mean())
        st.metric("Consistency Score", f"{consistency:.2f}")


def display_correlation_overview(title_features, thumbnail_features, engagement_data,
                                 analyzer, correlation_threshold):
    """Display correlation analysis overview"""

    st.subheader("🔗 Feature Correlation Analysis")

    # Combine all features
    all_features = pd.DataFrame()
    if title_features is not None:
        all_features = pd.concat([all_features, title_features], axis=1)
    if thumbnail_features is not None:
        all_features = pd.concat([all_features, thumbnail_features], axis=1)

    if all_features.empty:
        st.warning("No features available for correlation analysis")
        return

    # Calculate correlations - ensure only numeric columns
    numeric_engagement = engagement_data.select_dtypes(include=[np.number])
    analysis_data = pd.concat([all_features, numeric_engagement], axis=1)

    correlations = analyzer.calculate_correlations(
        analysis_data,
        target_columns=['views', 'likes', 'comments', 'engagement_rate', 'viral_score']
    )

    # Create two columns with better spacing
    col1, col2 = st.columns([4, 2])

    with col1:
        # Make heatmap larger and more readable
        fig = analyzer.plot_correlation_heatmap(correlations, "All Features")
        fig.update_layout(
            height=800,  # Increased height
            width=1000,  # Increased width
            font=dict(size=12),
            margin=dict(l=250, r=100, t=100, b=100)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Top Correlations")
        top_correlations = analyzer.get_top_correlations(correlations, top_n=15)

        # Color code correlations
        def color_correlation(val):
            if isinstance(val, (int, float)):
                if abs(val) >= 0.5:
                    return 'background-color: #ff9999' if val < 0 else 'background-color: #99ff99'
                elif abs(val) >= 0.3:
                    return 'background-color: #ffcc99' if val < 0 else 'background-color: #ccffcc'
            return ''

        styled_df = top_correlations.style.applymap(color_correlation, subset=['Correlation'])
        st.dataframe(styled_df, use_container_width=True, height=600)


def display_video_performance_analysis(engagement_data, video_metadata):
    """Display individual video performance analysis"""
    st.subheader("📹 Individual Video Performance")

    # Merge engagement data with metadata
    video_analysis = engagement_data.merge(video_metadata, on='video_id')

    # Sort by views
    video_analysis = video_analysis.sort_values('views', ascending=False)

    # Top performing videos
    st.markdown("### 🏆 Top Performing Videos")
    top_videos = video_analysis.head(10)[['title', 'views', 'likes', 'engagement_rate', 'days_since_publish']]
    top_videos['engagement_rate'] = (top_videos['engagement_rate'] * 100).round(2).astype(str) + '%'
    st.dataframe(top_videos, use_container_width=True)

    # Performance scatter plot
    st.markdown("### 📊 Views vs Engagement Rate")
    fig = px.scatter(
        video_analysis,
        x='views',
        y='engagement_rate',
        hover_data=['title'],
        title="Video Performance Distribution",
        labels={'engagement_rate': 'Engagement Rate', 'views': 'Views'},
        color='viral_score',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Recent videos performance
    st.markdown("### 🆕 Recent Videos Performance")
    recent_videos = video_analysis.nsmallest(10, 'days_since_publish')[
        ['title', 'views', 'likes', 'days_since_publish', 'views_per_day']
    ]
    st.dataframe(recent_videos, use_container_width=True)


def display_title_analysis(title_features, engagement_data, video_metadata,
                           analyzer, correlation_threshold):
    """Display title-specific analysis"""
    st.header("📝 Title Feature Analysis")

    if title_features is None:
        st.error("Title features not available")
        return

    # Ensure we only use numeric columns for correlation
    numeric_engagement = engagement_data.select_dtypes(include=[np.number])
    analysis_data = pd.concat([title_features, numeric_engagement], axis=1)

    correlations = analyzer.calculate_correlations(
        analysis_data,
        target_columns=['views', 'likes', 'comments', 'engagement_rate']
    )

    # Main visualization with better layout
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = analyzer.plot_correlation_heatmap(correlations, "Title Features")
        fig.update_layout(height=700, width=900)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Feature Statistics")

        # Key insights
        insights = analyzer.generate_insights(correlations, correlation_threshold)
        for metric, metric_insights in insights.items():
            if metric_insights:
                st.markdown(f"**{metric.title()}:**")
                for insight in metric_insights:
                    st.write(f"• {insight}")

        # Download correlation matrix
        csv_buffer = io.StringIO()
        correlations.to_csv(csv_buffer)
        st.download_button(
            label="📥 Download Title Correlations",
            data=csv_buffer.getvalue(),
            file_name="title_correlations.csv",
            mime="text/csv"
        )


def display_thumbnail_analysis(thumbnail_features, engagement_data, video_metadata,
                               analyzer, correlation_threshold):
    """Display thumbnail-specific analysis"""
    st.header("🖼️ Thumbnail Feature Analysis")

    if thumbnail_features is None:
        st.error("Thumbnail features not available")
        return

    # Ensure numeric data only
    numeric_engagement = engagement_data.select_dtypes(include=[np.number])
    analysis_data = pd.concat([thumbnail_features, numeric_engagement], axis=1)

    correlations = analyzer.calculate_correlations(
        analysis_data,
        target_columns=['views', 'likes', 'comments', 'engagement_rate']
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = analyzer.plot_correlation_heatmap(correlations, "Thumbnail Features")
        fig.update_layout(height=700, width=900)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎨 Visual Insights")

        insights = analyzer.generate_insights(correlations, correlation_threshold)
        for metric, metric_insights in insights.items():
            if metric_insights:
                st.markdown(f"**{metric.title()}:**")
                for insight in metric_insights:
                    st.write(f"• {insight}")


def display_detailed_title_analysis(title_features, engagement_data, video_metadata, analyzer):
    """Detailed title analysis with feature breakdown"""

    # Merge with video metadata to show titles
    analysis_df = pd.concat([
        video_metadata[['title']].reset_index(drop=True),
        title_features.reset_index(drop=True),
        engagement_data.reset_index(drop=True)
    ], axis=1)

    # Feature importance for each metric
    metrics = ['views', 'likes', 'comments', 'engagement_rate']

    for i, metric in enumerate(metrics):
        if i % 2 == 0:
            col1, col2 = st.columns(2)

        with col1 if i % 2 == 0 else col2:
            st.subheader(f"Top Features for {metric.title()}")

            # Calculate correlations for this metric - only numeric columns
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
            if metric in numeric_cols:
                correlations = analysis_df[numeric_cols].corr()[metric].abs().sort_values(ascending=False)

                # Filter out the metric itself and get top features
                top_features = correlations.drop(metric, errors='ignore').head(8)

                # Create bar chart
                fig = go.Figure(data=go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    marker_color='lightblue'
                ))

                fig.update_layout(
                    title=f"Feature Importance for {metric.title()}",
                    xaxis_title="Correlation Strength",
                    height=300,
                    margin=dict(l=150, r=50, t=50, b=50)
                )

                st.plotly_chart(fig, use_container_width=True)

    # Show sample of titles with high-impact features
    st.subheader("📝 Sample Titles with Feature Analysis")

    # Select a few interesting features
    sample_features = ['sentiment_compound', 'word_count', 'power_words_count', 'is_question']
    available_features = [f for f in sample_features if f in analysis_df.columns]

    if available_features:
        sample_df = analysis_df[['title', 'views', 'engagement_rate'] + available_features].head(10)
        st.dataframe(sample_df, use_container_width=True)


def display_detailed_thumbnail_analysis(thumbnail_features, engagement_data, video_metadata, analyzer):
    """Detailed thumbnail analysis"""

    st.subheader("🎨 Color Analysis Impact")

    # Merge with video metadata
    analysis_df = pd.concat([
        video_metadata[['title']].reset_index(drop=True),
        thumbnail_features.reset_index(drop=True),
        engagement_data.reset_index(drop=True)
    ], axis=1)

    # Color feature analysis
    color_features = [col for col in thumbnail_features.columns if
                      'color' in col.lower() or 'brightness' in col.lower() or 'contrast' in col.lower()]

    if color_features:
        # Only use numeric columns for correlation
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        color_features_numeric = [f for f in color_features if f in numeric_cols]

        if color_features_numeric and 'views' in numeric_cols:
            color_correlations = analysis_df[color_features_numeric + ['views']].corr()['views'].abs().sort_values(
                ascending=False)

            fig = px.bar(
                x=color_correlations.values[1:],  # Exclude self-correlation
                y=color_correlations.index[1:],
                orientation='h',
                title="Color Features Impact on Views"
            )
            st.plotly_chart(fig, use_container_width=True)


def display_advanced_analytics_tab(title_features, thumbnail_features, engagement_data,
                                   advanced_analytics, include_outliers, enable_clustering):
    """Advanced analytics and insights"""

    st.subheader("🔬 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Performance Distribution")

        # Views distribution
        fig = px.histogram(
            engagement_data,
            x='views',
            nbins=20,
            title="Views Distribution",
            labels={'views': 'Views', 'count': 'Number of Videos'}
        )
        st.plotly_chart(fig, use_container_width=True)

        if include_outliers:
            st.markdown("#### 🎯 Outlier Analysis")
            outliers = advanced_analytics.detect_outliers(engagement_data, 'views')
            st.write(f"Found {len(outliers)} outlier videos")
            if len(outliers) > 0:
                st.dataframe(outliers[['views', 'likes', 'engagement_rate']].head())

    with col2:
        st.markdown("#### ⏰ Temporal Patterns")

        # Engagement over time
        if 'days_since_publish' in engagement_data.columns:
            fig = px.scatter(
                engagement_data,
                x='days_since_publish',
                y='views',
                title="Views vs Days Since Publication",
                trendline="ols",
                labels={'days_since_publish': 'Days Since Published', 'views': 'Views'}
            )
            st.plotly_chart(fig, use_container_width=True)

        if enable_clustering:
            st.markdown("#### 🎲 Content Clustering")
            if title_features is not None and len(title_features) > 10:
                try:
                    clusters = advanced_analytics.perform_clustering(title_features, n_clusters=3)
                    cluster_summary = pd.DataFrame({
                        'Cluster': range(len(clusters)),
                        'Videos': [len(cluster) for cluster in clusters],
                        'Avg Views': [engagement_data.iloc[cluster]['views'].mean() for cluster in clusters]
                    })
                    st.dataframe(cluster_summary)
                except Exception as e:
                    st.warning(f"Clustering failed: {str(e)}")


def display_temporal_analysis(channel_data, title_features, thumbnail_features,
                              engagement_data, video_metadata, advanced_analytics):
    """Temporal analysis of channel performance"""

    st.header("⏰ Temporal Analysis")

    # Prepare temporal data
    temporal_df = pd.DataFrame(channel_data)
    temporal_df['published_at'] = pd.to_datetime(temporal_df['published_at'])
    temporal_df = temporal_df.merge(engagement_data, on='video_id')

    # Time-based visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Performance Over Time")

        # Monthly aggregation
        temporal_df['month'] = temporal_df['published_at'].dt.to_period('M')
        monthly_stats = temporal_df.groupby('month').agg({
            'views': 'mean',
            'engagement_rate': 'mean',
            'video_id': 'count'
        }).reset_index()
        monthly_stats['month'] = monthly_stats['month'].astype(str)

        fig = px.line(
            monthly_stats,
            x='month',
            y='views',
            title="Average Views per Month"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Upload Frequency Impact")

        fig = px.bar(
            monthly_stats,
            x='month',
            y='video_id',
            title="Videos Uploaded per Month"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_competitive_analysis(channel_data, engagement_data, advanced_analytics):
    """Competitive analysis features"""

    st.header("🏆 Competitive Analysis")

    st.info("This feature requires multiple channels for comparison. Currently showing single channel analysis.")

    # Performance benchmarks
    st.subheader("📊 Performance Benchmarks")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Views Percentile (vs Industry)",
            "Requires industry data",
            help="Comparison with industry averages"
        )

    with col2:
        st.metric(
            "Engagement Percentile",
            "Requires comparison data",
            help="Engagement rate vs similar channels"
        )

    with col3:
        st.metric(
            "Growth Rate",
            "Requires historical data",
            help="Channel growth trajectory"
        )


def display_recommendations(title_features, thumbnail_features, engagement_data, analyzer):
    """Generate actionable recommendations"""

    st.header("💡 Optimization Recommendations")

    # Combine features for analysis
    all_features = pd.DataFrame()
    if title_features is not None:
        all_features = pd.concat([all_features, title_features], axis=1)
    if thumbnail_features is not None:
        all_features = pd.concat([all_features, thumbnail_features], axis=1)

    if all_features.empty:
        st.warning("No features available for recommendations")
        return

    # Calculate correlations - only numeric columns
    numeric_engagement = engagement_data.select_dtypes(include=[np.number])
    analysis_data = pd.concat([all_features, numeric_engagement], axis=1)
    correlations = analysis_data.corr()['views'].abs().sort_values(ascending=False)

    # Generate recommendations
    recommendations = []

    # Top positive correlations
    top_positive = correlations[correlations > 0.3].head(5)
    if len(top_positive) > 0:
        recommendations.append("🎯 **Leverage High-Impact Features:**")
        for feature, corr in top_positive.items():
            if feature != 'views':
                recommendations.append(f"   • Focus on optimizing {feature} (correlation: {corr:.3f})")

    # Display recommendations
    for rec in recommendations:
        if rec.startswith("🎯"):
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
        else:
            st.write(rec)

    # Additional specific recommendations based on feature analysis
    st.subheader("📋 Specific Optimization Tips")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Title Optimization")
        if title_features is not None:
            # Analyze title patterns
            avg_word_count = title_features['word_count'].mean() if 'word_count' in title_features.columns else 0
            avg_sentiment = title_features[
                'sentiment_compound'].mean() if 'sentiment_compound' in title_features.columns else 0

            st.write(f"• Average word count: {avg_word_count:.1f}")
            st.write(f"• Average sentiment: {avg_sentiment:.3f}")

            if avg_word_count < 8:
                st.write("💡 Consider using longer, more descriptive titles")
            if avg_sentiment < 0.2:
                st.write("💡 Try more positive/engaging language in titles")

    with col2:
        st.markdown("### 🖼️ Thumbnail Optimization")
        if thumbnail_features is not None:
            # Analyze thumbnail patterns
            avg_faces = thumbnail_features['num_faces'].mean() if 'num_faces' in thumbnail_features.columns else 0
            avg_brightness = thumbnail_features[
                'brightness_mean'].mean() if 'brightness_mean' in thumbnail_features.columns else 0

            st.write(f"• Average faces detected: {avg_faces:.1f}")
            st.write(f"• Average brightness: {avg_brightness:.1f}")

            if avg_faces < 0.5:
                st.write("💡 Consider including faces in thumbnails")
            if avg_brightness < 100:
                st.write("💡 Try using brighter, more vibrant thumbnails")


def display_export_options(title_features, thumbnail_features, engagement_data,
                           video_metadata, export_format):
    """Export analysis results"""

    st.header("📥 Export Results")

    col1, col2, col3 = st.columns(3)

    # Prepare export data
    export_data_dict = {}

    if title_features is not None:
        export_data_dict['title_features'] = title_features
    if thumbnail_features is not None:
        export_data_dict['thumbnail_features'] = thumbnail_features
    if engagement_data is not None:
        export_data_dict['engagement_data'] = engagement_data
    if video_metadata is not None:
        export_data_dict['video_metadata'] = video_metadata

    with col1:
        if st.button("📊 Export Analysis Results"):
            if export_format == "CSV":
                # Create ZIP file with multiple CSVs
                import zipfile
                from io import BytesIO

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for name, df in export_data_dict.items():
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        zip_file.writestr(f"{name}.csv", csv_buffer.getvalue())

                st.download_button(
                    label="Download All CSVs (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="youtube_analysis_results.zip",
                    mime="application/zip"
                )

            elif export_format == "Excel":
                # Create Excel with multiple sheets
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    for name, df in export_data_dict.items():
                        df.to_excel(writer, sheet_name=name[:31], index=False)

                st.download_button(
                    label="Download Excel File",
                    data=excel_buffer.getvalue(),
                    file_name="youtube_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            elif export_format == "JSON":
                # Export as JSON
                import json
                json_data = {}
                for name, df in export_data_dict.items():
                    json_data[name] = df.to_dict('records')

                json_str = json.dumps(json_data, indent=2, default=str)

                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="youtube_analysis_results.json",
                    mime="application/json"
                )

    with col2:
        if st.button("📈 Export Correlations"):
            st.info("Select analysis tab to export specific correlations")

    with col3:
        if st.button("📋 Export Report"):
            # Generate text report
            report = generate_analysis_report(
                title_features, thumbnail_features,
                engagement_data, video_metadata
            )
            st.download_button(
                label="Download Report",
                data=report,
                file_name="youtube_analysis_report.txt",
                mime="text/plain"
            )


def generate_analysis_report(title_features, thumbnail_features, engagement_data, video_metadata):
    """Generate a comprehensive text report"""
    report_lines = [
        "YOUTUBE CHANNEL ANALYSIS REPORT",
        "=" * 50,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "CHANNEL OVERVIEW",
        "-" * 20,
        f"Total videos analyzed: {len(engagement_data)}",
        f"Total views: {engagement_data['views'].sum():,.0f}",
        f"Average views: {engagement_data['views'].mean():,.0f}",
        f"Average engagement rate: {engagement_data['engagement_rate'].mean() * 100:.2f}%",
        "",
        "TOP PERFORMING VIDEOS",
        "-" * 20
    ]

    # Add top videos
    if video_metadata is not None:
        top_videos = engagement_data.merge(video_metadata, on='video_id').nlargest(5, 'views')
        for idx, video in top_videos.iterrows():
            report_lines.append(f"{idx + 1}. {video['title'][:60]}...")
            report_lines.append(f"   Views: {video['views']:,.0f} | Likes: {video['likes']:,.0f}")
            report_lines.append("")

    # Add feature analysis summary
    if title_features is not None:
        report_lines.extend([
            "TITLE ANALYSIS SUMMARY",
            "-" * 20,
            f"Average word count: {title_features['word_count'].mean():.1f}" if 'word_count' in title_features.columns else "",
            f"Questions in titles: {(title_features['is_question'].sum() / len(title_features) * 100):.1f}%" if 'is_question' in title_features.columns else "",
            ""
        ])

    if thumbnail_features is not None:
        report_lines.extend([
            "THUMBNAIL ANALYSIS SUMMARY",
            "-" * 20,
            f"Videos with faces: {(thumbnail_features['num_faces'] > 0).sum() / len(thumbnail_features) * 100:.1f}%" if 'num_faces' in thumbnail_features.columns else "",
            f"Average brightness: {thumbnail_features['brightness_mean'].mean():.1f}" if 'brightness_mean' in thumbnail_features.columns else "",
            ""
        ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    main()