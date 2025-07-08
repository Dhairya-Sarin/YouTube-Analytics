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

# Custom CSS for better layout and styling
# Replace your existing CSS section with this improved version:

st.markdown("""
<style>
    /* FULL WIDTH LAYOUT */
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 0rem;
        max-width: none;
        width: 100%;
    }

    /* Remove default streamlit padding */
    .css-18e3th9 {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Make sidebar narrower to give more space to main content */
    .css-1d391kg {
        width: 300px;
    }

    /* Ensure main content uses full available width */
    .css-1y4p8pa {
        max-width: none;
        width: 100%;
    }

    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }

    /* IMPROVED METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #FF0000;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 0.3rem 0;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card h3 {
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }

    .metric-card h2 {
        font-size: 1.8rem;
        margin: 0;
        font-weight: bold;
        color: #333;
    }

    /* IMPROVED INSIGHT BOXES */
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    /* IMPROVED DOWNLOAD SECTION */
    .download-section {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff9800;
        margin: 2rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    /* BETTER TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        padding: 4px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding-left: 16px;
        padding-right: 16px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF0000;
        color: white;
    }

    /* FULL WIDTH CHARTS */
    .js-plotly-plot {
        width: 100% !important;
    }

    .plotly-graph-div {
        width: 100% !important;
    }

    /* BETTER DATAFRAMES */
    .dataframe {
        width: 100% !important;
        font-size: 0.9rem;
    }

    .stDataFrame {
        width: 100%;
    }

    .stDataFrame > div {
        width: 100%;
        overflow-x: auto;
    }

    /* IMPROVED BUTTONS */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 2px solid #FF0000;
        background-color: #FF0000;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
        font-size: 0.95rem;
    }

    .stButton > button:hover {
        background-color: #cc0000;
        border-color: #cc0000;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(255, 0, 0, 0.3);
    }

    /* DOWNLOAD BUTTONS */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #28a745;
        background-color: #28a745;
        color: white;
        font-weight: 500;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.2rem 0;
    }

    .stDownloadButton > button:hover {
        background-color: #218838;
        border-color: #218838;
    }

    /* SECTION HEADERS */
    h1, h2, h3 {
        color: #333;
        font-weight: 600;
    }

    h2 {
        border-bottom: 2px solid #FF0000;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    h3 {
        color: #666;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* SIDEBAR IMPROVEMENTS */
    .css-1d391kg {
        background-color: #f8f9fa;
        padding: 1rem;
    }

    .css-1d391kg .stSelectbox > div > div {
        background-color: white;
    }

    .css-1d391kg .stTextInput > div > div > input {
        background-color: white;
    }

    /* BETTER SPACING */
    .element-container {
        margin-bottom: 1rem;
    }

    /* RESPONSIVE COLUMNS */
    .row-widget {
        width: 100%;
    }

    /* IMPROVED ALERTS */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* BETTER TEXT READABILITY */
    .stMarkdown {
        line-height: 1.6;
    }

    /* FULL WIDTH CONTAINERS */
    .stContainer {
        max-width: none !important;
        width: 100% !important;
    }

    /* Remove any max-width constraints */
    .css-1y4p8pa, .css-18e3th9, .main .block-container {
        max-width: none !important;
        width: 100% !important;
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
        max_videos = st.slider("Max Videos to Analyze", 10, 500, 100, step=10)

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

    # Analysis button - make it more prominent
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Comprehensive Analysis", type="primary", use_container_width=True):
            run_analysis(api_key, channel_input, max_videos, analysis_type, feature_groups,
                         correlation_threshold, include_outliers, enable_clustering, export_format)


def display_welcome_screen():
    """Display welcome screen with instructions"""
    # Use columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
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
        """)

    with col2:
        st.markdown("""
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

        # Step 3: Prepare engagement data
        status_text.text("📊 Preparing engagement metrics...")
        engagement_data = prepare_engagement_data(channel_data)
        progress_bar.progress(0.8)

        # Step 4: Run analysis
        status_text.text("🧮 Running correlation analysis...")

        if analysis_type == "Complete Analysis":
            display_complete_analysis(title_features, thumbnail_features, engagement_data,
                                      analyzer, advanced_analytics, correlation_threshold,
                                      include_outliers, enable_clustering)
        elif analysis_type == "Title Features Only":
            display_title_analysis(title_features, engagement_data, analyzer, correlation_threshold)
        elif analysis_type == "Thumbnail Features Only":
            display_thumbnail_analysis(thumbnail_features, engagement_data, analyzer, correlation_threshold)
        elif analysis_type == "Temporal Analysis":
            display_temporal_analysis(channel_data, title_features, thumbnail_features,
                                      engagement_data, advanced_analytics)
        elif analysis_type == "Competitive Analysis":
            display_competitive_analysis(channel_data, engagement_data, advanced_analytics)

        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")

        # Export options
        display_export_options(title_features, thumbnail_features, engagement_data, export_format)

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)


def prepare_engagement_data(channel_data):
    """Prepare engagement metrics dataframe"""
    engagement_metrics = []
    valid_count = 0

    for video in channel_data:
        try:
            # Extract key fields from the flat dictionary structure
            views = max(video.get('view_count', 0), 1)
            likes = video.get('like_count', 0)
            comments = video.get('comment_count', 0)
            published_str = video.get('published_at')

            # Convert and validate timestamp
            published_ts = pd.to_datetime(published_str, errors='coerce')
            if pd.isna(published_ts):
                continue  # Skip if invalid timestamp

            # FIX: Handle timezone-aware timestamps properly
            if published_ts.tz is not None:
                # Convert to timezone-naive UTC
                published_ts = published_ts.tz_convert('UTC').tz_localize(None)

            # Calculate days since publish using timezone-naive timestamps
            current_time = pd.Timestamp.now()
            days_since_publish = (current_time - published_ts).days

            if days_since_publish <= 0:
                continue  # Skip videos published today or in future

            # Engagement rates
            like_rate = likes / views
            comment_rate = comments / views
            engagement_rate = (likes + comments) / views

            # Performance scores
            total_engagement = likes + comments
            viral_score = np.log10(views + 1) * engagement_rate

            # Append valid data
            engagement_metrics.append({
                'video_id': video['video_id'],
                'title': video.get('title'),
                'published_at': published_ts,
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
                'views_per_day': views / days_since_publish,
                'subscriber_count': video.get('channel_subscriber_count', 0),
                'category_id': video.get('category_id'),
                'language': video.get('language')
            })

            valid_count += 1

        except Exception as e:
            print(f"[WARN] Skipping video due to error: {e}")
            continue

    print(f"[INFO] Total valid videos processed: {valid_count}")
    return pd.DataFrame(engagement_metrics)


def display_complete_analysis(title_features, thumbnail_features, engagement_data,
                              analyzer, advanced_analytics, correlation_threshold,
                              include_outliers, enable_clustering):
    """Display comprehensive analysis"""

    st.header("🔍 Complete Channel Analysis")

    # Overview metrics
    display_overview_metrics(engagement_data)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Correlation Analysis",
        "📝 Title Insights",
        "🖼️ Thumbnail Insights",
        "🔬 Advanced Analytics",
        "💡 Recommendations",
        "📥 Downloads"
    ])

    with tab1:
        display_correlation_overview(title_features, thumbnail_features, engagement_data,
                                     analyzer, correlation_threshold)

    with tab2:
        if title_features is not None:
            display_detailed_title_analysis(title_features, engagement_data, analyzer)
        else:
            st.info("Title analysis not included in current configuration")

    with tab3:
        if thumbnail_features is not None:
            display_detailed_thumbnail_analysis(thumbnail_features, engagement_data, analyzer)
        else:
            st.info("Thumbnail analysis not included in current configuration")

    with tab4:
        display_advanced_analytics_tab(title_features, thumbnail_features, engagement_data,
                                       advanced_analytics, include_outliers, enable_clustering)

    with tab5:
        display_recommendations(title_features, thumbnail_features, engagement_data, analyzer)

    with tab6:
        display_comprehensive_downloads(title_features, thumbnail_features, engagement_data, analyzer)


def display_overview_metrics(engagement_data):
    """Display key channel metrics"""
    st.subheader("📈 Channel Overview")

    # Create 5 columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        avg_views = engagement_data['views'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF0000; margin: 0;">📺 Average Views</h3>
            <h2 style="margin: 0;">{avg_views:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_engagement = engagement_data['engagement_rate'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF0000; margin: 0;">💬 Avg Engagement</h3>
            <h2 style="margin: 0;">{avg_engagement:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_views = engagement_data['views'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF0000; margin: 0;">🎯 Total Views</h3>
            <h2 style="margin: 0;">{total_views:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        top_video_views = engagement_data['views'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF0000; margin: 0;">🏆 Best Video</h3>
            <h2 style="margin: 0;">{top_video_views:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        consistency = 1 - (engagement_data['views'].std() / engagement_data['views'].mean())
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF0000; margin: 0;">📊 Consistency</h3>
            <h2 style="margin: 0;">{consistency:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)


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

    # Calculate correlations
    analysis_data = pd.concat([all_features, engagement_data.select_dtypes(include=[np.number])], axis=1)
    correlations = analyzer.calculate_correlations(
        analysis_data,
        target_columns=['views', 'likes', 'comments', 'engagement_rate', 'viral_score']
    )

    # Display correlation heatmap
    col1, col2 = st.columns([3, 1])

    with col1:
        fig = analyzer.plot_correlation_heatmap(correlations, "All Features")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Top Correlations")
        top_correlations = analyzer.get_top_correlations(correlations, top_n=15)

        # Color code correlations
        def color_correlation(val):
            if abs(val) >= 0.5:
                return 'background-color: #ff9999' if val < 0 else 'background-color: #99ff99'
            elif abs(val) >= 0.3:
                return 'background-color: #ffcc99' if val < 0 else 'background-color: #ccffcc'
            return ''

        styled_df = top_correlations.style.applymap(color_correlation, subset=['Correlation'])
        st.dataframe(styled_df, use_container_width=True)


def display_title_analysis(title_features, engagement_data, analyzer, correlation_threshold):
    """Display title-specific analysis"""
    st.header("📝 Title Feature Analysis")

    if title_features is None:
        st.error("Title features not available")
        return

    # Correlation analysis
    analysis_data = pd.concat([title_features, engagement_data.select_dtypes(include=[np.number])], axis=1)
    correlations = analyzer.calculate_correlations(
        analysis_data,
        target_columns=['views', 'likes', 'comments', 'engagement_rate']
    )

    # Main visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = analyzer.plot_correlation_heatmap(correlations, "Title Features")
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


def display_thumbnail_analysis(thumbnail_features, engagement_data, analyzer, correlation_threshold):
    """Display thumbnail-specific analysis"""
    st.header("🖼️ Thumbnail Feature Analysis")

    if thumbnail_features is None:
        st.error("Thumbnail features not available")
        return

    # Similar structure to title analysis but for thumbnails
    analysis_data = pd.concat([thumbnail_features, engagement_data.select_dtypes(include=[np.number])], axis=1)
    correlations = analyzer.calculate_correlations(
        analysis_data,
        target_columns=['views', 'likes', 'comments', 'engagement_rate']
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = analyzer.plot_correlation_heatmap(correlations, "Thumbnail Features")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎨 Visual Insights")

        insights = analyzer.generate_insights(correlations, correlation_threshold)
        for metric, metric_insights in insights.items():
            if metric_insights:
                st.markdown(f"**{metric.title()}:**")
                for insight in metric_insights:
                    st.write(f"• {insight}")


def display_detailed_title_analysis(title_features, engagement_data, analyzer):
    """Detailed title analysis with feature breakdown"""

    # Feature importance for each metric
    metrics = ['views', 'likes', 'comments', 'engagement_rate']

    for i, metric in enumerate(metrics):
        if i % 2 == 0:
            col1, col2 = st.columns(2)

        with col1 if i % 2 == 0 else col2:
            st.subheader(f"Top Features for {metric.title()}")

            # FIX: Only use numeric columns for correlation
            combined_data = pd.concat([title_features, engagement_data], axis=1)

            # Select only numeric columns
            numeric_data = combined_data.select_dtypes(include=[np.number])

            # Check if the metric exists in numeric data
            if metric not in numeric_data.columns:
                st.warning(f"Metric '{metric}' not found in numeric data")
                continue

            # Calculate correlations for this metric
            correlations = numeric_data.corr()[metric].abs().sort_values(ascending=False)

            # Filter out the metric itself and get top features
            top_features = correlations.drop(metric, errors='ignore').head(8)

            if len(top_features) == 0:
                st.info(f"No correlations found for {metric}")
                continue

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
                height=400,
                margin=dict(l=150, r=50, t=50, b=50)
            )

            st.plotly_chart(fig, use_container_width=True)


def display_detailed_thumbnail_analysis(thumbnail_features, engagement_data, analyzer):
    """Detailed thumbnail analysis"""

    st.subheader("🎨 Color Analysis Impact")

    # Color feature analysis
    color_features = [col for col in thumbnail_features.columns if
                      'color' in col.lower() or 'brightness' in col.lower() or 'contrast' in col.lower()]

    if color_features:
        combined_data = pd.concat([thumbnail_features[color_features], engagement_data], axis=1)

        # FIX: Only use numeric columns for correlation
        numeric_data = combined_data.select_dtypes(include=[np.number])

        if 'views' in numeric_data.columns:
            color_correlations = numeric_data.corr()['views'].abs().sort_values(ascending=False)

            fig = px.bar(
                x=color_correlations.values[1:],  # Exclude self-correlation
                y=color_correlations.index[1:],
                orientation='h',
                title="Color Features Impact on Views"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Views data not available for correlation analysis")


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
            title="Views Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        if include_outliers:
            st.markdown("#### 🎯 Outlier Analysis")
            outliers = advanced_analytics.detect_outliers(engagement_data, 'views')
            st.write(f"Found {len(outliers)} outlier videos")
            if len(outliers) > 0:
                st.dataframe(outliers[['views', 'likes', 'engagement_rate']].head(), use_container_width=True)

    with col2:
        st.markdown("#### ⏰ Temporal Patterns")

        # Engagement over time
        if 'days_since_publish' in engagement_data.columns:
            fig = px.scatter(
                engagement_data,
                x='days_since_publish',
                y='views',
                title="Views vs Days Since Publication"
            )
            st.plotly_chart(fig, use_container_width=True)

        if enable_clustering:
            st.markdown("#### 🎲 Content Clustering")
            if title_features is not None and len(title_features) > 10:
                clusters = advanced_analytics.perform_clustering(title_features, n_clusters=3)
                cluster_summary = pd.DataFrame({
                    'Cluster': range(len(clusters)),
                    'Videos': [len(cluster) for cluster in clusters],
                    'Avg Views': [engagement_data.iloc[cluster]['views'].mean() for cluster in clusters]
                })
                st.dataframe(cluster_summary, use_container_width=True)


def display_temporal_analysis(channel_data, title_features, thumbnail_features,
                              engagement_data, advanced_analytics):
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

    # Calculate correlations - FIX: Use only numeric data
    analysis_data = pd.concat([all_features, engagement_data.select_dtypes(include=[np.number])], axis=1)

    # Select only numeric columns for correlation
    numeric_data = analysis_data.select_dtypes(include=[np.number])

    # Check if 'views' column exists
    if 'views' not in numeric_data.columns:
        st.warning("Views data not available for generating recommendations")
        return

    correlations = numeric_data.corr()['views'].abs().sort_values(ascending=False)

    # Generate recommendations
    recommendations = []

    # Top positive correlations
    top_positive = correlations[correlations > 0.3].head(5)
    if len(top_positive) > 0:
        recommendations.append("🎯 **Leverage High-Impact Features:**")
        for feature, corr in top_positive.items():
            if feature != 'views':
                recommendations.append(f"   • Focus on optimizing {feature} (correlation: {corr:.3f})")

    # Add more recommendation categories
    if len(correlations[correlations > 0.5]) > 0:
        strong_correlations = correlations[correlations > 0.5]
        recommendations.append("\n🔥 **Strong Performance Drivers:**")
        for feature, corr in strong_correlations.items():
            if feature != 'views':
                recommendations.append(f"   • {feature} shows strong correlation ({corr:.3f}) - prioritize this!")

    # Moderate correlations
    moderate_correlations = correlations[(correlations > 0.3) & (correlations <= 0.5)]
    if len(moderate_correlations) > 0:
        recommendations.append("\n⚡ **Moderate Impact Opportunities:**")
        for feature, corr in moderate_correlations.head(3).items():
            if feature != 'views':
                recommendations.append(f"   • Consider improving {feature} (correlation: {corr:.3f})")

    # Display recommendations in attractive boxes
    if recommendations:
        for rec in recommendations:
            if rec.startswith(("🎯", "🔥", "⚡")):
                st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
            else:
                st.write(rec)
    else:
        st.info("No strong correlations found to generate specific recommendations.")

        # Fallback general recommendations
        st.markdown("""
        ### General YouTube Optimization Tips:
        - **Titles**: Use clear, descriptive titles with relevant keywords
        - **Thumbnails**: Ensure high contrast and readable text
        - **Engagement**: Encourage likes and comments in your content
        - **Consistency**: Maintain regular upload schedule
        - **Quality**: Focus on content that provides value to viewers
        """)


def display_comprehensive_downloads(title_features, thumbnail_features, engagement_data, analyzer):
    """Comprehensive download section with all analysis results"""

    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.header("📥 Download Analysis Results")
    st.markdown("Get all your analysis data in various formats for further exploration.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Prepare all data for download
    all_features = pd.DataFrame()
    if title_features is not None:
        all_features = pd.concat([all_features, title_features], axis=1)
    if thumbnail_features is not None:
        all_features = pd.concat([all_features, thumbnail_features], axis=1)

    # Calculate correlations if we have features
    correlations = None
    top_correlations = None
    if not all_features.empty:
        analysis_data = pd.concat([all_features, engagement_data.select_dtypes(include=[np.number])], axis=1)
        correlations = analyzer.calculate_correlations(
            analysis_data,
            target_columns=['views', 'likes', 'comments', 'engagement_rate', 'viral_score']
        )
        top_correlations = analyzer.get_top_correlations(correlations, top_n=50)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Correlation Data")

        if correlations is not None:
            # Full correlation matrix - CSV
            csv_buffer = io.StringIO()
            correlations.to_csv(csv_buffer)
            st.download_button(
                label="📥 Download Full Correlation Matrix (CSV)",
                data=csv_buffer.getvalue(),
                file_name="full_correlation_matrix.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Full correlation matrix - Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                correlations.to_excel(writer, sheet_name='Full_Correlation_Matrix')
                if top_correlations is not None:
                    top_correlations.to_excel(writer, sheet_name='Top_50_Correlations', index=False)
                if title_features is not None:
                    title_correlations = analyzer.calculate_correlations(
                        pd.concat([title_features, engagement_data.select_dtypes(include=[np.number])], axis=1),
                        target_columns=['views', 'likes', 'comments', 'engagement_rate']
                    )
                    title_correlations.to_excel(writer, sheet_name='Title_Correlations')
                if thumbnail_features is not None:
                    thumb_correlations = analyzer.calculate_correlations(
                        pd.concat([thumbnail_features, engagement_data.select_dtypes(include=[np.number])], axis=1),
                        target_columns=['views', 'likes', 'comments', 'engagement_rate']
                    )
                    thumb_correlations.to_excel(writer, sheet_name='Thumbnail_Correlations')
                engagement_data.to_excel(writer, sheet_name='Engagement_Data', index=False)
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download Complete Analysis (Excel)",
                data=excel_buffer.getvalue(),
                file_name="youtube_complete_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # Top correlations only - CSV
            if top_correlations is not None:
                csv_buffer = io.StringIO()
                top_correlations.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Top 50 Correlations (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="top_50_correlations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No correlation data available for download")

    with col2:
        st.subheader("📈 Feature Data")

        # Title features
        if title_features is not None:
            csv_buffer = io.StringIO()
            title_features.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Title Features (CSV)",
                data=csv_buffer.getvalue(),
                file_name="title_features.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Thumbnail features
        if thumbnail_features is not None:
            csv_buffer = io.StringIO()
            thumbnail_features.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Thumbnail Features (CSV)",
                data=csv_buffer.getvalue(),
                file_name="thumbnail_features.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Engagement data
        csv_buffer = io.StringIO()
        engagement_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Engagement Data (CSV)",
            data=csv_buffer.getvalue(),
            file_name="engagement_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Combined dataset
        if not all_features.empty:
            combined_data = pd.concat([all_features, engagement_data], axis=1)
            csv_buffer = io.StringIO()
            combined_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Combined Dataset (CSV)",
                data=csv_buffer.getvalue(),
                file_name="combined_youtube_data.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Summary report
    st.markdown("---")
    st.subheader("📋 Analysis Report")

    if correlations is not None:
        # Generate text report
        insights = analyzer.generate_insights(correlations, 0.3)

        report_lines = [
            "YOUTUBE CHANNEL ANALYSIS REPORT",
            "=" * 50,
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            f"Total videos analyzed: {len(engagement_data)}",
            f"Average views: {engagement_data['views'].mean():,.0f}",
            f"Median views: {engagement_data['views'].median():,.0f}",
            f"Total features analyzed: {len(all_features.columns) if not all_features.empty else 0}",
            "",
            "KEY FINDINGS",
            "-" * 15
        ]

        # Add insights
        for metric, metric_insights in insights.items():
            if metric_insights:
                report_lines.append(f"\n{metric.upper()} INSIGHTS:")
                for insight in metric_insights:
                    report_lines.append(f"  • {insight}")

        # Add top correlations
        report_lines.extend([
            "",
            "TOP CORRELATIONS",
            "-" * 20
        ])

        if top_correlations is not None:
            for _, row in top_correlations.head(10).iterrows():
                direction = "↗️" if row['Correlation'] > 0 else "↘️"
                report_lines.append(f"{direction} {row['Feature']} → {row['Metric']}: {row['Correlation']:.3f}")

        report_text = "\n".join(report_lines)

        st.download_button(
            label="📥 Download Analysis Report (TXT)",
            data=report_text,
            file_name="youtube_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )


def display_export_options(title_features, thumbnail_features, engagement_data, export_format):
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

    with col1:
        if st.button("📊 Export Analysis Results"):
            if export_format == "CSV":
                # Create combined CSV
                combined_df = pd.concat(list(export_data_dict.values()), axis=1)
                csv_buffer = io.StringIO()
                combined_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="youtube_analysis_results.csv",
                    mime="text/csv"
                )

    with col2:
        if st.button("📈 Export Correlations"):
            st.info("Use the Downloads tab for correlation exports")

    with col3:
        if st.button("📋 Export Recommendations"):
            st.info("Use the Downloads tab for recommendation exports")


if __name__ == "__main__":
    main()