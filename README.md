# YouTube Channel Analyzer Pro

A comprehensive YouTube channel analytics tool that analyzes video titles, thumbnails, and engagement metrics to provide actionable insights for content optimization.

## Features

### 📝 Title Analysis (40+ metrics)
- **Linguistic Analysis**: Readability scores, sentiment analysis, text complexity
- **SEO Optimization**: Keyword density, trending terms, hashtag usage
- **Emotional Triggers**: Urgency indicators, curiosity gaps, power words
- **Structure Analysis**: Punctuation patterns, capitalization, question formats

### 🖼️ Thumbnail Analysis (35+ metrics)
- **Visual Composition**: Color analysis, contrast, brightness levels
- **Face Detection**: Number of faces, positioning, prominence
- **Text Detection**: Overlay text presence, readability, contrast
- **Design Elements**: Rule of thirds, symmetry, visual complexity

### 📊 Advanced Analytics
- **Correlation Analysis**: Identify features that drive views and engagement
- **Content Clustering**: Group similar content patterns
- **Temporal Analysis**: Performance trends over time
- **Outlier Detection**: Identify exceptionally performing videos

## Prerequisites

- Python 3.8 or higher
- YouTube Data API v3 key
- pip (Python package manager)

## Installation

1. **Clone the repository**
```bash
git clone <repository_url>
cd YoutubeAnalyticsService
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (for sentiment analysis)
```python
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Getting a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Library"
4. Search for "YouTube Data API v3" and click on it
5. Click "Enable" to enable the API
6. Go to "APIs & Services" > "Credentials"
7. Click "Create Credentials" > "API Key"
8. Copy your API key and keep it secure
9. (Optional) Click "Restrict Key" to add restrictions for security

## Running the Application

1. **Start the Streamlit app**
```bash
streamlit run app.py
```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to this URL

3. **Configure the analysis**
   - Enter your YouTube API key in the sidebar
   - Enter a channel name, URL, or handle (e.g., @channelname)
   - Select analysis type and features
   - Click "Start Comprehensive Analysis"

## Usage Guide

### Supported Channel Input Formats
- Channel name: `MrBeast`
- Channel URL: `https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA`
- Custom URL: `https://www.youtube.com/c/MrBeast6000`
- Handle: `@MrBeast` or `https://www.youtube.com/@MrBeast`

### Analysis Types
1. **Complete Analysis**: Full analysis of both titles and thumbnails
2. **Title Features Only**: Focus on linguistic and textual features
3. **Thumbnail Features Only**: Focus on visual elements
4. **Temporal Analysis**: Performance trends over time
5. **Competitive Analysis**: Benchmarking (requires multiple channels)

### Feature Groups
- **Basic Metrics**: Core statistics like views, likes, comments
- **NLP Features**: Natural language processing of titles
- **Visual Features**: Computer vision analysis of thumbnails
- **Engagement Patterns**: Interaction rates and viral indicators
- **Temporal Features**: Time-based performance metrics
- **Advanced Analytics**: Clustering, outliers, complex patterns

## Troubleshooting

### Common Issues

1. **"Error fetching channel videos"**
   - Verify your API key is correct
   - Check if the channel name/URL is valid
   - Ensure YouTube Data API v3 is enabled in Google Cloud Console

2. **Missing sentiment analysis**
   - Run: `pip install nltk vaderSentiment`
   - Download NLTK data: `python -c "import nltk; nltk.download('vader_lexicon')"`

3. **Thumbnail analysis errors**
   - Ensure OpenCV is properly installed: `pip install opencv-python`
   - Check internet connection (thumbnails are downloaded)

4. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### API Quota Management
- YouTube API has daily quota limits (typically 10,000 units)
- Each video data request uses ~3 units
- Analyzing 100 videos uses ~300 units
- Monitor usage in Google Cloud Console

## Data Export Options

The tool supports multiple export formats:
- **CSV**: Individual files for each data type
- **Excel**: Multi-sheet workbook with all data
- **JSON**: Structured data for programmatic use

## Advanced Configuration

### Environment Variables (Optional)
Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_api_key_here
DEFAULT_MAX_VIDEOS=100
CACHE_DURATION_HOURS=24
```

### Custom Configuration
Edit `config/config.yaml` (if created):
```yaml
api:
  timeout: 10
  max_retries: 3
  
analysis:
  correlation_threshold: 0.3
  max_videos: 500
  cache_duration_hours: 24
  
features:
  enable_sentiment_analysis: true
  enable_face_detection: true
  enable_text_detection: true
```

## Performance Optimization

1. **Caching**: Results are cached to avoid redundant API calls
2. **Batch Processing**: Features are extracted in batches for efficiency
3. **Selective Analysis**: Choose only needed feature groups
4. **Video Limits**: Start with fewer videos for faster analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YouTube Data API v3 for video metadata
- Streamlit for the web interface
- OpenCV for computer vision features
- NLTK and VaderSentiment for text analysis
- Plotly for interactive visualizations

## Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with details
4. Include error messages and steps to reproduce

## Future Enhancements

- [ ] Real-time monitoring dashboard
- [ ] Competitor comparison features
- [ ] A/B testing recommendations
- [ ] Machine learning predictions
- [ ] Custom metric definitions
- [ ] API for programmatic access
- [ ] Batch channel analysis
- [ ] Historical data tracking