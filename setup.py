#!/usr/bin/env python
"""
Quick setup script for YouTube Channel Analyzer
"""

import os
import sys
import subprocess


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'=' * 50}")
    print(f"🔧 {description}...")
    print(f"{'=' * 50}")

    try:
        if isinstance(command, str):
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(command, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False


def main():
    print("""
    YouTube Channel Analyzer - Setup Script
    ======================================
    This script will help you set up the project.
    """)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Your version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version}")

    # Create necessary directories
    directories = ['data', 'data/cache', 'config', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Install requirements
    print("\nInstalling dependencies...")
    if run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                   "Installing requirements"):
        print("✅ All dependencies installed!")
    else:
        print("⚠️  Some dependencies failed to install. Check the errors above.")

    # Download NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded!")
    except Exception as e:
        print(f"⚠️  NLTK data download failed: {e}")
        print("   You can manually download it later using: python -c \"import nltk; nltk.download('vader_lexicon')\"")

    # Create sample config file
    config_content = """# YouTube Channel Analyzer Configuration
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
"""

    config_path = os.path.join('config', 'config.yaml')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"✅ Created sample config file: {config_path}")

    # Create .env template
    env_template = """# YouTube API Configuration
# Get your API key from: https://console.cloud.google.com/
YOUTUBE_API_KEY=your_api_key_here

# Optional settings
DEFAULT_MAX_VIDEOS=100
CACHE_DURATION_HOURS=24
"""

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ Created .env template file")

    # Final instructions
    print("""
    ========================================
    ✅ Setup completed successfully!
    ========================================

    Next steps:
    1. Get your YouTube API key from: https://console.cloud.google.com/
    2. Add your API key to the .env file or enter it in the app
    3. Run the application: streamlit run app.py

    The app will open in your browser at http://localhost:8501

    For help, check the README.md file or run with --help
    """)

    # Ask if user wants to start the app
    response = input("\nWould you like to start the app now? (y/n): ").lower()
    if response == 'y':
        print("\nStarting YouTube Channel Analyzer...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()