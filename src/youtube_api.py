from googleapiclient.discovery import build
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import re


class YouTubeAPI:
    def __init__(self, api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def get_channel_id(self, channel_input: str) -> str:
        """Get channel ID from various input formats"""
        # Clean the input
        channel_input = channel_input.strip()

        # Direct channel ID (starts with UC)
        if channel_input.startswith('UC') and len(channel_input) == 24:
            return channel_input

        # Full YouTube URL patterns
        if 'youtube.com' in channel_input:
            if '/channel/' in channel_input:
                return channel_input.split('/channel/')[-1].split('?')[0].split('/')[0]
            elif '/c/' in channel_input:
                custom_name = channel_input.split('/c/')[-1].split('?')[0].split('/')[0]
                return self._search_channel_by_name(custom_name)
            elif '/user/' in channel_input:
                username = channel_input.split('/user/')[-1].split('?')[0].split('/')[0]
                return self._search_channel_by_name(username)
            elif '/@' in channel_input:
                handle = channel_input.split('/@')[-1].split('?')[0].split('/')[0]
                return self._search_channel_by_name(handle)

        # Handle format (starts with @)
        if channel_input.startswith('@'):
            return self._search_channel_by_name(channel_input[1:])

        # Plain channel name
        return self._search_channel_by_name(channel_input)

    def _search_channel_by_name(self, name: str) -> str:
        """Search for channel by name/username"""
        try:
            # First try search
            request = self.youtube.search().list(
                part="snippet",
                type="channel",
                q=name,
                maxResults=5
            )
            response = request.execute()

            if response['items']:
                # Look for exact match first
                for item in response['items']:
                    channel_title = item['snippet']['title'].lower()
                    if name.lower() in channel_title or channel_title in name.lower():
                        return item['snippet']['channelId']
                # If no exact match, return first result
                return response['items'][0]['snippet']['channelId']

            raise ValueError(f"Channel not found: {name}")

        except Exception as e:
            raise ValueError(f"Error finding channel '{name}': {str(e)}")

    def get_channel_videos(self, channel_input: str, max_videos: int = 100) -> List[Dict[str, Any]]:
        """Get video data from a channel"""
        try:
            channel_id = self.get_channel_id(channel_input)

            # Get channel info
            channel_request = self.youtube.channels().list(
                part="contentDetails,statistics,snippet",
                id=channel_id
            )
            channel_response = channel_request.execute()

            if not channel_response['items']:
                raise ValueError("Channel not found")

            channel_info = channel_response['items'][0]
            uploads_playlist_id = channel_info['contentDetails']['relatedPlaylists']['uploads']
            channel_subscriber_count = int(channel_info['statistics'].get('subscriberCount', 0))

            # Get video IDs from uploads playlist
            video_ids = []
            next_page_token = None

            while len(video_ids) < max_videos:
                request = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_videos - len(video_ids)),
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    if video_id not in video_ids:  # Avoid duplicates
                        video_ids.append(video_id)

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

            # Get detailed video information in batches
            videos_data = []
            for i in range(0, len(video_ids), 50):  # API allows max 50 IDs per request
                batch_ids = video_ids[i:i + 50]

                request = self.youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=','.join(batch_ids)
                )
                response = request.execute()

                for video in response['items']:
                    # Parse duration
                    duration_str = video['contentDetails']['duration']
                    duration_seconds = self._parse_duration(duration_str)

                    # Get thumbnail URL (highest quality available)
                    thumbnails = video['snippet']['thumbnails']
                    thumbnail_url = (
                            thumbnails.get('maxres', {}).get('url') or
                            thumbnails.get('high', {}).get('url') or
                            thumbnails.get('medium', {}).get('url') or
                            thumbnails.get('default', {}).get('url', '')
                    )

                    video_data = {
                        'video_id': video['id'],
                        'title': video['snippet']['title'],
                        'description': video['snippet']['description'],
                        'published_at': video['snippet']['publishedAt'],
                        'thumbnail_url': thumbnail_url,
                        'view_count': int(video['statistics'].get('viewCount', 0)),
                        'like_count': int(video['statistics'].get('likeCount', 0)),
                        'comment_count': int(video['statistics'].get('commentCount', 0)),
                        'duration_seconds': duration_seconds,
                        'channel_subscriber_count': channel_subscriber_count,
                        'category_id': video['snippet'].get('categoryId', ''),
                        'tags': video['snippet'].get('tags', []),
                        'language': video['snippet'].get('defaultLanguage', 'en')
                    }
                    videos_data.append(video_data)

            return videos_data

        except Exception as e:
            raise Exception(f"Error fetching channel videos: {str(e)}")

    def _parse_duration(self, duration_str: str) -> int:
        """Parse YouTube duration format (PT4M13S) to seconds"""
        import re

        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds