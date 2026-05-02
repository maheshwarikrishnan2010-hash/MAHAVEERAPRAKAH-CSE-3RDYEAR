import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import YOUTUBE_API_KEY, SEARCH_URL, COMMENT_URL, VIDEO_URL

session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)


def fetch_videos(query="electric vehicle review", max_results=1000000):

    videos = []
    next_page_token = None

    while len(videos) < max_results:
        limit = min(50, max_results - len(videos))
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "publishedAfter": "2025-01-01T00:00:00Z",
            "publishedBefore": "2026-12-31T23:59:59Z",
            "maxResults": limit,
            "key": YOUTUBE_API_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = session.get(SEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        if not items:
            break
            
        video_ids = [item["id"]["videoId"] for item in items]
        
        if video_ids:
            # Batch fetch video statistics and content details
            video_params = {
                "part": "snippet,statistics",
                "id": ",".join(video_ids),
                "key": YOUTUBE_API_KEY
            }
            vid_response = session.get(VIDEO_URL, params=video_params, timeout=30)
            if vid_response.status_code == 200:
                vid_data = vid_response.json()
                for item in vid_data.get("items", []):
                    snippet = item.get("snippet", {})
                    statistics = item.get("statistics", {})
                    videos.append({
                        "video_id": item.get("id"),
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "tags": ", ".join(snippet.get("tags", [])),
                        "category": snippet.get("categoryId"),
                        "views": statistics.get("viewCount"),
                        "likes": statistics.get("likeCount"),
                        "comments_count": statistics.get("commentCount")
                    })

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(videos)


def fetch_comments(video_id, max_results=1000000):

    comments = []
    next_page_token = None

    while len(comments) < max_results:
        limit = min(100, max_results - len(comments))
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": limit,
            "key": YOUTUBE_API_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = session.get(COMMENT_URL, params=params, timeout=30)
        if response.status_code != 200:
            break
            
        data = response.json()

        for item in data.get("items", []):
            try:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_text": snippet.get("textOriginal", ""),
                    "comment_likes": snippet.get("likeCount", 0),
                    "comment_timestamp": snippet.get("publishedAt", "")
                })
            except KeyError:
                continue
                
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments