import os
from dotenv import load_dotenv

# Path to the .env file located at project for data analyst/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

EV_API_URL = "https://data.wa.gov/resource/f6w7-q2d2.json"

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
COMMENT_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"