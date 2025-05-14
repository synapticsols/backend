import datetime
import requests
from urllib.parse import urlparse
from utils.time_utils import is_within_24_hours

async def fetch_reddit_posts(link: str):
    headers = {"User-Agent": "DefenseAI/0.1"}

    # Parse the link safely
    parsed_url = urlparse(link)
    path_parts = parsed_url.path.strip('/').split('/')

    if 'r' not in path_parts:
        raise Exception("Invalid Reddit link. No subreddit found.")

    try:
        subreddit_index = path_parts.index('r') + 1
        subreddit = path_parts[subreddit_index]
    except Exception:
        raise Exception("Failed to extract subreddit from the link.")

    # Build the API URL
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=50"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch posts from Reddit. Status code: {response.status_code}")

    data = response.json()

    posts = []
    for post in data.get("data", {}).get("children", []):
        post_time = datetime.datetime.utcfromtimestamp(post["data"]["created_utc"])
        if is_within_24_hours(post_time):
            posts.append({
                "title": post["data"].get("title", ""),
                "content": post["data"].get("selftext", ""),
                "timestamp": post_time.isoformat()
            })

    return {"source": "reddit", "posts": posts}
