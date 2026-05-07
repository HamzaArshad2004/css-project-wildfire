"""
Enhanced Reddit Data Collection Script
Collects wildfire-related posts and saves them grouped by day

Required environment variables (set in .env or shell):
    REDDIT_CLIENT_ID     - Reddit app client ID
    REDDIT_CLIENT_SECRET - Reddit app client secret
    REDDIT_USER_AGENT    - Descriptive user-agent string
"""

import os
import praw
import json
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import time
import sys

# ========= CONFIG =========
SUBREDDITS = [
    "LosAngeles",
    "California",
    "wildfire",
    "socal"
]

KEYWORDS = ["fire", "wildfire", "smoke", "evacu", "evacuation"]

START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 2, 5, tzinfo=timezone.utc)

SEARCH_LIMIT = 500

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "reddit_wildfire_posts_by_day.json"
# ==========================


def load_project_env(env_path: Path):
    """Load KEY=VALUE pairs from .env into process environment if missing."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_project_env(PROJECT_ROOT / ".env")

# Credentials loaded from environment variables.
# Copy .env.example to .env and fill in your values, or export them in your shell.
CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "crisis-analysis-bot/1.0")

def matches_keywords(text: str) -> bool:
    """Check if text contains any of the target keywords"""
    if not text:
        return False
    text = text.lower()
    return any(k in text for k in KEYWORDS)

def collect_top_comments(post, max_comments=3):
    """
    Collect top comments for additional context
    (Optional - currently disabled for speed)
    """
    try:
        post.comments.replace_more(limit=0)
        comments = []
        for comment in post.comments[:max_comments]:
            if hasattr(comment, 'body'):
                comments.append({
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(
                        comment.created_utc, tz=timezone.utc
                    ).isoformat()
                })
        return comments
    except:
        return []

def search_with_retry(subreddit, query, max_retries=3, **kwargs):
    """Search with exponential backoff for rate limits"""
    for attempt in range(max_retries):
        try:
            return list(subreddit.search(query=query, **kwargs))
        except Exception as e:
            error_msg = str(e).lower()
            if ("rate" in error_msg or "429" in error_msg) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                print(f"  ⚠️  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Error: {e}")
                return []
    return []

def main():
    print("=" * 70)
    print("REDDIT DATA COLLECTION - LA WILDFIRES")
    print("=" * 70)

    # Validate credentials
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ Reddit credentials not found.")
        print("   Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        print("   See .env.example for details.")
        sys.exit(1)

    # Initialize Reddit API
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        # Test connection
        reddit.user.me()
        print("✓ Reddit API connected successfully\n")
    except Exception as e:
        print(f"❌ Failed to connect to Reddit API: {e}")
        print("Please check your credentials.")
        sys.exit(1)
    
    posts_by_day = defaultdict(list)
    total_scanned = 0
    total_matched = 0
    
    # Track metadata
    metadata = {
        'collection_date': datetime.now(timezone.utc).isoformat(),
        'date_range': {
            'start': START_DATE.isoformat(),
            'end': END_DATE.isoformat()
        },
        'subreddits': SUBREDDITS,
        'keywords': KEYWORDS,
        'search_limit_per_sub': SEARCH_LIMIT
    }
    
    # Collect from each subreddit
    for subreddit_name in SUBREDDITS:
        print(f"📡 Searching r/{subreddit_name}...")
        subreddit = reddit.subreddit(subreddit_name)
        query = " OR ".join(KEYWORDS)
        
        posts = search_with_retry(
            subreddit,
            query=query,
            sort="new",
            time_filter="all",
            limit=SEARCH_LIMIT
        )
        
        sub_matched = 0
        for post in posts:
            total_scanned += 1
            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            
            # Filter by date range
            if created < START_DATE or created > END_DATE:
                continue
            
            # Check keyword match
            text = f"{post.title or ''} {post.selftext or ''}"
            if not matches_keywords(text):
                continue
            
            total_matched += 1
            sub_matched += 1
            day_str = created.strftime("%Y-%m-%d")
            
            post_data = {
                "subreddit": subreddit_name,
                "post_id": post.id,
                "title": post.title,
                "text": post.selftext,
                "author": str(post.author),
                "created_utc": created.isoformat(),
                "score": post.score,
                "num_comments": post.num_comments,
                "url": post.url,
                "upvote_ratio": getattr(post, 'upvote_ratio', None),
                "is_self": post.is_self,
                # Optionally collect comments (disabled for speed)
                # "top_comments": collect_top_comments(post, max_comments=3)
            }
            
            posts_by_day[day_str].append(post_data)
        
        print(f"  ✓ Found {sub_matched} matching posts")
        time.sleep(2)  # Be nice to Reddit API
    
    # Update metadata
    metadata['total_scanned'] = total_scanned
    metadata['total_matched'] = total_matched
    metadata['days_with_data'] = len(posts_by_day)
    
    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Posts scanned: {total_scanned}")
    print(f"Posts matched: {total_matched}")
    print(f"Days with data: {len(posts_by_day)}")
    print(f"Avg posts/day: {total_matched / max(len(posts_by_day), 1):.1f}")
    
    # Save with metadata
    output = {
        'metadata': metadata,
        'posts_by_day': posts_by_day
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("SAMPLE DATA (First 3 days)")
    print("=" * 70)
    for i, (day, plist) in enumerate(sorted(posts_by_day.items())[:3], start=1):
        print(f"\n📅 {day} ({len(plist)} posts):")
        for p in plist[:3]:
            title = p['title'][:60] + "..." if len(p['title']) > 60 else p['title']
            print(f"  • [{p['subreddit']}] {title}")

if __name__ == "__main__":
    main()