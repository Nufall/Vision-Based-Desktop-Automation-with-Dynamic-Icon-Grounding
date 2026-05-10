import os
import json
import winreg
import requests

API_URL = "https://jsonplaceholder.typicode.com/posts"


def get_desktop_path():
    """Read actual Desktop path from registry (handles OneDrive redirection)."""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        desktop, _ = winreg.QueryValueEx(key, "Desktop")
        winreg.CloseKey(key)
        return desktop
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Desktop")


OUTPUT_DIR = os.path.join(get_desktop_path(), "tjm-project")
CACHE_FILE = os.path.join(OUTPUT_DIR, ".posts_cache.json")
TIMEOUT = 10


def fetch_posts():
    """Fetch posts from API with timeout. Returns list or raises on failure."""
    response = requests.get(API_URL, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()[:10]


def load_cache():
    """Return cached posts if available, else None."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(posts):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f)


def main():
    posts = None

    try:
        print("Fetching posts from API...")
        posts = fetch_posts()
        save_cache(posts)
        print("Fetched from API.")
    except requests.exceptions.ConnectionError:
        print("ERROR: No network connection or DNS failure.")
    except requests.exceptions.Timeout:
        print(f"ERROR: Request timed out after {TIMEOUT}s.")
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: API returned {e.response.status_code}.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed — {e}")

    if posts is None:
        cached = load_cache()
        if cached:
            print("Falling back to cached data from last successful fetch.")
        else:
            print("No cached data available.")
            return


if __name__ == "__main__":
    main()
