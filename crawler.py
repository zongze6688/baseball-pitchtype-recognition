import re
from bs4 import BeautifulSoup
import pandas as pd
import shutil
from pathlib import Path
import subprocess
import os
import requests
try:
    import ffmpeg as ffmpeg_lib  # ffmpeg-python
except Exception:
    ffmpeg_lib = None

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parents[2]
df = pd.read_csv(SCRIPT_DIR / 'data' / 'statcast_data.csv')
download_dir = WORKSPACE_DIR / 'data' / 'videos_tiny'
os.makedirs(download_dir, exist_ok=True)
clipped_dir = WORKSPACE_DIR / 'data' / 'videos_clip'
os.makedirs(clipped_dir, exist_ok=True)

# playId
def extract_play_id(url):
    match = re.search(r'playId=([a-f0-9\-]+)', url)
    return match.group(1) if match else None

def get_video_url_from_page(play_id):
    url = f'https://baseballsavant.mlb.com/sporty-videos?playId={play_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"error request : {play_id}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    source_tag = soup.find('source', {'type': 'video/mp4'})

    if source_tag and 'src' in source_tag.attrs:
        return source_tag['src']
    else:
        print(f"cannot find <source> tag: {play_id}")
        return None

def check_exist(play_id, dir):
    dir = os.path.join(dir, f'{play_id}.mp4')
    if os.path.exists(dir):
        print(f'video {str(dir).split('/')[-1]}.mp4 already exists')
        return True
    else:
        return False

def download_video(video_url, play_id, output_dir):
    output_path = os.path.join(output_dir, f'{play_id}.mp4')
    if os.path.exists(output_path):
        print(f'video {play_id}.mp4 already exists')
        return output_path

    print(f'downloading {play_id}.mp4')

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://baseballsavant.mlb.com/'
    }

    try:
        with requests.get(video_url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                    if chunk:
                        f.write(chunk)
        print(f'video saved as {play_id}.mp4')
        return output_path
    except Exception as e:
        print(f"error downloading : {e}")
        return None

def clip_video_ffmpeg(input_path, play_id, output_dir):
    output_path = os.path.join(output_dir, f'{play_id}.mp4')
    if os.path.exists(output_path):
        print(f'clipped {play_id}.mp4 already exists')
        return False

    print(f'clipping 5s for: {play_id}.mp4')

    try:
        if ffmpeg_lib is not None and hasattr(ffmpeg_lib, "input"):
            (
                ffmpeg_lib
                .input(input_path, ss=0, to=5)
                .output(output_path, c='copy')  
                .run(overwrite_output=True, quiet=False)
            )
        else:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg binary not found; install ffmpeg or ffmpeg-python")
            cmd = [
                "ffmpeg",
                "-ss", "0",
                "-to", "5",
                "-i", input_path,
                "-c", "copy",
                "-y",
                output_path,
            ]
            subprocess.run(cmd, check=True)

        print(f'video saved as  {play_id}.mp4')
        return True
    except Exception as e:
        print(f"error clipping: {e}")
        return False


def main():
    for index, row in df.iterrows():
        
        video_link = row['VideoLink']
        play_id = extract_play_id(video_link)
        if not play_id:
            print(f"cannot extract playId: {video_link}")
            continue
        
        if check_exist(play_id, clipped_dir):
                    continue
        print(f' playId: {play_id}')
        video_url = get_video_url_from_page(play_id)
        if not video_url:
            continue

        downloaded_path = download_video(video_url, play_id, clipped_dir)
        if downloaded_path:
            clip_success = clip_video_ffmpeg(downloaded_path, play_id, clipped_dir)
            if clip_success:
                try:
                    os.remove(downloaded_path)
                    print(f'raw video {play_id}.mp4 removed')
                except Exception as e:
                    print(f"removal failed: {e}")

if __name__ == '__main__':
    main()
