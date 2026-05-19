import json
import subprocess
import os
from tqdm import tqdm

def fetch_metadata(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    # Fields: title, view_count, comment_count, upload_date, duration, like_count
    cmd = [
        'yt-dlp', 
        '--skip-download', 
        '--print', '%(view_count)s|%(comment_count)s|%(upload_date)s|%(duration)s|%(like_count)s',
        video_url
    ]
    if os.path.exists('cookies.txt'):
        cmd.extend(['--cookies', 'cookies.txt'])
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = res.stdout.strip().split('|')
        if len(output) >= 5:
            return {
                "view_count": int(output[0]) if output[0] != 'NA' else 0,
                "comment_count": int(output[1]) if output[1] != 'NA' else 0,
                "upload_date": output[2],
                "duration": int(output[3]) if output[3] != 'NA' else 0,
                "like_count": int(output[4]) if output[4] != 'NA' else 0
            }
    except Exception as e:
        print(f"Error fetching metadata for {video_id}: {e}")
    return None

def main():
    info_path = 'config/video_info.json'
    if not os.path.exists(info_path):
        print("video_info.json not found.")
        return

    with open(info_path, 'r', encoding='utf-8') as f:
        video_info = json.load(f)

    print(f"Fetching metadata for {len(video_info)} videos...")
    for v in tqdm(video_info):
        # Always refetch to ensure we have all data
        meta = fetch_metadata(v['video_id'])
        if meta:
            v.update(meta)

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, indent=4, ensure_ascii=False)
    
    print("\nMetadata updated in config/video_info.json")

if __name__ == "__main__":
    main()
