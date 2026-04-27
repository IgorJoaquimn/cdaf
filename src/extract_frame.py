import json
import subprocess
import os

def time_to_seconds(t_str):
    parts = list(map(int, t_str.split(':')))
    if len(parts) == 2: return parts[0] * 60 + parts[1]
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def main():
    with open('config/settings.json', 'r') as f:
        videos = json.load(f)
    
    os.makedirs('data/frames', exist_ok=True)

    for v in videos:
        vid = v['video_id']
        t_str = v['game_start_time']
        seconds = time_to_seconds(t_str)
        
        output_path = f"data/frames/{vid}_start.png"
        print(f"Extraindo frame para {vid} em {t_str}...")

        # Obtém a URL real do vídeo (formato 18 costuma ser 360p ou 720p estável)
        try:
            url_res = subprocess.run(
                ['yt-dlp', '-f', 'best', '--get-url', f'https://www.youtube.com/watch?v={vid}'],
                capture_output=True, text=True, check=True
            )
            video_url = url_res.stdout.strip()

            # ffmpeg extrai 1 frame no timestamp específico
            subprocess.run([
                'ffmpeg', '-ss', str(seconds), '-i', video_url,
                '-frames:v', '1', '-q:v', '2', '-y', output_path
            ], check=True, capture_output=True)
            
            print(f"Sucesso: {output_path}")
        except Exception as e:
            print(f"Erro ao extrair frame de {vid}: {e}")

if __name__ == "__main__":
    main()
