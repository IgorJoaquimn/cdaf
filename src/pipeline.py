import json
import pandas as pd
import os
import subprocess
import glob

def time_to_seconds(t_str):
    if not t_str: return 0
    parts = list(map(int, t_str.split(':')))
    if len(parts) == 2: return parts[0] * 60 + parts[1]
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def s2t(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0: return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def get_match_data(ts, v_start, sh_start):
    # Lógica simplificada sem o fim das metades
    if ts < v_start:
        m = (ts - v_start) / 60
        return "Pré-jogo", f"{int(m)}'"
    
    if ts < sh_start:
        # Tudo entre o início do jogo e o início do 2T é considerado 1º Tempo/Intervalo
        m = (ts - v_start) / 60
        label = f"{int(m)}'" if m <= 45 else f"45+{int(m-45)}'"
        return "1º Tempo", label
    
    # 2º Tempo
    m_sh = (ts - sh_start) / 60
    m_total = m_sh + 45
    label = f"{int(m_total)}'" if m_total <= 90 else f"90+{int(m_total-90)}'"
    return "2º Tempo", label

def process_video(video_config, info_list):
    video_id = video_config['video_id']
    v_start = time_to_seconds(video_config['game_start_time'])
    sh_start = time_to_seconds(video_config['second_half_start'])
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    raw_dir = 'data/raw'
    os.makedirs(raw_dir, exist_ok=True)
    raw_file = os.path.join(raw_dir, f"chat_{video_id}.live_chat.json")
    
    print(f"\n>>> Processando: {video_id}")

    # 1. Título
    if not any(i.get('video_id') == video_id for i in info_list):
        cmd_title = ['yt-dlp', '--get-title']
        if os.path.exists('cookies.txt'): cmd_title.extend(['--cookies', 'cookies.txt'])
        cmd_title.append(video_url)
        res = subprocess.run(cmd_title, capture_output=True, text=True)
        title = res.stdout.strip() or "Título não encontrado"
        info_list.append({"video_id": video_id, "title": title})
    else:
        title = next(i['title'] for i in info_list if i['video_id'] == video_id)
    print(f"Título: {title}")

    # 2. Download
    if not os.path.exists(raw_file):
        print(f"Baixando chat...")
        cmd_dl = ['yt-dlp', '--skip-download', '--write-subs', '--sub-langs', 'live_chat', '--output', os.path.join(raw_dir, f"chat_{video_id}")]
        if os.path.exists('cookies.txt'): cmd_dl.extend(['--cookies', 'cookies.txt'])
        cmd_dl.append(video_url)
        subprocess.run(cmd_dl, capture_output=True)

    # 3. Parsing
    if os.path.exists(raw_file):
        print(f"Parsing mensagens...")
        mensagens = []
        with open(raw_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    replay_action = data.get('replayChatItemAction', {})
                    actions = replay_action.get('actions', [])
                    if not actions: continue
                    item = actions[0].get('addChatItemAction', {}).get('item', {})
                    renderer = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer')
                    if renderer:
                        msg = "".join([p.get('text', '') for p in renderer.get('message', {}).get('runs', [])])
                        ts_video = int(replay_action.get('videoOffsetTimeMsec', '0')) / 1000
                        periodo, label = get_match_data(ts_video, v_start, sh_start)
                        if msg:
                            mensagens.append({
                                'timestamp_video_segundos': ts_video,
                                'timestamp_jogo_segundos': ts_video - v_start,
                                'label_partida': label, 
                                'periodo': periodo,
                                'autor': renderer.get('authorName', {}).get('simpleText', 'Desconhecido'),
                                'mensagem': msg
                            })
                except Exception: continue
        
        if mensagens:
            df = pd.DataFrame(mensagens).sort_values(by='timestamp_video_segundos')
            os.makedirs('data/processed', exist_ok=True)
            df.to_csv(f"data/processed/chat_{video_id}_processed.csv", index=False)
            print(f"Concluído: {len(df)} mensagens.")
    else:
        print(f"Erro: Arquivo bruto {raw_file} não encontrado.")

def main():
    with open('config/settings.json', 'r') as f:
        videos = json.load(f)
    
    info_path = 'config/video_info.json'
    info_list = []
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            data = json.load(f)
            info_list = data if isinstance(data, list) else [data]

    for video in videos:
        process_video(video, info_list)

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
