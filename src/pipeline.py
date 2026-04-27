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

def get_match_data(ts, start, fh_end, sh_start, sh_end):
    if ts < start: return "Pré-jogo", f"{int((ts-start)//60)}'"
    if ts <= fh_end:
        m = (ts - start) / 60
        return "1º Tempo", f"{int(m)}'" if m <= 45 else f"45+{int(m-45)}'"
    if ts < sh_start: return "Intervalo", "Int"
    if ts <= sh_end:
        m_total = ((ts - sh_start) / 60) + 45
        return "2º Tempo", f"{int(m_total)}'" if m_total <= 90 else f"90+{int(m_total-90)}'"
    return "Pós-jogo", "Pós"

def process_video(video_config, info_list):
    video_id = video_config['video_id']
    v_start = time_to_seconds(video_config['game_start_time'])
    fh_end = time_to_seconds(video_config['first_half_end'])
    sh_start = time_to_seconds(video_config['second_half_start'])
    sh_end = time_to_seconds(video_config['second_half_end'])
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    raw_dir = 'data/raw'
    os.makedirs(raw_dir, exist_ok=True)
    raw_file = os.path.join(raw_dir, f"chat_{video_id}.live_chat.json")
    
    print(f"\n>>> Processando: {video_id}")

    # 1. Título
    if not any(i.get('video_id') == video_id for i in info_list):
        print(f"Obtendo título...")
        cmd_title = ['yt-dlp', '--get-title']
        if os.path.exists('cookies.txt'):
            cmd_title.extend(['--cookies', 'cookies.txt'])
        cmd_title.append(video_url)
        
        res = subprocess.run(cmd_title, capture_output=True, text=True)
        title = res.stdout.strip() or "Título não encontrado"
        info_list.append({"video_id": video_id, "title": title})
    else:
        title = next(i['title'] for i in info_list if i['video_id'] == video_id)
        print(f"Título: {title}")

    # 2. Download se necessário
    if not os.path.exists(raw_file):
        print(f"Baixando chat...")
        cmd_dl = ['yt-dlp', '--skip-download', '--write-subs', '--sub-langs', 'live_chat', '--output', os.path.join(raw_dir, f"chat_{video_id}")]
        if os.path.exists('cookies.txt'):
            cmd_dl.extend(['--cookies', 'cookies.txt'])
        cmd_dl.append(video_url)
        
        subprocess.run(cmd_dl)

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
                        periodo, label = get_match_data(ts_video, v_start, fh_end, sh_start, sh_end)
                        if msg:
                            mensagens.append({
                                'timestamp_video_segundos': ts_video,
                                'timestamp_jogo_segundos': ts_video - v_start,
                                'label_partida': label, 'periodo': periodo,
                                'autor': renderer.get('authorName', {}).get('simpleText', 'Desconhecido'),
                                'mensagem': msg
                            })
                except Exception: continue
        
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
            if isinstance(data, list):
                info_list = data
            elif isinstance(data, dict):
                info_list = [data]

    for video in videos:
        process_video(video, info_list)

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
