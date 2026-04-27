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
    if ts < start:
        return "Pré-jogo", f"{int((ts-start)//60)}'"
    if ts <= fh_end:
        m = (ts - start) / 60
        label = f"{int(m)}'" if m <= 45 else f"45+{int(m-45)}'"
        return "1º Tempo", label
    if ts < sh_start:
        return "Intervalo", "Int"
    if ts <= sh_end:
        m_sh = (ts - sh_start) / 60
        m_total = m_sh + 45
        label = f"{int(m_total)}'" if m_total <= 90 else f"90+{int(m_total-90)}'"
        return "2º Tempo", label
    return "Pós-jogo", "Pós"

def main():
    config_path = 'config/settings.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    v_start = time_to_seconds(config.get('game_start_time'))
    fh_end = time_to_seconds(config.get('first_half_end'))
    sh_start = time_to_seconds(config.get('second_half_start'))
    sh_end = time_to_seconds(config.get('second_half_end'))
    
    raw_file = glob.glob(f"data/raw/chat_download*.live_chat.json")[0]

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
                    message_text = "".join([part.get('text', '') for part in renderer.get('message', {}).get('runs', [])])
                    author = renderer.get('authorName', {}).get('simpleText', 'Desconhecido')
                    ts_video = int(replay_action.get('videoOffsetTimeMsec', '0')) / 1000
                    
                    periodo, match_label = get_match_data(ts_video, v_start, fh_end, sh_start, sh_end)
                    
                    if message_text:
                        mensagens.append({
                            'timestamp_video_segundos': ts_video,
                            'timestamp_jogo_segundos': ts_video - v_start,
                            'label_partida': match_label,
                            'periodo': periodo,
                            'autor': author,
                            'mensagem': message_text
                        })
            except Exception: continue

    df_chat = pd.DataFrame(mensagens)
    output_path = os.path.join('data/processed', f"chat_{config['video_id']}_processed.csv")
    df_chat.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Pipeline concluído com labels de futebol.")

if __name__ == "__main__":
    main()
