import json
import os
import glob
import pandas as pd
from tqdm import tqdm

def process_events(events_dir, output_path):
    all_events = []
    
    json_files = glob.glob(os.path.join(events_dir, "*.json"))
    print(f"Encontrados {len(json_files)} arquivos JSON.")
    
    for file_path in tqdm(json_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for i in range(len(data)):
            event = data[i]
            pos_events = event.get('POSSESSION_EVENTS', {})
            game_events = event.get('GAME_EVENTS', {})
            ball = event.get('BALL', [])
            stadium = event.get('STADIUM_METADATA', {})
            
            if not ball:
                continue
                
            # Extrair dados básicos
            e_data = {
                'game_id': event.get('GAME_ID'),
                'period': game_events.get('PERIOD'),
                'timestamp': event.get('EVENT_TIME'),
                'clock': game_events.get('START_FORMATTED_GAME_CLOCK'),
                'team_id': game_events.get('TEAM_ID'),
                'player_id': game_events.get('PLAYER_ID'),
                'type': pos_events.get('POSSESSION_EVENT_TYPE'),
                'outcome': pos_events.get('PASS_OUTCOME_TYPE'),
                'shot_outcome': pos_events.get('SHOT_OUTCOME_TYPE'),
                'start_x': ball[0]['X'],
                'start_y': ball[0]['Y'],
                'home_team': game_events.get('HOME_TEAM'),
                'attacking_direction': stadium.get('TEAM_ATTACKING_DIRECTION'),
                'pitch_length': stadium.get('PITCH_LENGTH'),
                'pitch_width': stadium.get('PITCH_WIDTH')
            }
            
            # Tentar pegar o end_x, end_y do próximo evento se for do mesmo time
            if i + 1 < len(data):
                next_event = data[i+1]
                next_ball = next_event.get('BALL', [])
                next_team = next_event.get('GAME_EVENTS', {}).get('TEAM_ID')
                
                if next_ball and next_team == e_data['team_id']:
                    e_data['end_x'] = next_ball[0]['X']
                    e_data['end_y'] = next_ball[0]['Y']
                else:
                    e_data['end_x'] = None
                    e_data['end_y'] = None
            else:
                e_data['end_x'] = None
                e_data['end_y'] = None
                
            all_events.append(e_data)
            
    df = pd.DataFrame(all_events)
    df.to_parquet(output_path, index=False)
    print(f"Salvo {len(df)} eventos em {output_path}")

if __name__ == "__main__":
    process_events('data/events/2024-2025', 'data/events.parquet')
