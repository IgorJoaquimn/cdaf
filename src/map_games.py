import json
import glob
import os
import pandas as pd
import re

def normalize_name(name):
    if not name: return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    # Common mappings
    mapping = {
        'eintrachtfrankfurt': 'frankfurt',
        'bayernmnchen': 'bayern',
        'bayernmunich': 'bayern',
        'bayerleverkusen': 'leverkusen',
        'borussiadortmund': 'dortmund',
        'rbleipzig': 'leipzig',
        'werderbremen': 'werder',
        'vfbstuttgart': 'stuttgart',
        'borussiamnchengladbach': 'gladbach',
        'fcheidenheim': 'heidenheim',
        'fcsanktpauli': 'stpauli',
        'holsteinkiel': 'kiel'
    }
    for k, v in mapping.items():
        if k in name: return v
    return name

def get_match_summary_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    gid = data[0]['GAME_ID']
    sid = data[0]['STADIUM_METADATA']['STADIUM_ID']
    
    goals_home = 0
    goals_away = 0
    teams = set()
    for e in data:
        tid = e['GAME_EVENTS'].get('TEAM_ID')
        if tid: teams.add(tid)
        if e['POSSESSION_EVENTS'].get('POSSESSION_EVENT_TYPE') == 'SH' and e['POSSESSION_EVENTS'].get('SHOT_OUTCOME_TYPE') == 'G':
            if e['GAME_EVENTS'].get('HOME_TEAM'): goals_home += 1
            else: goals_away += 1
    
    # Tentar identificar home/away team id
    home_tid = None
    away_tid = None
    for tid in teams:
        for e in data:
            if e['GAME_EVENTS'].get('TEAM_ID') == tid:
                if e['GAME_EVENTS'].get('HOME_TEAM'): home_tid = tid
                else: away_tid = tid
                break
    
    return {
        'game_id': gid,
        'stadium_id': sid,
        'score': f'{goals_home}-{goals_away}',
        'home_tid': home_tid,
        'away_tid': away_tid
    }

def main():
    # 1. Load Event Data Summaries
    print("Extracting event summaries...")
    event_summaries = []
    for f in sorted(glob.glob('data/events/2024-2025/*.json')):
        event_summaries.append(get_match_summary_from_json(f))
    
    # 2. Load FotMob Matches
    print("Loading FotMob data...")
    df_fotmob = pd.read_csv('data/processed/consolidated/fotmob_matches.csv')
    df_fotmob['score_key'] = df_fotmob['home_score'].astype(str) + '-' + df_fotmob['away_score'].astype(str)
    
    # 3. Load Video Info
    print("Loading Video info...")
    video_info = json.load(open('config/video_info.json'))
    
    # 4. Map GameID -> FotMob -> VideoID
    mapping = []
    
    for ev in event_summaries:
        # Tentar encontrar matches com o mesmo score
        possible_matches = df_fotmob[df_fotmob['score_key'] == ev['score']]
        
        match_found = None
        for _, m in possible_matches.iterrows():
            # Tentar encontrar o vídeo correspondente pelo título
            home_norm = normalize_name(m['home_team'])
            away_norm = normalize_name(m['away_team'])
            
            for v in video_info:
                title_norm = normalize_name(v['title'])
                if home_norm in title_norm and away_norm in title_norm:
                    # Checar se a rodada bate (opcional mas bom)
                    round_match = re.search(r'RODADA (\d+)', v['title'])
                    if round_match:
                        v_round = int(round_match.group(1))
                        if v_round == m['round']:
                            match_found = {'video_id': v['video_id'], 'title': v['title'], 'round': m['round'], 'teams': f"{m['home_team']} x {m['away_team']}"}
                            break
                elif "PLAYOFF" in title_norm and ("ELVERSBERG" in m['home_team'].upper() or "ELVERSBERG" in m['away_team'].upper()):
                     # Caso especial playoff
                     match_found = {'video_id': v['video_id'], 'title': v['title'], 'round': 'Playoff', 'teams': f"{m['home_team']} x {m['away_team']}"}
            
            if match_found: break
            
        if match_found:
            mapping.append({
                'game_id': ev['game_id'],
                'video_id': match_found['video_id'],
                'video_title': match_found['title'],
                'teams': match_found['teams'],
                'score': ev['score']
            })
        else:
            print(f"Could not map GameID {ev['game_id']} (Score {ev['score']})")

    # Save mapping
    with open('data/game_video_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=4)
    
    print(f"\nMapped {len(mapping)} / {len(event_summaries)} games.")
    print("Mapping saved to data/game_video_mapping.json")

if __name__ == "__main__":
    main()
