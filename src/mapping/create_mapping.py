import os
import json
import re

def create_mapping():
    # 1. Load the mapped teams
    with open('scratch/team_mapping.json', 'r') as f:
        team_mapping = json.load(f)
    
    # Create inverse mapping (Name to ID)
    name_to_id = {}
    for tid, name in team_mapping.items():
        # normalize names for easier matching
        norm = name.lower()
        norm = re.sub(r'[^a-z0-9]', '', norm)
        name_to_id[norm] = tid
        
    # Manual overrides for some common names in video titles
    overrides = {
        'freiburg': 'SC Freiburg',
        'leverkusen': 'Bayer Leverkusen',
        'bayern': 'Bayern Munich', 'bayern de munique': 'Bayern Munich',
        'gladbach': 'Borussia Mönchengladbach',
        'dortmund': 'Borussia Dortmund',
        'leipzig': 'RB Leipzig',
        'hoffenheim': 'TSG Hoffenheim',
        'frankfurt': 'Eintracht Frankfurt',
        'wolfsburg': 'VfL Wolfsburg',
        'stpauli': 'FC St. Pauli',
        'st pauli': 'FC St. Pauli',
        'st. pauli': 'FC St. Pauli',
        'heidenheim': '1. FC Heidenheim',
        'mainz': 'Mainz 05',
        'augsburg': 'FC Augsburg',
        'bremen': 'Werder Bremen',
        'bochum': 'VfL Bochum',
        'kiel': 'Holstein Kiel',
        'union': 'Union Berlin',
        'stuttgart': 'VfB Stuttgart'
    }
    for k, v in overrides.items():
        norm = k.lower()
        norm = re.sub(r'[^a-z0-9]', '', norm)
        if v in team_mapping.values():
            for tid, name in team_mapping.items():
                if name == v:
                    name_to_id[norm] = tid
                    break
        
    # 2. Load the match results
    with open('scratch/match_results.json', 'r') as f:
        match_results = json.load(f)
        
    # 3. Load videos
    with open('config/video_info.json', 'r') as f:
        videos = json.load(f)
        
    mapping = {}
    unmapped = []
    
    for video in videos:
        vid = video['video_id']
        title = video['title']
        
        # Extract HOME X AWAY
        # E.g. JOGO COMPLETO: MAINZ 05 X BAYER LEVERKUSEN | ...
        match = re.search(r'(?:JOGO COMPLETO: )?(.+?) X (.+?)(?: \|)', title)
        if not match:
            unmapped.append((vid, title, "Could not parse teams from title"))
            continue
            
        home_str = match.group(1).strip()
        away_str = match.group(2).strip()
        
        h_norm = re.sub(r'[^a-z0-9]', '', home_str.lower())
        a_norm = re.sub(r'[^a-z0-9]', '', away_str.lower())
        
        home_id = name_to_id.get(h_norm)
        away_id = name_to_id.get(a_norm)
        
        if not home_id or not away_id:
            unmapped.append((vid, title, f"Team not found in mapping: {home_str}({home_id}) vs {away_str}({away_id})"))
            continue
            
        # Find the GAME_ID where home_id plays away_id
        candidate_games = []
        for gid, data in match_results.items():
            if str(data['home_team_id']) == str(home_id) and str(data['away_team_id']) == str(away_id):
                candidate_games.append(gid)
                
        if len(candidate_games) == 1:
            mapping[vid] = candidate_games[0]
        elif len(candidate_games) == 0:
            unmapped.append((vid, title, "No match found with this home/away configuration in the 306 games"))
        else:
            unmapped.append((vid, title, f"Multiple matches found: {candidate_games}"))
            
    # Save the mapping
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/video_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
        
    print(f"Successfully mapped {len(mapping)} videos.")
    print(f"Failed to map {len(unmapped)} videos.")
    for u in unmapped:
        print(f"  - {u[0]} ({u[1]}): {u[2]}")

if __name__ == "__main__":
    create_mapping()
