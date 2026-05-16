import pandas as pd
import json
import re
def normalize_name(name):
    if not name: return ""
    name = name.lower()

    # Mapeamentos de times conhecidos para termos simples
    if 'bayern' in name: return 'bayern'
    if 'leverkusen' in name: return 'leverkusen'
    if 'dortmund' in name: return 'dortmund'
    if 'leipzig' in name: return 'leipzig'
    if 'werder' in name: return 'werder'
    if 'stuttgart' in name: return 'stuttgart'
    if 'gladbach' in name: return 'gladbach'
    if 'heidenheim' in name: return 'heidenheim'
    if 'st. pauli' in name or 'sankt pauli' in name: return 'stpauli'
    if 'frankfurt' in name: return 'frankfurt'
    if 'bochum' in name: return 'bochum'
    if 'freiburg' in name: return 'freiburg'
    if 'union berlin' in name: return 'union'
    if 'mainz' in name: return 'mainz'
    if 'hoffenheim' in name: return 'hoffenheim'
    if 'wolfsburg' in name: return 'wolfsburg'
    if 'augsburg' in name: return 'augsburg'
    if 'kiel' in name: return 'kiel'

    return re.sub(r'[^a-z]', '', name)
def main():
    df_teams = pd.read_csv('data/game_id_to_teams.csv')
    video_info = json.load(open('config/video_info.json'))

    results = []
    for _, row in df_teams.iterrows():
        h_norm = normalize_name(row['home_team'])
        a_norm = normalize_name(row['away_team'])
        rd = row['round']

        print(f"Checking GameID {row['game_id']}: {row['home_team']} ({h_norm}) x {row['away_team']} ({a_norm}) RD {rd}")

        match_found = None
        for v in video_info:
            title = v['title'].lower()
            if (h_norm in title or row['home_team'].lower() in title) and \
               (a_norm in title or row['away_team'].lower() in title):

                round_match = re.search(r'rodada (\d+)', title)
                if round_match:
                    v_rd = int(round_match.group(1))
                    if v_rd == rd:
                        match_found = v
                        print(f"  -> Found RD match: {v['title']}")
                        break
                    else:
                        pass # Wrong round
                else:
                    match_found = v
                    print(f"  -> Found Team match (no RD in title): {v['title']}")

        if match_found:
            results.append({
                'game_id': int(row['game_id']),
                'video_id': match_found['video_id'],
                'video_title': match_found['title'],
                'teams': f"{row['home_team']} x {row['away_team']}",
                'round': int(row['round']),
                'date': row['date']
            })
        else:
            print(f"  !! No match found for GameID {row['game_id']}")            
    with open('data/game_video_mapping_final.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Final mapping complete: {len(results)} games mapped to videos.")

if __name__ == "__main__":
    main()
