import pandas as pd
import numpy as np

def main():
    df_events = pd.read_csv('data/event_stats_summary.csv')
    df_fotmob = pd.read_csv('data/processed/consolidated/fotmob_match_stats.csv')
    
    mapping = []
    used_json_gids = set()
    
    # Sort FotMob games to try to map more "unique" scores first
    df_fotmob['score_freq'] = df_fotmob.groupby(['home_score', 'away_score'])['home_team'].transform('count')
    df_fotmob = df_fotmob.sort_values('score_freq')
    
    for _, f in df_fotmob.iterrows():
        # Possible JSONs with same score
        mask = (df_events['goals_h'] == f['home_score']) & (df_events['goals_a'] == f['away_score'])
        possible = df_events[mask & ~df_events['game_id'].isin(used_json_gids)].copy()
        
        if len(possible) == 0:
            # Try swapped score
            mask_inv = (df_events['goals_h'] == f['away_score']) & (df_events['goals_a'] == f['home_score'])
            possible = df_events[mask_inv & ~df_events['game_id'].isin(used_json_gids)].copy()
            swapped = True
        else:
            swapped = False
            
        if len(possible) > 0:
            # Find the one with closest stats
            if not swapped:
                possible['diff'] = abs(possible['shots_h'] - f['home_total_shots']) + \
                                   abs(possible['shots_a'] - f['away_total_shots']) + \
                                   (abs(possible['passes_h'] - f['home_passes']) + \
                                    abs(possible['passes_a'] - f['away_passes'])) / 20
            else:
                possible['diff'] = abs(possible['shots_h'] - f['away_total_shots']) + \
                                   abs(possible['shots_a'] - f['home_total_shots']) + \
                                   (abs(possible['passes_h'] - f['away_passes']) + \
                                    abs(possible['passes_a'] - f['home_passes'])) / 20
            
            best = possible.sort_values('diff').iloc[0]
            mapping.append({
                'game_id': int(best['game_id']),
                'home_team': f['home_team'],
                'away_team': f['away_team'],
                'round': int(f['round']),
                'date': f['date'],
                'swapped': swapped,
                'match_diff': best['diff']
            })
            used_json_gids.add(best['game_id'])
            
    df_mapping = pd.DataFrame(mapping)
    df_mapping.to_csv('data/game_id_to_teams.csv', index=False)
    print(f"Mapped {len(df_mapping)} / {len(df_fotmob)} matches using global optimization.")

if __name__ == "__main__":
    main()
