import pandas as pd
import json

def main():
    df_events = pd.read_csv('data/event_stats_summary.csv')
    df_fotmob = pd.read_csv('data/processed/consolidated/fotmob_match_stats.csv')
    
    # Clean FotMob team names for comparison
    def clean(s): return str(s).lower().replace(' ', '')
    
    mapping = []
    
    for _, ev in df_events.iterrows():
        # Match by Score AND (Shots or Passes with some tolerance)
        mask = (df_fotmob['home_score'] == ev['goals_h']) & \
               (df_fotmob['away_score'] == ev['goals_a']) & \
               (abs(df_fotmob['home_total_shots'] - ev['shots_h']) <= 2) & \
               (abs(df_fotmob['away_total_shots'] - ev['shots_a']) <= 2)
        
        matches = df_fotmob[mask]
        
        if len(matches) == 1:
            m = matches.iloc[0]
            mapping.append({
                'game_id': int(ev['game_id']),
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'round': int(m['round']),
                'date': m['date']
            })
        elif len(matches) > 1:
            # Try to disambiguate by passes
            matches['pass_diff'] = abs(matches['home_passes'] - ev['passes_h']) + abs(matches['away_passes'] - ev['passes_a'])
            m = matches.sort_values('pass_diff').iloc[0]
            mapping.append({
                'game_id': int(ev['game_id']),
                'home_team': m['home_team'],
                'away_team': m['away_team'],
                'round': int(m['round']),
                'date': m['date']
            })
            
    df_mapping = pd.DataFrame(mapping)
    df_mapping.to_csv('data/game_id_to_teams.csv', index=False)
    print(f"Mapped {len(df_mapping)} / {len(df_events)} games to FotMob matches.")

if __name__ == "__main__":
    main()
