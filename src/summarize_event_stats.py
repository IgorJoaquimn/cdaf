import json
import glob
import os
import pandas as pd

def get_stats(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    gid = data[0]['GAME_ID']
    
    shots_h = 0
    shots_a = 0
    goals_h = 0
    goals_a = 0
    passes_h = 0
    passes_a = 0
    
    for e in data:
        p_type = e['POSSESSION_EVENTS'].get('POSSESSION_EVENT_TYPE')
        outcome = e['POSSESSION_EVENTS'].get('SHOT_OUTCOME_TYPE')
        is_home = e['GAME_EVENTS'].get('HOME_TEAM')
        
        if p_type == 'SH':
            if is_home: 
                shots_h += 1
                if outcome == 'G': goals_h += 1
            else: 
                shots_a += 1
                if outcome == 'G': goals_a += 1
        elif p_type == 'PA':
            if is_home: passes_h += 1
            else: passes_a += 1
            
    return {
        'game_id': gid,
        'goals_h': goals_h,
        'goals_a': goals_a,
        'shots_h': shots_h,
        'shots_a': shots_a,
        'passes_h': passes_h,
        'passes_a': passes_a
    }

def main():
    results = []
    for f in sorted(glob.glob('data/events/2024-2025/*.json')):
        results.append(get_stats(f))
    
    df_events = pd.DataFrame(results)
    df_events.to_csv('data/event_stats_summary.csv', index=False)
    print("Event stats summary saved to data/event_stats_summary.csv")

if __name__ == "__main__":
    main()
