import json
import requests
import pandas as pd
import re
import time
from tqdm import tqdm
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def normalize_name(name):
    if not name: return ""
    name = name.lower()
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

def search_sofascore(query):
    try:
        # Replace non-alphanumeric with space for better searching
        query_encoded = requests.utils.quote(query)
        url = f"https://api.sofascore.com/api/v1/search/all?q={query_encoded}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            for res in results:
                if res.get('type') == 'event':
                    return res.get('entity', {}).get('id')
    except Exception as e:
        pass
    return None

def fetch_momentum(match_id):
    try:
        url = f"https://api.sofascore.com/api/v1/event/{match_id}/graph"
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json().get('graphPoints', [])
    except Exception as e:
        pass
    return []

def main():
    # 1. Load Data
    with open('config/video_info.json', 'r') as f:
        video_info = json.load(f)
    df_fotmob = pd.read_csv('data/processed/consolidated/fotmob_matches.csv')
    
    all_momentum_data = []
    
    print(f"Processing {len(video_info)} videos...")
    
    for v in tqdm(video_info):
        video_id = v['video_id']
        title = v['title'].lower()
        
        # 2. Map Video to FotMob Match
        match_row = None
        for _, m in df_fotmob.iterrows():
            h_norm = normalize_name(m['home_team'])
            a_norm = normalize_name(m['away_team'])
            
            if h_norm in title and a_norm in title:
                match_row = m
                break
        
        if match_row is not None:
            # 3. Search SofaScore using FotMob Team Names
            query = f"{match_row['home_team']} {match_row['away_team']}"
            match_id = search_sofascore(query)
            
            if match_id:
                points = fetch_momentum(match_id)
                for p in points:
                    all_momentum_data.append({
                        'video_id': video_id,
                        'minute': p['minute'],
                        'momentum': p['value']
                    })
                time.sleep(0.2)
            else:
                print(f"  !! SofaScore ID not found for: {query}")
        else:
            print(f"  !! FotMob match not found for title: {v['title']}")

    if all_momentum_data:
        df = pd.DataFrame(all_momentum_data)
        df.to_parquet('data/sofascore_momentum.parquet', index=False)
        print(f"\nSaved {len(df)} momentum points to data/sofascore_momentum.parquet")
    else:
        print("No momentum data fetched.")

if __name__ == "__main__":
    main()
