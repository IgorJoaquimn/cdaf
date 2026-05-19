import pandas as pd
import json

def filter_events():
    # Carregar mapeamento para pegar os game_ids relevantes
    with open('data/game_video_mapping_final.json', 'r') as f:
        mapping = json.load(f)
    
    video_game_ids = {item['game_id'] for item in mapping}
    print(f"Filtrando para {len(video_game_ids)} partidas com vídeo.")
    
    # Carregar o parquet completo
    df = pd.read_parquet('data/events.parquet')
    original_len = len(df)
    
    # Filtrar
    df_filtered = df[df['game_id'].isin(video_game_ids)]
    new_len = len(df_filtered)
    
    # Salvar de volta
    df_filtered.to_parquet('data/events.parquet', index=False)
    print(f"Eventos reduzidos de {original_len} para {new_len}.")
    print(f"Partidas únicas restantes: {df_filtered['game_id'].nunique()}")

if __name__ == "__main__":
    filter_events()
