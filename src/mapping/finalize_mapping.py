import json
import pandas as pd
import re

def normalize(name):
    if not name: return ""
    name = str(name).lower()
    # Remover acentos e caracteres especiais
    import unidecode
    name = unidecode.unidecode(name)
    name = re.sub(r'[^a-z0-9]', '', name)
    
    replacements = {
        'eintrachtfrankfurt': 'frankfurt',
        'frankfurt': 'frankfurt',
        'bayernmnchen': 'bayern',
        'bayernmunchen': 'bayern',
        'bayernmunich': 'bayern',
        'bayerndemunique': 'bayern',
        'bayerleverkusen': 'leverkusen',
        'borussiadortmund': 'dortmund',
        'rbleipzig': 'leipzig',
        'werderbremen': 'werder',
        'vfbstuttgart': 'stuttgart',
        'borussiamnchengladbach': 'gladbach',
        'monchengladbach': 'gladbach',
        'fcheidenheim': 'heidenheim',
        'fcsanktpauli': 'stpauli',
        'stpauli': 'stpauli',
        'holsteinkiel': 'kiel',
        'mainz05': 'mainz',
        'unionberlin': 'union',
        'fortunadsseldorf': 'dusseldorf',
        'vflbochum': 'bochum',
        'vflwolfsburg': 'wolfsburg',
        'tsghoffenheim': 'hoffenheim',
        'scfreiburg': 'freiburg',
        'fcaugsburg': 'augsburg',
        'svwerderbremen': 'werder',
        'rb leipzig': 'leipzig',
        'borussia mgladbach': 'gladbach'
    }
    for k, v in replacements.items():
        if k in name: return v
    return name

def main():
    # Carregar dados
    df_games = pd.read_csv('data/game_id_to_teams.csv')
    with open('config/video_info.json', 'r') as f:
        video_info = json.load(f)
    
    mapping = []
    video_mapped_count = 0
    
    for v in video_info:
        title = v['title'].upper()
        # Extrair times do título (ex: TIME A X TIME B)
        # Tenta vários padrões
        match_teams = re.search(r'JOGO COMPLETO: (.*?) X (.*?) (?:\||RODADA|VALE|$)', title)
            
        if not match_teams:
            print(f"Não foi possível extrair times de: {v['title']}")
            continue
            
        v_home_norm = normalize(match_teams.group(1).strip())
        v_away_norm = normalize(match_teams.group(2).strip())
        
        if "FRANKFURT" in title and "BAYERN" in title:
            print(f"DEBUG Frankfurt/Bayern: title_h={v_home_norm}, title_a={v_away_norm}")
            print(f"DEBUG df_games sample:\n{df_games[(df_games['home_team'].str.contains('Frankfurt')) & (df_games['away_team'].str.contains('Bayern'))][['home_team', 'away_team', 'h_norm', 'a_norm']]}")

        # Extrair rodada
        v_round = None
        match_round = re.search(r'RODADA (\d+)', title)
        if match_round:
            v_round = int(match_round.group(1))
        
        # Tentar encontrar no dataframe
        # Ordem importa: v_home em home_team E v_away em away_team
        df_games['h_norm'] = df_games['home_team'].apply(normalize)
        df_games['a_norm'] = df_games['away_team'].apply(normalize)
        
        mask = (df_games['h_norm'] == v_home_norm) & (df_games['a_norm'] == v_away_norm)
        
        possible = df_games[mask].copy()
        
        if len(possible) == 0:
            # Tenta ordem inversa just in case
            mask_inv = (df_games['h_norm'] == v_away_norm) & (df_games['a_norm'] == v_home_norm)
            possible = df_games[mask_inv].copy()
            
        if v_round and len(possible) > 1:
            possible = possible[possible['round'] == v_round]
            
        if len(possible) == 1:
            res = possible.iloc[0]
            mapping.append({
                'game_id': int(res['game_id']),
                'video_id': v['video_id'],
                'video_title': v['title'],
                'teams': f"{res['home_team']} x {res['away_team']}",
                'round': int(res['round']),
                'date': res['date']
            })
            video_mapped_count += 1
        elif len(possible) > 1:
             # Se ainda ambíguo, pega o com round mais próximo ou o primeiro
             res = possible.iloc[0]
             mapping.append({
                'game_id': int(res['game_id']),
                'video_id': v['video_id'],
                'video_title': v['title'],
                'teams': f"{res['home_team']} x {res['away_team']}",
                'round': int(res['round']),
                'date': res['date']
             })
             video_mapped_count += 1
        else:
             print(f"Nenhuma partida encontrada para {v['title']} ({v_home_norm} x {v_away_norm})")

    # Salvar resultado
    with open('data/game_video_mapping_final.json', 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print(f"\nFinalizado: Mapeados {video_mapped_count} vídeos de {len(video_info)}.")

if __name__ == "__main__":
    main()
