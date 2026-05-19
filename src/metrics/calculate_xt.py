import pandas as pd
import numpy as np
import json
import os

def calculate_xt_grid(df, grid_size=(16, 12), n_iterations=10):
    # Filtrar apenas eventos válidos para o grid
    df = df.copy()
    
    # Normalização de Coordenadas
    df['nx_start'] = df.apply(lambda r: -r['start_x'] if r['attacking_direction'] == 'L' else r['start_x'], axis=1)
    df['ny_start'] = df.apply(lambda r: -r['start_y'] if r['attacking_direction'] == 'L' else r['start_y'], axis=1)
    df['nx_end'] = df.apply(lambda r: -r['end_x'] if r['attacking_direction'] == 'L' and r['end_x'] is not None else r['end_x'], axis=1)
    df['ny_end'] = df.apply(lambda r: -r['end_y'] if r['attacking_direction'] == 'L' and r['end_y'] is not None else r['end_y'], axis=1)
    
    df['nx_start'] = (df['nx_start'] + df['pitch_length'] / 2).clip(0, df['pitch_length'])
    df['ny_start'] = (df['ny_start'] + df['pitch_width'] / 2).clip(0, df['pitch_width'])
    
    # Mapear para indices do grid
    df['x_idx'] = (df['nx_start'] / df['pitch_length'] * (grid_size[0] - 1)).astype(int)
    df['y_idx'] = (df['ny_start'] / df['pitch_width'] * (grid_size[1] - 1)).astype(int)
    
    # Mesma coisa para o fim
    df_moves = df[df['nx_end'].notnull()].copy()
    df_moves['nx_end'] = (df_moves['nx_end'] + df_moves['pitch_length'] / 2).clip(0, df_moves['pitch_length'])
    df_moves['ny_end'] = (df_moves['ny_end'] + df_moves['pitch_width'] / 2).clip(0, df_moves['pitch_width'])
    df_moves['x_idx_end'] = (df_moves['nx_end'] / df_moves['pitch_length'] * (grid_size[0] - 1)).astype(int)
    df_moves['y_idx_end'] = (df_moves['ny_end'] / df_moves['pitch_width'] * (grid_size[1] - 1)).astype(int)
    
    # 1. Probabilidade de Chute vs Movimento
    # Contar chutes e movimentos por célula
    shot_counts = df[df['type'] == 'SH'].groupby(['x_idx', 'y_idx']).size().unstack(fill_value=0).reindex(index=range(grid_size[0]), columns=range(grid_size[1]), fill_value=0).values
    move_counts = df[df['type'].isin(['PA', 'BC', 'CR', 'IT'])].groupby(['x_idx', 'y_idx']).size().unstack(fill_value=0).reindex(index=range(grid_size[0]), columns=range(grid_size[1]), fill_value=0).values
    
    total_actions = shot_counts + move_counts
    # Evitar divisão por zero
    total_actions_safe = np.where(total_actions == 0, 1, total_actions)
    
    p_shot = shot_counts / total_actions_safe
    p_move = move_counts / total_actions_safe
    
    # 2. Probabilidade de Gol dado o Chute
    goal_counts = df[(df['type'] == 'SH') & (df['shot_outcome'] == 'G')].groupby(['x_idx', 'y_idx']).size().unstack(fill_value=0).reindex(index=range(grid_size[0]), columns=range(grid_size[1]), fill_value=0).values
    # Suavização simples para células sem chutes
    p_goal_if_shot = goal_counts / np.where(shot_counts == 0, 1, shot_counts)
    
    # 3. Matriz de Transição de Movimentos (T)
    # T[start_idx, end_idx]
    # Simplificando: vamos achatar o grid para grid_size[0]*grid_size[1]
    n_cells = grid_size[0] * grid_size[1]
    transition_matrix = np.zeros((n_cells, n_cells))
    
    for _, row in df_moves[df_moves['type'].isin(['PA', 'BC', 'CR', 'IT'])].iterrows():
        start_cell = int(row['x_idx'] * grid_size[1] + row['y_idx'])
        end_cell = int(row['x_idx_end'] * grid_size[1] + row['y_idx_end'])
        transition_matrix[start_cell, end_cell] += 1
        
    # Normalizar matriz de transição
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / np.where(row_sums[:, None] == 0, 1, row_sums[:, None])
    
    # 4. Resolver Iterativamente
    # xt = (p_shot * p_goal_if_shot) + (p_move * T * xt)
    xt = np.zeros(n_cells)
    s_grid = (p_shot * p_goal_if_shot).flatten()
    m_grid = p_move.flatten()
    
    for _ in range(n_iterations):
        xt = s_grid + m_grid * np.dot(transition_matrix, xt)
        
    return xt.reshape(grid_size)

def apply_xt(df, xt_grid, grid_size=(16, 12)):
    # Normalizar e mapear
    df['nx_start'] = df.apply(lambda r: -r['start_x'] if r['attacking_direction'] == 'L' else r['start_x'], axis=1)
    df['ny_start'] = df.apply(lambda r: -r['start_y'] if r['attacking_direction'] == 'L' else r['start_y'], axis=1)
    df['nx_end'] = df.apply(lambda r: -r['end_x'] if r['attacking_direction'] == 'L' and r['end_x'] is not None else r['end_x'], axis=1)
    df['ny_end'] = df.apply(lambda r: -r['end_y'] if r['attacking_direction'] == 'L' and r['end_y'] is not None else r['end_y'], axis=1)
    
    df['nx_start'] = (df['nx_start'] + df['pitch_length'] / 2).clip(0, df['pitch_length'])
    df['ny_start'] = (df['ny_start'] + df['pitch_width'] / 2).clip(0, df['pitch_width'])
    
    df['x_idx'] = (df['nx_start'] / df['pitch_length'] * (grid_size[0] - 1)).astype(int)
    df['y_idx'] = (df['ny_start'] / df['pitch_width'] * (grid_size[1] - 1)).astype(int)
    
    df['xt_start'] = df.apply(lambda r: xt_grid[r['x_idx'], r['y_idx']], axis=1)
    
    # Para movimentos bem-sucedidos (Passes completos)
    df['xt_value'] = 0.0
    
    # Garantir que temos end_x e end_y
    mask_has_end = df['nx_end'].notnull()
    mask_pass = (df['type'] == 'PA') & (df['outcome'] == 'C')
    mask_other_move = df['type'].isin(['BC', 'CR'])
    
    df_moves = df[mask_has_end & (mask_pass | mask_other_move)].copy()
    
    if not df_moves.empty:
        df_moves['nx_end'] = (df_moves['nx_end'] + df_moves['pitch_length'] / 2).clip(0, df_moves['pitch_length'])
        df_moves['ny_end'] = (df_moves['ny_end'] + df_moves['pitch_width'] / 2).clip(0, df_moves['pitch_width'])
        df_moves['x_idx_end'] = (df_moves['nx_end'] / df_moves['pitch_length'] * (grid_size[0] - 1)).astype(int)
        df_moves['y_idx_end'] = (df_moves['ny_end'] / df_moves['pitch_width'] * (grid_size[1] - 1)).astype(int)
        
        df_moves['xt_end'] = df_moves.apply(lambda r: xt_grid[r['x_idx_end'], r['y_idx_end']], axis=1)
        df_moves['xt_value'] = df_moves['xt_end'] - df_moves['xt_start']
        
        # Atualizar o dataframe original
        df.loc[df_moves.index, 'xt_value'] = df_moves['xt_value']
    
    return df

def calculate_momentum(df, window_size_sec=300):
    # Momentum = Sum(xT_home) - Sum(xT_away) em uma janela deslizante
    # Ou simplesmente xT_team_i em relação ao tempo
    
    match_momentums = []
    
    for game_id, group in df.groupby('game_id'):
        group = group.sort_values('timestamp')
        
        # Criar bins de tempo (ex: a cada 1 minuto)
        group['time_bin'] = (group['timestamp'] // 60).astype(int)
        
        momentum = group.groupby(['time_bin', 'home_team'])['xt_value'].sum().unstack(fill_value=0)
        
        # Se faltar algum time no unstack
        if True not in momentum.columns: momentum[True] = 0.0
        if False not in momentum.columns: momentum[False] = 0.0
        
        momentum['momentum'] = momentum[True] - momentum[False]
        # Suavizar com janela deslizante
        momentum['momentum_smooth'] = momentum['momentum'].rolling(window=5, min_periods=1, center=True).mean()
        
        momentum['game_id'] = game_id
        match_momentums.append(momentum.reset_index())
        
    return pd.concat(match_momentums)

def main():
    df = pd.read_parquet('data/events.parquet')
    grid_size = (16, 12)
    
    print("Calculando Grid de xT...")
    xt_grid = calculate_xt_grid(df, grid_size=grid_size)
    np.save('data/xt_grid.npy', xt_grid)
    
    print("Aplicando xT aos eventos...")
    df_with_xt = apply_xt(df, xt_grid, grid_size=grid_size)
    df_with_xt.to_parquet('data/events_with_xt.parquet', index=False)
    
    print("Calculando Momentum...")
    df_momentum = calculate_momentum(df_with_xt)
    df_momentum.to_parquet('data/match_momentum.parquet', index=False)
    
    print(f"Salvo momentum em data/match_momentum.parquet")
    
    # Exemplo de Top Players por xT total
    top_players = df_with_xt.groupby('player_id')['xt_value'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 Jogadores por xT total:")
    print(top_players)

if __name__ == "__main__":
    main()
