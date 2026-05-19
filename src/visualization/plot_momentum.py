import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_momentum(momentum_path, game_id, output_dir='reports'):
    df = pd.read_parquet(momentum_path)
    match_df = df[df['game_id'] == game_id].sort_values('time_bin')
    
    if match_df.empty:
        print(f"Jogo {game_id} não encontrado.")
        return

    plt.figure(figsize=(15, 6))
    
    # Plotar momentum suavizado
    sns.lineplot(data=match_df, x='time_bin', y='momentum_smooth', color='blue', label='Momentum (Home - Away)')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Preencher área
    plt.fill_between(match_df['time_bin'], match_df['momentum_smooth'], 0, 
                     where=(match_df['momentum_smooth'] >= 0), color='blue', alpha=0.3)
    plt.fill_between(match_df['time_bin'], match_df['momentum_smooth'], 0, 
                     where=(match_df['momentum_smooth'] < 0), color='red', alpha=0.3)
    
    plt.title(f'Match Momentum (xT) - Game ID: {game_id}')
    plt.xlabel('Minutos')
    plt.ylabel('xT Momentum')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'momentum_{game_id}.png'))
    print(f"Gráfico salvo em {output_dir}/momentum_{game_id}.png")

if __name__ == "__main__":
    # Usar o primeiro game_id disponível
    df = pd.read_parquet('data/match_momentum.parquet')
    first_game = df['game_id'].iloc[0]
    plot_momentum('data/match_momentum.parquet', first_game)
