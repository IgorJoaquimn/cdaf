# CDAF - Análise de Chat Live (YouTube)

Este projeto automatiza a coleta e análise de mensagens de chat ao vivo do YouTube, com foco em partidas de futebol. Ele extrai as mensagens, categoriza-as por períodos do jogo e gera visualizações de engajamento baseadas em picos e tendências.

## 🚀 Funcionalidades

- **Coleta Automatizada**: Utiliza `yt-dlp` para baixar o histórico de chat e metadados.
- **Refinamento de Tempo (Vision)**: Script avançado que usa OCR e processamento de imagem para detectar o início exato da partida pelo placar da transmissão.
- **Pipeline Unificado**: Processa dados brutos em um CSV estruturado com cronômetro oficial de TV (0-90'+).
- **Análise Visual**: Notebook Jupyter com gráficos de volume bruto e tendências suavizadas (Média Móvel).

## 📁 Estrutura do Projeto

```text
├── config/
│   ├── settings.json      # Configuração principal (Vídeos e Timestamps)
│   └── video_info.json    # Títulos dos vídeos extraídos
├── data/
│   ├── raw/               # JSONs brutos do YouTube
│   ├── processed/         # CSVs processados com tempos oficiais
│   └── verification/      # Frames e ROIs usados para validar o OCR
├── src/
│   ├── pipeline.py        # Coleta e processamento de chat
│   └── refine_start_times.py # Refinamento automático do início via Vision
├── analysis.ipynb         # Notebook para visualização de dados
└── pyproject.toml         # Dependências (uv)
```

## 🛠️ Instalação e Uso

1. **Configurar os vídeos**:
   Edite `config/settings.json` com os IDs dos vídeos e estimativas iniciais.

2. **Refinar o início (Opcional, mas recomendado)**:
   ```bash
   uv run src/refine_start_times.py
   ```
   *Usa OCR para ler o placar e ajustar o game_start_time automaticamente.*

3. **Executar o Pipeline**:
   ```bash
   uv run src/pipeline.py
   ```

4. **Gerar a Análise**:
   ```bash
   uv run jupyter nbconvert --to html --execute analysis.ipynb
   ```

---
*Desenvolvido para análise de engajamento digital em transmissões esportivas.*
