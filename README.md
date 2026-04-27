# CDAF - Análise de Chat Live (YouTube)

Este projeto automatiza a coleta e análise de mensagens de chat ao vivo do YouTube, com foco em partidas de futebol. Ele extrai as mensagens, categoriza-as por períodos do jogo (1º Tempo, Intervalo, 2º Tempo) e gera visualizações de volume e engajamento.

## 🚀 Funcionalidades

- **Coleta Automatizada**: Utiliza `yt-dlp` para baixar o histórico de chat.
- **Pipeline Unificado**: Processa dados brutos em um CSV estruturado com timestamps de vídeo e de partida (0-90'+).
- **Detecção de Picos (Hype)**: Identifica aumentos súbitos de volume que indicam momentos importantes (gols, faltas, etc).
- **Análise Visual**: Notebook Jupyter com gráficos de volume bruto e tendências suavizadas (Média Móvel).

## 📁 Estrutura do Projeto

```text
├── config/
│   ├── settings.json      # Configurações (ID do vídeo, tempo de início)
│   └── video_info.json    # Títulos dos vídeos extraídos
├── data/
│   ├── raw/               # JSONs brutos do YouTube
│   ├── processed/         # CSVs processados com tempos oficiais
│   └── frames/            # Frames extraídos para verificação/OCR
├── src/
│   ├── pipeline.py        # Coleta e processamento de chat
│   ├── extract_frame.py   # Extração de frames no início do jogo
│   └── extract_time.py    # OCR para ler o placar (Vision)
├── analysis.ipynb         # Notebook para visualização de dados
└── pyproject.toml         # Dependências (uv)
```

## 🛠️ Instalação e Uso

...

### 🔍 Vision (Opcional)

Para extrair e validar o tempo do jogo diretamente do placar na tela:

1. **Extrair o frame**:
   ```bash
   uv run src/extract_frame.py
   ```

2. **Ler o tempo via OCR**:
   ```bash
   uv run src/extract_time.py data/frames/NOME_DO_ARQUIVO.png
   ```


## 📊 Gráficos Gerados

1. **Volume Bruto**: Mostra picos instantâneos minuto a minuto.
2. **Tendência de Engajamento**: Utiliza média móvel para destacar o "clima" geral do chat e identificar inícios de "hype".

---
*Desenvolvido para análise de engajamento digital em transmissões esportivas.*
