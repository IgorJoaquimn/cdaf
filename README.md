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
│   └── settings.json      # Configurações (ID do vídeo, tempo de início)
├── data/
│   ├── raw/               # JSONs brutos do YouTube (ignorado no git)
│   └── processed/         # CSVs processados com tempos oficiais
├── src/
│   └── pipeline.py        # Script principal de coleta e processamento
├── analysis.ipynb         # Notebook para visualização de dados
└── pyproject.toml         # Dependências do projeto (gerenciado pelo uv)
```

## 🛠️ Instalação e Uso

Este projeto utiliza o [uv](https://github.com/astral-sh/uv) para gerenciamento de pacotes.

1. **Configurar o vídeo**:
   Edite o arquivo `config/settings.json` com o ID do vídeo e os tempos de início/fim dos tempos.

2. **Executar o Pipeline**:
   ```bash
   uv run src/pipeline.py
   ```

3. **Gerar a Análise**:
   ```bash
   uv run jupyter nbconvert --to html --execute analysis.ipynb
   ```
   Isso gerará um arquivo `analysis.html` com todos os gráficos.

## 📊 Gráficos Gerados

1. **Volume Bruto**: Mostra picos instantâneos minuto a minuto.
2. **Tendência de Engajamento**: Utiliza média móvel para destacar o "clima" geral do chat e identificar inícios de "hype".

---
*Desenvolvido para análise de engajamento digital em transmissões esportivas.*
