# CDAF - Análise de Chat Live (YouTube - Bundesliga)

Este projeto automatiza o ciclo completo de coleta, sincronização e análise de mensagens de chat ao vivo do YouTube, especificamente para transmissões de futebol (CazéTV).

## 🚀 Funcionalidades

- **Coleta de Chat Sincronizada**: Extração de mensagens com timestamps precisos.
- **Refinamento via Vision (OCR)**: Sincronização automática com o cronômetro da TV.
- **NLP Preprocessing**: Limpeza paralela de ruído, gírias e acrônimos esportivos.
- **Seasonal EDA**: Análise estatística de engajamento de toda a temporada.

## 📁 Estrutura do Projeto

```text
├── config/
│   ├── settings.json      # Configuração dos vídeos
│   └── video_info.json    # Títulos oficiais extraídos
├── data/
│   ├── processed/         # CSVs individuais por partida
│   │   └── consolidated/  # Arquivos Parquet consolidados para EDA
│   └── verification/      # Frames/ROIs para validar OCR
├── notebooks/
│   ├── analysis.ipynb     # Exploratory Data Analysis (EDA)
│   └── preprocessing.ipynb # Pipeline de limpeza NLP
├── reports/
│   ├── analysis.html      # Relatório estatístico gerado
│   └── preprocessing.html # Relatório de limpeza gerado
├── src/
│   ├── pipeline.py        # Coleta e processamento principal
│   └── refine_start_times.py # Sincronização via Vision
└── README.md
```

## 🛠️ Como Usar

1.  **Refinar Tempos**: `uv run src/refine_start_times.py`
2.  **Coletar Dados**: `uv run src/pipeline.py`
3.  **Executar Análises**:
    ```bash
    uv run jupyter nbconvert --to html --execute notebooks/analysis.ipynb --output-dir reports/
    uv run jupyter nbconvert --to html --execute notebooks/preprocessing.ipynb --output-dir reports/
    ```

---
*Projeto organizado para escalabilidade e clareza analítica.*
