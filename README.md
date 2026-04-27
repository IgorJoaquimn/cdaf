# CDAF - Análise de Chat Live (YouTube - Bundesliga)

Este projeto automatiza o ciclo completo de coleta, sincronização e análise de mensagens de chat ao vivo do YouTube, especificamente para transmissões de futebol (CazéTV). 

O diferencial técnico deste projeto é a **sincronização matemática** entre o tempo real do vídeo e o cronômetro oficial da partida (0-90'+) através de Computer Vision.

## 🚀 Funcionalidades

- **Coleta de Chat Sincronizada**: Extração de centenas de milhares de mensagens com timestamps precisos.
- **Refinamento via Vision (OCR)**: Algoritmo que "assiste" ao vídeo, localiza o placar e sincroniza o chat com o cronômetro de TV.
- **Pipeline Multi-Vídeo**: Suporte para processamento em massa de playlists inteiras.
- **Análise Exploratória (EDA)**: Dashboard estatístico com heatmaps de engajamento, detecção de picos de "hype" e comportamento de usuários.

## 📁 Estrutura do Projeto

```text
├── config/
│   ├── settings.json      # Configuração dos vídeos (IDs e metadados)
│   └── video_info.json    # Títulos oficiais extraídos
├── src/
│   ├── pipeline.py        # Coleta e processamento de mensagens
│   └── refine_start_times.py # Refinamento via OCR (Vision)
├── analysis.ipynb         # Notebook de análise estatística
├── pyproject.toml         # Gerenciamento de dependências (uv)
└── README.md              # Documentação
```

## 🛠️ Desafios e Soluções (Log de Desenvolvimento)

Durante o desenvolvimento, superamos os seguintes obstáculos técnicos:

1.  **Mudanças na Estrutura do YouTube**: A biblioteca inicial `chat-downloader` falhou. Resolvemos migrando para o `yt-dlp` com extração de metadados JSON, o que se mostrou mais robusto.
2.  **Rate Limiting (HTTP 429)**: Ao processar 69 vídeos em paralelo, o YouTube bloqueou o IP. Implementamos suporte a **Cookies de Sessão** e **Delays Aleatórios** para bypassar a detecção de bots.
3.  **Scoreboards Complexos**: O OCR falhou inicialmente em layouts de "Multi-Live" (telas reduzidas). Criamos uma **Estratégia de Multi-ROI** com melhoria de contraste (CLAHE) e upscaling, atingindo 100% de precisão na detecção do tempo.
4.  **Acréscimos e Intervalos**: Gols e picos de chat ocorrem frequentemente nos acréscimos (ex: 45+2'). Criamos uma lógica de **Broadcasting Time** que converte segundos lineares em labels oficiais de futebol.

## 📊 Como Usar

Este projeto utiliza o [uv](https://github.com/astral-sh/uv).

1.  **Instalação**: `uv sync`
2.  **Configuração**: Adicione os IDs dos vídeos em `config/settings.json`.
3.  **Bypass de Bot (Opcional)**: Salve seus cookies do YouTube em `cookies.txt`.
4.  **Refinamento de Tempo**:
    ```bash
    uv run src/refine_start_times.py
    ```
5.  **Coleta e Processamento**:
    ```bash
    uv run src/pipeline.py
    ```
6.  **Gerar Análise**:
    ```bash
    uv run jupyter nbconvert --to html --execute analysis.ipynb
    ```

---
*Projeto desenvolvido para fins de pesquisa em engajamento digital e análise de dados esportivos.*
