# CDAF - Análise de Chat Live (YouTube - Bundesliga)

Este projeto automatiza o ciclo completo de coleta, sincronização, análise de sentimento (via destilação de conhecimento de LLM para BERT) e correlação com eventos de campo em transmissões de futebol da CazéTV.

## 🚀 Funcionalidades

- **Coleta de Chat Sincronizada**: Extração de mensagens com timestamps e alinhamento temporal com o cronômetro do jogo.
- **Refinamento via Vision (OCR)**: Sincronização automática com o cronômetro da transmissão.
- **Análise de Sentimento (LLM → BERT)**: Classificação multi-classe (Positivo, Negativo, Neutro) utilizando a técnica de *Knowledge Distillation*:
  - **Teacher Model**: Gemma 4 (E2B) ou Llama 3.1 com prompt estruturado de Chain-of-Thought (CoT) calibrado no futebol brasileiro.
  - **Student Model**: BERTimbau-base treinado de forma supervisionada sobre o dataset gerado pelo professor.
- **Cruzamento Contínuo**: Alinhamento minuto a minuto entre sentimento do chat (Polaridade/WSI) e métricas táticas de campo (xT e Momentum) e eventos discretos (gols, cartões, subs).

---

## 📁 Estrutura do Projeto

```text
├── config/
│   ├── settings.json                # Configuração dos vídeos e tempos
│   └── video_info.json              # Títulos oficiais extraídos
├── data/
│   ├── processed/                   # CSVs de chat individuais por partida
│   │   └── consolidated/            # Arquivos Parquet e bases de modelagem
│   │       ├── golden_set_consensus.csv # Golden set humano (652 comments)
│   │       └── synthetic_dataset_for_bert.csv # Base de treino (2k comments)
│   └── events/                      # Eventos estruturados da Bundesliga
├── notebooks/
│   ├── analysis.ipynb               # Análise exploratória inicial (EDA)
│   ├── TP3_Progress_Report.ipynb    # Notebook de avaliação e comparação de métricas
│   └── sentiment_eda.ipynb          # Análise exploratória do sentimento do chat
├── reports/
│   ├── cdaf_delivery_package.zip    # Pacote ZIP com os dados não rastreados pelo Git
│   ├── confusion_matrix_bertimbau.png # Matriz de confusão do BERTimbau
│   ├── confusion_matrix_llama.png   # Matriz de confusão do Llama 3.1
│   └── sentiment_distribution.png   # Distribuição de classes nos datasets
├── src/
│   ├── data_collection/             # Scripts de coleta de metadados/vídeos
│   ├── metrics/
│   │   └── calculate_xt_momentum.py # Cálculo de xT (Expected Threat) e Momentum
│   ├── sentiment/
│   │   ├── system_prompt.txt        # Prompt CoT do modelo professor
│   │   ├── eval_prompt.py           # Avaliação de LLMs no Golden Set
│   │   ├── generate_gold_standard_vllm.py # Geração da base sintética de 2k
│   │   ├── train_sentiment.py       # Fine-tuning do BERTimbau (Estudante)
│   │   └── classify_sentiment.py    # Classificação em larga escala do chat inteiro
│   └── pipeline.py                  # Pipeline principal de parsing do chat
└── README.md
```

---

## 📦 Entrega e Pacote de Dados (`reports/cdaf_delivery_package.zip`)

Como os arquivos de dados volumosos estão listados no `.gitignore`, os conjuntos de dados necessários para análise das RQs e replicação dos modelos foram compactados no arquivo `reports/cdaf_delivery_package.zip`.

### Como extrair:
Após dar `git pull`, extraia os arquivos do pacote ZIP diretamente para a raiz do repositório:
```bash
unzip reports/cdaf_delivery_package.zip -d .
```
Isso criará a pasta `data/` com os seguintes arquivos:
1.  **`data/match_minute_metrics.csv`**: Tabela alinhada minuto a minuto (0 a 90) para as 63 partidas mapeadas contendo:
    - Volume de mensagens do chat (positivo, negativo, neutro, total).
    - Polaridade e WSI (Engagement-Weighted Sentiment Index).
    - xT das equipes de casa/fora e Momentum de jogo.
    - Ocorrência de eventos discretos (gols, cartões amarelos/vermelhos e substituições).
2.  **`data/chat_comments_with_sentiment.csv`**: O conjunto completo de **192.979 comentários** classificados automaticamente pelo modelo BERTimbau destilado.
3.  **`data/golden_set_consensus.csv`**: As 652 mensagens com 100% de consenso humano, utilizadas como validação.
4.  **`data/synthetic_dataset_for_bert.csv`**: As 2.000 mensagens rotuladas pelo LLM utilizadas no treino do BERTimbau.

---

## 🛠️ Como Executar a Modelagem de Sentimento

### 1. Avaliar um Modelo / Prompt no Golden Set:
Para testar a acurácia de classificação de um LLM no conjunto de validação humano:
```bash
uv run python src/sentiment/eval_prompt.py
```

### 2. Gerar Base de Treino Sintética:
Para rotular um lote de comentários do chat usando o LLM (via vLLM) e gerar novos exemplos de treinamento:
```bash
uv run python src/sentiment/generate_gold_standard_vllm.py
```

### 3. Treinar o BERTimbau (Student):
Para rodar o ajuste fino supervisionado do BERTimbau sobre as mensagens geradas pelo professor e validá-lo no Golden Set humano:
```bash
uv run python src/sentiment/train_sentiment.py
```

### 4. Classificar o Chat Completo (Modo Turbo):
Para classificar o sentimento do dataset completo de 246k comentários utilizando o modelo treinado:
```bash
uv run python src/sentiment/classify_sentiment.py
```

---
*Projeto acadêmico de Ciência de Dados Aplicada ao Futebol (UFMG).*
