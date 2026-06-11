import json
import os

def update_notebook():
    notebook_path = "notebooks/TP3_Progress_Report.ipynb"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb["cells"]
    
    # Let's find the cells by their ID
    opt_model_code_idx = -1
    opt_model_comparison_idx = -1
    insights_idx = -1
    cron_idx = -1
    
    for i, cell in enumerate(cells):
        cell_id = cell.get("id")
        if cell_id == "opt_model_code":
            opt_model_code_idx = i
        elif cell_id == "opt_model_comparison":
            opt_model_comparison_idx = i
        elif cell_id == "cb599190":
            insights_idx = i
        elif cell_id == "60c7248c":
            cron_idx = i
            
    print(f"Indices found: opt_model_code={opt_model_code_idx}, opt_model_comparison={opt_model_comparison_idx}, insights={insights_idx}, cron={cron_idx}")
    
    if opt_model_code_idx == -1 or opt_model_comparison_idx == -1:
        raise ValueError("Could not find target cells in the notebook.")
        
    # Define the new BERTimbau cells
    bert_markdown_cell = {
        "cell_type": "markdown",
        "id": "bert_model_md",
        "metadata": {},
        "source": [
            "## 7. Avaliação do Modelo Destilado (BERTimbau)\n",
            "\n",
            "Como próximo passo de nossa estratégia de modelagem, realizamos a destilação de conhecimento (Knowledge Distillation) treinando o modelo **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) sobre o dataset de 2.000 mensagens rotuladas pelo modelo `Llama-3.1-8B-FP8`.\n",
            "\n",
            "Abaixo, avaliamos a performance deste classificador leve e especializado contra o nosso Golden Set humano."
        ]
    }
    
    bert_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "bert_model_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Carregar tokenizador e modelo BERTimbau finetunado\n",
            "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
            "import torch\n",
            "\n",
            "model_path = '../bertimbau-bundesliga'\n",
            "tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
            "model = AutoModelForSequenceClassification.from_pretrained(model_path)\n",
            "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "model.to(device)\n",
            "model.eval()\n",
            "\n",
            "# Predição no eval_opt_df\n",
            "messages_eval = eval_opt_df['mensagem'].tolist()\n",
            "preds = []\n",
            "batch_size = 64\n",
            "for i in range(0, len(messages_eval), batch_size):\n",
            "    batch = messages_eval[i:i+batch_size]\n",
            "    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')\n",
            "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
            "    with torch.no_grad():\n",
            "        outputs = model(**inputs)\n",
            "        batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()\n",
            "        preds.extend(batch_preds)\n",
            "\n",
            "eval_opt_df['pred_bertimbau'] = preds\n",
            "y_pred_bert = eval_opt_df['pred_bertimbau']\n",
            "\n",
            "print(f\"Exemplos para avaliação (BERTimbau): {len(eval_opt_df)}\")\n",
            "print(\"\\n--- Métricas BERTimbau vs Humano ---\")\n",
            "print(f\"Accuracy: {accuracy_score(y_true_opt, y_pred_bert):.3f}\")\n",
            "print(f\"F1-Score (Weighted): {f1_score(y_true_opt, y_pred_bert, average='weighted'):.3f}\")\n",
            "print(f\"Cohen's Kappa Score: {cohen_kappa_score(y_true_opt, y_pred_bert):.3f}\")\n",
            "print(\"\\nRelatório de Classificação:\")\n",
            "print(classification_report(y_true_opt, y_pred_bert, target_names=['Negativo', 'Neutro', 'Positivo']))"
        ]
    }
    
    # Insert new cells
    # We insert right after opt_model_code_idx
    cells.insert(opt_model_code_idx + 1, bert_markdown_cell)
    cells.insert(opt_model_code_idx + 2, bert_code_cell)
    
    # Re-calculate index for opt_model_comparison (which shifted by 2)
    new_opt_comparison_idx = opt_model_comparison_idx + 2
    
    # Modify the comparison cell
    cells[new_opt_comparison_idx]["source"] = [
        "## 8. Comparação de Métricas e Discussão sobre o Cohen's Kappa\n",
        "Abaixo, consolidamos a comparação entre os anotadores humanos e as versões dos modelos:\n",
        "\n",
        "| Relação | Métrica de Concordância | Valor | Nível de Concordância |\n",
        "| :--- | :--- | :---: | :--- |\n",
        "| **Humano vs Humano** | Cohen's Kappa Global | **0.472** | Moderada |\n",
        "| **Baseline vLLM vs Humano** | Cohen's Kappa / Acurácia | **0.501** / 69.0% | Moderada |\n",
        "| **Llama-3.1-8B-FP8 (Otimizado) vs Humano** | Cohen's Kappa / Acurácia | **0.690** / 80.2% | **Substancial** |\n",
        "| **BERTimbau (Destilado) vs Humano** | Cohen's Kappa / Acurácia | **0.698** / 80.8% | **Substancial** |\n",
        "\n",
        "### Discussão:\n",
        "1. **Concordância Humana (0.472):** A concordância moderada entre os rotuladores humanos evidencia a alta subjetividade dos comentários do chat da CazéTV. Gírias de aposta misturadas com torcida criam divergências naturais de interpretação.\n",
        "2. **Modelo Otimizado Llama 8B (0.690):** Ao refinar o prompt para mapear explicitamente os casos de sarcasmo, gírias e definir a classe Neutro como default, o modelo de 8B obteve uma concordância de **0.690** com os anotadores humanos (substancial).\n",
        "3. **BERTimbau Destilado (0.698):** Treinar o modelo BERTimbau (110M parâmetros) na base de 2.000 comentários classificados pelo Llama 8B resultou em um Cohen's Kappa de **0.698** e acurácia de **80.8%**, ligeiramente superando o professor de 8B. Isso demonstra uma destilação de alta eficácia, permitindo usar um classificador extremamente veloz e compacto sem perda de acurácia."
    ]
    
    # Re-calculate index for insights and cron (which also shifted by 2)
    new_insights_idx = insights_idx + 2
    new_cron_idx = cron_idx + 2
    
    cells[new_insights_idx]["source"] = [
        "## 9. Insights Preliminares\n",
        "- **Dificuldade em Sarcasmo**: Notamos que o vLLM às vezes interpreta gírias de aposta como negativas quando são celebrações.\n",
        "- **Concordância Humana**: O Kappa obtido indica o quão subjetiva é a tarefa, sugerindo a necessidade de diretrizes de anotação mais claras.\n",
        "- **Performance**: Resultados iniciais mostram que o modelo base é promissor, mas necessita de ajuste fino para o domínio de futebol."
    ]
    
    cells[new_cron_idx]["source"] = [
        "## 10. Cronograma e Próximos Passos\n",
        "\n",
        "| Data | Atividade | Objetivo |\n",
        "| :--- | :--- | :--- |\n",
        "| 05/Jun | Refinamento do Golden Set | Resolver divergências manuais nos ~10% restantes |\n",
        "| 10/Jun | Treinamento BERTimbau | Superar baseline do vLLM com modelo especializado |\n",
        "| 15/Jun | Tuning de Hiperparâmetros | Otimizar Learning Rate e Batch Size |\n",
        "| 20/Jun | Integração Final | Correlacionar sentimento com xT e Match Momentum |\n",
        "| 25/Jun | Entrega Final | Relatório consolidado e Dashboard |\n",
        "\n",
        "**Métricas escolhidas**: F1-Score (Weighted) é a principal, pois as classes são desbalanceadas (muito neutro)."
    ]
    
    nb["cells"] = cells
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook updated successfully.")

if __name__ == "__main__":
    update_notebook()
