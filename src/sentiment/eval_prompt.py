import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
from vllm import LLM, SamplingParams
import json
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score

def run_eval(system_prompt, model_name="google/gemma-4-E2B-it"):
    # 1. Load Golden Set
    df = pd.read_csv('data/processed/consolidated/golden_set_consensus.csv')
    messages = df['mensagem'].tolist()
    y_true = df['label'].tolist()

    # 2. Init LLM
    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name, 
        trust_remote_code=True, 
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=2048,
    )

    tokenizer = llm.get_tokenizer()

    def build_prompt(msg):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analise o comentário: \"{msg}\""}
        ]
        # Disable thinking natively in Gemma 4's template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return prompt

    prompts = [build_prompt(m) for m in messages]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=250)
    
    print(f"Generating labels for {len(prompts)} examples with {model_name}...")
    outputs = llm.generate(prompts, sampling_params)

    y_pred = []
    reasonings = []
    for output in outputs:
        text = output.outputs[0].text
        
        # Extract thoughts and json string
        thought = ""
        json_str = ""
        if "```json" in text:
            parts = text.split("```json")
            thought = parts[0].replace("thought", "").strip()
            json_str = parts[1].split("```")[0].strip()
        elif "{" in text:
            parts = text.split("{", 1)
            thought = parts[0].replace("thought", "").strip()
            json_str = "{" + parts[1].rsplit("}", 1)[0] + "}"
        else:
            thought = text
            json_str = ""

        try:
            data = json.loads(json_str)
            y_pred.append(int(data.get('sentimento', 1)))
            raciocinio_final = (thought + "\n---\n" + data.get('raciocinio', '')).strip()
            reasonings.append(raciocinio_final)
        except Exception as e:
            # Fallback parsing
            match = re.search(r'"sentimento":\s*(\d)', text)
            if match:
                y_pred.append(int(match.group(1)))
                reasonings.append(f"Fallback parse from: {text}")
            else:
                y_pred.append(1)
                reasonings.append(f"Error parsing: {text} | {str(e)}")

    # 3. Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n=== EVALUATION RESULTS ({model_name}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Weighted): {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=['Negativo', 'Neutro', 'Positivo']))
    
    # Save results for comparison
    df['pred'] = y_pred
    df['reasoning'] = reasonings
    df.to_csv('data/processed/consolidated/llama_eval_results.csv', index=False)
    
    return acc, f1, kappa

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "system_prompt.txt")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
        
    run_eval(system_prompt)
