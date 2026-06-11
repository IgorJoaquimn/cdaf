import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import pandas as pd
from vllm import LLM, SamplingParams
import json
import re

def main():
    # 1. Configurações
    model_name = "google/gemma-4-E2B-it"
    input_path = 'data/processed/consolidated/bundesliga_2425_cazetv_chat_cleaned.parquet'
    output_path = 'data/processed/consolidated/synthetic_dataset_for_bert.csv'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "system_prompt.txt")

    print(f"Carregando vLLM com o modelo: {model_name}...")
    llm = LLM(
        model=model_name, 
        trust_remote_code=True, 
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=2048,
    )

    tokenizer = llm.get_tokenizer()

    # 2. Carregar e preparar dados
    print(f"Carregando comentários de: {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Filtrar comentários vazios
    df = df[df['mensagem'].notna() & (df['mensagem'].str.strip() != '')]
    
    # Amostrar 2000 comentários com seed aleatória para reprodutibilidade
    print("Amostrando 2000 comentários...")
    df_sample = df.sample(n=2000, random_state=42).copy()
    messages = df_sample['mensagem'].tolist()

    # 3. Carregar o Prompt de Sistema
    print(f"Carregando prompt de: {prompt_path}...")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    def build_prompt(msg):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analise o comentário: \"{msg}\""}
        ]
        # Disable thinking natively for direct JSON generation
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return prompt

    prompts = [build_prompt(m) for m in messages]

    # 4. Inferência via vLLM
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=250,
    )
    
    print(f"Iniciando classificação JSON de {len(prompts)} mensagens com {model_name}...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Extração dos Resultados
    labels = []
    reasonings = []

    for output in outputs:
        text = output.outputs[0].text
        try:
            clean_json = text.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_json)
            labels.append(int(data.get('sentimento', 1)))
            reasonings.append(data.get('raciocinio', ''))
        except Exception as e:
            # Fallback parsing
            match = re.search(r'"sentimento":\s*(\d)', text)
            if match:
                labels.append(int(match.group(1)))
                reasonings.append(f"Fallback parse from: {text}")
            else:
                labels.append(1)
                reasonings.append(f"Error parsing: {text} | {str(e)}")

    df_sample['label'] = labels
    df_sample['reasoning'] = reasonings

    # 6. Salvar novo dataset para o Bertimbau
    df_sample.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nConcluído!")
    print(f"Dataset de treino sintético gerado com sucesso: {output_path}")
    print(f"\nDistribuição de sentimentos ({model_name}):")
    print(df_sample['label'].value_counts())

if __name__ == "__main__":
    main()
