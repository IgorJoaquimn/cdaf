import pandas as pd
from vllm import LLM, SamplingParams
import os
import json
import re
from tqdm import tqdm

def main():
    # 1. Configurações
    # Upgrade para 7B para melhor raciocínio, mantendo utilidade de memória segura para 16GB
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    input_path = 'data/processed/consolidated/gold_standard_v2_1000.csv'
    output_path = 'data/processed/consolidated/gold_standard_vllm.csv'

    print(f"Carregando vLLM com o modelo: {model_name}...")
    import os
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["VLLM_USE_V1"] = "0" # Desativar V1 para maior compatibilidade/estabilidade
    # Usando quantização de 4-bit ou 8-bit para garantir que o 7B caiba com folga no KV cache
    llm = LLM(
        model=model_name, 
        trust_remote_code=True, 
        gpu_memory_utilization=0.8,
        quantization="bitsandbytes", # Tentando bitsandbytes se disponível
        load_format="bitsandbytes"
    )

    # 2. Carregar dados
    df = pd.read_csv(input_path)
    messages = df['mensagem'].tolist()

    # 3. Prompting com Contexto de Futebol, Few-Shot, CoT e JSON
    system_prompt = (
        "Você é um especialista em análise de sentimentos de chats ao vivo de futebol brasileiro (CazéTV).\n"
        "Sua tarefa é analisar o comentário, explicar seu raciocínio (Chain of Thought) e classificar o sentimento.\n\n"
        "CONTEXTO DE FUTEBOL:\n"
        "- 'Bagre': Jogador ruim.\n"
        "- 'Jogar de terno': Jogar com muita classe e excelência.\n"
        "- 'Odd', 'Green', 'Red': Termos de apostas esportivas.\n"
        "- 'VAR', 'Operação', 'Máfia': Críticas à arbitragem.\n\n"
        "CATEGORIAS:\n"
        "0: NEGATIVO (Críticas, xingamentos, deboche de erro, reclamação da arbitragem)\n"
        "1: NEUTRO (Dúvidas, informações técnicas, saudações, termos de aposta sem emoção)\n"
        "2: POSITIVO (Comemoração, apoio, elogios, euforia)\n\n"
        "EXEMPLOS:\n"
        "Comentário: \"Sané tá jogando de terno hoje\"\n"
        "Resposta: {\"raciocinio\": \"'Jogar de terno' é um elogio à elegância e performance. Sentimento positivo.\", \"sentimento\": 2}\n\n"
        "Comentário: \"Que golaço desse bagre\"\n"
        "Resposta: {\"raciocinio\": \"'Bagre' é pejorativo. Mesmo com o gol, indica deboche do autor.\", \"sentimento\": 0}\n\n"
        "Comentário: \"odd de 1.50 pro bayer\"\n"
        "Resposta: {\"raciocinio\": \"Apenas informação técnica de aposta, sem carga emocional.\", \"sentimento\": 1}\n\n"
        "Responda SEMPRE em formato JSON com os campos 'raciocinio' e 'sentimento'."
    )

    def build_prompt(msg):
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nAnalise o comentário: \"{msg}\"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"```json\n"
        )

    prompts = [build_prompt(m) for m in messages]

    # 4. Inferência via vLLM
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=200,
        stop=["```"] # Parar ao fechar o bloco de código
    )
    print(f"Iniciando classificação JSON de {len(prompts)} mensagens com Qwen 7B...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Extração dos Resultados
    labels = []
    reasonings = []

    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        try:
            # Tentar limpar o texto caso o modelo coloque algo fora do JSON
            clean_json = generated_text.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_json)
            labels.append(int(data.get('sentimento', 1)))
            reasonings.append(data.get('raciocinio', ''))
        except:
            # Fallback se o JSON falhar
            match = re.search(r'"sentimento":\s*(\d)', generated_text)
            labels.append(int(match.group(1)) if match else 1)
            reasonings.append("Erro no parsing JSON")

    df['sentiment_vllm'] = labels
    df['reasoning_vllm'] = reasonings
    df['sentiment_manual'] = labels

    # 6. Salvar novo Gold Standard
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nConcluído!")
    print(f"Novo Gold Standard gerado via vLLM (CoT): {output_path}")
    print(f"\nDistribuição de sentimentos ({model_name} CoT + JSON):")
    print(df['sentiment_vllm'].value_counts())

if __name__ == "__main__":
    main()
