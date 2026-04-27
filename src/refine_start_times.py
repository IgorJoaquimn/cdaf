import json
import subprocess
import os
import cv2
import pytesseract
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def t2s(t):
    parts = list(map(int, t.split(':')))
    if len(parts) == 2: return parts[0] * 60 + parts[1]
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def s2t(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0: return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def get_ocr_time(image_path, video_id, attempt_offset_min):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    # Expandimos o ROI para pegar o placar mesmo se ele estiver ligeiramente deslocado
    # Em alguns multi-lives o placar pode não estar colado na borda absoluta
    roi_top_left = img[int(h*0.03):int(h*0.12), int(w*0.03):int(w*0.16)]
    
    # Salva o ROI para debug
    roi_dir = 'data/frames_refine/roi'
    os.makedirs(roi_dir, exist_ok=True)
    roi_path = os.path.join(roi_dir, f"{video_id}_off{attempt_offset_min}m_roi.png")
    cv2.imwrite(roi_path, roi_top_left)
    
    gray = cv2.cvtColor(roi_top_left, cv2.COLOR_BGR2GRAY)
    
    # Upscale agressivo para ajudar o Tesseract
    scaled = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    
    # Diversos métodos de processamento para lidar com diferentes placares
    methods = [
        # 1. Invertido com Otsu (bom para texto claro em fundo escuro)
        ("otsu_inv", cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
        # 2. Invertido com limiar fixo agressivo (bom para o fundo translúcido da CazéTV)
        ("fixed_120_inv", cv2.threshold(scaled, 120, 255, cv2.THRESH_BINARY_INV)[1]),
        ("fixed_180_inv", cv2.threshold(scaled, 180, 255, cv2.THRESH_BINARY_INV)[1]),
        # 3. Normal com Otsu
        ("otsu", cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        # 4. Adaptive Thresholding (bom para iluminação irregular)
        ("adaptive_inv", cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4))
    ]
    
    for name, processed in methods:
        for psm in [7, 6, 11]:
            text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
            matches = re.findall(r'\d{1,2}:\d{2}', text)
            if matches:
                m_time = t2s(matches[0])
                # Validação: o tempo no placar deve ser aproximadamente igual ao offset que adicionamos
                # Damos uma margem de manobra de +/- 10 minutos para acomodar pre-games e delays
                offset_sec = attempt_offset_min * 60
                if (offset_sec - 600) < m_time < (offset_sec + 600):
                    return m_time
    return None

def download_and_extract(v, offset_min):
    v_id = v['video_id']
    current_start_sec = t2s(v['game_start_time'])
    target_video_sec = current_start_sec + (offset_min * 60)
    
    full_dir = 'data/frames_refine/full'
    os.makedirs(full_dir, exist_ok=True)
    frame_path = os.path.join(full_dir, f"{v_id}_off{offset_min}m.png")
    
    if os.path.exists(frame_path):
        return v_id, frame_path, target_video_sec

    cmd = ['yt-dlp', '-f', 'best', '--get-url']
    if os.path.exists('cookies.txt'):
        cmd.extend(['--cookies', 'cookies.txt'])
    cmd.append(f'https://www.youtube.com/watch?v={v_id}')

    try:
        import time
        import random
        time.sleep(random.uniform(1.0, 2.0))
        
        url_res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        v_url = url_res.stdout.strip()
        subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
        return v_id, frame_path, target_video_sec
    except subprocess.CalledProcessError as e:
        print(f"Erro no yt-dlp para {v_id}. Possível Rate Limit (429). {e}")
        return v_id, None, target_video_sec
    except Exception as e:
        print(f"Erro genérico no download de {v_id}: {e}")
        return v_id, None, target_video_sec

def run_pass(videos, offset_min):
    if not videos:
        return [], []
        
    print(f"\n--- Sondagem aos +{offset_min} minutos ({len(videos)} vídeos) ---")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda v: download_and_extract(v, offset_min), videos))
    
    success_configs = []
    failed_videos = []
    video_map = {v['video_id']: v for v in videos}

    for v_id, frame_path, target_sec in results:
        v_orig = video_map[v_id]
        if not frame_path:
            failed_videos.append(v_orig)
            continue
            
        match_time_sec = get_ocr_time(frame_path, v_id, offset_min)
        if match_time_sec:
            real_start_sec = target_sec - match_time_sec
            real_start_str = s2t(real_start_sec)
            print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} -> Início Real {real_start_str}")
            v_ref = v_orig.copy()
            v_ref['game_start_time'] = real_start_str
            success_configs.append(v_ref)
        else:
            print(f"   [FALHA] {v_id}: Placar não encontrado/legível.")
            failed_videos.append(v_orig)
            
    return success_configs, failed_videos

def main():
    # Usar o arquivo atual de config como base
    config_path = 'config/settings_refined_full.json'
    if not os.path.exists(config_path):
        config_path = 'config/settings.json'
        
    with open(config_path, 'r') as f:
        all_videos = json.load(f)

    # Identificar quais vídeos ainda precisam de refinamento.
    # Podemos assumir que se falhou antes, o 'game_start_time' ainda é o inicial.
    # Vamos pegar a lista de arquivos falhos que você gerou:
    failed_ids = ["vgAGutUBzvo", "-sXQpbHwhnM", "zLMB45iOuTc", "bhOemLm_D8A", "N4fxZQu_hF8", 
                  "Q_snbPJ8QYk", "SWhmFEwPBgk", "PAGjzBjC8-w", "pdPSp8agY_0", "MFDzkRNlhYY", 
                  "BqCHPpIg8Sg", "GNy-aVzpM2c", "bAlo5Eh3ovY", "fbxD_0msVU4", "OZ2gA_kmBB0", 
                  "bcMPWKHTBEA", "7AH2hDDulZM", "QNZwsNZQeHI", "KSHbl1LZxuo", "P8qb8yV2ce8", 
                  "C5JAmx1RoSY", "Pne4Fl0lWHI", "Z-ddfBl5GW4", "VGo6TfFDKAM", "tAKBwvKP-Po", 
                  "EvlcrG3u6-U", "EXYr-et6-z8"]
                  
    target_videos = [v for v in all_videos if v['video_id'] in failed_ids]
    successful_videos = [v for v in all_videos if v['video_id'] not in failed_ids]

    # Estratégia de Múltiplas Sondagens
    offsets_to_try = [15, 25, 35, 45] # Testa em 15m, 25m, 35m, e 45m de jogo
    
    current_batch = target_videos
    for offset in offsets_to_try:
        success, failed = run_pass(current_batch, offset)
        successful_videos.extend(success)
        current_batch = failed
        
        if not current_batch:
            break # Todos processados!

    final_failed = current_batch
    
    # Reconstroi a lista final mantendo os falhos com o tempo original
    final_refined = successful_videos + final_failed
    
    # Restaura a ordem original (opcional, mas bom para consistência)
    video_order = {v['video_id']: i for i, v in enumerate(all_videos)}
    final_refined.sort(key=lambda v: video_order.get(v['video_id'], 999))
    
    output_path = 'config/settings_refined_full.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_refined, f, indent=4, ensure_ascii=False)
    
    print(f"\n==== Resumo Final ====")
    print(f"Total Sucesso (Recuperados): {len(target_videos) - len(final_failed)}")
    print(f"Total Falha Definitiva: {len(final_failed)}")
    if final_failed:
        print(f"Falharam: {[v['video_id'] for v in final_failed]}")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
