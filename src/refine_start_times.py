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

def get_ocr_time(image_path, video_id, offset_min):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    # Lista de ROIs possíveis (Top-Left, um pouco mais para o centro, e uma área de Multi-Live)
    rois = [
        ("standard", img[int(h*0.03):int(h*0.12), int(w*0.03):int(w*0.16)]),
        ("wide", img[int(h*0.02):int(h*0.15), int(w*0.02):int(w*0.25)]),
        ("center_ish", img[int(h*0.05):int(h*0.15), int(w*0.10):int(w*0.30)])
    ]
    
    roi_dir = 'data/verification/roi'
    os.makedirs(roi_dir, exist_ok=True)
    
    for roi_name, roi in rois:
        if roi is None or roi.size == 0: continue
        
        # Salvamos o ROI base para debug
        cv2.imwrite(os.path.join(roi_dir, f"{video_id}_{roi_name}_off{offset_min}m_raw.png"), roi)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Aumento de contraste via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Upscale
        scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        # Métodos de Thresholding
        methods = [
            ("otsu_inv", cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ("fixed_150_inv", cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY_INV)[1]),
            ("adaptive", cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)),
            ("gray", scaled) # Tenta no cinza puro também
        ]
        
        for m_name, processed in methods:
            # Salvamos os processados para os falhos para vermos o que o Tesseract vê
            for psm in [7, 6]:
                text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
                matches = re.findall(r'\d{1,2}:\d{2}', text)
                if matches:
                    m_time = t2s(matches[0])
                    # Validação expandida: aceitamos qualquer tempo entre 1 min e 90 min na busca manual
                    if 60 < m_time < 5400:
                        return m_time
    return None

def download_and_extract(v, offset_min):
    v_id = v['video_id']
    current_start_sec = t2s(v['game_start_time'])
    target_video_sec = current_start_sec + (offset_min * 60)
    
    full_dir = 'data/verification/full'
    os.makedirs(full_dir, exist_ok=True)
    frame_path = os.path.join(full_dir, f"{v_id}_off{offset_min}m.png")
    
    if os.path.exists(frame_path):
        return v_id, frame_path, target_video_sec

    cmd = ['yt-dlp', '-f', 'best', '--get-url']
    if os.path.exists('cookies.txt'):
        cmd.extend(['--cookies', 'cookies.txt'])
    cmd.append(f'https://www.youtube.com/watch?v={v_id}')

    try:
        url_res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        v_url = url_res.stdout.strip()
        subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
        return v_id, frame_path, target_video_sec
    except Exception:
        return v_id, None, target_video_sec

def run_pass(videos, offset_min):
    if not videos: return [], []
    print(f"\n--- Processando {len(videos)} vídeos aos +{offset_min}m ---")
    
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
            print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} -> Início {real_start_str}")
            v_ref = v_orig.copy()
            v_ref['game_start_time'] = real_start_str
            success_configs.append(v_ref)
        else:
            failed_videos.append(v_orig)
            
    return success_configs, failed_videos

def main():
    config_path = 'config/settings_refined_full.json'
    if not os.path.exists(config_path):
        config_path = 'config/settings.json'
    with open(config_path, 'r') as f:
        all_videos = json.load(f)

    # Pegamos os IDs que falharam na última rodada (pode automatizar pegando todos os não refinados)
    # Por segurança, vamos rodar apenas nos que falharam definitivamente
    failed_ids = ["bhOemLm_D8A", "N4fxZQu_hF8", "Q_snbPJ8QYk", "PAGjzBjC8-w", "pdPSp8agY_0", 
                  "MFDzkRNlhYY", "BqCHPpIg8Sg", "fbxD_0msVU4", "OZ2gA_kmBB0", "bcMPWKHTBEA", 
                  "7AH2hDDulZM", "QNZwsNZQeHI", "KSHbl1LZxuo", "P8qb8yV2ce8", "C5JAmx1RoSY"]
                  
    target_videos = [v for v in all_videos if v['video_id'] in failed_ids]
    successful_videos = [v for v in all_videos if v['video_id'] not in failed_ids]

    # Tentamos múltiplas vezes com o novo OCR agressivo
    current_batch = target_videos
    for offset in [15, 25, 35, 45, 55]:
        success, failed = run_pass(current_batch, offset)
        successful_videos.extend(success)
        current_batch = failed
        if not current_batch: break

    # Salva o arquivo final
    final_refined = successful_videos + current_batch
    video_order = {v['video_id']: i for i, v in enumerate(all_videos)}
    final_refined.sort(key=lambda v: video_order.get(v['video_id'], 999))
    
    with open('config/settings.json', 'w', encoding='utf-8') as f:
        json.dump(final_refined, f, indent=4, ensure_ascii=False)
    
    print(f"\nFinalizado! Recuperados: {len(target_videos) - len(current_batch)} de {len(target_videos)}")

if __name__ == "__main__":
    main()
