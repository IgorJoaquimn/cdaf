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

def get_ocr_time(image_path, video_id, attempt):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    # ROI (Top-left area where clock usually is)
    roi = img[int(h*0.03):int(h*0.10), int(w*0.04):int(w*0.14)]
    
    # Salva o ROI em pasta específica
    roi_dir = 'data/frames_refine/roi'
    os.makedirs(roi_dir, exist_ok=True)
    roi_path = os.path.join(roi_dir, f"{video_id}_att{attempt}_roi.png")
    cv2.imwrite(roi_path, roi)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    
    methods = [
        ("otsu_inv", cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
        ("fixed_inv", cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY_INV)[1]),
        ("otsu", cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("fixed", cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)[1])
    ]
    
    for name, processed in methods:
        for psm in [7, 6, 11]:
            text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
            matches = re.findall(r'\d{1,2}:\d{2}', text)
            if matches:
                m_time = t2s(matches[0])
                # Validação: aceitamos tempo coerente com a tentativa
                min_expected = 300 if attempt == 1 else 900
                max_expected = 900 if attempt == 1 else 1500
                if min_expected < m_time < max_expected:
                    return m_time
    return None

def download_and_extract(v, attempt=1):
    v_id = v['video_id']
    current_start_sec = t2s(v['game_start_time'])
    # Se attempt 1 -> +10min, attempt 2 -> +20min
    offset = 600 * attempt
    target_video_sec = current_start_sec + offset
    
    full_dir = 'data/frames_refine/full'
    os.makedirs(full_dir, exist_ok=True)
    frame_path = os.path.join(full_dir, f"{v_id}_att{attempt}.png")
    
    if os.path.exists(frame_path):
        return v_id, frame_path, target_video_sec

    try:
        url_res = subprocess.run(['yt-dlp', '-f', 'best', '--get-url', f'https://www.youtube.com/watch?v={v_id}'], capture_output=True, text=True, check=True)
        v_url = url_res.stdout.strip()
        subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
        return v_id, frame_path, target_video_sec
    except Exception:
        return v_id, None, target_video_sec

def run_pass(videos, attempt):
    print(f"\n--- Passagem {attempt} (+{attempt*10}min) ---")
    
    # 1. Download
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda v: download_and_extract(v, attempt), videos))
    
    # 2. OCR e Refinamento
    success_configs = []
    failed_videos = []
    video_map = {v['video_id']: v for v in videos}

    for v_id, frame_path, target_sec in results:
        v_orig = video_map[v_id]
        if not frame_path:
            failed_videos.append(v_orig)
            continue
            
        match_time_sec = get_ocr_time(frame_path, v_id, attempt)
        if match_time_sec:
            real_start_sec = target_sec - match_time_sec
            real_start_str = s2t(real_start_sec)
            print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} -> Início {real_start_str}")
            v_ref = v_orig.copy()
            v_ref['game_start_time'] = real_start_str
            success_configs.append(v_ref)
        else:
            print(f"   [FALHA] {v_id}: OCR não leu o tempo.")
            failed_videos.append(v_orig)
            
    return success_configs, failed_videos

def main():
    with open('config/settings.json', 'r') as f:
        all_videos = json.load(f)

    # Primeira Passagem (+10 min)
    success_1, failed_1 = run_pass(all_videos, attempt=1)
    
    # Segunda Passagem (+20 min) para os que falharam
    success_2, final_failed = run_pass(failed_1, attempt=2)
    
    final_refined = success_1 + success_2 + final_failed
    
    # Salva
    output_path = 'config/settings_refined_full.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_refined, f, indent=4, ensure_ascii=False)
    
    print(f"\nFinalizado!")
    print(f"Total Sucesso: {len(success_1 + success_2)}")
    print(f"Total Falha: {len(final_failed)}")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
