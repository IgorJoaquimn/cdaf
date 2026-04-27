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

def get_ocr_time(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    # ROI mais generosa para garantir que nada seja cortado
    roi = img[int(h*0.03):int(h*0.10), int(w*0.04):int(w*0.14)]
    
    debug_roi_path = image_path.replace('.png', '_roi.png')
    cv2.imwrite(debug_roi_path, roi)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    
    # Testamos diferentes binarizações
    # Como o texto é branco em fundo escuro, BINARY normal costuma ser bom
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
                if 300 < m_time < 1500: # Sanity check (entre 5 e 25 min)
                    return m_time
    return None

def download_and_extract(v):
    v_id = v['video_id']
    current_start_sec = t2s(v['game_start_time'])
    target_video_sec = current_start_sec + 600 # 10 min depois
    
    frame_path = os.path.join('data/frames_refine', f"{v_id}_refine_check.png")
    
    if os.path.exists(frame_path):
        return v_id, frame_path, target_video_sec

    try:
        url_res = subprocess.run(['yt-dlp', '-f', 'best', '--get-url', f'https://www.youtube.com/watch?v={v_id}'], capture_output=True, text=True, check=True)
        v_url = url_res.stdout.strip()
        subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
        return v_id, frame_path, target_video_sec
    except Exception as e:
        print(f"Erro no download de {v_id}: {e}")
        return v_id, None, target_video_sec

def main():
    with open('config/settings.json', 'r') as f:
        videos = json.load(f)
    
    frames_dir = 'data/frames_refine'
    os.makedirs(frames_dir, exist_ok=True)

    print(f"--- [1/2] Baixando frames em paralelo ---")
    results = []
    # Usar 4 workers para não sobrecarregar mas ser rápido
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(download_and_extract, videos))

    print(f"--- [2/2] Processando OCR ---")
    refined_settings = []
    
    # Mapeia configs originais para facilitar busca
    video_map = {v['video_id']: v for v in videos}

    for v_id, frame_path, target_sec in results:
        v_orig = video_map[v_id]
        if not frame_path:
            refined_settings.append(v_orig)
            continue
            
        match_time_sec = get_ocr_time(frame_path)
        if match_time_sec:
            real_start_sec = target_sec - match_time_sec
            real_start_str = s2t(real_start_sec)
            print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} -> Início {real_start_str}")
            v_ref = v_orig.copy()
            v_ref['game_start_time'] = real_start_str
            refined_settings.append(v_ref)
        else:
            print(f"   [FALHA] {v_id}: OCR falhou.")
            refined_settings.append(v_orig)

    output_path = 'config/settings_refined_full.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refined_settings, f, indent=4, ensure_ascii=False)
    
    print(f"\nFinalizado! Total processado: {len(refined_settings)}")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
