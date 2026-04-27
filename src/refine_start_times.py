import json
import subprocess
import os
import cv2
import pytesseract
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def t2s(t):
    if not t: return 0
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
    
    # Múltiplos ROIs para garantir que pegamos o placar em layouts diferentes
    rois = [
        ("std", img[int(h*0.03):int(h*0.12), int(w*0.03):int(w*0.16)]),
        ("wide", img[int(h*0.02):int(h*0.15), int(w*0.02):int(w*0.25)])
    ]
    
    roi_dir = 'data/verification/roi'
    os.makedirs(roi_dir, exist_ok=True)
    
    for roi_name, roi in rois:
        cv2.imwrite(os.path.join(roi_dir, f"{video_id}_sh_off{offset_min}m_{roi_name}_raw.png"), roi)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        scaled = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        
        methods = [
            ("otsu_inv", cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ("fixed_120_inv", cv2.threshold(scaled, 120, 255, cv2.THRESH_BINARY_INV)[1]),
            ("adaptive", cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)),
            ("gray", scaled)
        ]
        
        for name, processed in methods:
            for psm in [7, 6]:
                text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
                matches = re.findall(r'\d{1,2}:\d{2}', text)
                if matches:
                    m_time = t2s(matches[0])
                    # No 2º tempo, o placar deve ser >= 45:00
                    if 2700 <= m_time < 5400: # 45min a 90min
                        return m_time
    return None

def download_and_extract(v, offset_min):
    v_id = v['video_id']
    # A base é o início do 1º tempo que já sabemos ser exato
    base_1t = t2s(v['game_start_time'])
    target_video_sec = base_1t + (offset_min * 60)
    
    full_dir = 'data/verification/full_sh'
    os.makedirs(full_dir, exist_ok=True)
    frame_path = os.path.join(full_dir, f"{v_id}_off{offset_min}m.png")
    
    if os.path.exists(frame_path):
        return v_id, frame_path, target_video_sec

    cmd = ['yt-dlp', '-f', 'best', '--get-url']
    if os.path.exists('cookies.txt'): cmd.extend(['--cookies', 'cookies.txt'])
    cmd.append(f'https://www.youtube.com/watch?v={v_id}')

    try:
        url_res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        v_url = url_res.stdout.strip()
        subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
        return v_id, frame_path, target_video_sec
    except Exception: return v_id, None, target_video_sec

def run_pass(videos, offset_min):
    if not videos: return [], []
    print(f"\n--- Sondagem 2º Tempo aos +{offset_min}m ({len(videos)} vídeos) ---")
    
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
            # Placar (ex: 50:00 = 3000s) - Início do 2T no placar (45:00 = 2700s) = Tempo decorrido no 2T (300s)
            elapsed_sh = match_time_sec - 2700
            real_sh_start_sec = target_sec - elapsed_sh
            print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} -> 2º Tempo inicia em {s2t(real_sh_start_sec)}")
            v_ref = v_orig.copy()
            v_ref['second_half_start'] = s2t(real_sh_start_sec)
            success_configs.append(v_ref)
        else:
            failed_videos.append(v_orig)
            
    return success_configs, failed_videos

def main():
    with open('config/settings.json', 'r') as f:
        all_videos = json.load(f)

    # Pegamos os que não tem 'second_half_start' definido
    target_videos = [v for v in all_videos if not v.get('second_half_start')]
    successful_videos = [v for v in all_videos if v.get('second_half_start')]

    print(f"Vídeos para processar: {len(target_videos)}")

    # Proba em tempos diferentes para fugir de comerciais/lineups
    for offset in [65, 75, 85, 95]:
        success, failed = run_pass(target_videos, offset)
        successful_videos.extend(success)
        target_videos = failed
        if not target_videos: break

    # Salva
    final_list = successful_videos + target_videos
    video_order = {v['video_id']: i for i, v in enumerate(all_videos)}
    final_list.sort(key=lambda v: video_order.get(v['video_id'], 999))
    
    with open('config/settings.json', 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)
    
    print(f"\nFinalizado! Recuperados nesta rodada: {len(success)}")
    print(f"Total com 2º Tempo: {len([v for v in final_list if v.get('second_half_start')])} de {len(final_list)}")

if __name__ == "__main__":
    main()
