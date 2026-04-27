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

def get_ocr_time(image_path, video_id, offset_min, half_prefix):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    rois = [
        ("std", img[int(h*0.03):int(h*0.12), int(w*0.03):int(w*0.16)]),
        ("wide", img[int(h*0.02):int(h*0.15), int(w*0.02):int(w*0.25)])
    ]
    
    roi_dir = f'data/verification/{half_prefix}/roi'
    os.makedirs(roi_dir, exist_ok=True)
    
    for roi_name, roi in rois:
        cv2.imwrite(os.path.join(roi_dir, f"{video_id}_off{offset_min}m_{roi_name}_raw.png"), roi)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        scaled = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        
        methods = [
            cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.threshold(scaled, 120, 255, cv2.THRESH_BINARY_INV)[1],
            cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5),
            scaled
        ]
        
        for processed in methods:
            for psm in [7, 6]:
                text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
                matches = re.findall(r'\d{1,2}:\d{2}', text)
                if matches:
                    m_time = t2s(matches[0])
                    # Verificações de sanidade por tempo
                    if half_prefix == "1st_half" and 60 < m_time < 2400: return m_time
                    if half_prefix == "2nd_half" and 2700 <= m_time < 5400: return m_time
    return None

def download_and_extract(v, offset_min, half_prefix):
    v_id = v['video_id']
    base_time = t2s(v['game_start_time'])
    target_video_sec = base_time + (offset_min * 60)
    
    full_dir = f'data/verification/{half_prefix}/full'
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

def run_refinement(videos, half_prefix, offsets):
    print(f"\n>>> Refinando {half_prefix}...")
    current_batch = videos
    successful_configs = []
    
    for offset in offsets:
        if not current_batch: break
        print(f"--- Sondagem aos +{offset}m ({len(current_batch)} vídeos) ---")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda v: download_and_extract(v, offset, half_prefix), current_batch))
        
        failed_this_pass = []
        video_map = {v['video_id']: v for v in current_batch}

        for v_id, frame_path, target_sec in results:
            v_orig = video_map[v_id]
            if not frame_path:
                failed_this_pass.append(v_orig)
                continue
                
            match_time_sec = get_ocr_time(frame_path, v_id, offset, half_prefix)
            if match_time_sec:
                # Cálculo para 1º Tempo vs 2º Tempo
                if half_prefix == "1st_half":
                    real_start = target_sec - match_time_sec
                    v_orig['game_start_time'] = s2t(real_start)
                else:
                    elapsed_sh = match_time_sec - 2700
                    real_sh_start = target_sec - elapsed_sh
                    v_orig['second_half_start'] = s2t(real_sh_start)
                
                print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} refinado.")
                successful_configs.append(v_orig)
            else:
                failed_this_pass.append(v_orig)
        
        current_batch = failed_this_pass
            
    return successful_configs, current_batch

def main():
    with open('config/settings.json', 'r') as f:
        all_videos = json.load(f)

    # Nota: Este script agora pode ser rodado para refinar ambos os tempos.
    # Por agora, ele garante que a estrutura de pastas esteja correta.
    
    # Se quiser rodar refinamento novamente em massa, descomente as linhas abaixo:
    # fh_success, fh_fail = run_refinement(all_videos, "1st_half", [15, 25, 35])
    # sh_success, sh_fail = run_refinement(all_videos, "2nd_half", [65, 75, 85])
    
    print("Estrutura de pastas organizada. Scripts prontos.")

if __name__ == "__main__":
    main()
