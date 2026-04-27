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

def get_ocr_time(image_path, video_id, mode):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    
    rois = [
        ("std", img[int(h*0.03):int(h*0.12), int(w*0.03):int(w*0.16)]),
        ("wide", img[int(h*0.02):int(h*0.15), int(w*0.02):int(w*0.25)])
    ]
    
    for roi_name, roi in rois:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        scaled = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        
        methods = [
            cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY_INV)[1],
            cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        ]
        
        for processed in methods:
            for psm in [7, 6]:
                text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
                matches = re.findall(r'\d{1,2}:\d{2}', text)
                if matches:
                    m_time = t2s(matches[0])
                    # No 2º tempo, o relógio deve estar entre 45 e 65 min
                    if mode == "sh" and 2700 <= m_time < 3900:
                        return m_time
                    # No 1º tempo, entre 1 e 25 min
                    if mode == "fh" and 60 < m_time < 1500:
                        return m_time
    return None

def download_and_extract(v, mode="fh", offset_min=15):
    v_id = v['video_id']
    # Se for mode "sh", estimamos o início do 2T como Início_1T + 65 min (45+15+5 de folga)
    base_time = t2s(v['game_start_time']) if mode == "fh" else t2s(v['game_start_time']) + 3900
    target_video_sec = base_time + (offset_min * 60) if mode == "fh" else base_time
    
    full_dir = f'data/verification/full_{mode}'
    os.makedirs(full_dir, exist_ok=True)
    frame_path = os.path.join(full_dir, f"{v_id}.png")
    
    cmd = ['yt-dlp', '-f', 'best', '--get-url']
    if os.path.exists('cookies.txt'): cmd.extend(['--cookies', 'cookies.txt'])
    cmd.append(f'https://www.youtube.com/watch?v={v_id}')

    try:
        url_res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        v_url = url_res.stdout.strip()
        subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
        return v_id, frame_path, target_video_sec
    except Exception: return v_id, None, target_video_sec

def main():
    with open('config/settings.json', 'r') as f:
        videos = json.load(f)

    # 1. Refinar 2º Tempo (Só para os que não tem ou para todos)
    print(f"--- Refinando Início do 2º Tempo ({len(videos)} vídeos) ---")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda v: download_and_extract(v, mode="sh"), videos))

    refined_list = []
    for v_id, frame_path, target_sec in results:
        v_orig = next(v for v in videos if v['video_id'] == v_id)
        if not frame_path:
            refined_list.append(v_orig)
            continue
            
        # Tenta ler o tempo (esperamos algo como 45:00 ou superior)
        match_time_sec = get_ocr_time(frame_path, v_id, mode="sh")
        if match_time_sec:
            # Cálculo: Tempo_Video - (Tempo_Placar - 45min) = Início_Real_2T
            # Mas o placar no 2T já começa em 45, então:
            # Real_SH_Start = Target_Video_Sec - (Match_Time_Sec - 2700)
            real_sh_start_sec = target_sec - (match_time_sec - 2700)
            print(f"   [OK] {v_id}: Placar {s2t(match_time_sec)} -> 2º Tempo inicia em {s2t(real_sh_start_sec)}")
            v_orig['second_half_start'] = s2t(real_sh_start_sec)
        else:
            print(f"   [FALHA] {v_id}: Não detectou início do 2º Tempo.")
        
        refined_list.append(v_orig)

    with open('config/settings.json', 'w', encoding='utf-8') as f:
        json.dump(refined_list, f, indent=4, ensure_ascii=False)
    print("\nProcesso concluído! Settings atualizado.")

if __name__ == "__main__":
    main()
