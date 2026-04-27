import json
import subprocess
import os
import cv2
import pytesseract
import re
import numpy as np

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
    
    # O placar nos vídeos da Bundesliga/CazéTV fica bem no cantinho
    # Vamos pegar uma área pequena mas suficiente
    # Time is usually in the very first segment of the scoreboard
    # Coordinates roughly: Top 5% to 10%, Left 5% to 15%
    roi = img[int(h*0.04):int(h*0.085), int(w*0.055):int(w*0.115)]
    
    # Debug: salvar o ROI para ver o que o script está tentando ler
    debug_roi_path = image_path.replace('.png', '_roi.png')
    cv2.imwrite(debug_roi_path, roi)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Técnicas de processamento
    # 1. Resize (importante para Tesseract)
    scaled = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    
    # 2. Thresholding variados
    methods = [
        cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY_INV)[1],
        cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ]
    
    for processed in methods:
        # Tesseract configurations
        # PSM 7: Treat the image as a single text line.
        # PSM 6: Assume a single uniform block of text.
        for psm in [7, 6, 11]:
            text = pytesseract.image_to_string(processed, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
            matches = re.findall(r'\d{1,2}:\d{2}', text)
            if matches:
                m_time = t2s(matches[0])
                # Sanity check: deve estar perto de 10 min (600s)
                if 300 < m_time < 1200:
                    return m_time
    return None

def main():
    with open('config/settings.json', 'r') as f:
        videos = json.load(f)
    
    test_videos = videos[:10]
    refined_settings = []
    frames_dir = 'data/frames_refine'
    os.makedirs(frames_dir, exist_ok=True)

    print(f"--- Iniciando Refinamento com OCR Otimizado ---")

    for v in test_videos:
        v_id = v['video_id']
        current_start_sec = t2s(v['game_start_time'])
        target_video_sec = current_start_sec + 600 
        
        frame_path = os.path.join(frames_dir, f"{v_id}_refine_check.png")
        print(f"\nProcessando {v_id}...")

        # Se o frame não existe, baixa. Se existe, usa o local para testar o OCR.
        if not os.path.exists(frame_path):
            try:
                url_res = subprocess.run(['yt-dlp', '-f', 'best', '--get-url', f'https://www.youtube.com/watch?v={v_id}'], capture_output=True, text=True, check=True)
                v_url = url_res.stdout.strip()
                subprocess.run(['ffmpeg', '-ss', str(target_video_sec), '-i', v_url, '-frames:v', '1', '-q:v', '2', '-y', frame_path], check=True, capture_output=True)
            except Exception as e:
                print(f"   [ERRO] Download falhou: {e}")
                refined_settings.append(v)
                continue

        # Tentar ler o tempo
        match_time_sec = get_ocr_time(frame_path)
        
        if match_time_sec:
            real_start_sec = target_video_sec - match_time_sec
            real_start_str = s2t(real_start_sec)
            print(f"   [OK] Placar: {s2t(match_time_sec)} | Início: {real_start_str}")
            v_refined = v.copy()
            v_refined['game_start_time'] = real_start_str
            refined_settings.append(v_refined)
        else:
            print(f"   [FALHA] OCR não leu o tempo em {frame_path}")
            refined_settings.append(v)

    output_path = 'config/settings_refined_10.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refined_settings, f, indent=4, ensure_ascii=False)
    print(f"\nFinalizado! Veja config/settings_refined_10.json")

if __name__ == "__main__":
    main()
