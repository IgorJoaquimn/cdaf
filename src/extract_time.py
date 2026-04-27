import cv2
import pytesseract
import sys
import re

def extract_time(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return
    
    h, w = img.shape[:2]
    # The scoreboard is usually in the top-left corner
    # Try different crops
    crops = [
        img[0:int(h*0.2), 0:int(w*0.3)],
        img[0:int(h*0.15), 0:int(w*0.25)],
        img[int(h*0.05):int(h*0.15), int(w*0.05):int(w*0.2)]
    ]
    
    found = False
    for i, cropped in enumerate(crops):
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Try raw grayscale, binary threshold, and inverted threshold
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        for name, processed_img in [("Gray", gray), ("Thresh1", thresh1), ("Thresh2", thresh2)]:
            # Scale up to improve OCR
            scaled = cv2.resize(processed_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # Try different PSM modes
            for psm in [11, 6]:
                # We limit the allowed characters to numbers and colon to reduce noise
                text = pytesseract.image_to_string(scaled, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789:')
                matches = re.findall(r'\b\d{2}:\d{2}\b', text)
                if matches:
                    print(f"{image_path} -> Found time: {matches[0]}")
                    found = True
                    break
            if found: break
        if found: break
        
    if not found:
        print(f"{image_path} -> No time found.")

for arg in sys.argv[1:]:
    extract_time(arg)
