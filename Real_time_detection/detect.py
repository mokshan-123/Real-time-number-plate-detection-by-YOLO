import cv2
import torch
import pytesseract
from ultralytics import YOLO

# ── Tesseract path ───────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Load model ───────────────────────────────────────────────
model = YOLO("E:/Github/Number plate detection/Source_Files_and_final_weights/Final_weights/best.pt")
model.to('cpu')
torch.set_num_threads(4)

# ── OCR function ─────────────────────────────────────────────
def read_plate(crop):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2)
    _, thresh = cv2.threshold(resized, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text   = pytesseract.image_to_string(thresh, config=config).strip()
    return text

# ── Open camera ──────────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Could not open camera — try index 3")
    exit()

print("Running... Press Q to quit")

frame_count = 0
last_boxes  = []
last_texts  = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Run YOLO every 3rd frame ──────────────────────────────
    if frame_count % 3 == 0:
        results    = model.predict(
            source  = frame,
            imgsz   = 256,
            conf    = 0.4,
            verbose = False,
            half    = False
        )
        last_boxes = results[0].boxes

        # ── Run OCR every 6th frame ───────────────────────────
        # Tesseract is fast enough to run more frequently
        if frame_count % 6 == 0 and last_boxes is not None:
            last_texts = {}
            for i, box in enumerate(last_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pad = 4
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(frame.shape[1], x2 + pad)
                y2p = min(frame.shape[0], y2 + pad)

                plate_crop = frame[y1p:y2p, x1p:x2p]

                if plate_crop.size == 0:
                    continue

                text = read_plate(plate_crop)

                if len(text) >= 4:   # ignore very short/empty reads
                    last_texts[i] = text

    # ── Draw boxes and OCR text on every frame ────────────────
    if last_boxes is not None:
        for i, box in enumerate(last_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])

            # green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)

            # detection label
            det_label = f"Plate {conf_score:.2f}"
            (lw, lh), _ = cv2.getTextSize(det_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), (0, 255, 80), -1)
            cv2.putText(frame, det_label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # OCR text below box in orange
            if i in last_texts:
                ocr_label = last_texts[i]
                (tw, th), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y2), (x1+tw+4, y2+th+8), (255, 140, 0), -1)
                cv2.putText(frame, ocr_label, (x1+2, y2+th+2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Info overlay ─────────────────────────────────────────
    cv2.putText(frame, f"Plates: {len(last_boxes) if last_boxes else 0}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)

    cv2.imshow("License Plate Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")