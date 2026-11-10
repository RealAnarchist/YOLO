import time
import cv2
from ultralytics import YOLO

DEVICE = "cuda:0"   # change to "cuda:0" for the GPU

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

frames = 0
start = time.time()
prev = start

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, device=DEVICE, verbose=False)
    annotated = results[0].plot()

    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    frames += 1
    infer_ms = results[0].speed["inference"]

    text = f"{DEVICE}  FPS: {fps:.1f}  inference: {infer_ms:.1f} ms"
    cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("YOLO webcam", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

total = time.time() - start
print(f"Device: {DEVICE}")
print(f"Average FPS: {frames / total:.1f}")

cap.release()
cv2.destroyAllWindows()