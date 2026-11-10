
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("pencil_test2.jpg", iou=0.5)
annotated = results[0].plot()
h, w = annotated.shape[:2]
annotated = cv2.resize(annotated, (int(w * 800 / h), 800))

cv2.imshow("Pencil detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()