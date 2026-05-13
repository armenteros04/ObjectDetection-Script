import cv2
from ultralytics import YOLO
import time
import sys

model = YOLO("yolov8n.pt")

# line_y = 300
line_y = 400
counter = 0
track_history = {}
already_counted = set()

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("videos/car.mp4")

if not cap.isOpened():
    print("ERROR: There is no access to the camera/video.")
    exit()

print("--- COM3548 FINAL PROJECT - Real Time Object Detection ---")

while cap.isOpened():
    start_time = time.time()

    success, frame = cap.read()
    if not success:
        print("\nError in the frame capture.")
        break

    objetos_vistos = []

    results = model.track(frame,
                          persist=True,
                          tracker="botsort.yaml",
                          verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Centro del objeto

            label = model.names[cls]
            objetos_vistos.append(f"{label}({track_id})")

            if cy > line_y and track_id not in already_counted:
                counter += 1
                already_counted.add(track_id)
                sys.stdout.write(f"\n[⚠️] {label} ID:{track_id} cross the line.\n")

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id} {label}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    info = f"\nFPS: {fps:.1f} | Objects: {', '.join(objetos_vistos)} | Total: {counter}"
    sys.stdout.write('\r' + info.ljust(100))
    sys.stdout.flush()

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)
    cv2.putText(frame, f"Counter: {counter}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("COM3548 - Real Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n\nFinal count: {counter}")