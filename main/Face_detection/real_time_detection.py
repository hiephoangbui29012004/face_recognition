import cv2
import dlib
import time
import os

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
save_dir = BASE_DIR / "outputs"
os.makedirs(save_dir, exist_ok=True)


detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

last_save_time = 0
save_interval = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    current_time = time.time()
    if len(faces) > 0 and (current_time - last_save_time >= save_interval):
        for i, face in enumerate(faces):
            x1 = max(0, face.left())
            y1 = max(0, face.top())
            x2 = min(frame.shape[1], face.right())
            y2 = min(frame.shape[0], face.bottom())

            face_img = frame[y1:y2, x1:x2]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"face_{timestamp}_{i}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"Đã lưu ảnh khuôn mặt: {filename}")

        last_save_time = current_time

    cv2.imshow("Dlib HOG Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()