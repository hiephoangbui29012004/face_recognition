import cv2
import dlib
import os
import time
import numpy as np
from deepface import DeepFace
from pathlib import Path

# --- CONFIG ---
SAVE_DIR = os.path.join(os.getcwd(), "Face_detection", "outputs")
VECTOR_DIR = os.path.join(os.getcwd(), "Feature_extractor", "output_vector")
REF_IMG_DIR = os.path.join(os.getcwd(), "../reference_images")
REF_VEC_DIR = "../reference_vectors"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.6

# --- SETUP ---
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(REF_VEC_DIR, exist_ok=True)

# --- FUNC: Cosine Distance ---
def find_cosine_distance(source_representation, test_representation):
    a = np.dot(source_representation, test_representation)
    b = np.sum(source_representation ** 2)
    c = np.sum(test_representation ** 2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# --- FUNC: Feature Extraction ---
def extract_feature(image_path):
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
        if embedding_objs and len(embedding_objs) > 0:
            return np.array(embedding_objs[0]['embedding'])  # ép kiểu thành np.array
        else:
            print(f"[!] Không trích xuất được đặc trưng từ {image_path}")
            return None
    except Exception as e:
        print(f"[!] Lỗi khi trích xuất đặc trưng: {e}")
        return None

# --- FUNC: Chuẩn bị vector mẫu (1 lần hoặc nếu chưa có) ---
def prepare_reference_vectors():
    for filename in os.listdir(REF_IMG_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            vec_path = os.path.join(REF_VEC_DIR, f"{name}.npy")
            if not os.path.exists(vec_path):
                img_path = os.path.join(REF_IMG_DIR, filename)
                vec = extract_feature(img_path)
                if vec is not None:
                    np.save(vec_path, vec)
                    print(f"[✓] Đã lưu vector mẫu: {vec_path}")

# --- FUNC: So sánh với tất cả vector mẫu ---
def compare_with_all_references(test_vector):
    for vec_file in os.listdir(REF_VEC_DIR):
        if vec_file.endswith(".npy"):
            ref_vector = np.load(os.path.join(REF_VEC_DIR, vec_file))
            dist = find_cosine_distance(ref_vector, test_vector)
            if dist <= THRESHOLD:
                name = os.path.splitext(vec_file)[0]  # tên người (từ tên file)
                print(f"[=] Khớp với {name} (distance={dist:.4f})")
                return name  # <-- trả về tên người
    return None  #

# --- MAIN LOOP ---
def main():
    prepare_reference_vectors()
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    last_save_time = 0
    save_interval = 5

    if not cap.isOpened():
        print("[ERROR] Không mở được webcam! Có thể do webcam đang bận hoặc không có thiết bị webcam.")
    else:
        print("[INFO] Webcam mở thành công.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)

        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_crop = rgb[y1:y2, x1:x2]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            face_img_path = os.path.join(SAVE_DIR, f"face_{timestamp}.jpg")
            cv2.imwrite(face_img_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

            test_vector = extract_feature(face_img_path)
            if test_vector is not None:
                matched_name = compare_with_all_references(test_vector)
                label = matched_name if matched_name else "Unknown"
            else:
                label = "Detection Failed"

            cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)


        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
