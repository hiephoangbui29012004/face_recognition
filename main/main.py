import cv2
import dlib
import os
import time
import numpy as np
import tensorflow as tf
import csv
import shutil

# Cấu hình GPU cho TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("[INFO] Đã bật tăng trưởng bộ nhớ GPU.")
    except RuntimeError as e:
        print(f"[ERROR] Lỗi cấu hình GPU: {e}")

from deepface import DeepFace

# --- CẤU HÌNH CHUNG ---
SAVE_DIR = os.path.join(os.getcwd(), "Face_detection", "outputs")
REF_IMG_DIR = "../reference_images"  # Thư mục chứa ảnh mẫu
REF_VEC_DIR = "../reference_vectors"  # Thư mục lưu trữ vector đặc trưng của ảnh mẫu
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"
THRESHOLD = 0.6  # Ngưỡng khoảng cách cosine để xác định sự khớp

# --- Cấu hình lưu trữ dữ liệu điểm danh ---
ATTENDANCE_CSV_FILE = os.path.join(os.getcwd(), "attendance.csv")

# --- KHỞI TẠO THƯ MỤC ---
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REF_VEC_DIR, exist_ok=True)

# Xây dựng mô hình DeepFace
print("[INFO] Đang xây dựng mô hình DeepFace...")
facenet_model = DeepFace.build_model(MODEL_NAME)
print(f"[INFO] Đã xây dựng xong mô hình DeepFace ({MODEL_NAME}).")


# --- HÀM TÍNH KHOẢNG CÁCH COSINE ---
def find_cosine_distance(source_representation, test_representation):
    a = np.dot(source_representation, test_representation)
    b = np.sum(source_representation ** 2)
    c = np.sum(test_representation ** 2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# --- HÀM TRÍCH XUẤT ĐẶC TRƯNG KHUÔN MẶT ---
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
            return np.array(embedding_objs[0]['embedding'])
        return None
    except Exception as e:
        print(f"[!] Lỗi khi trích xuất đặc trưng: {e}")
        return None

# --- Chuẩn Bị Vector Mẫu (Không Xóa File Cũ) ---
def prepare_reference_vectors():
    if not os.path.exists(REF_IMG_DIR):
        print(f"[ERROR] Thư mục ảnh mẫu không tồn tại: {REF_IMG_DIR}")
        print("Vui lòng tạo thư mục này và đặt ảnh mẫu vào.")
        return

    print(f"[INFO] Bắt đầu chuẩn bị vector mẫu từ {REF_IMG_DIR}...")
    for filename in os.listdir(REF_IMG_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            vec_path = os.path.join(REF_VEC_DIR, f"{name}.npy")
            
            # Kiểm tra nếu vector mẫu đã tồn tại, thì bỏ qua
            if os.path.exists(vec_path):
                print(f"[INFO] Vector cho {name} đã tồn tại, bỏ qua.")
                continue

            img_path = os.path.join(REF_IMG_DIR, filename)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"[!] Không thể đọc ảnh: {img_path}, bỏ qua.")
                continue
            
            vec = extract_feature(img_path)
            if vec is not None:
                np.save(vec_path, vec)
                print(f"[✓] Đã lưu vector mẫu: {vec_path}")
            else:
                print(f"[!] Không thể tạo vector cho ảnh {filename}")
    print("[INFO] Hoàn tất chuẩn bị vector mẫu.")

# --- HÀM SO SÁNH VỚI TẤT CẢ VECTOR MẪU ---
def compare_with_all_references(test_vector):
    if not os.path.exists(REF_VEC_DIR) or not os.listdir(REF_VEC_DIR):
        print("[WARNING] Thư mục vector mẫu rỗng hoặc không tồn tại. Vui lòng chạy prepare_reference_vectors trước.")
        return None

    for vec_file in os.listdir(REF_VEC_DIR):
        if vec_file.endswith(".npy"):
            ref_vector = np.load(os.path.join(REF_VEC_DIR, vec_file))
            dist = find_cosine_distance(ref_vector, test_vector)
            if dist <= THRESHOLD:
                name = os.path.splitext(vec_file)[0]
                print(f"[=] Khớp với {name} (distance={dist:.4f})")
                return name
    return None

# --- HÀM LƯU DỮ LIỆU VÀO CSV ---
def save_to_csv(data_entry, filename=ATTENDANCE_CSV_FILE):
    headers = ["ID_NhanVien", "Ten_NhanVien", "ThoiGian", "TrangThai", "GhiChu"]
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_entry)
    print(f"[✓] Đã ghi vào CSV: {data_entry}")

# --- HÀM CHÍNH ---
def main():
    print("[INFO] Bắt đầu quá trình nhận diện...")
    prepare_reference_vectors()
    print("[INFO] Đã hoàn tất chuẩn bị vector tham chiếu.")

    print("[INFO] Đang khởi tạo bộ phát hiện khuôn mặt dlib và webcam...")
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0) # 0 là ID của webcam mặc định
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[INFO] Đã khởi tạo xong bộ phát hiện khuôn mặt và webcam.")
    
    last_processed_person = {}
    processing_interval_seconds = 10 

    if not cap.isOpened():
        print("[ERROR] Không mở được webcam! Có thể do webcam đang bận hoặc không có thiết bị webcam.")
        print("Vui lòng kiểm tra quyền truy cập webcam trong cài đặt hệ thống (Privacy & Security -> Camera trên macOS).")
        return
    else:
        print("[INFO] Webcam mở thành công. Bắt đầu nhận diện...")
    
    last_recognition_time = 0
    recognition_interval = 1 # Khoảng thời gian tối thiểu giữa hai lần nhận diện liên tiếp

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Mất kết nối webcam hoặc luồng video kết thúc.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)
        current_time = time.time()

        for face in faces:
            # Kiểm tra thời gian để tránh nhận diện quá nhanh
            if current_time - last_recognition_time < recognition_interval:
                continue
            
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            
            # Vẽ hình chữ nhật quanh khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Cắt ảnh khuôn mặt để xử lý
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = rgb[y1:y2, x1:x2]
            
            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                print("[WARNING] Ảnh crop khuôn mặt bị rỗng, bỏ qua.")
                continue
            
            # Lưu tạm ảnh crop để DeepFace xử lý
            temp_face_path = os.path.join(SAVE_DIR, "temp_face_crop.jpg")
            cv2.imwrite(temp_face_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)) 
            
            # Trích xuất vector đặc trưng từ khuôn mặt vừa crop
            test_vector = extract_feature(temp_face_path)
            
            # Xóa ảnh tạm sau khi sử dụng
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)

            label = "Unknown"
            if test_vector is not None:
                matched_name = compare_with_all_references(test_vector)
                if matched_name:
                    label = matched_name
                    # Kiểm tra thời gian để tránh ghi điểm danh liên tục cho cùng một người
                    if matched_name not in last_processed_person or \
                       (current_time - last_processed_person[matched_name]) > processing_interval_seconds:
                        
                        id_nhan_vien = f"ID_{matched_name.replace(' ', '_')}"
                        thoi_gian = time.strftime("%Y-%m-%d %H:%M:%S")
                        data_entry = {
                            "ID_NhanVien": id_nhan_vien,
                            "Ten_NhanVien": matched_name,
                            "ThoiGian": thoi_gian,
                            "TrangThai": "Check-in",
                            "GhiChu": ""
                        }

                        save_to_csv(data_entry) # Lưu vào CSV

                        last_processed_person[matched_name] = current_time
            
            # Hiển thị tên (hoặc "Unknown") lên khung hình
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
            
            last_recognition_time = current_time # Cập nhật thời gian nhận diện cuối cùng

        # Hiển thị khung hình từ webcam
        cv2.imshow("Face Recognition", frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng webcam và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
