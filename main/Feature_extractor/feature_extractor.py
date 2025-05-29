import os
import numpy as np
from deepface import DeepFace
import logging

INPUT_DIR = "../Face_detection/outputs"
OUTPUT_DIR = "output_vector"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = True

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Chuyển level về WARNING, chỉ hiển thị cảnh báo và lỗi, bỏ log INFO thường

def extract_features_to_vectors(input_dir, output_dir, model_name, detector_backend, enforce_detection=True):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục output: {output_dir}")
        except OSError as e:
            print(f"Không thể tạo thư mục output {output_dir}: {e}")
            return

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) 
                   if os.path.isfile(os.path.join(input_dir, f)) and 
                   f.lower().endswith(valid_extensions) and not f.startswith('.')]

    if not image_files:
        print(f"Không tìm thấy ảnh hợp lệ trong thư mục {input_dir}")
        return

    processed_count = 0
    error_count = 0

    for filename in image_files:
        file_path = os.path.join(input_dir, filename)
        try:
            embedding_objs = DeepFace.represent(
                img_path=file_path,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
                align=True
            )
            if embedding_objs and len(embedding_objs) > 0:
                vector = embedding_objs[0]['embedding']
                base_filename = os.path.splitext(filename)[0]
                output_filename = f"{base_filename}.npy"
                output_path = os.path.join(output_dir, output_filename)

                np.save(output_path, vector)
                print(f"[Thành công] Lưu vector cho {filename}")
                processed_count += 1
            else:
                logging.warning(f"Không trích xuất được vector từ ảnh: {filename}")
                error_count += 1
        except ValueError as ve:
            logging.warning(f"Lỗi xử lý ảnh {filename}: {ve}")
            error_count += 1
        except Exception as e:
            logging.error(f"Lỗi khi xử lý ảnh {filename}: {e}")
            error_count += 1

    print("--- Kết thúc ---")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Thư mục input '{INPUT_DIR}' không tồn tại!")
    else:
        extract_features_to_vectors(INPUT_DIR, OUTPUT_DIR, MODEL_NAME, DETECTOR_BACKEND, ENFORCE_DETECTION)
