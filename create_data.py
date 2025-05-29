import os
import sys
from pathlib import Path
from glob import glob


# Đảm bảo thêm đường dẫn `main/` vào sys.path
BASE_DIR = Path(__file__).resolve().parent
MAIN_DIR = BASE_DIR / "main"
sys.path.append(str(MAIN_DIR))  # Bắt buộc!

# Giờ mới import
from Feature_extractor.feature_extractor import extract_features_to_vectors

# Cấu hình
IMAGE_DIR = "reference_images"         # Thư mục ảnh mẫu
OUTPUT_DIR = "reference_vectors"       # Nơi lưu vector .npy
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "dlib"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Duyệt qua từng ảnh
image_paths = glob(os.path.join(IMAGE_DIR, "*.*"))
if not image_paths:
    print(f"[x] Không có ảnh trong {IMAGE_DIR}")
    sys.exit()

for img_path in image_paths:
    name = Path(img_path).stem  # Lấy tên file ảnh không có đuôi
    # Trích xuất tất cả vector trong folder IMAGE_DIR
vectors = extract_features_to_vectors(IMAGE_DIR, OUTPUT_DIR, MODEL_NAME, DETECTOR_BACKEND)
if vectors is not None:
    print(f"[✓] Đã trích xuất vectors cho folder {IMAGE_DIR}")
else:
    print("[x] Trích xuất vectors thất bại")
