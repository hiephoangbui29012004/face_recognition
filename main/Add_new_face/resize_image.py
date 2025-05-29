import os
import cv2
import numpy as np
from PIL import Image

# Đường dẫn nguồn và nơi lưu của macbook
path = 'D:/Project VXL/dataset'
path_save = 'D:/Project VXL/dataset_save'

# Kích thước resize
newsize = (200, 200)

# Chọn chế độ lưu: 'gray' hoặc 'rgb'
SAVE_MODE = 'rgb'  # Đổi thành 'gray' nếu muốn lưu grayscale

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, path)
                save_path = os.path.join(path_save, rel_path)

                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Đọc ảnh với PIL và chuyển về RGB
                pil_img = Image.open(input_path).convert('RGB')
                img = np.array(pil_img)

                if SAVE_MODE == 'gray':
                    # Chuyển về grayscale 8-bit
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    resized_img = cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)
                    # Đảm bảo ảnh là 8-bit
                    resized_img = resized_img.astype(np.uint8)
                else:
                    # Giữ nguyên RGB
                    resized_img = cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)
                    resized_img = resized_img.astype(np.uint8)

                # Lưu ảnh
                success = cv2.imwrite(save_path, resized_img)
                if success:
                    # Kiểm tra ảnh đã lưu
                    check_img = cv2.imread(save_path)
                    print(f" Đã resize và lưu: {save_path}")
                    print(f" Shape ảnh đã lưu: {check_img.shape}")
                    print(f" Kiểu dữ liệu: {check_img.dtype}")
                else:
                    print(f" Không thể lưu: {save_path}")

            except Exception as e:
                print(f" Lỗi xử lý ảnh {file}: {e}")

print("\n Hoàn tất resize ảnh!")
