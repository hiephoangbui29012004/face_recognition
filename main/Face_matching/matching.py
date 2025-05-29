import os
import numpy as np

# Cấu hình
REF_VEC_DIR = "../reference_vectors"            # Thư mục chứa vector mẫu đã có sẵn (CSDL)
TEST_VEC_DIR = "../Feature_extractor/output_vector"  # Thư mục chứa vector mới từ webcam
THRESHOLD = 0.8

# Hàm tính khoảng cách cosine
def find_cosine_distance(vec1, vec2):
    a = np.dot(vec1, vec2)
    b = np.linalg.norm(vec1)
    c = np.linalg.norm(vec2)
    return 1 - (a / (b * c))

# Hàm so sánh 1 vector với toàn bộ mẫu trong CSDL
def compare_with_all_references(test_vector_or_path):
    import numpy as np
    if isinstance(test_vector_or_path, str):
        test_vector = np.load(test_vector_or_path).astype(np.float32)
    else:
        test_vector = test_vector_or_path.astype(np.float32)

    for vec_file in os.listdir(REF_VEC_DIR):
        if vec_file.endswith(".npy"):
            ref_vector = np.load(os.path.join(REF_VEC_DIR, vec_file)).astype(np.float32)
            dist = find_cosine_distance(ref_vector, test_vector)
            if dist <= THRESHOLD:
                name = os.path.splitext(vec_file)[0]
                print(f"[=] Khớp với {name} (distance={dist:.4f})")
                return name, dist
    return None, None

def get_latest_test_vector():
    npy_files = [f for f in os.listdir(TEST_VEC_DIR) if f.endswith(".npy")]
    if not npy_files:
        print("[!] Không tìm thấy vector nào trong output.")
        return None

    npy_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEST_VEC_DIR, f)), reverse=True)
    latest_path = os.path.join(TEST_VEC_DIR, npy_files[0])
    print(f"[i] Vector mới nhất: {latest_path}")
    return np.load(latest_path).astype(np.float32)  # Ép kiểu float ngay khi load


def match_latest_vector():
    test_vector = get_latest_test_vector()
    if test_vector is None:
        return None, None
    return compare_with_all_references(test_vector)


# Hàm chính
def main():
    name, distance = match_latest_vector()
    if name:
        print(f"[✓] Nhận diện: {name} (distance = {distance:.4f})")
    else:
        print("[x] Không nhận diện được khuôn mặt (Unknown)")


if __name__ == "__main__":
    main()
