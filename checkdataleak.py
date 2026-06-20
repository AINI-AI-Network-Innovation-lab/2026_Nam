import numpy as np
import os

FILE_ORIGINAL = "trained_model_weights1.5B.npz"
FILE_RETRAIN = "retrained_model_weights1.5B.npz"

def check_diff():
    if not os.path.exists(FILE_ORIGINAL) or not os.path.exists(FILE_RETRAIN):
        print("❌ Không tìm thấy đủ file để so sánh.")
        return

    print(f"Comparing:\n1. {FILE_ORIGINAL}\n2. {FILE_RETRAIN}")
    
    w1 = np.load(FILE_ORIGINAL)
    w2 = np.load(FILE_RETRAIN)
    
    # Lấy thử lớp đầu tiên (arr_0) để so sánh
    layer1_orig = w1['arr_0']
    layer1_retr = w2['arr_0']
    
    diff = np.sum(np.abs(layer1_orig - layer1_retr))
    
    print("-" * 30)
    if diff == 0:
        print("😱 KẾT QUẢ: HAI FILE GIỐNG HỆT NHAU 100%!")
        print("-> Nguyên nhân: Code Retrain chưa lưu được weight mới, hoặc bạn copy nhầm file.")
    else:
        print(f"✅ KẾT QUẢ: Hai file có khác nhau (Diff score: {diff})")
        print("-> Nguyên nhân: Có thể do lỗi load thứ tự (Sorting Bug) trong file Evaluate.")

if __name__ == "__main__":
    check_diff()
