import numpy as np
import os

def check():
    if not os.path.exists('y_train_data.npy'):
        print("❌ Không tìm thấy file y_train_data.npy")
        return

    y = np.load('y_train_data.npy')
    unique, counts = np.unique(y, return_counts=True)
    
    print("📊 THỐNG KÊ DỮ LIỆU CỦA BẠN:")
    print(f"Tổng số mẫu: {len(y)}")
    print("-" * 30)
    print("Phân bố các lớp (Class Distribution):")
    
    has_data = False
    for label, count in zip(unique, counts):
        print(f"  - Class {label}: {count} mẫu ({count/len(y)*100:.1f}%)")
        if count > 0: has_data = True
        
    print("-" * 30)
    
    if len(unique) == 1:
        print("🚨 CẢNH BÁO ĐỎ: Bạn chỉ có 1 loại nhãn duy nhất!")
        print("   -> Mô hình sẽ LUÔN LUÔN dự đoán ra nhãn này bất kể đầu vào là gì.")
        print("   -> Giải pháp: Kiểm tra lại dataset, đảm bảo có đủ file .txt chứa số 1 (Swimming) và 2 (Out of water).")
    elif len(unique) < 3:
        print("⚠️ CẢNH BÁO VÀNG: Bạn bị thiếu lớp dữ liệu (Cần đủ 3 lớp 0, 1, 2).")
    else:
        print("✅ Dữ liệu có vẻ ổn về mặt số lượng lớp.")

if __name__ == "__main__":
    check()
    