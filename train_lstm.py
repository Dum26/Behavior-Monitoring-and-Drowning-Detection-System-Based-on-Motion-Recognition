import numpy as np
import os
import sys

# Kiểm tra xem thư viện TensorFlow đã được cài đặt chưa
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    print(f"✅ Đã tìm thấy TensorFlow phiên bản: {tf.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài đặt TensorFlow.")
    print("👉 Hãy chạy lệnh: pip install tensorflow")
    sys.exit(1)

# --- 1. CẤU HÌNH THAM SỐ ---
# Các thông số này PHẢI khớp với file make_data_yolo.py
SEQUENCE_LENGTH = 30    # Độ dài chuỗi (số frames)
FEATURE_VECTOR_SIZE = 51 # Kích thước vector đặc trưng (17 khớp * 3 thông số)

# Tên file model sẽ lưu
MODEL_SAVE_PATH = 'action_classifier_lstm.h5'
BEST_MODEL_SAVE_PATH = 'best_action_classifier.h5'

def build_lstm_model(input_shape, num_classes):
    """
    Xây dựng kiến trúc mô hình LSTM.
    Kiến trúc này được tinh chỉnh để hoạt động tốt với dữ liệu xương khớp (skeleton).
    """
    model = Sequential()

    # Layer LSTM 1: Trả về chuỗi (return_sequences=True) để lớp sau tiếp tục xử lý
    # Input shape: (30, 51)
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Dropout giúp chống học vẹt (Overfitting)

    # Layer LSTM 2: Trả về kết quả tóm tắt cuối cùng (return_sequences=False)
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    # Layer Dense: Lớp nơ-ron kết nối đầy đủ để học các đặc trưng phi tuyến tính
    model.add(Dense(units=32, activation='relu'))

    # Output Layer: Trả về xác suất cho từng lớp (dùng Softmax cho phân loại đa lớp)
    model.add(Dense(units=num_classes, activation='softmax'))

    # Biên dịch mô hình
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model():
    print("\n--- BƯỚC 1: TẢI DỮ LIỆU ---")
    
    # Kiểm tra file dữ liệu có tồn tại không
    if not os.path.exists('X_train_data.npy') or not os.path.exists('y_train_data.npy'):
        print("❌ LỖI: Không tìm thấy file dữ liệu .npy!")
        print("👉 Bạn CẦN chạy file 'make_data_yolo.py' trước để tạo dữ liệu từ ảnh.")
        return

    # Tải dữ liệu từ file .npy
    print("⏳ Đang đọc file X_train_data.npy và y_train_data.npy...")
    X = np.load('X_train_data.npy')
    y = np.load('y_train_data.npy')

    # In thông tin dữ liệu để kiểm tra
    print(f"   - Tổng số mẫu dữ liệu: {X.shape[0]}")
    print(f"   - Kích thước chuỗi (Frames): {X.shape[1]}")
    print(f"   - Đặc trưng mỗi frame: {X.shape[2]}")

    # Tự động xác định số lượng lớp (Classes) từ dữ liệu
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"   - Số lượng lớp tìm thấy: {num_classes}")
    print(f"   - Các nhãn ID: {unique_classes}")

    print("\n--- BƯỚC 2: CHUẨN BỊ TRAINING ---")
    
    # Chuyển đổi nhãn sang dạng One-hot vector 
    # Ví dụ: nếu có 3 lớp, nhãn 1 sẽ thành [0, 1, 0]
    y_one_hot = to_categorical(y, num_classes=num_classes)

    # Chia dữ liệu: 80% để học (Train), 20% để kiểm tra (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    print(f"   - Dữ liệu dùng để Train: {X_train.shape[0]} mẫu")
    print(f"   - Dữ liệu dùng để Test: {X_test.shape[0]} mẫu")

    # Xây dựng mô hình
    model = build_lstm_model((SEQUENCE_LENGTH, FEATURE_VECTOR_SIZE), num_classes)
    model.summary() # In cấu trúc mô hình ra màn hình

    # Thiết lập các chiến lược Training (Callbacks)
    callbacks = [
        # Dừng sớm nếu sau 15 vòng (patience) mà độ lỗi không giảm thêm -> Tiết kiệm thời gian
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        
        # Luôn lưu lại phiên bản mô hình tốt nhất (có val_loss thấp nhất) trong quá trình chạy
        ModelCheckpoint(BEST_MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("\n--- BƯỚC 3: BẮT ĐẦU HUẤN LUYỆN (TRAINING) ---")
    print("🚀 Quá trình này có thể mất vài phút tùy vào lượng dữ liệu...")
    
    # Bắt đầu training
    history = model.fit(
        X_train, y_train,
        epochs=100,         # Số vòng lặp tối đa (nếu không bị dừng sớm)
        batch_size=32,      # Số lượng mẫu học mỗi lần cập nhật trọng số
        validation_data=(X_test, y_test), # Dữ liệu để kiểm tra chéo
        callbacks=callbacks
    )

    print("\n--- BƯỚC 4: ĐÁNH GIÁ KẾT QUẢ ---")
    # Đánh giá độ chính xác cuối cùng trên tập Test
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"📊 Kết quả trên tập Test:")
    print(f"   - Loss: {loss:.4f}")
    print(f"   - Accuracy (Độ chính xác): {accuracy*100:.2f}%")

    # Lưu mô hình cuối cùng vào file .h5
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Đã lưu mô hình thành công tại: {MODEL_SAVE_PATH}")
    print(f"👉 Bây giờ bạn có thể chạy file 'detect_people.py' hoặc 'run_demo.py' để kiểm tra kết quả!")

if __name__ == '__main__':
    train_model()