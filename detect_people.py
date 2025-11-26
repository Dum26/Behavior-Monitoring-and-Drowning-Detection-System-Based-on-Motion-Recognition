import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import sys
import os
import time

# Thư viện cho LSTM (TensorFlow/Keras)
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ Cảnh báo: Không tìm thấy thư viện TensorFlow. Chức năng phân loại hành động (LSTM) sẽ bị vô hiệu hóa.")
    TENSORFLOW_AVAILABLE = False


# --- 1. CẤU HÌNH HỆ THỐNG ---
POSE_MODEL_PATH = 'yolov8n-pose.pt' 
LSTM_MODEL_PATH = 'action_classifier_lstm.h5' # File LSTM đã huấn luyện
TARGET_CLASS_ID = 0 
CONFIDENCE_THRESHOLD = 0.5 

# Cấu hình cho Bài toán 3 (LSTM)
SEQUENCE_LENGTH = 30 # Độ dài chuỗi (số khung hình)
NUM_KEYPOINTS = 17 
FEATURE_VECTOR_SIZE = NUM_KEYPOINTS * 3 # 51 chiều (x, y, conf)
# ⚠️ QUAN TRỌNG: Thứ tự nhãn phải khớp với lúc train (xem log train_lstm.py để chắc chắn)
# Giả sử: 0=Drowning, 1=Swimming, 2=Out_of_water (Cần kiểm tra lại file y_train_data.npy nếu nghi ngờ)
LABELS = {0: 'DROWNING', 1: 'SWIMMING', 2: 'OUT_OF_WATER'}

# Khởi tạo Bộ đệm Chuỗi và Mô hình
sequence_buffers = {} # {track_id: list_of_keypoint_vectors}
pose_model = None
action_model = None


try:
    pose_model = YOLO(POSE_MODEL_PATH) 
    print(f"✅ Đã tải mô hình Pose Estimation: {POSE_MODEL_PATH}")
    
    if TENSORFLOW_AVAILABLE:
        # Cố gắng tải mô hình LSTM
        if os.path.exists(LSTM_MODEL_PATH):
            action_model = load_model(LSTM_MODEL_PATH)
            print(f"✅ Đã tải mô hình LSTM: {LSTM_MODEL_PATH}")
        else:
            print(f"❌ Cảnh báo: Không tìm thấy file LSTM '{LSTM_MODEL_PATH}'. Chỉ chạy Ước lượng Tư thế.")
    
except Exception as e:
    print(f"❌ LỖI KHỞI TẠO MÔ HÌNH: {e}")
    sys.exit(1)


def pre_process_keypoints(kpts_norm_combined):
    """
    Tiền xử lý: Chuẩn hóa tương đối (Relative Normalization) và xử lý dữ liệu bị thiếu.
    Sử dụng khớp hông (ID 12) làm gốc chuẩn hóa.
    :param kpts_norm_combined: mảng NumPy [17, 3] (x_norm, y_norm, conf)
    :return: vector đặc trưng 51 chiều đã xử lý
    """
    # Khớp gốc để chuẩn hóa (Ví dụ: Khớp hông phải ID 12)
    ROOT_KEYPOINT_ID = 12 
    
    # Kiểm tra độ tin cậy của khớp gốc
    if kpts_norm_combined[ROOT_KEYPOINT_ID, 2] < 0.1:
        # Nếu khớp gốc không đáng tin cậy, trả về vector 0
        return np.zeros(FEATURE_VECTOR_SIZE)
    
    root_x = kpts_norm_combined[ROOT_KEYPOINT_ID, 0]
    root_y = kpts_norm_combined[ROOT_KEYPOINT_ID, 1]
    
    processed_kpts = kpts_norm_combined.copy()

    # Chuẩn hóa Tương đối (Dịch chuyển)
    for i in range(NUM_KEYPOINTS):
        # x' = x - x_root, y' = y - y_root
        processed_kpts[i, 0] = kpts_norm_combined[i, 0] - root_x 
        processed_kpts[i, 1] = kpts_norm_combined[i, 1] - root_y 
        
        # Xử lý thiếu dữ liệu (Zeroing): Nếu conf quá thấp (< 0.1), đặt tọa độ về 0
        if kpts_norm_combined[i, 2] < 0.1:
             processed_kpts[i, 0] = 0.0
             processed_kpts[i, 1] = 0.0

    return processed_kpts.flatten() # Vector 51 chiều


def detect_and_classify(frame):
    """
    Thực hiện Pose Estimation, Tiền xử lý, Tạo chuỗi và Phân loại (nếu có LSTM).
    """
    
    # 1. Chạy mô hình Pose Estimation VỚI TRACKING (persist=True)
    # persist=True giúp duy trì ID của người qua các frame
    results = pose_model.track(frame, classes=[TARGET_CLASS_ID], conf=CONFIDENCE_THRESHOLD, persist=True, verbose=False)
    
    # Kiểm tra xem có phát hiện được ai không
    if results[0].boxes.id is None:
        return frame

    # Lấy danh sách ID của các đối tượng trong khung hình
    track_ids = results[0].boxes.id.int().cpu().tolist()
    
    # Lặp qua từng người dựa trên ID của họ
    for i, track_id in enumerate(track_ids):
        # Lấy dữ liệu Keypoint
        keypoints_xyn = results[0].keypoints.xyn.cpu().numpy()[i]
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()[i]
        keypoints_conf = results[0].keypoints.conf.cpu().numpy()[i]
        
        # Mở rộng chiều cho conf để có kích thước (17, 1)
        keypoints_conf_expanded = keypoints_conf[:, np.newaxis] 

        # Ghép lại thành vector (17, 3)
        keypoints_norm_combined = np.hstack((keypoints_xyn, keypoints_conf_expanded)) # [17, 3]
        keypoints_pixel_data = np.hstack((keypoints_xy, keypoints_conf_expanded))     # [17, 3] (dùng để vẽ)
        
        # 2. Tiền xử lý và trích xuất Đặc trưng
        feature_vector_51 = pre_process_keypoints(keypoints_norm_combined)
        
        # 3. TẠO BỘ ĐỆM CHUỖI (Sequence Buffer) CHO TỪNG ID
        if track_id not in sequence_buffers:
            sequence_buffers[track_id] = []
            
        sequence_buffers[track_id].append(feature_vector_51)
        
        # Cắt bớt phần tử cũ nếu chuỗi dài hơn SEQUENCE_LENGTH
        if len(sequence_buffers[track_id]) > SEQUENCE_LENGTH:
            sequence_buffers[track_id] = sequence_buffers[track_id][-SEQUENCE_LENGTH:] 
        
        action_label = "Thinking..."
        confidence_score = 0.0
        color = (255, 165, 0) # Màu Cam
        
        # 4. PHÂN LOẠI HÀNH ĐỘNG (Dự đoán bằng LSTM)
        if action_model and len(sequence_buffers[track_id]) == SEQUENCE_LENGTH:
            sequence_data = np.array(sequence_buffers[track_id], dtype=np.float32)
            input_sequence = np.expand_dims(sequence_data, axis=0) # Thêm batch dimension (1, 30, 51)
            
            # Dự đoán
            prediction = action_model.predict(input_sequence, verbose=0)[0]
            predicted_class_index = np.argmax(prediction)
            
            # Kiểm tra xem index có nằm trong LABELS không
            if predicted_class_index in LABELS:
                action_label = LABELS[predicted_class_index]
            else:
                action_label = f"Class {predicted_class_index}"
                
            confidence_score = prediction[predicted_class_index]
            
            # 5. HIỂN THỊ CẢNH BÁO
            color = (0, 255, 0) # Xanh lá (Bình thường/Out of water)
            if action_label == 'DROWNING':
                color = (0, 0, 255) # Đỏ (Cảnh báo nguy hiểm)
            elif action_label == 'SWIMMING':
                 color = (255, 255, 0) # Vàng

        # --- VẼ TRỰC QUAN ---
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        x1, y1, x2, y2 = boxes[i]
        
        # Vẽ Bounding Box và Nhãn kèm ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text_label = f"ID:{track_id} {action_label}"
        if action_model and action_label != "Thinking...":
            text_label += f" ({confidence_score:.2f})"
            
        cv2.putText(frame, text_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Vẽ Keypoints 
        for kp in keypoints_pixel_data:
            x, y, conf = int(kp[0]), int(kp[1]), kp[2]
            if conf > 0.5: 
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1) 
                
    return frame

def process_video_stream(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ LỖI: Không thể mở nguồn video/camera: {source}.")
        return

    print(f"🎬 Bắt đầu Giám sát Đuối nước từ nguồn: {source}. Nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video đã kết thúc.")
            break

        processed_frame = detect_and_classify(frame)

        cv2.imshow('Drowning Detector (YOLOv8-Pose + LSTM)', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8-LSTM Drowning Detector")
    parser.add_argument('--source', type=str, default='0', 
                        help='Input source: 0 for default camera, or path to a video file.')
    
    args = parser.parse_args()
    
    # Xử lý input: số (camera ID) hay chuỗi (đường dẫn file)
    try:
        source_id = int(args.source)
        process_video_stream(source_id)
    except ValueError:
        process_video_stream(args.source)