import cv2
import numpy as np
import argparse
import time
from collections import deque, Counter
from ultralytics import YOLO
import sys
import os

# Cố gắng import TensorFlow để load model LSTM
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ LỖI: Chưa cài TensorFlow. Chạy: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# --- CẤU HÌNH HỆ THỐNG ---
POSE_MODEL_PATH = 'yolov8n-pose.pt' 
LSTM_MODEL_PATH = 'action_classifier_lstm.h5' 

# ⚠️ QUAN TRỌNG: Nếu nhận diện bị NGƯỢC (Bơi thành Đuối), hãy thử đổi chỗ tên trong danh sách này
LABELS = {
    0: 'DROWNING', 
    1: 'SWIMMING', 
    2: 'OUT_OF_WATER'
}

# Cấu hình màu sắc (BGR)
COLORS = {
    'DROWNING': (0, 0, 255),      # ĐỎ
    'SWIMMING': (0, 255, 0),      # XANH LÁ
    'OUT_OF_WATER': (0, 255, 255),# VÀNG
    'Unknown': (128, 128, 128)    # XÁM
}

# Cấu hình Voting
HISTORY_LENGTH = 15
LSTM_CONFIDENCE_THRESHOLD = 0.6 # Chỉ tin nếu xác suất > 60%
voting_buffer = {} 

def run_system(source):
    print("⏳ Đang tải YOLOv8-Pose...")
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
    except Exception as e:
        print(f"❌ Lỗi tải YOLO: {e}")
        return
    
    action_model = None
    if TENSORFLOW_AVAILABLE:
        print(f"⏳ Đang tải LSTM Model: {LSTM_MODEL_PATH}...")
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                action_model = load_model(LSTM_MODEL_PATH)
                print("✅ Đã tải xong LSTM!")
            except Exception as e:
                 print(f"❌ Lỗi khi load file h5: {e}")
                 return
        else:
            print("❌ Không tìm thấy file model. Hãy chạy train_lstm.py trước!")
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Không mở được nguồn: {source}")
        return

    print("🚀 HỆ THỐNG BẮT ĐẦU! Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Tracking với YOLO
        results = pose_model.track(frame, persist=True, verbose=False, conf=0.5)
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for i, track_id in enumerate(track_ids):
                # --- A. TRÍCH XUẤT ---
                kpts_xyn = results[0].keypoints.xyn.cpu().numpy()[i]
                kpts_conf = results[0].keypoints.conf.cpu().numpy()[i]
                vector_51 = np.column_stack((kpts_xyn, kpts_conf)).flatten()
                input_sequence = np.tile(vector_51, (30, 1)).reshape(1, 30, 51)
                
                final_label = "Scanning..."
                confidence = 0.0
                probs = [] # Để hiển thị debug
                
                # --- B. LSTM DỰ ĐOÁN ---
                if action_model:
                    pred = action_model.predict(input_sequence, verbose=0)[0]
                    current_label_idx = np.argmax(pred)
                    current_conf = pred[current_label_idx]
                    probs = pred # Lưu lại để vẽ
                    
                    # Chỉ thêm vào bộ nhớ nếu độ tin cậy đủ cao
                    if current_conf > LSTM_CONFIDENCE_THRESHOLD:
                        if track_id not in voting_buffer:
                            voting_buffer[track_id] = deque(maxlen=HISTORY_LENGTH)
                        voting_buffer[track_id].append(current_label_idx)
                    
                    # Voting
                    if track_id in voting_buffer and len(voting_buffer[track_id]) > 0:
                        votes = Counter(voting_buffer[track_id])
                        winner_idx, count = votes.most_common(1)[0]
                        # Nếu nhãn thắng cuộc chiếm ưu thế
                        if count >= 1: # Lấy kết quả phổ biến nhất
                            final_label = LABELS.get(winner_idx, "Unknown")
                            confidence = pred[winner_idx] # Lấy conf của frame hiện tại cho nhãn đó

                # --- C. VẼ KẾT QUẢ ---
                box = results[0].boxes.xyxy.cpu().numpy()[i].astype(int)
                color = COLORS.get(final_label, (255, 255, 255))
                
                # Khung người
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Nhãn chính
                text = f"ID:{track_id} {final_label}"
                if action_model and final_label != "Scanning...":
                     text += f" ({confidence:.2f})"
                
                # Vị trí vẽ chữ
                text_y = box[1] - 10 if box[1] - 10 > 20 else box[1] + 20
                cv2.putText(frame, text, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # --- D. VẼ DEBUG (XÁC SUẤT CHI TIẾT) ---
                # Giúp bạn biết tại sao nó nhận sai
                if len(probs) > 0:
                    dy = 25
                    for idx, prob in enumerate(probs):
                        label_name = LABELS.get(idx, str(idx))
                        debug_text = f"{label_name}: {prob:.2f}"
                        # Vẽ chữ nhỏ bên cạnh hộp
                        cv2.putText(frame, debug_text, (box[2] + 5, box[1] + dy * (idx + 1)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Vẽ xương khớp
                kpts_pixel = results[0].keypoints.xy.cpu().numpy()[i]
                for kp in kpts_pixel:
                    x, y = int(kp[0]), int(kp[1])
                    if x > 0 and y > 0:
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        cv2.imshow('Drowning Detection (Debug Mode)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Path to video file or 0 for Webcam')
    args = parser.parse_args()
    try:
        src = int(args.source)
    except ValueError:
        src = args.source
    run_system(src)