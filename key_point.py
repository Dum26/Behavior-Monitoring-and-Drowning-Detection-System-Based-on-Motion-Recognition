import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import sys
import os

# --- 1. CẤU HÌNH MÔ HÌNH POSE ---
POSE_MODEL_PATH = 'yolov8n-pose.pt' 
TARGET_CLASS_ID = 0 # Lớp 'person'
CONFIDENCE_THRESHOLD = 0.5 

try:
    # Tải mô hình YOLOv8-Pose
    pose_model = YOLO(POSE_MODEL_PATH) 
    print(f"✅ Đã tải mô hình Pose Estimation: {POSE_MODEL_PATH}")
except Exception as e:
    print(f"❌ LỖI TẢI MÔ HÌNH: Không tìm thấy {POSE_MODEL_PATH}. Lỗi: {e}")
    sys.exit(1)


def estimate_pose_and_extract_keypoints(frame):
    """
    Thực hiện ước lượng tư thế, vẽ các khớp, và trích xuất vector đặc trưng 51 chiều.
    
    Returns:
        frame: Khung hình đã vẽ khớp và Bounding Box.
        keypoints_features: Danh sách các vector đặc trưng 51 chiều (cho LSTM).
    """
    keypoints_features = [] 
    
    # 1. Chạy mô hình dự đoán
    results = pose_model(frame, classes=[TARGET_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    for result in results:
        # Lấy các thành phần riêng biệt từ Keypoints object
        
        # [N_People, 17, 2] -> x, y chuẩn hóa (0.0 - 1.0)
        keypoints_xyn = result.keypoints.xyn.cpu().numpy()      
        # [N_People, 17, 2] -> x, y pixel
        keypoints_xy = result.keypoints.xy.cpu().numpy()        
        # [N_People, 17] -> c độ tin cậy
        keypoints_conf = result.keypoints.conf.cpu().numpy()    

        # Thêm chiều mới cho conf để ghép (dstack yêu cầu kích thước [N, 17, 1])
        keypoints_conf_expanded = keypoints_conf[:, :, np.newaxis] 

        # Ghép xyn và conf lại để có vector chuẩn hóa [x_norm, y_norm, conf]
        # Kích thước: [N_People, 17, 3]
        keypoints_norm_combined = np.dstack((keypoints_xyn, keypoints_conf_expanded))
        
        # Ghép xy và conf lại để có vector pixel [x_pixel, y_pixel, conf] (dùng cho vẽ)
        # Kích thước: [N_People, 17, 3]
        keypoints_pixel_data = np.dstack((keypoints_xy, keypoints_conf_expanded))

        for i in range(keypoints_norm_combined.shape[0]):
            kpts_norm = keypoints_norm_combined[i]
            kpts_pixel = keypoints_pixel_data[i]
            
            # --- 🎯 TRÍCH XUẤT ĐẶC TRƯNG CHUỖI THỜI GIAN (51 chiều) ---
            
            # TODO: NÊN THỰC HIỆN CHUẨN HÓA TƯƠNG ĐỐI (Ví dụ: so với khớp hông) Ở ĐÂY
            # Hiện tại, ta sử dụng vector thô (x_norm, y_norm, conf)
            
            # Làm phẳng mảng 17x3 để có vector 51 chiều: [x1, y1, c1, x2, y2, c2, ...]
            feature_vector = kpts_norm.flatten() 
            keypoints_features.append(feature_vector)
            
            # --- VẼ TRỰC QUAN (Visualisation) ---
            
            # 1. Vẽ Khớp
            for kp in kpts_pixel:
                x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                if conf > 0.5: 
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1) # Chấm vàng
            
            # 2. Vẽ Bounding Box
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Xanh dương

    return frame, keypoints_features

## 🎥 HÀM CHÍNH XỬ LÝ LUỒNG VIDEO/CAMERA

def process_video_stream(source):
    """
    Xử lý luồng đầu vào, gọi hàm ước lượng tư thế và hiển thị kết quả.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ LỖI: Không thể mở nguồn video/camera: {source}.")
        if isinstance(source, str) and not os.path.exists(source):
            print(f"❌ LỖI: Đường dẫn file '{source}' không tồn tại.")
        return

    print(f"🎬 Bắt đầu Ước lượng Tư thế từ nguồn: {source}. Nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video đã kết thúc hoặc lỗi đọc frame.")
            break

        processed_frame, current_keypoints_features = estimate_pose_and_extract_keypoints(frame)

        # TO DO: Ở bước này, current_keypoints_features sẽ được đưa vào bộ đệm (Buffer)
        # để cung cấp chuỗi thời gian cho mô hình LSTM (Bài toán 3)

        cv2.imshow('YOLOv8 Pose Estimation (Keypoints)', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 2. CHẠY CHƯƠNG TRÌNH VỚI ARGPARSE ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimator")
    parser.add_argument('--source', type=str, default='0', 
                        help='Input source: 0 for default camera, or path to a video file.')
    
    args = parser.parse_args()
    
    # Xử lý input: số (camera ID) hay chuỗi (đường dẫn file)
    try:
        source_id = int(args.source)
        process_video_stream(source_id)
    except ValueError:
        process_video_stream(args.source)
# python key_point.py --source "D:\thi_nghiem_AI\dataset\video\drowning_1.mp4"