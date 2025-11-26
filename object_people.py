import cv2
from ultralytics import YOLO
import argparse
import sys
import os # Import os để kiểm tra đường dẫn file

# --- Cấu hình Mô hình và Nhận dạng ---
# Đảm bảo đường dẫn này đúng với vị trí file yolov8n.pt của bạn
MODEL_PATH = 'models/yolov8n.pt' 

# Lớp 'person' trong bộ dữ liệu COCO là class ID 0
TARGET_CLASS_ID = 0 
CLASS_NAME = 'person'
CONFIDENCE_THRESHOLD = 0.5 

# Tải mô hình YOLOv8
try:
    model = YOLO(MODEL_PATH) 
except Exception as e:
    print("----------------------------------------------------------------")
    print(f"LỖI TẢI MÔ HÌNH: Không tìm thấy {MODEL_PATH}")
    print("Vui lòng đảm bảo file 'yolov8n.pt' nằm trong thư mục 'models'.")
    print(f"Chi tiết lỗi: {e}")
    print("----------------------------------------------------------------")
    sys.exit(1) # Thoát chương trình nếu không tải được mô hình

def detect_people_and_draw_bbox(frame):
    """
    Thực hiện phát hiện đối tượng người trên một khung hình và vẽ bounding box.
    """
    # 1. Chạy mô hình dự đoán
    # Sử dụng 'classes=0' và 'conf' trực tiếp trong hàm model()
    results = model(frame, classes=[TARGET_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Duyệt qua các kết quả dự đoán
    for result in results:
        # Lấy thông tin bounding box và độ tin cậy
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) 
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            
            # Vẽ bounding box (màu xanh lá cây)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ghi nhãn lên hộp (tên lớp và độ tin cậy)
            label = f"{CLASS_NAME}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return frame

## 🎥 Hàm Chính để Xử lý Luồng Camera/Video

def process_video_stream(source):
    """
    Khởi động luồng camera hoặc đọc từ file video.
    :param source: 0 (hoặc số khác) cho camera, hoặc đường dẫn file video.
    """
    # Khởi tạo VideoCapture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở nguồn video/camera: {source}.")
        
        # Nếu source là file, kiểm tra xem file có tồn tại không
        if isinstance(source, str) and not os.path.exists(source):
            print(f"LỖI: Đường dẫn file '{source}' không tồn tại.")
        return

    print(f"Bắt đầu phát hiện người từ nguồn: {source}. Nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã kết thúc video hoặc lỗi đọc frame.")
            break

        # Thực hiện phát hiện và vẽ bounding box
        processed_frame = detect_people_and_draw_bbox(frame)

        # Hiển thị kết quả
        cv2.imshow('YOLOv8 People Detection', processed_frame)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- CHẠY CHƯƠNG TRÌNH VỚI ARGPARSE ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 People Detector")
    
    parser.add_argument('--source', type=str, default='0', 
                        help='Input source: 0 for default camera, or path to a video file.')
    
    args = parser.parse_args()

    # Xử lý input: số (camera ID) hay chuỗi (đường dẫn file)
    try:
        source_id = int(args.source)
        process_video_stream(source_id)
    except ValueError:
        process_video_stream(args.source)
# python object_people.py
# python object_people.py --source "D:\thi_nghiem_AI\dataset\video\drowning_1.mp4"