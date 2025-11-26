import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm # Cần cài đặt: pip install tqdm

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục train chứa 2 folder con là 'images' và 'labels'
DATASET_ROOT = r"D:\thi_nghiem_AI\dataset\train" 
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")

# Chuỗi giả lập (Nhân bản 1 ảnh thành 30 frames)
SEQ_LENGTH = 30  

print("⏳ Đang tải mô hình YOLOv8-Pose...")
model = YOLO('yolov8n-pose.pt')

def get_feature_vector(image_path):
    """Trích xuất 17 khớp xương và làm phẳng thành vector 51 chiều"""
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        
        # Chạy YOLO Pose
        results = model(img, verbose=False)
        
        for result in results:
            if result.keypoints is None: continue
            
            # Kiểm tra nếu không tìm thấy người nào
            if result.keypoints.xyn.shape[0] == 0:
                continue
                
            # Lấy người đầu tiên tìm thấy
            keypoints_xyn = result.keypoints.xyn.cpu().numpy()[0]  # (17, 2)
            keypoints_conf = result.keypoints.conf.cpu().numpy()[0] # (17,)
            
            # Ghép lại: [x, y, conf] -> Shape (17, 3)
            keypoints_combined = np.column_stack((keypoints_xyn, keypoints_conf))
            
            return keypoints_combined.flatten() # 51 chiều
    except Exception as e:
        pass
    return None

def create_dataset():
    X_data = []
    y_data = []
    
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục 'images' hoặc 'labels' tại {DATASET_ROOT}")
        return

    print(f" Bắt đầu quét dữ liệu...")
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    count_success = 0
    count_skipped = 0

    for img_file in tqdm(image_files, desc="Đang xử lý"):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_file)
        img_path = os.path.join(IMAGES_DIR, img_file)

        if not os.path.exists(label_path): continue
            
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines: continue
                
                # Lấy Class ID từ dòng đầu tiên
                first_line = lines[0].strip().split()
                class_id = int(first_line[0]) 
                
                # Trích xuất đặc trưng
                vector_51 = get_feature_vector(img_path)
                
                if vector_51 is not None:
                    # Nhân bản thành chuỗi 30 frames
                    sequence = np.tile(vector_51, (SEQ_LENGTH, 1))
                    X_data.append(sequence)
                    y_data.append(class_id)
                    count_success += 1
                else:
                    count_skipped += 1
        except:
            continue

    X = np.array(X_data)
    y = np.array(y_data)

    print(f"\n✅ Hoàn tất! Số lượng mẫu: {count_success}")
    if len(X) > 0:
        np.save('X_train_data.npy', X)
        np.save('y_train_data.npy', y)
        print("💾 Đã lưu file .npy thành công.")

if __name__ == "__main__":
    create_dataset()