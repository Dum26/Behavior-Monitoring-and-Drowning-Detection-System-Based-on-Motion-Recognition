# ========================================================
# Train YOLOv8 Model for Swimming and Drowning Detection
# Author: Your Name
# ========================================================

from ultralytics import YOLO
import torch
import os

# --------------------------------------------------------
# Kiểm tra GPU (nếu có)
# --------------------------------------------------------
print("PyTorch version:", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------------
# Nạp mô hình YOLOv8 có sẵn
# --------------------------------------------------------
model_path = "models/yolov8n.pt"  # model YOLO base
if not os.path.exists(model_path):
    print("⚠️ Model yolov8n.pt not found. It will be downloaded automatically.")
model = YOLO(model_path)

# --------------------------------------------------------
# Huấn luyện mô hình với dữ liệu của bạn
# --------------------------------------------------------
results = model.train(
    data="dataset/data.yaml",   # file cấu hình dataset
    epochs=50,                  # số vòng lặp huấn luyện (bạn có thể tăng lên 100+ nếu dataset lớn)
    imgsz=640,                  # kích thước ảnh huấn luyện
    batch=8,                    # số lượng ảnh mỗi batch
    device=device,              # tự động chọn CPU hoặc GPU
    name="drowning_detector",   # tên thư mục kết quả
    workers=2,                  # số luồng đọc dữ liệu
    project="runs/train"        # nơi lưu kết quả
)

# --------------------------------------------------------
# In ra kết quả huấn luyện
# --------------------------------------------------------
print("\n Training complete!")
print(" Results saved in:", results.save_dir)
print(" Best model path:", os.path.join(results.save_dir, "weights", "best.pt"))
