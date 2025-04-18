import os
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox  # Corrected import
from tqdm import tqdm  # Import tqdm for progress bar

# === Config ===
image_folder = "../5/CrowdHuman Cropped/Dataset CrowdHuman/non crowd"
yolo_model_path = "weights/crowdhuman_yolov5m.pt"
output_folder = "annotations/non crowd"  # Folder to save the annotations

# === Create output folder ===
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# === Load model ===
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolo_model_path, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(416, s=stride)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

# === Run on each image ===
for filename in tqdm(sorted(os.listdir(image_folder)), desc="Processing images"):  # Use tqdm for progress
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path = os.path.join(image_folder, filename)
    orig_img = cv2.imread(path)
    if orig_img is None:
        print(f"Failed to read {filename}")
        continue

    # Preprocess image
    img = letterbox(orig_img, new_shape=imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0, 1])  # Class 0: person, Class 1: head

    annotations = []  # To store annotations for this image

    if pred[0] is not None and len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], orig_img.shape).round()

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)

            # Convert to YOLOv12 format (normalized coordinates)
            h, w, _ = orig_img.shape
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # Add annotation (class_id x_center y_center width height)
            annotations.append(f"{int(cls)} {x_center} {y_center} {width} {height}")

            # Draw bounding box and center point
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, orig_img, label=label, color=[0, 255, 0], line_thickness=2)

    # Show result
    cv2.imshow("Detection", orig_img)

    # Save annotations in YOLOv12 format
    annotation_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
    with open(annotation_file, 'w') as f:
        f.write("\n".join(annotations))

    # key = cv2.waitKey(0)
    # if key == 27:  # ESC to quit early
    #     break

cv2.destroyAllWindows()
