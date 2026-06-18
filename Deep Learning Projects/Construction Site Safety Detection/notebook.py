import os
import yaml
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from ultralytics import YOLO

BASE_PATH = 'archive/css-data'
IMG_Train = os.path.join(BASE_PATH, 'train/images')
IMG_Test = os.path.join(BASE_PATH,'test/images')
IMG_Val = os.path.join(BASE_PATH, 'valid/images')

LBL_Train = os.path.join(BASE_PATH, 'train/labels')
LBL_Test = os.path.join(BASE_PATH,'test/lsbels')
LBL_Val = os.path.join(BASE_PATH,'valid/labels')

data_yaml = {
    'path' : 'archive/css-data',
    'train' : 'train/images',
    'test' :  'test/images',
    'val' : 'valid/images',
    'nc' : 10,
    'names' : [
        'Hardhat',
        'Mask',
        'NO-Hardhat',
        'NO-Mask',
        'NO-Safety Vest',
        'Person',
        'Safety Cone',
        'Safety Vest',
        'machinery',
        'vehicle'
    ]
}
import os
import yaml

# Create working directory
os.makedirs("working", exist_ok=True)

# Save data.yaml
yaml_file_path = os.path.join("working", "data.yaml")

with open(yaml_file_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✅ data.yaml successfully created at: {yaml_file_path}")

# Read and verify the file
print("\nChecking data.yaml...")

print("YAML Exists:", os.path.exists(yaml_file_path))

with open(yaml_file_path, "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

print("Classes:", class_names)

print("\nContents of data.yaml:\n")
print(data)

def load_images(path):
    exts = ('.jpg','.jpeg','.png')
    return [f for f in glob(path + "/*") if f.lower().endswith(exts)]
train_images = load_images(IMG_Train)
test_images = load_images(IMG_Test)
valid_images = load_images(IMG_Val)

print('Train Images;',len(train_images))
print('Test Images;', len(test_images))
print('Valid Images;', len(valid_images))

train_labels = glob(LBL_Train + '/*.txt')
test_labels = glob(LBL_Test + '/*.txt')
valid_labels = glob(LBL_Val + '/*.txt')

print('Train Labels;', len(train_labels))
print('Test Labels;', len(test_labels))
print('Valid Labels;', len(valid_labels))

from collections import Counter
counts = []
for file in train_labels:
    with open(file) as f:
        for line in f:
            counts.append(int(line.split()[0]))
counter = Counter(counts)
counter

plt.figure(figsize=(10,8))
for i in range(6):
    img = Image.open(random.choice(train_images))
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.show()

def show_bbox(img_path):
    img = Image.open(img_path)
    w,h = img.size
    plt.imshow(img)
    label_path = img_path.replace('image','labels').replace('.jpg','.txt')
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())
                x1 = (x - bw/2) * w
                y1 = (x - bh/2) * h

                rect = plt.Rectangle(
                    (x1,y1), bw*w, bh*h,
                    edgecolor='red',facecolor = 'none',linewidth=2
                )
                plt.gca().add_patch(rect)

    plt.axis('off')
    plt.show()
for _ in range(10):
    show_bbox(random.choice(train_images))

bbox_counts = []
for file in train_labels:
    with open(file) as f:
        bbox_counts.append(len(f.readlines()))
pd.Series(bbox_counts).describe()

sns.histplot(bbox_counts, bins=10)
plt.title('Bounding Boxes per Image')
plt.show()

import cv2
img_path = random.choice(train_images)
img = cv2.imread(img_path)

flip = cv2.flip(img, 1)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Orignal')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(flip, cv2.COLOR_BGR2RGB))
plt.title('Flipped')

plt.show()

import torch
def get_device():
    if not torch.cuda.is_available():
        print('CUDA not available, Using CPU')
        return 'CPU'
    device_id = 0
    cap = torch.cuda.get_device_capability(device_id)
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f' Using GPU: {gpu_name} (Computer Capability {cap[0]}.{cap[1]})')
    return device_id
device = get_device()

model = YOLO('yolo11s.pt')

model.train(
    data = yaml_file_path,
    epochs = 30,
    imgsz = 640,
    batch = 16,
    device = device,
    workers = 1,
    project = 'Construction Site Safety',
    name = 'Construction Site Safety Detection',
    exist_ok = True,
    mosaic = 1.0,
    mixup = 0.2,
    copy_paste = 0.1,
    fliplr = 0.5,
    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.2,
    scale = 0.5,
    translate = 0.1,
    degrees = 15.0,
    shear = 10.0,
    perspective = 0.0005
)

metrics = model.val(verbose=True)
print(metrics)

print(f'nmAP50: {metrics.box.map50}')
print(f'mAP50-95: {metrics.box.map}')
print(f'Precission: {metrics.box.mp}')
print(f'Recall: {metrics.box.mr}')

import matplotlib.image as mpimg
metrics_path = '/archve/working/runs/detect/Construction Site Safety/Construction Site Safety Detection/results.png'
cm_path = '/archive/working/runs/detect/Construction Site Safety/Construction Site Safety Detection/confusion_matrix.png'


if os.path.exists(metrics_path):
    plt.figure(figsize=(20,10),facecolor='white')
    img_metrics = mpimg.imread(metrics_path)
    plt.imshow(img_metrics)
    plt.axis('off')
    plt.title('Training And Validation Metrics (Loss Accuray Per Epochs)', fontsize=18,pad=20)
    plt.show()
else:
    print(f'Metrics Plot Not Found at {metrics_path}')

# Confusion Matrix
if os.path.exists(cm_path):
    plt.figure(figsize=(20,10),facecolor='white')
    img_metrics = mpimg.imread(cm_path)
    plt.imshow(img_metrics)
    plt.axis('off')
    plt.title('Validation Confusion Matrix', fontsize=18,pad=20)
    plt.show()
else:
    print(f'Confusion Matrix Plot Not Found at {cm_path}')

from IPython.display import display
preds = glob('/archive/working/runs/detect/predict/*.jpg')
for img in preds[:5]:
    display(Image.open(img))

model.export(format='onnx')

from IPython.display import FileLink
FileLink(r'runs/detect/Construction Site Safety/Construction Site Safety Detection/weights/best.pt')

FileLink(r'runs/detect/Construction Site Safety/Construction Site Safety Detection/weights/best.onnx')