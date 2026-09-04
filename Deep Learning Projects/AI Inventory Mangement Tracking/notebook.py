import os
for root, dirs, files in os.walk ('archive'):
    print(root)

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

BASE_PATH = 'archive/'
IMG_TRAIN = os.path.join(BASE_PATH, 'train/images')
IMG_VAL = os.path.join(BASE_PATH, 'valid/images')
IMG_Test = os.path.join(BASE_PATH, 'test/images')

LBL_TRAIN = os.path.join(BASE_PATH, 'train/labels')
LBL_VAL = os.path.join(BASE_PATH, 'valid/labels')
LBL_Test = os.path.join(BASE_PATH, 'test/labels')


YAML_PATH = os.path.join(BASE_PATH, 'data.yaml')

data_yaml = {
    'train':IMG_TRAIN,
    'val': IMG_VAL,
    'test': IMG_Test,
    'nc':1,
    'names': ['void_detection - v1 2025-04-06 7-16pm']
}
YAML_PATH = 'data.yaml'
with open(YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f)
print('YAML Created at:', YAML_PATH)

print('YAML Exists:', os.path.exists(YAML_PATH))
with open (YAML_PATH, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']
print('Classes:', class_names)

def load_images(path):
    exts = ('.jpg','.jpeg','.png')
    return [f for f in glob(path + "/*") if f.lower().endswith(exts)]
train_images = load_images(IMG_TRAIN)
val_images = load_images(IMG_VAL)

print('Train Images;', len(train_images))
print("Val Images:" , len(val_images))

train_labels = glob(LBL_TRAIN + '/*.txt')
val_labels = glob(LBL_VAL + '/*.txt')

print('Train Labels:', len(train_labels))
print('Val Labels:', len(val_labels))

from collections import Counter
counts = []
for file in train_labels:
    with open(file) as f:
        for line in f:
            counts.append(int(line.split()[0]))
counter = Counter(counts)
counter

plt.figure(figsize=(8,5))
sns.barplot(
    x = [class_names[i] for i in counter.keys()],
    y = list(counter.values())
)
plt.xticks(rotation=90)
plt.title('Class distribution')
plt.show()

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
    w , h =img.size
    plt.imshow(img)
    label_path = img_path.replace('image','labels').replace('.jpg','.txt')
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y , bw, bh = map(float, line.split())
                x1 = (x - bw/2) * w
                y1 = (x - bh/2) * h

                rect = plt.Rectangle(
                    (x1, y1), bw*w, bh*h,
                    edgecolor='red', facecolor='none',linewidth=2
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
plt.title('Bounding Boxes Per Image')
plt.show()

import cv2
img_path = random.choice(train_images)
img = cv2.imread(img_path)

flip = cv2.flip(img, 1)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('ORIGNAL')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(flip, cv2.COLOR_BGR2RGB))
plt.title('FLIPPED')

plt.show()

import torch
def get_device():
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return "CPU"
    device_id = 0
    cap = torch.cuda.get_device_capability(device_id)
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f' Using GPU: {gpu_name} (Computer Capability {cap[0]}.{cap[1]})')
    return device_id
device = get_device()


model  = YOLO('yolov8n.pt')

model.train(
    data = YAML_PATH,
    epochs = 30,
    imgsz = 640,
    batch = 16,
    device = device,
    workers = 1,
    project = 'AI Inventury Management Monitoring',
    name = 'AI Inventory Management Monitoring',
    exist_ok=True,
    mosaic = 1.0,
    mixup = 0.2,
    copy_paste = 0.1,
    fliplr = 0.5,
    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
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
print(f'Precision: {metrics.box.mp}')
print(f'Recall: {metrics.box.mr}')


import matplotlib.image as mpimg

metrics_path = '/runs/detect/AI Invenory Management/AI Inventroy Management Detection/results.png'


if os.path.exists(metrics_path):
    plt.figure(figsize=(20,10),facecolor='white')
    img_metrics = mpimg.imread(metrics_path)
    plt.imshow(img_metrics)
    plt.axis('off')
    plt.title('Training and Valiadtion Metrics(Loss Accuracy Per Epochs)', fontsize=18,pad=20)
    plt.show()
else:
    print(f'Metrics Plot not found at  {metrics_path}')

cm_path = '/runs/detect/AI Inventory Management/AI Inventory Management Detection/confusion_matrix.png'

# Confusion Matrix
if os.path.exists(cm_path):
    plt.figure(figsize=(20,10),facecolor='white')
    img_metrics = mpimg.imread(cm_path)
    plt.imshow(img_metrics)
    plt.axis('off')
    plt.title('Valiadtion Confusion Matrix', fontsize=18,pad=20)
    plt.show()
else:
    print(f' Confusion Metrics Plot not found at  {cm_path}')


model.export(format='onnx')

from IPython.display import FileLink
FileLink(r'runs/detect/AI Inventory Management/AI Inventory Management Detection/weights/best.pt')

FileLink(r'runs/detect/AI Inventory Management/AI Inventory Management Detection/weights/best.onnx')