import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_NAMES = ['full-faced', 'half-faced', 'invalid', 'no helmet']
NUM_CLASSES = 4


class HelmetDataset(Dataset):
    """
    
    Folder structure:
        dataset/train/images/*.jpg
        dataset/train/labels/*.txt
    """

    def __init__(self, images_dir, labels_dir, img_size=416, is_train=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size   = img_size
        self.is_train   = is_train

        self.samples = []
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path   = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir,
                                      os.path.splitext(fname)[0] + '.txt')
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))

        if is_train:
            self.samples = self.samples[:500]
        else:
            self.samples = self.samples[:100]

        print(f"[Dataset] Found {len(self.samples)} samples in {images_dir}")

        if is_train:
            aug_list = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                           contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15,
                                     sat_shift_limit=30,
                                     val_shift_limit=20, p=0.4),
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                         scale=(0.8, 1.2),
                         rotate=(-10, 10),
                         p=0.4),
            ]

            noise_transform = None
            if hasattr(A, 'GaussianNoise'):
                try:
                    noise_transform = A.GaussianNoise(var_limit=(10.0, 50.0), p=0.2)
                except Exception:
                    noise_transform = A.GaussianNoise(p=0.2)
            elif hasattr(A, 'GaussNoise'):
                noise_transform = A.GaussNoise(p=0.2)

            if noise_transform is not None:
                aug_list.append(noise_transform)

            aug_list += [
                A.Blur(blur_limit=3, p=0.2),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]

            self.transform = A.Compose(aug_list, bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes       = []
        class_labels = []

        with open(label_path) as f:
            for line in f.read().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                cls   = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                bboxes.append([cx, cy, w, h])
                class_labels.append(cls)

        result = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        image        = result['image']          # Tensor (3, H, W)
        bboxes       = result['bboxes']         # list of (cx,cy,w,h)
        class_labels = result['class_labels']   # list of ints

        if len(bboxes) > 0:
            target_boxes  = torch.tensor(bboxes,       dtype=torch.float32)
            target_labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            target_boxes  = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,),   dtype=torch.long)

        return image, target_boxes, target_labels


def collate_fn(batch):
    images, boxes_list, labels_list = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(boxes_list), list(labels_list)
