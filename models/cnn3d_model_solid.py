import os
import cv2
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict, Any


class ViolenceDataset(Dataset):
    """
    Dataset de videos con muestreo aleatorio de frames y normalización.
    Devuelve tensores con shape (C, T, H, W).
    """
    def __init__(self, video_paths: List[str], labels: List[int],
                 transform=None, clip_len: int = 30, frame_size: int = 112):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.clip_len = clip_len
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_paths)

    def _read_frame(self, cap, idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        return Image.fromarray(frame)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames = []

        if total_frames >= self.clip_len and self.clip_len > 0:
            indices = sorted(random.sample(range(total_frames), self.clip_len))
        else:
            indices = list(range(total_frames))

        for i in indices:
            img = self._read_frame(cap, i)
            if img is not None:
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                    img = transforms.Normalize([0.432, 0.398, 0.377],
                                               [0.228, 0.224, 0.225])(img)
                frames.append(img)

        cap.release()

        # Pad si faltan frames
        while len(frames) < self.clip_len:
            frames.append(torch.zeros(3, self.frame_size, self.frame_size))

        # (C, T, H, W)
        video_tensor = torch.stack(frames[:self.clip_len], dim=1).float()
        return video_tensor, torch.tensor(label, dtype=torch.long)


class ViolenceDetector(nn.Module):
    """
    3D-CNN con atención temporal y clasificador final.
    Entrada esperada: (B, 3, T, H, W)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 7, 7))
        )
        self.temp_attention = nn.Sequential(
            nn.Linear(256*7*7, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.float()
        feats = self.backbone(x)               # (B, C, T, H, W)
        B, C, T, H, W = feats.shape
        feats = feats.permute(0, 2, 1, 3, 4).reshape(B, T, C*H*W)  # (B, T, D)
        attn = torch.softmax(self.temp_attention(feats), dim=1)    # (B, T, 1)
        context = (feats * attn).sum(dim=1)                        # (B, D)
        return self.classifier(context)


# -------- Helpers de datos --------
def cargar_datos_desde_directorio(directorio: str) -> Tuple[List[str], List[int]]:
    """
    Espera subcarpetas 'Violence' y 'NonViolence' con .mp4.
    Retorna listas: rutas y etiquetas (1 = Violence, 0 = NonViolence).
    """
    paths, etiquetas = [], []
    for clase in ['Violence', 'NonViolence']:
        clase_dir = os.path.join(directorio, clase)
        if not os.path.isdir(clase_dir):
            continue
        for archivo in os.listdir(clase_dir):
            if archivo.endswith('.mp4'):
                paths.append(os.path.join(clase_dir, archivo))
                etiquetas.append(1 if clase == 'Violence' else 0)
    return paths, etiquetas


def build_transforms(frame_size: int = 112, aug: bool = True):
    if aug:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(size=frame_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.432, 0.398, 0.377], [0.228, 0.224, 0.225])
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.432, 0.398, 0.377], [0.228, 0.224, 0.225])
        ])

    val_tf = transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.432, 0.398, 0.377], [0.228, 0.224, 0.225])
    ])
    return train_tf, val_tf


def build_dataloaders(cfg: Dict[str, Any]):
    """
    Construye DataLoaders de train/val usando directorios del YAML.
    """

    clip_len = cfg['data']['clip_len']
    frame_size = cfg['data']['frame_size']
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']
    train_paths_all, train_labels_all = cargar_datos_desde_directorio(cfg['data']['train_dir'])
       
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths_all,
        train_labels_all,
        test_size=0.15,
        stratify=train_labels_all,   # mantiene balance de clases
        shuffle=True, 
        random_state=42              # para reproducibilidad
    )

    train_tf, val_tf = build_transforms(frame_size, aug=False)
    train_ds = ViolenceDataset(train_paths, train_labels, transform=train_tf,
                               clip_len=clip_len, frame_size=frame_size)
    val_ds = ViolenceDataset(val_paths, val_labels, transform=val_tf,
                             clip_len=clip_len, frame_size=frame_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_labels, val_labels

