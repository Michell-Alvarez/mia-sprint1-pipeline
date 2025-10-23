import torch
import torch.nn as nn
import yaml
import os
import pandas as pd
import argparse
# Usar ruta absoluta o encontrar la ruta correcta


# Crear un único parser
parser = argparse.ArgumentParser()

# Agregar ambos argumentos al mismo parser
parser.add_argument('--mode', type=str, default='fe_off', choices=['fe_off', 'fe_on'],
                    help="Modo de ejecución: con o sin feature engineering")
parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'solid'],
                    help="Modelo a ejecutar: baseline o solid")
                    
# Parsear argumentos una sola vez
args = parser.parse_args()

# Construir la ruta del archivo de configuración
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', f'config_{args.model}_{args.mode}.yaml')


class Baseline3DCNN(nn.Module):
    def __init__(self, config_path=config_path):
        
        super(Baseline3DCNN, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Parámetros del modelo
        frame_size = self.config['data']['frame_size']
        num_frames = self.config['data']['num_frames']
        channels = self.config['model']['channels']
        
        # Capas convolucionales
        self.conv1 = nn.Conv3d(3, channels[0], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Calcular tamaño automáticamente
        self.fc_input_size = self._calculate_fc_size(num_frames, frame_size)
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(self.config['model']['dropout']),
            nn.Linear(64, 2)
        )
        
    def _calculate_fc_size(self, T, H):
        x = torch.zeros(1, 3, T, H, H)
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool3(nn.functional.relu(self.conv3(x)))
        
        return x.numel()
    
    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool3(nn.functional.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
        
# Dataset simplificado para la nueva estructura
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, index_file, transform=None):
        self.df = pd.read_csv(index_file)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['video_path']
        label = self.df.iloc[idx]['label']
        
        # Captura de frames (igual que antes)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        indices = np.linspace(0, total_frames-1, self.config['data']['num_frames'], dtype=int)
        
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (self.config['data']['frame_size'], 
                                         self.config['data']['frame_size']))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < self.config['data']['num_frames']:
            frames += [torch.zeros_like(frames[0])] * (self.config['data']['num_frames'] - len(frames))
        
        frames = torch.stack(frames[:self.config['data']['num_frames']])
        return frames.permute(1, 0, 2, 3), label
