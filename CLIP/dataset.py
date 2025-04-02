import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from config import *

class AnimeDataset(Dataset):
    def __init__(self, metadata, processor):
        self.metadata = metadata
        self.processor = processor
        self.image_base = Path(CFG["data_root"])
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        try:
            # Construct the full image path
            img_path = self.image_base / item["file_name"]  
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('RGBA')  # Add format conversion
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Combine text features
            text_parts = [
                f"Character: {item['character']}",
                f"Appearance: {item['appearance']}",
                f"Personality: {item['personality']}" if item['personality'] else ""
            ]
            text = " | ".join([p for p in text_parts if p])
            
            return image, text
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            return None

# High-performance data loading processor
class DataProcessor:
    def __init__(self, cfg):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.train_meta, self.val_meta = self._load_metadata(cfg["metadata_path"])
        
    def _load_metadata(self, metadata_path):
        """Metadata loading method adapted for multi-level directory structures"""
        train_meta = []
        val_meta = []
        error_count = 0
        valid_count = 0
        
        with open(metadata_path, "r") as f:
            for line_idx, line in enumerate(tqdm(f, desc="Loading metadata")):
                try:
                    data = json.loads(line)
                    file_path = data["file_name"]
                    
                    # Validate path structure (should contain two levels of directories)
                    path_parts = file_path.split("/")
                    if len(path_parts) < 2:
                        raise ValueError(f"Insufficient path levels: {file_path}")
                        
                    numbered_folder = path_parts[1]  
                    if "_" not in numbered_folder:
                        raise ValueError(f"Second-level directory missing underscore: {numbered_folder}")
                        
                    # Extract folder index
                    folder_idx = int(numbered_folder.split("_")[1])
                    
                    # Validate full path
                    full_path = Path(CFG["data_root"]) / file_path
                    if not full_path.exists():
                        raise FileNotFoundError(f"File not found: {full_path}")
                    
                    # Split dataset based on folder index
                    if folder_idx <= CFG["train_folders"]:
                        train_meta.append(data)
                    else:
                        val_meta.append(data)
                        
                    valid_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Invalid metadata entry (line {line_idx+1}): {str(e)}")
                    continue
        
        print(f"\nValid data: {valid_count} entries | Invalid data: {error_count} entries")
        print(f"Training set: {len(train_meta)} entries | Validation set: {len(val_meta)} entries")
        return train_meta, val_meta

    def get_datasets(self):
        return (
            AnimeDataset(self.train_meta, self.processor),
            AnimeDataset(self.val_meta, self.processor)
        )