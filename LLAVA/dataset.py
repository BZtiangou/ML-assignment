import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer  # For cosine similarity calculation
from PIL import Image
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import traceback

class AniPersonaDataset(Dataset):
    def __init__(self, data_root, processor, max_length=128):
        try:
            self.data_root = Path(data_root)
            self.processor = processor
            self.max_length = max_length
            
            # Verify the existence of the metadata file
            metadata_path = self.data_root / 'metadata.jsonl'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            # Load metadata
            with open(metadata_path) as f:
                print(f"Loading metadata from {metadata_path}")
                self.metadata = [json.loads(line) for line in f]
                print(f"Successfully loaded {len(self.metadata)} metadata entries")
                
            # Build sample list
            self.samples = []
            for idx, data in enumerate(self.metadata):
                try:
                    img_path = self.data_root / data['file_name']
                    if not img_path.exists():
                        print(f"⚠️ Image not found: {img_path} (entry {idx})")
                        continue
                        
                    self.samples.append({
                        'image': str(img_path),
                        'personality': data.get('personality', '') or '',
                        'appearance': data['appearance']
                    })
                except KeyError as e:
                    print(f"❌ Missing key {e} in metadata entry {idx}")
                except Exception as e:
                    print(f"❌ Error processing entry {idx}: {str(e)}")
                    traceback.print_exc()
                    
            print(f"Final dataset size: {len(self.samples)} valid samples")

        except Exception as e:
            print("🔥 Dataset initialization failed!")
            traceback.print_exc()
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            
            # Load image
            image = Image.open(sample['image']).convert('RGB')
            
            # Build prompt
            prompt = f"Describe the personality traits of this anime character based on appearance: {sample['appearance']}"
            
            # Target text generation
            target = f"Personality: {sample['personality']}" if sample['personality'] else ""
            
            processed = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Validate processing results
            if processed.pixel_values.dim() != 4:
                raise ValueError(f"Invalid image tensor shape: {processed.pixel_values.shape}")
                
            if processed.input_ids.size(1) != self.max_length:
                raise ValueError(f"Text length mismatch: {processed.input_ids.size(1)} vs {self.max_length}")
                
            return processed, target
            
        except Exception as e:
            print(f"🔥 Error processing sample {idx} ({sample.get('image', 'unknown')}):")
            traceback.print_exc()
            return None  # Return None for collate_fn to handle


def collate_fn(batch):
    try:
        # Filter out invalid samples
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            raise ValueError("Empty batch after filtering")
            
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Debugging print
        print(f"Batch size: {len(batch)}")
        print(f"Input shapes - pixel: {inputs[0].pixel_values.shape}, ids: {inputs[0].input_ids.shape}")
        
        processed = {
            'pixel_values': torch.cat([i.pixel_values for i in inputs], 0),
            'input_ids': torch.cat([i.input_ids for i in inputs], 0),
            'attention_mask': torch.cat([i.attention_mask for i in inputs], 0)
        }
        
        return processed, targets
        
    except Exception as e:
        print("🔥 Collate function failed:")
        traceback.print_exc()
        raise