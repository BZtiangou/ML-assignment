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
from dataset import *
from eval import evaluate_model

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

def train():
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'  # Force using local cache
        
        # Initialize the model
        print("\n" + "="*40)
        print("Initializing model...")
        model_path = "/home/xyc/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369"
        
        # Verify the existence of model files
        required_shards = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors", 
            "model-00003-of-00003.safetensors"
        ]
        
        for shard in required_shards:
            shard_path = os.path.join(model_path, shard)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Model shard file missing: {shard_path}")

        # Load the model with sharded mode specified
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True
        )
        print("✅ Model loaded successfully")
        
        # Initialize the processor
        print("\n" + "="*40)
        print("Initializing processor...")
        processor = LlavaProcessor.from_pretrained(
            model_path,
            tokenizer_class="LlamaTokenizerFast",
            trust_remote_code=True
        )
        print("✅ Processor initialized")
        
        # Prepare the dataset
        print("\n" + "="*40)
        print("Preparing dataset...")
        dataset = AniPersonaDataset(
            data_root="/home/xyc/ML/data/AniPersonaCaps/images/",
            processor=processor
        )
        print(f"Dataset contains {len(dataset)} samples")
        
        # Data loader
        dataloader = DataLoader(dataset, 
                              batch_size=2, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                              num_workers=4)
        
        # Training configuration
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        num_epochs = 10
        
        # Training loop
        print("\n" + "="*40)
        print("Starting training...")
        for epoch in range(num_epochs):
            try:
                model.train()
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
                
                for batch_idx, (inputs, targets) in enumerate(progress_bar):
                    try:
                        # Debugging print
                        print("\n" + "-"*30)
                        print(f"Batch {batch_idx} inputs:")
                        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
                        print(f"Input IDs shape: {inputs['input_ids'].shape}")
                        print(f"Sample text: {processor.decode(inputs['input_ids'][0][:30])}...")
                        
                        # Move data to device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Generate target tokens
                        print("Processing targets...")
                        targets_encoding = processor(
                            text=targets,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=128
                        ).to(device)
                        
                        # Validate target encoding
                        if targets_encoding.input_ids.dim() != 2:
                            raise ValueError(f"Invalid target shape: {targets_encoding.input_ids.shape}")
                        
                        # Forward pass
                        print("Forward pass...")
                        outputs = model(**inputs, labels=targets_encoding.input_ids)
                        
                        # Check loss value
                        loss = outputs.loss
                        if torch.isnan(loss):
                            raise ValueError("NaN loss detected!")
                        if torch.isinf(loss):
                            raise ValueError("Inf loss detected!")
                            

                        # Backward pass
                        print("Backward pass...")
                        loss.backward()
                        
                        # Parameter update
                        print("Optimizer step...")
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Progress bar update
                        progress_bar.set_postfix({'loss': loss.item()})
                        
                        # Save checkpoint
                        if batch_idx % 100 == 0:
                            try:
                                save_path = f"llava_checkpoint_epoch{epoch+1}_step{batch_idx}.pth"
                                torch.save(model.state_dict(), save_path)
                                print(f"💾 Checkpoint saved to {save_path}")
                            except Exception as e:
                                print(f"🔥 Failed to save checkpoint: {str(e)}")
                                traceback.print_exc()
                                
                    except Exception as e:
                        print(f"🔥 Error in batch {batch_idx}:")
                        traceback.print_exc()
                        # Skip problematic batch and continue training
                        continue
                    # Validate after each epoch
                val_metrics = evaluate_model(model, processor, val_loader, device)
                print(f"Validation metrics | "
                      f"BLEU: {val_metrics['BLEU']:.4f} | "
                      f"ROUGE-1: {val_metrics['ROUGE-1']:.4f} | "
                      f"ROUGE-L: {val_metrics['ROUGE-L']:.4f} | "
                      f"Cosine similarity: {val_metrics['Cosine']:.4f}")
                
                # Save the best model based on cosine similarity
                if val_metrics['Cosine'] > best_cosine:
                    best_cosine = val_metrics['Cosine']
                    torch.save(model.state_dict(), "best_llava_model.pth")
                    print(f"New best model found, cosine similarity: {best_cosine:.4f}")
            except Exception as e:
                print(f"🔥 Error in epoch {epoch+1}:")
                traceback.print_exc()
                # Continue to the next epoch
                continue
                
    except Exception as e:
        print("\n🔥 Critical error in training setup:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("="*50)
    print("Starting training process")
    try:
        train()
    except Exception as e:
        print("\n💥 Fatal error occurred:")
        traceback.print_exc()
    print("Training process ended")