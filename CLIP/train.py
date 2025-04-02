import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
from config import *
from dataset import *

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def validate(model, val_loader, top_k=5):
    model.eval()
    correct = 0
    top_k_correct = 0
    total = 0
    cos_sim_sum = 0.0
    
    with torch.no_grad():
        for images, texts in val_loader:
            if images is None:
                continue
                
            images = images.to(DEVICE)
            texts = {k: v.to(DEVICE) for k, v in texts.items()}
            
            with torch.autocast(device_type=DEVICE, dtype=TORCH_DTYPE):
                outputs = model(
                    input_ids=texts["input_ids"],
                    attention_mask=texts["attention_mask"],
                    pixel_values=images
                )
                
                logits_per_image = outputs.logits_per_image  # [batch_size, batch_size]
                image_embeds = outputs.image_embeds          # [batch_size, embed_dim]
                text_embeds = outputs.text_embeds             # [batch_size, embed_dim]
                
                # 1. Calculate classification accuracy
                preds = logits_per_image.argmax(dim=1)
                correct += (preds == torch.arange(len(images), device=DEVICE)).sum().item()
                
                # 2. Calculate Top-K recall rate
                batch_size = len(images)
                _, topk_indices = torch.topk(logits_per_image, k=top_k, dim=1)
                targets = torch.arange(batch_size, device=DEVICE).view(-1, 1)
                topk_correct = torch.any(topk_indices == targets, dim=1).sum().item()
                top_k_correct += topk_correct
                
                # 3. Calculate cosine similarity (mean of correct pairs)
                correct_cos_sim = torch.einsum('i d, i d -> i', image_embeds, text_embeds)
                cos_sim_sum += correct_cos_sim.sum().item()
                
                total += batch_size
    
    accuracy = correct / total if total != 0 else 0.0
    top_k_recall = top_k_correct / total if total != 0 else 0.0
    avg_cos_sim = cos_sim_sum / total if total != 0 else 0.0
    
    return accuracy, top_k_recall, avg_cos_sim

# Enhanced data loader
def create_data_loader(dataset, cfg):
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        images, texts = zip(*batch)
        
        # Image processing
        image_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        )["pixel_values"]
        
        # Text processing
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=cfg["text_max_len"],
            truncation=True
        )
        
        return image_inputs, text_inputs
    
    processor = dataset.processor
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True
    )

# New validation function
def validate_inputs(texts, images):
    assert images.shape[1:] == (3, 224, 224), f"Image size error: {images.shape}"
    tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
    encoded = tokenizer(texts, padding="max_length", max_length=77, truncation=True)
    assert all(len(ids) == 77 for ids in encoded["input_ids"]), "Text length exceeds limit"

# Model initialization (with error handling)
def initialize_model():
    try:
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            projection_dim=512,
            logit_scale_init_value=16.0
        ).to(DEVICE)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise

# Training process
def train():
    # Initialize components
    processor = DataProcessor(CFG)
    train_dataset, val_dataset = processor.get_datasets()
    train_loader = create_data_loader(train_dataset, CFG)
    val_loader = create_data_loader(val_dataset, CFG)
    model = initialize_model()
    
    # Optimizer configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=0.05
    )
    
    # Learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader)*CFG["epochs"],
        eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0  # New best accuracy tracking
    # Training loop
    for epoch in range(CFG["epochs"]):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}")
        for images, texts in progress_bar:
            images = images.to(DEVICE)
            texts = {k: v.to(DEVICE) for k, v in texts.items()}
            
            # Mixed precision forward pass
            with torch.autocast(device_type=DEVICE, dtype=TORCH_DTYPE):
                outputs = model(
                    input_ids=texts["input_ids"],
                    attention_mask=texts["attention_mask"],
                    pixel_values=images
                )
                
                # Contrastive loss calculation
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                labels = torch.arange(len(images), device=DEVICE)
                loss = 0.5 * (
                    nn.functional.cross_entropy(logits_per_image, labels) +
                    nn.functional.cross_entropy(logits_per_text, labels)
                )
            
            # Backward pass optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        val_acc, val_topk, val_cos_sim = validate(model, val_loader, top_k=5)
        print(f"Validation results | Accuracy: {val_acc:.4f}, Top-5 recall: {val_topk:.4f}, Cosine similarity: {val_cos_sim:.4f}")
        # Save the best model based on cosine similarity + accuracy
        if val_cos_sim + val_acc > best_acc:
            best_acc = val_cos_sim + val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model found | Combined metric: {best_acc:.4f}")

if __name__ == "__main__":
    # New directory structure validation
    required_subdirs = [f"images_{i}" for i in range(1,31)]
    existing_dirs = next(os.walk(CFG["data_root"]))[1]
    missing = [d for d in required_subdirs if d not in existing_dirs]
    
    train()