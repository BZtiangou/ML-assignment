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

def evaluate_model(model, processor, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    cos_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight semantic encoding model
    
    # Initialize metric calculators
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4  # BLEU smoothing function
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Generate text
                generated_ids = model.generate(
                    inputs["input_ids"].to(device),
                    pixel_values=inputs["pixel_values"].to(device),
                    max_length=128,
                    num_beams=3,
                    early_stopping=True
                )
                
                # Decode generated text
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                all_preds.extend(preds)
                all_targets.extend(targets)
                
            except Exception as e:
                print(f"Evaluation batch {batch_idx} failed: {str(e)}")
                continue
    
    # Calculate text similarity metrics
    print("\nCalculating evaluation metrics...")
    bleu_scores = []
    rouge_scores = {"rouge1": [], "rougeL": []}
    target_embeddings = []
    pred_embeddings = []
    
    # Parallel processing to speed up
    for pred, target in zip(all_preds, all_targets):
        # BLEU
        ref = [target.split()]
        hyp = pred.split()
        bleu_scores.append(corpus_bleu([ref], [hyp], smoothing_function=smoothie))
        
        # ROUGE
        scores = scorer.score(target, pred)
        rouge_scores["rouge1"].append(scores['rouge1'].fmeasure)
        rouge_scores["rougeL"].append(scores['rougeL'].fmeasure)
        
        # Semantic embeddings
        target_embeddings.append(cos_model.encode(target, convert_to_tensor=True))
        pred_embeddings.append(cos_model.encode(pred, convert_to_tensor=True))
    
    # Calculate cosine similarity
    cos_sims = [torch.cosine_similarity(t, p, dim=0).item() 
               for t, p in zip(target_embeddings, pred_embeddings)]
    
    # Aggregate results
    metrics = {
        "BLEU": np.mean(bleu_scores),
        "ROUGE-1": np.mean(rouge_scores["rouge1"]),
        "ROUGE-L": np.mean(rouge_scores["rougeL"]),
        "Cosine": np.mean(cos_sims)
    }
    
    # Print examples
    print("\nGenerated examples:")
    for i in range(min(3, len(all_preds))):
        print(f"Reference: {all_targets[i]}")
        print(f"Generated: {all_preds[i]}\n")
    
    return metrics