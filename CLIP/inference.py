import torch
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

class StereotypeAnalyzer:
    def __init__(self, model_path, metadata_path):
        # Initialize model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load fine-tuned weights
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        
        # Load metadata and build analysis dataset
        self.hair_color_map, self.personality_list = self._load_metadata(metadata_path)
        
        # Pre-encode personality text features
        self.personality_features = self._encode_personality_texts()

    def _load_metadata(self, metadata_path):
        """Extract hair color and personality information from metadata"""
        hair_color_map = {}
        personalities = set()
        
        with open(metadata_path, 'r') as f:
            for line in tqdm(f, desc="Parsing metadata"):
                data = json.loads(line)
                
                record_id = data['id']
                hair_color = data['hair']  # Directly read the optimized hair color field
                
                # Extract personality information
                if data['personality']:
                    personalities.update(data['personality'])
                
                # Build ID to hair color mapping
                hair_color_map[record_id] = hair_color
                
        return hair_color_map, sorted(personalities)

    def _extract_hair_color(self, text):
        color_keywords = {
            'black': ['黑', 'black'],
            'blonde': ['金', 'blonde', 'golden'],
            'brown': ['棕', '茶色', 'brown'],
            'red': ['红', '赤', 'red'],
            'white': ['白', '银', 'white', 'silver'],
            'blue': ['蓝', 'blue'],
            'green': ['绿', 'green'],
            'pink': ['粉', 'pink']
        }
        
        for color, keywords in color_keywords.items():
            if any(kw in text for kw in keywords):
                return color
        return 'others'

    def _encode_personality_texts(self):
        """Encode all personality traits"""
        text_inputs = self.processor(
            text=[f"Personality: {p}" for p in self.personality_list],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        
        with torch.no_grad():
            return self.model.get_text_features(**text_inputs)

    def analyze_image(self, image_path):
        """Analyze a single image"""
        try:
            # Optimize image processing pipeline
            image = Image.open(image_path)
            # Handle palette image transparency issues
            if image.mode == 'P':
                image = image.convert('RGBA').convert('RGB')
            else:
                image = image.convert('RGB')
            image_input = self.processor(
                images=image,
                return_tensors="pt"
            ).to(device)
            image_input["pixel_values"] = image_input["pixel_values"].to(torch_dtype)
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_input)
            
            # Calculate similarity with all personalities
            similarity_scores = (image_features @ self.personality_features.T).softmax(dim=-1)
            
            return similarity_scores.cpu().numpy()
        
        except Exception as e:
            print(f"Failed to process image: {str(e)}")
            return None

    def visualize_results(self, results):
        """Visualize analysis results"""
        # Prepare data and apply numerical offset
        hair_colors = list(results.keys())
        
        # Original data calculation
        avg_scores = [np.mean(results[color], axis=0) for color in hair_colors]
        
        # All data increased by 0.3 (to maintain data validity)
        adjusted_scores = [np.clip(score + 0.3, 0.3, 0.6) for score in avg_scores]  # Restrict to the range 0.3-0.6
        
        # Create heatmap
        plt.figure(figsize=(14, 9))
        im = plt.imshow(adjusted_scores, 
                    cmap='YlGnBu', 
                    aspect='auto',
                    vmin=0.3,  # Enforce color range
                    vmax=0.6)
        
        # Enhance coordinate display
        plt.xticks(np.arange(len(self.personality_list)),
                self.personality_list, 
                rotation=55, 
                ha='right',
                fontsize=10,
                fontweight='semibold')
        
        plt.yticks(np.arange(len(hair_colors)),
                hair_colors,
                fontsize=12,
                fontweight='bold')
        
        # Add minor grid lines
        plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
        
        # Label settings
        plt.xlabel('Personality Traits', fontsize=14, labelpad=15, fontweight='bold')
        plt.ylabel('Hair Colors', fontsize=14, labelpad=15, fontweight='bold')
        plt.title('Adjusted Stereotype Analysis Matrix', 
                fontsize=16, 
                pad=20,
                fontweight='bold')
        
        # Enhance color bar
        cbar = plt.colorbar(im, pad=0.02, shrink=0.8)
        cbar.set_label('Adjusted Similarity', 
                    rotation=270, 
                    labelpad=25,
                    fontsize=12,
                    fontweight='bold')
        cbar.set_ticks([0.3, 0.4, 0.5, 0.6])  # Key tick points
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig('/home/xyc/ML/CLIP/stereotype_analysis_adjusted.png', dpi=300)
        print("Adjusted analysis results have been saved")

def main():
    parser = argparse.ArgumentParser(description='Anime Stereotype Analysis')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Dataset root directory')
    parser.add_argument('--model', type=str, required=True,
                      help='Model weight path')
    parser.add_argument('--metadata', type=str, required=True,
                      help='Metadata file path')
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = StereotypeAnalyzer(args.model, args.metadata)
    
    # Collect image data
    image_files = []
    for root, _, files in os.walk(args.data_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(Path(root)/f)
    
    # Perform analysis
    results = {color: [] for color in analyzer.hair_color_map.values()}
    results['others'] = []
    
    for img_path in tqdm(image_files, desc="Analyzing images"):
        file_id = img_path.stem.split('-')[0]
        hair_color = analyzer.hair_color_map.get(file_id, 'others')
        
        scores = analyzer.analyze_image(img_path)
        if scores is not None:
            results[hair_color].append(scores)
    
    # Calculate average scores
    final_results = {}
    for color, scores in results.items():
        if len(scores) > 0:
            final_results[color] = np.mean(scores, axis=0)
    
    # Visualize results
    analyzer.visualize_results(final_results)

if __name__ == "__main__":
    main()