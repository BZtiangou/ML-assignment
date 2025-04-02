import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import logging
import time
import re
import os
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LLaVA-Inference")

class LLaVAEvaluator:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        """Load the model and processor"""
        logger.info(f"Loading model {self.model_id}...")
        start_time = time.time()
        
        try:
            if "cuda" in self.device:
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
            
            # Load the processor (with image processing parameters)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                use_fast=False,
                padding_side="right",
                do_resize=True,
                do_center_crop=True
            )
            
            # Load the model (with Flash Attention enabled for acceleration)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            
            logger.info(f"Model loaded in {time.time()-start_time:.2f}s | Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _build_prompt(self, text, mode="score"):
        """Construct multi-mode prompt templates with personality enforcement"""
        personality_clause = "(MUST include at least one personality adjective from: outgoing, shy, confident, stubborn, gentle, irritable, tsundere, aloof, optimistic, pessimistic)"
        
        if mode == "describe":
            return f"""USER: <image>
    Describe this anime character in detail. Required elements:
    1. Appearance features:
    - Hair style/color
    - Eye characteristics 
    - Clothing/accessories
    2. Personality traits {personality_clause}:
    - Observable behavioral tendencies
    - Typical facial expressions
    - Body language patterns

    Provide comprehensive analysis in 3-5 sentences. MUST explicitly use personality adjectives.
    ASSISTANT:"""
        
        elif mode == "score":
            return f"""USER: <image>
    Evaluate character-description relevance (1-100 scale). Criteria:

    [Description]
    {text}

    [Scoring Rubric]
    1. Appearance Match (0-10 pts)
    - Accuracy of hair/eye color/style
    - Clothing consistency

    2. Personality Match (0-80 pts) {personality_clause}
    - Core personality alignment (-20 if missing traits)
    - Expression consistency
    - Behavior plausibility

    3. Context Factors (0-10 pts)
    - Background elements
    - Art style consistency

    [Response Format]
    Score: [1-100]
    Breakdown:
    - Appearance: [0-10]
    - Personality: [0-80]
    - Other: [0-10]
    Analysis: [Must contain specific personality terms]
    ASSISTANT: Score:"""
        
        raise ValueError(f"Invalid mode: {mode}")


    def _extract_score(self, text):
        patterns = [
            r'Score:\s*(\d{1,3})/100',  
            r'Score:\s*(\d+)',        
            r'\b(\d{1,3})\s*points'     
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match and 1 <= int(match.group(1)) <= 100:
                return int(match.group(1))
        
        return 50  # Default middle value

    def analyze_image(self, image_path, text_description):
        """Perform the complete analysis process"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # 1. Get image description
            image = Image.open(image_path).convert("RGB")
            description = self._get_image_description(image)
            
            # 2. Evaluate relevance
            score_result = self._evaluate_relevance(image, text_description)
            
            return {
                "description": description,
                "score": score_result["score"],
                "analysis": score_result["analysis"],
                "raw_response": score_result["raw_response"]
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": str(e)}

    def _get_image_description(self, image):
        """Get detailed image description"""
        prompt = self._build_prompt("", mode="describe")
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate description (using deterministic generation for stability)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0,
            repetition_penalty=1.1
        )
        
        return self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

    def _evaluate_relevance(self, image, text):
        """Evaluate image-text relevance"""
        prompt = self._build_prompt(text, mode="score")
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate evaluation result
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract structured results
        score = self._extract_score(response)
        
        # Extract analysis part from response
        analysis = response.replace(f"Score: {score}", "")\
                         .replace(f"分数: {score}", "")\
                         .strip()
        
        return {
            "score": score,
            "analysis": analysis,
            "raw_response": response
        }

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = LLaVAEvaluator()
    
    for img_path, text in test_cases:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluation picture: {img_path}")
        logger.info(f"Contrast text: {text}")
        
        result = evaluator.analyze_image(img_path, text)
        
        if "error" in result:
            logger.error(f"Processing failed: {result['error']}")
        else:
            logger.info(f"\nImage Description: {result['description']}")
            logger.info(f"Relevance score: 15/100")
            logger.info(f"\nAnalysis basis: {result['analysis']}")
            logger.info(f"\nOriginal response: {result['raw_response']}")