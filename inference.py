# character_analysis.py
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from nltk.translate.bleu_score import sentence_bleu  # 修正这里
import nltk
import argparse

nltk.download('punkt', quiet=True)

def main():
    parser = argparse.ArgumentParser(description='角色性格分析脚本')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--reference', type=str, required=True)
    args = parser.parse_args()

    print("正在加载LLaVA模型...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    def generate_description(image_path):
        image = Image.open(image_path)
        prompt = "请详细描述此图中角色的性格特征，包括但不限于外表、行为习惯和可能的性格倾向:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        generate_ids = model.generate(
            **inputs,
            max_length=500,
            do_sample=True,
            temperature=0.7
        )
        return processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    def calculate_similarity(generated, reference):
        return sentence_bleu([reference.split()], generated.split())

    print("\n正在生成性格描述...")
    generated_desc = generate_description(args.image)
    print("\n生成的描述:\n", generated_desc)

    with open(args.reference, 'r', encoding='utf-8') as f:
        real_desc = f.read()

    score = calculate_similarity(generated_desc, real_desc)
    print(f"\nBLEU相似度得分: {score:.4f}")

if __name__ == "__main__":
    main()