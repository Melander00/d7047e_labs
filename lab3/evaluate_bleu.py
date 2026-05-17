import os
import torch
import json
from tqdm import tqdm
from dataset import get_loaders
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict

# Import our models
from Combinations.Base import BaseCaption
from Combinations.ResNet import ResNetCaption
from Combinations.Attention import AttentionCaption

def calculate_bleu_scores(model_name, model_class, dataset, test_subset, device, limit=None):
    print(f"\nEvaluating {model_name.upper()} model...")
    
    # 1. Initialize and load model
    vocab = dataset.vocab
    model = model_class(len(vocab)).to(device)
    weights_path = f"SavedModels/{model_name}/best_comb.pth"
    
    if not os.path.exists(weights_path):
        print(f"⚠️  No saved weights found at {weights_path}. Skipping.")
        return None
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Build full mapping of image_name -> all 5 references (tokenized)
    # Since each image has 5 different captions in Flickr8k, we want to evaluate
    # against all 5 references to get an accurate BLEU score.
    img_to_refs = defaultdict(list)
    for img_name, cap in zip(dataset.images, dataset.captions):
        # Tokenize reference caption (lowercased, split by space)
        ref_tokens = vocab.tokenizer(cap)
        img_to_refs[img_name].append(ref_tokens)

    # 3. Evaluate on the test subset
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    # Define smoothing function to handle short sentences / zero n-gram matches
    chencherry = SmoothingFunction()

    # We evaluate up to the limit if specified (e.g. for a quick test)
    indices = test_subset.indices
    if limit:
        indices = indices[:limit]
        print(f"Running quick evaluation on {limit} test images...")

    with torch.no_grad():
        for idx in tqdm(indices):
            # Get the image tensor and image name
            img_tensor, _ = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]
            img_name = dataset.images[idx]
            
            # Generate caption
            generated_caption = model.generate_caption(img_tensor, vocab)
            
            # Tokenize generated caption
            hypothesis = vocab.tokenizer(generated_caption)
            
            # Get all 5 reference token lists for this image
            references = img_to_refs[img_name]
            
            # Calculate BLEU scores with smoothing
            b1 = sentence_bleu(references, hypothesis, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1)
            b2 = sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
            b3 = sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
            b4 = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
            
            bleu1_scores.append(b1)
            bleu2_scores.append(b2)
            bleu3_scores.append(b3)
            bleu4_scores.append(b4)

    # Calculate average scores
    avg_b1 = sum(bleu1_scores) / len(bleu1_scores) * 100
    avg_b2 = sum(bleu2_scores) / len(bleu2_scores) * 100
    avg_b3 = sum(bleu3_scores) / len(bleu3_scores) * 100
    avg_b4 = sum(bleu4_scores) / len(bleu4_scores) * 100

    print(f"📊 {model_name.upper()} Scores -> BLEU-1: {avg_b1:.2f} | BLEU-2: {avg_b2:.2f} | BLEU-3: {avg_b3:.2f} | BLEU-4: {avg_b4:.2f}")
    
    return {
        "BLEU-1": avg_b1,
        "BLEU-2": avg_b2,
        "BLEU-3": avg_b3,
        "BLEU-4": avg_b4
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloaders
    try:
        loaders, dataset = get_loaders("Data")
        test_subset = loaders[2].dataset # This is the Subset object for the test split
        print(f"Loaded test split with {len(test_subset)} samples.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("   → Make sure the 'Data/' folder exists with 'Images/' and 'captions.txt'.")
        return

    models_to_evaluate = {
        "base": BaseCaption,
        "resnet": ResNetCaption,
        "attention": AttentionCaption
    }

    results = {}
    
    # Change limit to None to evaluate the ENTIRE test set (takes ~2-3 mins on CPU)
    # We set limit=200 here for a quick but highly representative evaluate run!
    eval_limit = 200 

    for model_name, model_class in models_to_evaluate.items():
        scores = calculate_bleu_scores(model_name, model_class, dataset, test_subset, device, limit=eval_limit)
        if scores:
            results[model_name] = scores

    # 4. Print beautiful comparison table
    if results:
        print("\n" + "="*65)
        print("🏆  FINAL QUANTITATIVE EVALUATION RESULTS (BLEU SCORES)  🏆")
        print("="*65)
        print(f"{'Model Name':<15} | {'BLEU-1 (%)':<10} | {'BLEU-2 (%)':<10} | {'BLEU-3 (%)':<10} | {'BLEU-4 (%)':<10}")
        print("-"*65)
        for model_name, scores in results.items():
            print(f"{model_name.upper():<15} | {scores['BLEU-1']:<10.2f} | {scores['BLEU-2']:<10.2f} | {scores['BLEU-3']:<10.2f} | {scores['BLEU-4']:<10.2f}")
        print("="*65)
        print("💡 Tip: Add this exact table to your final lab report report!")
        print("="*65)

if __name__ == "__main__":
    main()
