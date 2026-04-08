import json
import os
import subprocess

from tqdm import tqdm

FP_1K = "./data/amazon_raw/amazon_cells_labelled.txt"
FP_25K = "./data/amazon_raw/amazon_cells_labelled_LARGE_25K.txt"

def load_amazon_reviews(
    file_path = FP_1K,
    show_progress_bar = True
):
    reviews = []

    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(tqdm(file, desc="Loading reviews", unit=" reviews", leave=False, disable=not show_progress_bar)):
            text, label = line.split("\t")
            label = int(label.replace("\n", ""))
            reviews.append((text, label))

    return reviews


def save_extracts_to_disk(
    extracts: list[tuple[str, int]],
    outdir: str = "./data/precompute",
    outname: str = "amazon.jsonl",
    show_progress_bar = True,
):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, outname), "w", encoding="utf-8") as file:
        for text, label in tqdm(extracts, desc="Saving data", unit=" reviews", leave=False, disable=not show_progress_bar):
            entry = {
                "text": text, 
                "label": label
            }
            file.write(json.dumps(entry)+"\n")



def get_amazon_file_path(
    use_25k_set,
    raw_file = None,
    show_progress_bars = True,
):
    precompute_dir = "./data/precompute"
    precompute_name = "amazon_25k.jsonl" if use_25k_set else "amazon_1k.jsonl"
    precompute_file = os.path.join(precompute_dir, precompute_name)

    if os.path.exists(precompute_file):
        return precompute_file
    
    if raw_file is None:
        raw_file = FP_25K if use_25k_set else FP_1K

    reviews = load_amazon_reviews(file_path=raw_file, show_progress_bar=show_progress_bars)
    save_extracts_to_disk(extracts=reviews, outdir=precompute_dir, outname=precompute_name, show_progress_bar=show_progress_bars)

    return precompute_file

if __name__ == "__main__":
    x = get_amazon_file_path(use_25k_set=True)
    print(x)