import json
import os
import subprocess

from tqdm import tqdm


def load_yelp_reviews_json(
    json_file = "./data/yelp_raw/yelp_academic_dataset_review.json", 
    max_lines = -1,
    show_progress_bar = True,
):
    """
    Loads the JSON file containing the yelp reviews. 
    
    Only loads as many entries as `max_lines`.
    Set `max_lines = -1` for full dataset (all rows).
    """
    reviews = []        

    with open(json_file, "r", encoding="utf-8") as file:
        for i, line in enumerate(tqdm(file, desc="Loading reviews", unit=" reviews", leave=False, disable=not show_progress_bar)):
            if max_lines >= 0 and i >= max_lines:
                break
            reviews.append(json.loads(line.strip()))
    
    return reviews



def extract_fields(
    reviews: list,
    show_progress_bar = True,
): 
    
    extracts: list[tuple[str, int]] = []

    for review in tqdm(reviews, desc="Extracting info", unit=" reviews", leave=False, disable=not show_progress_bar):
        try:
            text = str(review['text'])
            stars = int(review['stars'])
        except KeyError:
            # This happens in case stars or text are undefined. If so we simply skip this entry.
            continue

        if text == "": # We can't do sentiment analysis on empty text.
            continue

        extracts.append((text, stars))

    return extracts
        

def stars_to_label(stars: int):
    if stars <= 2:
        return 0 # <-- Changed from -1 to 0 to match the provided datasets. 
    if stars >= 4:
        return 1
    return None # <-- The provided datasets didn't have a "neutral" sentiment

def save_extracts_to_disk(
    extracts: list[tuple[str, int]],
    outdir: str = "./data/precompute",
    outname: str = "reviews.jsonl",
    show_progress_bar = True,
):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, outname), "w", encoding="utf-8", newline="\n") as file:
        for text, stars in tqdm(extracts, desc="Saving data", unit=" reviews", leave=False, disable=not show_progress_bar):
            label = stars_to_label(stars)

            if label is None:
                continue

            entry = {
                "text": text, 
                "label": label
            }
            file.write(json.dumps(entry)+"\n")
    
    
def get_yelp_file_path(
    raw_file = "./data/yelp_raw/yelp_academic_dataset_review.json", 
    entries = -1,
    show_progress_bars = True
):
    precompute_dir = "./data/precompute"
    precompute_name = f"reviews_{entries}.jsonl" if entries >= 0 else "reviews_all.jsonl"
    precompute_file = os.path.join(precompute_dir, precompute_name)

    if os.path.exists(precompute_file):
        return precompute_file

    reviews = load_yelp_reviews_json(raw_file, max_lines=entries, show_progress_bar=show_progress_bars)
    extracts = extract_fields(reviews, show_progress_bar=show_progress_bars)
    save_extracts_to_disk(extracts, outdir=precompute_dir, outname=precompute_name, show_progress_bar=show_progress_bars)

    return precompute_file


if __name__ == "__main__":
    fp = get_yelp_file_path()
    print(fp)