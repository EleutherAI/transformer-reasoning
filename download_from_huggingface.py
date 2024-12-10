from transformers import AutoTokenizer
from datasets import load_dataset

def download_resources():
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama_multihop_tokenizer")
    print("✓ Tokenizer downloaded")

    # Download datasets
    print("\nDownloading datasets...")
    for N in [1000, 10000, 15000, 25000]:
        dataset = load_dataset(f"EleutherAI/profiles_dataset_{N}_uniform_r17")
        print("✓ Datasets downloaded")

if __name__ == "__main__":
    download_resources()