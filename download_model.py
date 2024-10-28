from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os 
import argparse


def download_hf_model_dynamic(model_repo):
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    local_snapshots_dir = f"{os.getenv('MODEL_DIR')}/{model_repo}"
    try:
        snapshot_path = snapshot_download(
            repo_id=model_repo,
            cache_dir=local_snapshots_dir,
            token=hf_token
        )
        print(f"The entire repository has been downloaded to\n: {snapshot_path}", "\npaste this path into the model config file")
    except Exception as e:
        print(f"Failed to download the model: {str(e)}")



def main():
    parser = argparse.ArgumentParser(description='Download a model from Hugging Face Hub.')
    parser.add_argument('--model', type=str, help='Model name to download')
    args = parser.parse_args()

    download_hf_model_dynamic(args.model)
    

if __name__ == "__main__":
    load_dotenv()
    main()