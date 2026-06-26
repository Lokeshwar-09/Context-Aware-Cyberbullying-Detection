import os
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi, create_repo

def main():
    print("=" * 60)
    print("   Upload Trained Model to Hugging Face Hub")
    print("=" * 60)
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        print(f"Error: Local folder '{model_dir}' not found.")
        print("Please run this script from the project root directory where the 'models' folder exists.")
        return
        
    required_files = ["model.safetensors", "config.json", "vocab.txt", "label_encoder.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing_files:
        print(f"Warning: The following expected files are missing from '{model_dir}': {missing_files}")
        confirm = input("Do you still want to proceed? (y/n): ").strip().lower()
        if confirm != 'y':
            return

    token = input("Enter your Hugging Face Write Token (get one from https://huggingface.co/settings/tokens): ").strip()
    if not token:
        print("Error: Hugging Face token is required to upload models.")
        return

    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        username = user_info["name"]
        print(f"Authenticated successfully as: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    repo_name = f"{username}/context-aware-cyberbullying-detection"
    print(f"Target repository: https://huggingface.co/{repo_name}")
    
    try:
        print("Creating Hugging Face model repository (if it doesn't exist)...")
        create_repo(repo_id=repo_name, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    print("Uploading model files to Hugging Face...")
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your model has been successfully uploaded to Hugging Face!")
        print(f"Repository Link: https://huggingface.co/{repo_name}")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. If your Hugging Face username is NOT 'Lokeshwar-09', add the following secret in your Streamlit Cloud app settings (under Secrets):")
        print(f'   HF_MODEL_REPO = "{repo_name}"')
        print("2. Reboot your Streamlit Cloud application.")
        print("=" * 60)
    except Exception as e:
        print(f"Error uploading folder: {e}")

if __name__ == "__main__":
    main()
