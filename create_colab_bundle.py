import shutil
from pathlib import Path

def create_colab_bundle():
    print("Creating Colab bundle...")
    bundle_dir = Path("colab_bundle")
    bundle_dir.mkdir(exist_ok=True)
    
    # Copy necessary directories
    dirs_to_copy = ["src", "configs"]
    for d in dirs_to_copy:
        shutil.copytree(d, bundle_dir / d, dirs_exist_ok=True)
        
    # Copy processed data
    shutil.copytree(Path("data/processed"), bundle_dir / "data/processed", dirs_exist_ok=True)
    
    # Zip it up
    print("Zipping...")
    shutil.make_archive("nba_predict_colab", "zip", bundle_dir)
    
    # Clean up
    shutil.rmtree(bundle_dir)
    print("Done! Upload 'nba_predict_colab.zip' to Google Colab.")

if __name__ == "__main__":
    create_colab_bundle()
