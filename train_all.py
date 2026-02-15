"""
All-in-One Training Script
Automatically downloads datasets and trains the AI detection model.

Usage:
    python train_all.py
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display progress."""
    print("\n" + "="*60)
    print(f"🚀 {description}")
    print("="*60)

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)

    print(f"✅ {description} completed successfully!")
    return result

def main():
    print("\n" + "="*60)
    print("🎯 AI Text Detection - Complete Training Pipeline")
    print("="*60)
    print("\nThis script will:")
    print("1. Download training datasets from Hugging Face")
    print("2. Prepare and process the data")
    print("3. Train the DeBERTa classifier on your GPU")
    print("4. Save the fine-tuned model")
    print("\nEstimated time: 2-3 hours total")
    print("="*60)

    # Check if data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Step 1: Prepare training data
    run_command(
        "python -m training.prepare_data",
        "Step 1/2: Downloading and preparing training datasets"
    )

    # Step 2: Train the classifier
    run_command(
        "python -m training.train_classifier",
        "Step 2/2: Training DeBERTa classifier on GPU"
    )

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("\n✅ Your fine-tuned model is ready!")
    print(f"📁 Model saved to: models/detector/best/")
    print("\n🚀 Next steps:")
    print("  1. Test the web UI: streamlit run app.py")
    print("  2. Test the API: uvicorn api:app --reload")
    print("  3. Run detailed analysis: python analyze_text.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
