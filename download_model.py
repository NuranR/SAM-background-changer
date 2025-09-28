import os

def check_model():
    """Check if the SAM 2 model is properly placed"""
    
    expected_filename = "sam2.1_hiera_small.pt"
    current_dir = os.getcwd()
    
    print("=" * 50)
    print("SAM 2 Model Setup Instructions")
    print("=" * 50)
    print()
    print(f"📁 Current directory: {current_dir}")
    print()
    print("📥 Model file location:")
    print(f"   Place your downloaded model file here: {os.path.join(current_dir, expected_filename)}")
    print()
    print("🔍 Expected filename: sam2.1_hiera_small.pt")
    print()
    
    if os.path.exists(expected_filename):
        file_size = os.path.getsize(expected_filename) / (1024 * 1024)  # Size in MB
        print(f"✅ Model found! ({file_size:.1f} MB)")
        print("   Your model is ready to use.")
    else:
        print("❌ Model not found!")
        print()
        print("📋 Steps to add the model:")
        print("   1. Download the SAM 2.1 Hiera Small model from Hugging Face")
        print("   2. Rename it to: sam2.1_hiera_small.pt")
        print(f"   3. Place it in: {current_dir}")
        print()
        print("🔗 Hugging Face model links:")
        print("   - facebook/sam2-hiera-small")
        print("   - facebook/sam2.1-hiera-small")
    
    print("=" * 50)

if __name__ == "__main__":
    check_model()