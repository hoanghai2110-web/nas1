
#!/usr/bin/env python3
"""
🇻🇳 ULTRA-VIETNAMESE PRETRAINING RUNNER 🇻🇳
Main script to run ULTRA Vietnamese pretraining
DESIGNED TO OUTPERFORM GEMMA, LLAMA, CHATGPT!
"""

import sys
import time
import torch
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import torch
        import numpy as np
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install torch numpy")
        return False

def main():
    """
    ULTRA-Main execution function
    """
    print("🇻🇳" + "="*60 + "🇻🇳")
    print("     ULTRA-VIETNAMESE AI PRETRAINING")
    print("  🎯 OUTPERFORM GEMMA, LLAMA, CHATGPT!")
    print("🇻🇳" + "="*60 + "🇻🇳")
    
    # Check system requirements
    print("\n🔍 CHECKING SYSTEM REQUIREMENTS...")
    if not check_requirements():
        return
    
    try:
        # Import after checking dependencies
        from pretrain import main as pretrain_main
        
        print("\n🚀 STARTING ULTRA-PRETRAINING...")
        start_time = time.time()
        
        # Run ULTRA pretraining
        pretrain_main()
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "="*70)
        print("🎉 ULTRA-PRETRAINING HOÀN THÀNH!")
        print(f"⏱️  Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print("🏆 MODEL SẴN SÀNG VƯỢT QUA COMPETITORS!")
        print("="*70)
        
        # Show final stats
        if Path('ultra_best_model.pt').exists():
            model_size = Path('ultra_best_model.pt').stat().st_size / (1024*1024)
            print(f"💾 Model size: {model_size:.1f} MB")
        
        if Path('ultra_tokenizer.pkl').exists():
            print("✅ ULTRA-Tokenizer saved")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Test generation with your prompts")
        print("2. Fine-tune on specific Vietnamese tasks")
        print("3. Deploy and showcase Vietnamese AI superiority!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install torch numpy")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TROUBLESHOOTING:")
        print("1. Check if you have enough disk space")
        print("2. Reduce batch_size if out of memory")
        print("3. Ensure data files exist (data.txt, data1.txt, data2.txt)")

if __name__ == "__main__":
    main()
