#!/usr/bin/env python3
"""
Pre-download models for Model Merger API
This script downloads and caches popular models to avoid download delays during merging
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache directory
CACHE_DIR = Path("./model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Popular models to pre-download (smaller models first for testing)
MODELS_TO_DOWNLOAD = [
    {
        'id': 'microsoft-dialoGPT-small',
        'hf_id': 'microsoft/DialoGPT-small',
        'size': '117MB',
        'priority': 1  # Download first for testing
    },
    {
        'id': 'gpt2-small',
        'hf_id': 'gpt2',
        'size': '548MB', 
        'priority': 2
    },
    {
        'id': 'distilgpt2',
        'hf_id': 'distilgpt2',
        'size': '353MB',
        'priority': 3
    },
    # Larger production models (download only if requested)
    {
        'id': 'mistral-7b-instruct',
        'hf_id': 'mistralai/Mistral-7B-Instruct-v0.1',
        'size': '13.5GB',
        'priority': 10
    },
    {
        'id': 'zephyr-7b-beta',
        'hf_id': 'HuggingFaceH4/zephyr-7b-beta',
        'size': '13.5GB',
        'priority': 11
    }
]

def check_disk_space():
    """Check available disk space"""
    statvfs = os.statvfs('.')
    free_bytes = statvfs.f_frsize * statvfs.f_bavail
    free_gb = free_bytes / (1024**3)
    
    logger.info(f"💾 Available disk space: {free_gb:.1f}GB")
    
    if free_gb < 50:
        logger.warning("⚠️ Low disk space! Need at least 50GB for large models")
        return False
    return True

def check_gpu_memory():
    """Check GPU memory availability"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        logger.info(f"🎮 GPU Memory: {gpu_memory_gb:.1f}GB")
        return gpu_memory_gb > 6  # Need at least 6GB for 7B models
    else:
        logger.info("🖥️ No GPU available - using CPU (slower)")
        return False

def download_model(model_info, force=False):
    """Download and cache a single model"""
    model_id = model_info['id']
    hf_id = model_info['hf_id']
    size = model_info['size']
    
    cache_path = CACHE_DIR / model_id
    
    # Check if already downloaded
    if cache_path.exists() and not force:
        logger.info(f"✅ {model_id} already cached at {cache_path}")
        return True
    
    logger.info(f"📥 Downloading {model_id} ({size}) from {hf_id}...")
    
    try:
        start_time = time.time()
        
        # Download tokenizer
        logger.info(f"  📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_path / "tokenizer")
        
        # Download model
        logger.info(f"  🧠 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            cache_dir=cache_path / "model",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,  # Reduce memory usage during download
            trust_remote_code=True
        )
        
        # Save to cache
        logger.info(f"  💾 Saving to cache...")
        tokenizer.save_pretrained(cache_path / "tokenizer")
        model.save_pretrained(cache_path / "model")
        
        # Create metadata
        metadata = {
            'id': model_id,
            'hf_id': hf_id,
            'size': size,
            'downloaded_at': time.time(),
            'cache_path': str(cache_path)
        }
        
        import json
        with open(cache_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ {model_id} downloaded successfully in {elapsed:.1f}s")
        
        # Clear memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_id}: {e}")
        return False

def download_test_models():
    """Download small models for testing"""
    logger.info("🧪 Downloading test models...")
    
    test_models = [m for m in MODELS_TO_DOWNLOAD if m['priority'] <= 5]
    
    for model_info in test_models:
        success = download_model(model_info)
        if not success:
            logger.error(f"Failed to download test model {model_info['id']}")
            return False
    
    logger.info("✅ Test models downloaded successfully!")
    return True

def download_production_models():
    """Download large production models"""
    logger.info("🏭 Downloading production models...")
    
    if not check_disk_space():
        logger.error("❌ Insufficient disk space for production models")
        return False
    
    gpu_available = check_gpu_memory()
    if not gpu_available:
        logger.warning("⚠️ No GPU available - production models will be slow")
    
    production_models = [m for m in MODELS_TO_DOWNLOAD if m['priority'] >= 10]
    
    for model_info in production_models:
        logger.info(f"📥 Starting download of {model_info['id']} ({model_info['size']})")
        success = download_model(model_info)
        if not success:
            logger.error(f"Failed to download production model {model_info['id']}")
            return False
    
    logger.info("✅ Production models downloaded successfully!")
    return True

def list_cached_models():
    """List all cached models"""
    logger.info("📋 Cached models:")
    
    if not CACHE_DIR.exists():
        logger.info("  No models cached yet")
        return
    
    for model_dir in CACHE_DIR.iterdir():
        if model_dir.is_dir():
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file) as f:
                    metadata = json.load(f)
                logger.info(f"  ✅ {metadata['id']} ({metadata['size']})")
            else:
                logger.info(f"  📁 {model_dir.name} (no metadata)")

def main():
    """Main download function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python download_models.py test      # Download small test models")
        print("  python download_models.py prod      # Download production models")
        print("  python download_models.py list      # List cached models")
        print("  python download_models.py all       # Download all models")
        return
    
    command = sys.argv[1].lower()
    
    logger.info("🚀 Model Downloader Starting...")
    logger.info(f"📁 Cache directory: {CACHE_DIR.absolute()}")
    
    if command == "test":
        download_test_models()
    elif command == "prod":
        download_production_models()
    elif command == "list":
        list_cached_models()
    elif command == "all":
        download_test_models()
        download_production_models()
    else:
        logger.error(f"Unknown command: {command}")
        return
    
    logger.info("🎉 Download process completed!")

if __name__ == '__main__':
    main()
