#!/usr/bin/env python3
"""
Production Model Merger API Startup Script
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import transformers
        import flask
        logger.info("✅ All dependencies are installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            logger.warning("⚠️ CUDA not available, using CPU (will be slower)")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("🚀 Starting Production Model Merger API")
    
    # Check if dependencies are installed
    if not check_dependencies():
        logger.info("Installing missing dependencies...")
        if not install_dependencies():
            logger.error("Failed to install dependencies. Exiting.")
            sys.exit(1)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("merged_models", exist_ok=True)
    
    # Start the API
    logger.info("🌟 Model Merger API is ready!")
    logger.info("📡 API will be available at: http://localhost:5007")
    logger.info("📚 Endpoints:")
    logger.info("   GET  /api/model-merger/models - List available models")
    logger.info("   GET  /api/model-merger/experiments - List experiments")
    logger.info("   POST /api/model-merger/experiments - Create new experiment")
    logger.info("   GET  /api/model-merger/experiments/<id>/download - Download merged model")
    
    # Import and run the app
    from app import app
    app.run(host='0.0.0.0', port=5007, debug=False)

if __name__ == '__main__':
    main()
