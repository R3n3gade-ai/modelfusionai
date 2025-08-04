# 🧠 Production Model Merger API

A production-ready API for merging Large Language Models (LLMs) using state-of-the-art algorithms.

## 🚀 Features

### **Merge Algorithms**
- **Linear (Model Soup)** - Simple weighted averaging of model parameters
- **SLERP** - Spherical Linear Interpolation for smooth 2-model merging
- **TIES** - Trim, Elect Sign, and Merge for multiple models
- **DARE** - Drop and Rescale for efficient parameter merging
- **MoE** - Mixture of Experts (coming soon)

### **Supported Models**
- **Mistral 7B Instruct** - General purpose instruction-following
- **Zephyr 7B Beta** - Helpful assistant model
- **Code Llama 7B** - Code generation and understanding
- **Llama 2 7B Chat** - Conversational AI
- **OpenChat 7B** - High-performance chat model

### **Production Features**
- ✅ Real-time progress tracking
- ✅ Asynchronous processing
- ✅ Model download and export
- ✅ Experiment management
- ✅ GPU acceleration support
- ✅ Error handling and recovery

## 📋 Requirements

### **System Requirements**
- Python 3.8+
- 16GB+ RAM (32GB+ recommended)
- 50GB+ free disk space
- CUDA-compatible GPU (optional but recommended)

### **Dependencies**
- PyTorch 2.0+
- Transformers 4.35+
- Flask 2.3+
- CUDA Toolkit (for GPU support)

## 🛠️ Installation

### **1. Clone Repository**
```bash
cd model-merger-api
```

### **2. Install Dependencies**
```bash
# Option 1: Automatic installation
python start.py

# Option 2: Manual installation
pip install -r requirements.txt
```

### **3. Start API Server**
```bash
python start.py
```

The API will be available at: `http://localhost:5007`

## 📡 API Endpoints

### **GET /api/model-merger/models**
List available models for merging
```json
{
  "success": true,
  "models": [
    {
      "id": "mistral-7b-instruct",
      "name": "Mistral 7B Instruct v0.1",
      "size": "7B",
      "architecture": "mistral",
      "download_size": "13.5GB"
    }
  ]
}
```

### **POST /api/model-merger/experiments**
Create a new model merging experiment
```json
{
  "name": "My Custom Model",
  "method": "linear",
  "models": [
    {"id": "mistral-7b-instruct", "weight": 0.6},
    {"id": "zephyr-7b-beta", "weight": 0.4}
  ],
  "parameters": {}
}
```

### **GET /api/model-merger/experiments**
List all experiments with status and progress

### **GET /api/model-merger/experiments/{id}/download**
Download merged model as ZIP file

## 🔬 Merge Methods

### **Linear (Model Soup)**
```python
merged_param = Σ(weight_i × model_i_param) / Σ(weight_i)
```
- **Best for**: General merging, multiple models
- **Parameters**: Model weights
- **Models**: 2-8 models

### **SLERP**
```python
merged_param = sin((1-α)θ)/sin(θ) × param1 + sin(αθ)/sin(θ) × param2
```
- **Best for**: Smooth interpolation between 2 models
- **Parameters**: Alpha (interpolation factor)
- **Models**: Exactly 2 models

### **TIES**
```python
# 1. Trim: Keep top-k% parameters by magnitude
# 2. Elect: Resolve sign conflicts via majority vote
# 3. Merge: Weighted average of task vectors
```
- **Best for**: Multiple specialized models
- **Parameters**: Density (sparsity level)
- **Models**: 2+ models

### **DARE**
```python
# 1. Drop: Random parameter dropout
# 2. Rescale: Compensate for dropped parameters
merged_param = base_param + Σ(weight_i × dropped_rescaled_task_vector_i)
```
- **Best for**: Efficient merging with regularization
- **Parameters**: Drop rate
- **Models**: 2+ models

## 🎯 Usage Examples

### **Frontend Integration**
The API is designed to work with the Metatron frontend Model Training Studio.

### **Direct API Usage**
```bash
# Start a merge experiment
curl -X POST http://localhost:5007/api/model-merger/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Code + Chat Model",
    "method": "linear",
    "models": [
      {"id": "code-llama-7b", "weight": 0.7},
      {"id": "llama2-7b-chat", "weight": 0.3}
    ]
  }'

# Check experiment status
curl http://localhost:5007/api/model-merger/experiments

# Download merged model
curl -O http://localhost:5007/api/model-merger/experiments/{id}/download
```

## ⚡ Performance

### **GPU Acceleration**
- Automatic CUDA detection
- Mixed precision support
- Memory optimization

### **Benchmarks**
- **7B Model Merge**: ~10-30 minutes (GPU) / ~1-3 hours (CPU)
- **Memory Usage**: ~20-40GB peak
- **Disk Space**: ~40GB per merged model

## 🔧 Configuration

### **Environment Variables**
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export HF_HOME=/path/to/cache  # HuggingFace cache
export TORCH_HOME=/path/to/torch  # PyTorch cache
```

## 🚨 Troubleshooting

### **Common Issues**
1. **Out of Memory**: Reduce batch size or use CPU
2. **Model Download Fails**: Check internet connection and HF token
3. **CUDA Errors**: Update GPU drivers and CUDA toolkit

### **Logs**
Check console output for detailed error messages and progress updates.

## 📄 License

This project is part of the Metatron AI Platform.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

---

**Ready to merge some models?** 🚀 Start the API and open the Metatron Model Training Studio!
