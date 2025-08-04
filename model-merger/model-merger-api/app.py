#!/usr/bin/env python3
"""
Production Model Merger API
Implements real model merging algorithms for LLMs
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import uuid
import threading
import time
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MODELS_DIR = Path("./models")
EXPERIMENTS_DIR = Path("./experiments")
MERGED_MODELS_DIR = Path("./merged_models")

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)
MERGED_MODELS_DIR.mkdir(exist_ok=True)

# Global state
experiments = {}
model_library = {}

# Persistence functions
def save_experiments():
    """Save experiments to disk"""
    try:
        experiments_file = EXPERIMENTS_DIR / 'experiments.json'
        with open(experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save experiments: {e}")

def load_experiments():
    """Load experiments from disk"""
    try:
        experiments_file = EXPERIMENTS_DIR / 'experiments.json'
        if experiments_file.exists():
            with open(experiments_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load experiments: {e}")
    return {}

class ModelMerger:
    """Production model merging implementation"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def linear_merge(self, models, weights=None):
        """Linear interpolation (Model Soup) merging - Following Sebastian Raschka's protocol"""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        logger.info("Instantiating merged model from first model config")
        # Create merged model from first model's config (Sebastian's approach)
        merged_model = AutoModelForCausalLM.from_config(models[0].config).to("cpu")

        # Count total parameters for progress tracking
        total_params = sum(1 for _ in models[0].parameters())
        logger.info(f"Merging {total_params} parameters across {len(models)} models")

        # Sebastian's exact merging protocol
        with torch.no_grad():
            if len(models) == 2:
                # Two model case - Sebastian's original formula
                alpha = weights[0]  # First model weight
                logger.info(f"Two-model merge with alpha={alpha}")

                for param1, param2, merged_param in zip(
                    models[0].parameters(), models[1].parameters(), merged_model.parameters()
                ):
                    # Sebastian's formula: alpha * param1 + (1 - alpha) * param2
                    merged_param.data = alpha * param1.data + (1 - alpha) * param2.data
            else:
                # Multi-model case - weighted average
                logger.info(f"Multi-model merge with weights: {weights}")

                # Initialize with first model weighted
                for param_first, merged_param in zip(models[0].parameters(), merged_model.parameters()):
                    merged_param.data = weights[0] * param_first.data

                # Add weighted parameters from remaining models
                for model, weight in zip(models[1:], weights[1:]):
                    for param, merged_param in zip(model.parameters(), merged_model.parameters()):
                        merged_param.data.add_(weight * param.data)

        return merged_model
    
    def slerp_merge(self, model1, model2, alpha=0.5):
        """Spherical Linear Interpolation merging"""
        merged_state_dict = {}
        
        for key in model1.state_dict().keys():
            param1 = model1.state_dict()[key]
            param2 = model2.state_dict()[key]
            
            # Flatten parameters for SLERP
            flat1 = param1.flatten()
            flat2 = param2.flatten()
            
            # Compute dot product
            dot = torch.dot(flat1, flat2)
            
            # Clamp to avoid numerical issues
            dot = torch.clamp(dot, -1.0, 1.0)
            
            # Compute angle
            theta = torch.acos(torch.abs(dot))
            
            # SLERP formula
            if theta < 1e-6:  # Vectors are nearly parallel
                merged_flat = (1 - alpha) * flat1 + alpha * flat2
            else:
                sin_theta = torch.sin(theta)
                merged_flat = (torch.sin((1 - alpha) * theta) * flat1 + 
                              torch.sin(alpha * theta) * flat2) / sin_theta
            
            # Reshape back to original shape
            merged_state_dict[key] = merged_flat.reshape(param1.shape)
        
        return merged_state_dict
    
    def ties_merge(self, models, weights=None, density=0.5):
        """TIES merging algorithm"""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        merged_state_dict = {}
        
        for key in models[0].state_dict().keys():
            # Collect all parameters for this layer
            params = [model.state_dict()[key] for model in models]
            
            # Compute task vectors (difference from base)
            base_param = params[0]  # Use first model as base
            task_vectors = [param - base_param for param in params[1:]]
            
            if not task_vectors:  # Only one model
                merged_state_dict[key] = base_param
                continue
            
            # Stack task vectors
            stacked = torch.stack(task_vectors)
            
            # Trim: Keep only top-k% of parameters by magnitude
            flat_stacked = stacked.flatten(1)
            magnitudes = torch.abs(flat_stacked)
            
            # Find threshold for top density% of parameters
            k = int(density * flat_stacked.shape[1])
            if k > 0:
                threshold = torch.topk(magnitudes.max(0)[0], k)[0][-1]
                mask = magnitudes.max(0)[0] >= threshold
            else:
                mask = torch.zeros_like(magnitudes.max(0)[0], dtype=torch.bool)
            
            # Elect: Resolve sign conflicts
            signs = torch.sign(flat_stacked)
            sign_consensus = torch.mode(signs, dim=0)[0]
            
            # Apply mask and consensus
            for i, tv in enumerate(task_vectors):
                flat_tv = tv.flatten()
                flat_tv = flat_tv * mask.float()
                flat_tv = flat_tv * (torch.sign(flat_tv) == sign_consensus).float()
                task_vectors[i] = flat_tv.reshape(tv.shape)
            
            # Merge: Weighted average of task vectors
            merged_tv = torch.zeros_like(base_param)
            for tv, weight in zip(task_vectors, weights[1:]):
                merged_tv += weight * tv
            
            merged_state_dict[key] = base_param + merged_tv
        
        return merged_state_dict
    
    def dare_merge(self, models, weights=None, drop_rate=0.1):
        """DARE merging algorithm"""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        merged_state_dict = {}
        
        for key in models[0].state_dict().keys():
            params = [model.state_dict()[key] for model in models]
            base_param = params[0]
            
            # Compute task vectors
            task_vectors = [param - base_param for param in params[1:]]
            
            if not task_vectors:
                merged_state_dict[key] = base_param
                continue
            
            # Drop and rescale
            merged_tv = torch.zeros_like(base_param)
            for tv, weight in zip(task_vectors, weights[1:]):
                # Random dropout
                mask = torch.rand_like(tv) > drop_rate
                dropped_tv = tv * mask.float()
                
                # Rescale to maintain expected magnitude
                if drop_rate < 1.0:
                    dropped_tv = dropped_tv / (1.0 - drop_rate)
                
                merged_tv += weight * dropped_tv
            
            merged_state_dict[key] = base_param + merged_tv
        
        return merged_state_dict

# Initialize merger
merger = ModelMerger()

@app.route('/api/model-merger/models', methods=['GET'])
def get_models():
    """Get available models for merging"""
    try:
        # Production model library with real HuggingFace models
        models = [
            # Small test models for faster debugging
            {
                'id': 'gpt2-small',
                'name': 'GPT-2 Small (124M)',
                'size': '124M',
                'type': 'test',
                'description': 'Small test model for quick merging',
                'huggingface_id': 'gpt2',
                'architecture': 'gpt2',
                'specialties': ['testing', 'small'],
                'download_size': '548MB',
                'context_length': 1024
            },
            {
                'id': 'distilgpt2',
                'name': 'DistilGPT-2 (82M)',
                'size': '82M',
                'type': 'test',
                'description': 'Smaller test model for quick merging',
                'huggingface_id': 'distilgpt2',
                'architecture': 'gpt2',
                'specialties': ['testing', 'small'],
                'download_size': '353MB',
                'context_length': 1024
            },
            # Large production models
            {
                'id': 'mistral-7b-instruct',
                'name': 'Mistral 7B Instruct v0.1',
                'size': '7B',
                'type': 'instruction',
                'description': 'General purpose instruction-following model',
                'huggingface_id': 'mistralai/Mistral-7B-Instruct-v0.1',
                'architecture': 'mistral',
                'specialties': ['general', 'instruction-following'],
                'download_size': '13.5GB',
                'context_length': 8192
            },
            {
                'id': 'zephyr-7b-beta',
                'name': 'Zephyr 7B Beta',
                'size': '7B',
                'type': 'chat',
                'description': 'Helpful assistant model fine-tuned from Mistral',
                'huggingface_id': 'HuggingFaceH4/zephyr-7b-beta',
                'architecture': 'mistral',
                'specialties': ['chat', 'helpfulness'],
                'download_size': '13.5GB',
                'context_length': 8192
            },
            {
                'id': 'code-llama-7b',
                'name': 'Code Llama 7B',
                'size': '7B',
                'type': 'code',
                'description': 'Code generation and understanding',
                'huggingface_id': 'codellama/CodeLlama-7b-hf',
                'architecture': 'llama',
                'specialties': ['coding', 'programming'],
                'download_size': '13.5GB',
                'context_length': 16384
            },
            {
                'id': 'llama2-7b-chat',
                'name': 'Llama 2 7B Chat',
                'size': '7B',
                'type': 'chat',
                'description': 'Conversational AI model',
                'huggingface_id': 'meta-llama/Llama-2-7b-chat-hf',
                'architecture': 'llama',
                'specialties': ['conversation', 'general'],
                'download_size': '13.5GB',
                'context_length': 4096
            },
            {
                'id': 'openchat-7b',
                'name': 'OpenChat 7B',
                'size': '7B',
                'type': 'chat',
                'description': 'High-performance chat model',
                'huggingface_id': 'openchat/openchat_3.5',
                'architecture': 'mistral',
                'specialties': ['chat', 'reasoning'],
                'download_size': '13.5GB',
                'context_length': 8192
            },
            # Gemma 3 Models (Latest Release!)
            {
                'id': 'gemma-3-1b-it',
                'name': 'Gemma 3 1B Instruct',
                'size': '1B',
                'type': 'chat',
                'description': 'Google\'s latest Gemma 3 ultra-efficient model with multimodal capabilities',
                'huggingface_id': 'google/gemma-3-1b-it',
                'architecture': 'gemma3',
                'specialties': ['chat', 'efficiency', 'multimodal', 'latest'],
                'download_size': '2.5GB',
                'context_length': 32768
            },
            {
                'id': 'gemma-3-4b-it',
                'name': 'Gemma 3 4B Instruct',
                'size': '4B',
                'type': 'chat',
                'description': 'Google\'s Gemma 3 balanced model with 128K context and multimodal support',
                'huggingface_id': 'google/gemma-3-4b-it',
                'architecture': 'gemma3',
                'specialties': ['chat', 'multimodal', 'long-context', 'latest'],
                'download_size': '8GB',
                'context_length': 131072
            },
            {
                'id': 'gemma-3-12b-it',
                'name': 'Gemma 3 12B Instruct',
                'size': '12B',
                'type': 'chat',
                'description': 'Google\'s powerful Gemma 3 model with 128K context and advanced reasoning',
                'huggingface_id': 'google/gemma-3-12b-it',
                'architecture': 'gemma3',
                'specialties': ['chat', 'reasoning', 'multimodal', 'long-context', 'latest'],
                'download_size': '24GB',
                'context_length': 131072
            },
            {
                'id': 'gemma-3-27b-it',
                'name': 'Gemma 3 27B Instruct',
                'size': '27B',
                'type': 'chat',
                'description': 'Google\'s flagship Gemma 3 model with 128K context and state-of-the-art performance',
                'huggingface_id': 'google/gemma-3-27b-it',
                'architecture': 'gemma3',
                'specialties': ['chat', 'reasoning', 'multimodal', 'long-context', 'flagship', 'latest'],
                'download_size': '54GB',
                'context_length': 131072
            },
            # Gemma 2 Models
            {
                'id': 'gemma-2-2b',
                'name': 'Gemma 2 2B',
                'size': '2B',
                'type': 'instruction',
                'description': 'Google\'s efficient Gemma 2 model for instruction following',
                'huggingface_id': 'google/gemma-2-2b',
                'architecture': 'gemma',
                'specialties': ['instruction-following', 'efficiency'],
                'download_size': '4.5GB',
                'context_length': 8192
            },
            {
                'id': 'gemma-2-9b',
                'name': 'Gemma 2 9B',
                'size': '9B',
                'type': 'instruction',
                'description': 'Google\'s powerful Gemma 2 model with enhanced capabilities',
                'huggingface_id': 'google/gemma-2-9b',
                'architecture': 'gemma',
                'specialties': ['instruction-following', 'reasoning'],
                'download_size': '18GB',
                'context_length': 8192
            },
            {
                'id': 'gemma-2-9b-it',
                'name': 'Gemma 2 9B Instruct',
                'size': '9B',
                'type': 'chat',
                'description': 'Instruction-tuned version of Gemma 2 9B for conversations',
                'huggingface_id': 'google/gemma-2-9b-it',
                'architecture': 'gemma',
                'specialties': ['chat', 'instruction-following'],
                'download_size': '18GB',
                'context_length': 8192
            },
            # DeepSeek Models
            {
                'id': 'deepseek-coder-6.7b',
                'name': 'DeepSeek Coder 6.7B',
                'size': '6.7B',
                'type': 'code',
                'description': 'Specialized coding model from DeepSeek',
                'huggingface_id': 'deepseek-ai/deepseek-coder-6.7b-base',
                'architecture': 'deepseek',
                'specialties': ['coding', 'programming', 'software-development'],
                'download_size': '13GB',
                'context_length': 16384
            },
            {
                'id': 'deepseek-coder-6.7b-instruct',
                'name': 'DeepSeek Coder 6.7B Instruct',
                'size': '6.7B',
                'type': 'code',
                'description': 'Instruction-tuned DeepSeek Coder for code generation',
                'huggingface_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
                'architecture': 'deepseek',
                'specialties': ['coding', 'instruction-following', 'code-generation'],
                'download_size': '13GB',
                'context_length': 16384
            },
            {
                'id': 'deepseek-llm-7b-chat',
                'name': 'DeepSeek LLM 7B Chat',
                'size': '7B',
                'type': 'chat',
                'description': 'General purpose chat model from DeepSeek',
                'huggingface_id': 'deepseek-ai/deepseek-llm-7b-chat',
                'architecture': 'deepseek',
                'specialties': ['chat', 'general', 'reasoning'],
                'download_size': '13.5GB',
                'context_length': 4096
            },
            # Qwen Models
            {
                'id': 'qwen2-7b',
                'name': 'Qwen2 7B',
                'size': '7B',
                'type': 'instruction',
                'description': 'Alibaba\'s Qwen2 base model with strong multilingual capabilities',
                'huggingface_id': 'Qwen/Qwen2-7B',
                'architecture': 'qwen',
                'specialties': ['multilingual', 'general', 'reasoning'],
                'download_size': '14GB',
                'context_length': 32768
            },
            {
                'id': 'qwen2-7b-instruct',
                'name': 'Qwen2 7B Instruct',
                'size': '7B',
                'type': 'chat',
                'description': 'Instruction-tuned Qwen2 for conversations and tasks',
                'huggingface_id': 'Qwen/Qwen2-7B-Instruct',
                'architecture': 'qwen',
                'specialties': ['chat', 'instruction-following', 'multilingual'],
                'download_size': '14GB',
                'context_length': 32768
            },
            {
                'id': 'qwen2.5-7b-instruct',
                'name': 'Qwen2.5 7B Instruct',
                'size': '7B',
                'type': 'chat',
                'description': 'Latest Qwen2.5 with improved performance and capabilities',
                'huggingface_id': 'Qwen/Qwen2.5-7B-Instruct',
                'architecture': 'qwen',
                'specialties': ['chat', 'reasoning', 'multilingual', 'latest'],
                'download_size': '14GB',
                'context_length': 32768
            },
            {
                'id': 'qwen2.5-coder-7b-instruct',
                'name': 'Qwen2.5 Coder 7B Instruct',
                'size': '7B',
                'type': 'code',
                'description': 'Specialized coding version of Qwen2.5 for programming tasks',
                'huggingface_id': 'Qwen/Qwen2.5-Coder-7B-Instruct',
                'architecture': 'qwen',
                'specialties': ['coding', 'programming', 'multilingual', 'latest'],
                'download_size': '14GB',
                'context_length': 32768
            }
        ]
        
        return jsonify({
            'success': True,
            'models': models,
            'total': len(models)
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-merger/experiments', methods=['GET'])
def get_experiments():
    """Get all experiments"""
    try:
        experiment_list = []
        for exp_id, exp_data in experiments.items():
            experiment_list.append({
                'id': exp_id,
                'name': exp_data['name'],
                'method': exp_data['method'],
                'models': exp_data['models'],
                'status': exp_data['status'],
                'created_at': exp_data['created_at'],
                'progress': exp_data.get('progress', 0),
                'error': exp_data.get('error'),
                'output_path': exp_data.get('output_path')
            })
        
        # Sort by creation time (newest first)
        experiment_list.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'experiments': experiment_list,
            'total': len(experiment_list)
        })
        
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_model_info(model_id):
    """Get model info by ID without Flask context"""
    # Direct model library lookup (no Flask context needed)
    models = [
        # Small test models for faster debugging
        {
            'id': 'gpt2-small',
            'name': 'GPT-2 Small (124M)',
            'size': '124M',
            'type': 'test',
            'description': 'Small test model for quick merging',
            'huggingface_id': 'gpt2',
            'architecture': 'gpt2',
            'specialties': ['testing', 'small'],
            'download_size': '548MB',
            'context_length': 1024
        },
        {
            'id': 'distilgpt2',
            'name': 'DistilGPT-2 (82M)',
            'size': '82M',
            'type': 'test',
            'description': 'Smaller test model for quick merging',
            'huggingface_id': 'distilgpt2',
            'architecture': 'gpt2',
            'specialties': ['testing', 'small'],
            'download_size': '353MB',
            'context_length': 1024
        },
        # Large production models
        {
            'id': 'mistral-7b-instruct',
            'name': 'Mistral 7B Instruct v0.1',
            'size': '7B',
            'type': 'instruction',
            'description': 'General purpose instruction-following model',
            'huggingface_id': 'mistralai/Mistral-7B-Instruct-v0.1',
            'architecture': 'mistral',
            'specialties': ['general', 'instruction-following'],
            'download_size': '13.5GB',
            'context_length': 8192
        },
        {
            'id': 'zephyr-7b-beta',
            'name': 'Zephyr 7B Beta',
            'size': '7B',
            'type': 'chat',
            'description': 'Helpful assistant model fine-tuned from Mistral',
            'huggingface_id': 'HuggingFaceH4/zephyr-7b-beta',
            'architecture': 'mistral',
            'specialties': ['chat', 'helpfulness'],
            'download_size': '13.5GB',
            'context_length': 8192
        },
        {
            'id': 'code-llama-7b',
            'name': 'Code Llama 7B',
            'size': '7B',
            'type': 'code',
            'description': 'Code generation and understanding',
            'huggingface_id': 'codellama/CodeLlama-7b-hf',
            'architecture': 'llama',
            'specialties': ['coding', 'programming'],
            'download_size': '13.5GB',
            'context_length': 16384
        },
        {
            'id': 'llama2-7b-chat',
            'name': 'Llama 2 7B Chat',
            'size': '7B',
            'type': 'chat',
            'description': 'Conversational AI model',
            'huggingface_id': 'meta-llama/Llama-2-7b-chat-hf',
            'architecture': 'llama',
            'specialties': ['conversation', 'general'],
            'download_size': '13.5GB',
            'context_length': 4096
        },
        {
            'id': 'openchat-7b',
            'name': 'OpenChat 7B',
            'size': '7B',
            'type': 'chat',
            'description': 'High-performance chat model',
            'huggingface_id': 'openchat/openchat_3.5',
            'architecture': 'mistral',
            'specialties': ['chat', 'reasoning'],
            'download_size': '13.5GB',
            'context_length': 8192
        },
        # Gemma 3 Models (Latest Release!)
        {
            'id': 'gemma-3-1b-it',
            'name': 'Gemma 3 1B Instruct',
            'size': '1B',
            'type': 'chat',
            'description': 'Google\'s latest Gemma 3 ultra-efficient model with multimodal capabilities',
            'huggingface_id': 'google/gemma-3-1b-it',
            'architecture': 'gemma3',
            'specialties': ['chat', 'efficiency', 'multimodal', 'latest'],
            'download_size': '2.5GB',
            'context_length': 32768
        },
        {
            'id': 'gemma-3-4b-it',
            'name': 'Gemma 3 4B Instruct',
            'size': '4B',
            'type': 'chat',
            'description': 'Google\'s Gemma 3 balanced model with 128K context and multimodal support',
            'huggingface_id': 'google/gemma-3-4b-it',
            'architecture': 'gemma3',
            'specialties': ['chat', 'multimodal', 'long-context', 'latest'],
            'download_size': '8GB',
            'context_length': 131072
        },
        {
            'id': 'gemma-3-12b-it',
            'name': 'Gemma 3 12B Instruct',
            'size': '12B',
            'type': 'chat',
            'description': 'Google\'s powerful Gemma 3 model with 128K context and advanced reasoning',
            'huggingface_id': 'google/gemma-3-12b-it',
            'architecture': 'gemma3',
            'specialties': ['chat', 'reasoning', 'multimodal', 'long-context', 'latest'],
            'download_size': '24GB',
            'context_length': 131072
        },
        {
            'id': 'gemma-3-27b-it',
            'name': 'Gemma 3 27B Instruct',
            'size': '27B',
            'type': 'chat',
            'description': 'Google\'s flagship Gemma 3 model with 128K context and state-of-the-art performance',
            'huggingface_id': 'google/gemma-3-27b-it',
            'architecture': 'gemma3',
            'specialties': ['chat', 'reasoning', 'multimodal', 'long-context', 'flagship', 'latest'],
            'download_size': '54GB',
            'context_length': 131072
        },
        # Gemma 2 Models
        {
            'id': 'gemma-2-2b',
            'name': 'Gemma 2 2B',
            'size': '2B',
            'type': 'instruction',
            'description': 'Google\'s efficient Gemma 2 model for instruction following',
            'huggingface_id': 'google/gemma-2-2b',
            'architecture': 'gemma',
            'specialties': ['instruction-following', 'efficiency'],
            'download_size': '4.5GB',
            'context_length': 8192
        },
        {
            'id': 'gemma-2-9b',
            'name': 'Gemma 2 9B',
            'size': '9B',
            'type': 'instruction',
            'description': 'Google\'s powerful Gemma 2 model with enhanced capabilities',
            'huggingface_id': 'google/gemma-2-9b',
            'architecture': 'gemma',
            'specialties': ['instruction-following', 'reasoning'],
            'download_size': '18GB',
            'context_length': 8192
        },
        {
            'id': 'gemma-2-9b-it',
            'name': 'Gemma 2 9B Instruct',
            'size': '9B',
            'type': 'chat',
            'description': 'Instruction-tuned version of Gemma 2 9B for conversations',
            'huggingface_id': 'google/gemma-2-9b-it',
            'architecture': 'gemma',
            'specialties': ['chat', 'instruction-following'],
            'download_size': '18GB',
            'context_length': 8192
        },
        # DeepSeek Models
        {
            'id': 'deepseek-coder-6.7b',
            'name': 'DeepSeek Coder 6.7B',
            'size': '6.7B',
            'type': 'code',
            'description': 'Specialized coding model from DeepSeek',
            'huggingface_id': 'deepseek-ai/deepseek-coder-6.7b-base',
            'architecture': 'deepseek',
            'specialties': ['coding', 'programming', 'software-development'],
            'download_size': '13GB',
            'context_length': 16384
        },
        {
            'id': 'deepseek-coder-6.7b-instruct',
            'name': 'DeepSeek Coder 6.7B Instruct',
            'size': '6.7B',
            'type': 'code',
            'description': 'Instruction-tuned DeepSeek Coder for code generation',
            'huggingface_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
            'architecture': 'deepseek',
            'specialties': ['coding', 'instruction-following', 'code-generation'],
            'download_size': '13GB',
            'context_length': 16384
        },
        {
            'id': 'deepseek-llm-7b-chat',
            'name': 'DeepSeek LLM 7B Chat',
            'size': '7B',
            'type': 'chat',
            'description': 'General purpose chat model from DeepSeek',
            'huggingface_id': 'deepseek-ai/deepseek-llm-7b-chat',
            'architecture': 'deepseek',
            'specialties': ['chat', 'general', 'reasoning'],
            'download_size': '13.5GB',
            'context_length': 4096
        },
        # Qwen Models
        {
            'id': 'qwen2-7b',
            'name': 'Qwen2 7B',
            'size': '7B',
            'type': 'instruction',
            'description': 'Alibaba\'s Qwen2 base model with strong multilingual capabilities',
            'huggingface_id': 'Qwen/Qwen2-7B',
            'architecture': 'qwen',
            'specialties': ['multilingual', 'general', 'reasoning'],
            'download_size': '14GB',
            'context_length': 32768
        },
        {
            'id': 'qwen2-7b-instruct',
            'name': 'Qwen2 7B Instruct',
            'size': '7B',
            'type': 'chat',
            'description': 'Instruction-tuned Qwen2 for conversations and tasks',
            'huggingface_id': 'Qwen/Qwen2-7B-Instruct',
            'architecture': 'qwen',
            'specialties': ['chat', 'instruction-following', 'multilingual'],
            'download_size': '14GB',
            'context_length': 32768
        },
        {
            'id': 'qwen2.5-7b-instruct',
            'name': 'Qwen2.5 7B Instruct',
            'size': '7B',
            'type': 'chat',
            'description': 'Latest Qwen2.5 with improved performance and capabilities',
            'huggingface_id': 'Qwen/Qwen2.5-7B-Instruct',
            'architecture': 'qwen',
            'specialties': ['chat', 'reasoning', 'multilingual', 'latest'],
            'download_size': '14GB',
            'context_length': 32768
        },
        {
            'id': 'qwen2.5-coder-7b-instruct',
            'name': 'Qwen2.5 Coder 7B Instruct',
            'size': '7B',
            'type': 'code',
            'description': 'Specialized coding version of Qwen2.5 for programming tasks',
            'huggingface_id': 'Qwen/Qwen2.5-Coder-7B-Instruct',
            'architecture': 'qwen',
            'specialties': ['coding', 'programming', 'multilingual', 'latest'],
            'download_size': '14GB',
            'context_length': 32768
        }
    ]

    for model in models:
        if model['id'] == model_id:
            return model
    return None

def load_model(model_id):
    """Load a model from HuggingFace or local cache"""
    try:
        model_info = get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")

        hf_id = model_info['huggingface_id']
        logger.info(f"Loading model {hf_id}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise

def merge_models_async(experiment_id, config):
    """Asynchronously merge models"""
    try:
        experiments[experiment_id]['status'] = 'running'
        experiments[experiment_id]['progress'] = 5
        experiments[experiment_id]['current_step'] = 'Initializing merge process...'
        experiments[experiment_id]['estimated_time_remaining'] = '25-30 minutes'

        logger.info(f"🚀 Starting merge experiment {experiment_id}")
        logger.info(f"📋 Config: {config}")

        # Load models
        logger.info(f"📥 Loading {len(config['models'])} models for experiment {experiment_id}")
        experiments[experiment_id]['current_step'] = f"Downloading {len(config['models'])} models from HuggingFace..."
        experiments[experiment_id]['progress'] = 10

        loaded_models = []
        tokenizers = []

        for i, model_config in enumerate(config['models']):
            model_name = model_config.get('name', model_config['id'])
            progress = 10 + (i * 40 // len(config['models']))
            experiments[experiment_id]['progress'] = progress
            experiments[experiment_id]['current_step'] = f"Downloading {model_name} ({i+1}/{len(config['models'])})"
            experiments[experiment_id]['estimated_time_remaining'] = f"{max(20 - (progress//5), 5)}-{max(25 - (progress//4), 10)} minutes"

            logger.info(f"📥 Loading model {i+1}/{len(config['models'])}: {model_name}")
            model, tokenizer = load_model(model_config['id'])
            loaded_models.append(model)
            tokenizers.append(tokenizer)
            logger.info(f"✅ Model {model_name} loaded successfully")

        experiments[experiment_id]['progress'] = 50
        experiments[experiment_id]['current_step'] = 'All models downloaded. Starting merge process...'
        experiments[experiment_id]['estimated_time_remaining'] = '5-10 minutes'

        # Extract weights
        weights = [model_config.get('weight', 1.0) for model_config in config['models']]

        # Perform merge based on method
        method = config['method']
        logger.info(f"🧠 Merging models using {method} method with Sebastian Raschka's protocol")
        experiments[experiment_id]['current_step'] = f'Applying {method} merge algorithm...'
        experiments[experiment_id]['progress'] = 60

        if method == 'linear':
            # Sebastian's linear merge returns the complete merged model
            logger.info("🔬 Executing Sebastian's linear merge algorithm")
            experiments[experiment_id]['current_step'] = 'Running linear weight averaging (Sebastian Raschka protocol)...'
            merged_model = merger.linear_merge(loaded_models, weights)
        elif method == 'slerp':
            if len(loaded_models) != 2:
                raise ValueError("SLERP requires exactly 2 models")
            alpha = config.get('parameters', {}).get('alpha', 0.5)
            merged_state_dict = merger.slerp_merge(loaded_models[0], loaded_models[1], alpha)
            # Create merged model and load state dict
            merged_model = loaded_models[0]
            merged_model.load_state_dict(merged_state_dict)
        elif method == 'ties':
            density = config.get('parameters', {}).get('density', 0.5)
            merged_state_dict = merger.ties_merge(loaded_models, weights, density)
            # Create merged model and load state dict
            merged_model = loaded_models[0]
            merged_model.load_state_dict(merged_state_dict)
        elif method == 'dare':
            drop_rate = config.get('parameters', {}).get('drop_rate', 0.1)
            merged_state_dict = merger.dare_merge(loaded_models, weights, drop_rate)
            # Create merged model and load state dict
            merged_model = loaded_models[0]
            merged_model.load_state_dict(merged_state_dict)
        else:
            raise ValueError(f"Unknown merge method: {method}")

        experiments[experiment_id]['progress'] = 80
        experiments[experiment_id]['current_step'] = 'Merge complete! Saving merged model...'
        experiments[experiment_id]['estimated_time_remaining'] = '2-3 minutes'

        # Save merged model
        output_dir = MERGED_MODELS_DIR / experiment_id
        output_dir.mkdir(exist_ok=True)

        logger.info(f"💾 Saving merged model to {output_dir}")
        experiments[experiment_id]['current_step'] = 'Saving model weights and configuration...'
        merged_model.save_pretrained(output_dir)

        experiments[experiment_id]['progress'] = 90
        experiments[experiment_id]['current_step'] = 'Saving tokenizer...'
        tokenizers[0].save_pretrained(output_dir)

        experiments[experiment_id]['progress'] = 95
        experiments[experiment_id]['current_step'] = 'Creating experiment metadata...'
        # Save experiment metadata
        metadata = {
            'experiment_id': experiment_id,
            'name': config['name'],
            'method': method,
            'models': config['models'],
            'weights': weights,
            'parameters': config.get('parameters', {}),
            'created_at': experiments[experiment_id]['created_at'],
            'completed_at': datetime.now().isoformat()
        }

        with open(output_dir / 'experiment.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        experiments[experiment_id]['status'] = 'completed'
        experiments[experiment_id]['progress'] = 100
        experiments[experiment_id]['current_step'] = 'Merge completed successfully!'
        experiments[experiment_id]['estimated_time_remaining'] = 'Complete'
        experiments[experiment_id]['output_path'] = str(output_dir)
        save_experiments()

        logger.info(f"🎉 Experiment {experiment_id} completed successfully!")
        logger.info(f"📁 Merged model saved to: {output_dir}")
        logger.info(f"📊 Model size: {sum(p.numel() for p in merged_model.parameters())} parameters")

    except Exception as e:
        logger.error(f"Error in experiment {experiment_id}: {e}")
        experiments[experiment_id]['status'] = 'failed'
        experiments[experiment_id]['error'] = str(e)
        save_experiments()

@app.route('/api/model-merger/experiments', methods=['POST'])
def create_experiment():
    """Create a new model merging experiment"""
    try:
        config = request.get_json()

        # Validate config
        required_fields = ['name', 'method', 'models']
        for field in required_fields:
            if field not in config:
                return jsonify({
                    'success': False,
                    'error': f"Missing required field: {field}"
                }), 400

        if len(config['models']) < 2:
            return jsonify({
                'success': False,
                'error': "At least 2 models required for merging"
            }), 400

        # Validate method-specific requirements
        if config['method'] == 'slerp' and len(config['models']) != 2:
            return jsonify({
                'success': False,
                'error': "SLERP method requires exactly 2 models"
            }), 400

        # Create experiment
        experiment_id = str(uuid.uuid4())
        experiments[experiment_id] = {
            'id': experiment_id,
            'name': config['name'],
            'method': config['method'],
            'models': config['models'],
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat()
        }
        save_experiments()

        # Start merging in background thread
        logger.info(f"🚀 Starting background thread for experiment {experiment_id}")
        try:
            thread = threading.Thread(
                target=merge_models_async,
                args=(experiment_id, config)
            )
            thread.daemon = True
            thread.start()
            logger.info(f"✅ Background thread started successfully for {experiment_id}")
        except Exception as e:
            logger.error(f"❌ Failed to start background thread: {e}")
            experiments[experiment_id]['status'] = 'failed'
            experiments[experiment_id]['error'] = f"Failed to start merge: {e}"
            save_experiments()

        return jsonify({
            'success': True,
            'experiment_id': experiment_id,
            'message': 'Experiment started successfully'
        })

    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-merger/experiments/<experiment_id>/download', methods=['GET'])
def download_experiment(experiment_id):
    """Download merged model"""
    try:
        if experiment_id not in experiments:
            return jsonify({
                'success': False,
                'error': 'Experiment not found'
            }), 404

        experiment = experiments[experiment_id]
        if experiment['status'] != 'completed':
            return jsonify({
                'success': False,
                'error': 'Experiment not completed'
            }), 400

        output_path = experiment.get('output_path')
        if not output_path or not os.path.exists(output_path):
            return jsonify({
                'success': False,
                'error': 'Model files not found'
            }), 404

        # Create zip file of the model
        import zipfile
        zip_path = f"merged_models/{experiment_id}.zip"

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_path)
                    zipf.write(file_path, arcname)

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{experiment['name']}.zip",
            mimetype='application/zip'
        )

    except Exception as e:
        logger.error(f"Error downloading experiment {experiment_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load existing experiments
    experiments.update(load_experiments())
    logger.info(f"Loaded {len(experiments)} existing experiments")

    logger.info("Starting Production Model Merger API on port 5007")
    app.run(host='0.0.0.0', port=5007, debug=False)
