#!/usr/bin/env python3
"""
No-Code Model Merger API for Metatron Platform
Provides a user-friendly interface for merging LLMs without coding
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@dataclass
class MergeExperiment:
    """Represents a model merging experiment"""
    id: str
    name: str
    method: str  # slerp, ties, dare, linear, moe
    models: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    status: str  # pending, running, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class ModelMergerEngine:
    """Core engine for no-code model merging"""
    
    def __init__(self):
        self.experiments: Dict[str, MergeExperiment] = {}
        self.supported_methods = {
            'linear': 'Model Soup - Simple averaging of model weights',
            'slerp': 'Spherical Linear Interpolation - Smooth interpolation between two models',
            'ties': 'TIES - Trim, Elect Sign, and Merge for multiple models',
            'dare': 'DARE - Drop and Rescale for efficient merging',
            'moe': 'Mixture of Experts - Create expert routing system'
        }
        
        # Initialize model library
        self.model_library = self._load_model_library()
    
    def _load_model_library(self) -> List[Dict[str, Any]]:
        """Load available models for merging"""
        return [
            {
                'id': 'mistral-7b-instruct',
                'name': 'Mistral 7B Instruct',
                'size': '7B',
                'type': 'instruction',
                'description': 'General purpose instruction-following model',
                'huggingface_id': 'mistralai/Mistral-7B-Instruct-v0.1',
                'architecture': 'mistral',
                'specialties': ['general', 'instruction-following']
            },
            {
                'id': 'zephyr-7b-beta',
                'name': 'Zephyr 7B Beta',
                'size': '7B', 
                'type': 'chat',
                'description': 'Helpful assistant model',
                'huggingface_id': 'HuggingFaceH4/zephyr-7b-beta',
                'architecture': 'mistral',
                'specialties': ['chat', 'helpfulness']
            },
            {
                'id': 'code-llama-7b',
                'name': 'Code Llama 7B',
                'size': '7B',
                'type': 'code',
                'description': 'Code generation and understanding',
                'huggingface_id': 'codellama/CodeLlama-7b-hf',
                'architecture': 'llama',
                'specialties': ['coding', 'programming']
            },
            {
                'id': 'llama2-7b-chat',
                'name': 'Llama 2 7B Chat',
                'size': '7B',
                'type': 'chat',
                'description': 'Conversational AI model',
                'huggingface_id': 'meta-llama/Llama-2-7b-chat-hf',
                'architecture': 'llama',
                'specialties': ['conversation', 'general']
            }
        ]
    
    def create_experiment(self, config: Dict[str, Any]) -> str:
        """Create a new merging experiment"""
        experiment_id = str(uuid.uuid4())
        
        experiment = MergeExperiment(
            id=experiment_id,
            name=config.get('name', f'Merge Experiment {experiment_id[:8]}'),
            method=config['method'],
            models=config['models'],
            parameters=config.get('parameters', {}),
            status='pending',
            created_at=datetime.now().isoformat()
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {experiment_id}")
        
        return experiment_id
    
    def generate_merge_config(self, experiment: MergeExperiment) -> Dict[str, Any]:
        """Generate MergeKit configuration from experiment"""
        method = experiment.method
        models = experiment.models
        params = experiment.parameters
        
        if method == 'linear':
            return self._generate_linear_config(models, params)
        elif method == 'slerp':
            return self._generate_slerp_config(models, params)
        elif method == 'ties':
            return self._generate_ties_config(models, params)
        elif method == 'dare':
            return self._generate_dare_config(models, params)
        elif method == 'moe':
            return self._generate_moe_config(models, params)
        else:
            raise ValueError(f"Unsupported merge method: {method}")
    
    def _generate_linear_config(self, models: List[Dict], params: Dict) -> Dict:
        """Generate linear (model soup) configuration"""
        return {
            'merge_method': 'linear',
            'slices': [
                {
                    'sources': [
                        {
                            'model': model['huggingface_id'],
                            'layer_range': model.get('layer_range', [0, 32])
                        }
                        for model in models
                    ]
                }
            ],
            'base_model': models[0]['huggingface_id'],
            'parameters': {
                'weight': params.get('weights', [1.0] * len(models)),
                'normalize': params.get('normalize', True)
            },
            'dtype': params.get('dtype', 'float16')
        }
    
    def _generate_slerp_config(self, models: List[Dict], params: Dict) -> Dict:
        """Generate SLERP configuration"""
        if len(models) != 2:
            raise ValueError("SLERP requires exactly 2 models")
        
        return {
            'merge_method': 'slerp',
            'slices': [
                {
                    'sources': [
                        {'model': models[0]['huggingface_id']},
                        {'model': models[1]['huggingface_id']}
                    ]
                }
            ],
            'base_model': models[0]['huggingface_id'],
            'parameters': {
                't': params.get('interpolation_factor', 0.5)
            },
            'dtype': params.get('dtype', 'float16')
        }
    
    def _generate_ties_config(self, models: List[Dict], params: Dict) -> Dict:
        """Generate TIES configuration"""
        return {
            'merge_method': 'ties',
            'slices': [
                {
                    'sources': [
                        {'model': model['huggingface_id']}
                        for model in models
                    ]
                }
            ],
            'base_model': models[0]['huggingface_id'],
            'parameters': {
                'density': params.get('density', 0.5),
                'weight': params.get('weight', 1.0)
            },
            'dtype': params.get('dtype', 'float16')
        }
    
    def _generate_dare_config(self, models: List[Dict], params: Dict) -> Dict:
        """Generate DARE configuration"""
        return {
            'merge_method': 'dare_ties',
            'slices': [
                {
                    'sources': [
                        {'model': model['huggingface_id']}
                        for model in models
                    ]
                }
            ],
            'base_model': models[0]['huggingface_id'],
            'parameters': {
                'density': params.get('density', 0.5),
                'weight': params.get('weight', 1.0),
                'drop_rate': params.get('drop_rate', 0.1)
            },
            'dtype': params.get('dtype', 'float16')
        }
    
    def _generate_moe_config(self, models: List[Dict], params: Dict) -> Dict:
        """Generate MoE configuration"""
        experts = []
        for model in models[1:]:  # Skip base model
            experts.append({
                'source_model': model['huggingface_id'],
                'positive_prompts': model.get('specialties', ['general tasks'])
            })
        
        return {
            'base_model': models[0]['huggingface_id'],
            'gate_mode': params.get('gate_mode', 'hidden'),
            'dtype': params.get('dtype', 'float16'),
            'experts_per_token': params.get('experts_per_token', 2),
            'experts': experts
        }

# Initialize the merger engine
merger_engine = ModelMergerEngine()

@app.route('/api/model-merger/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'model-merger',
        'timestamp': datetime.now().isoformat(),
        'supported_methods': list(merger_engine.supported_methods.keys())
    })

@app.route('/api/model-merger/models', methods=['GET'])
def get_model_library():
    """Get available models for merging"""
    return jsonify({
        'success': True,
        'models': merger_engine.model_library,
        'count': len(merger_engine.model_library)
    })

@app.route('/api/model-merger/methods', methods=['GET'])
def get_merge_methods():
    """Get supported merge methods"""
    return jsonify({
        'success': True,
        'methods': merger_engine.supported_methods
    })

@app.route('/api/model-merger/experiments', methods=['POST'])
def create_experiment():
    """Create a new merging experiment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['method', 'models']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate method
        if data['method'] not in merger_engine.supported_methods:
            return jsonify({
                'success': False,
                'error': f'Unsupported method: {data["method"]}'
            }), 400
        
        # Create experiment
        experiment_id = merger_engine.create_experiment(data)
        
        return jsonify({
            'success': True,
            'experiment_id': experiment_id,
            'message': 'Experiment created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-merger/experiments/<experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    """Get experiment details"""
    if experiment_id not in merger_engine.experiments:
        return jsonify({
            'success': False,
            'error': 'Experiment not found'
        }), 404
    
    experiment = merger_engine.experiments[experiment_id]
    return jsonify({
        'success': True,
        'experiment': asdict(experiment)
    })

@app.route('/api/model-merger/experiments', methods=['GET'])
def list_experiments():
    """List all experiments"""
    experiments = [asdict(exp) for exp in merger_engine.experiments.values()]
    return jsonify({
        'success': True,
        'experiments': experiments,
        'count': len(experiments)
    })

if __name__ == '__main__':
    print("🧠 Starting Model Merger API...")
    print("🔗 Available at: http://localhost:5007")
    print("📋 Endpoints:")
    print("   GET  /api/model-merger/health")
    print("   GET  /api/model-merger/models")
    print("   GET  /api/model-merger/methods") 
    print("   POST /api/model-merger/experiments")
    print("   GET  /api/model-merger/experiments")
    
    app.run(host='0.0.0.0', port=5007, debug=True)
