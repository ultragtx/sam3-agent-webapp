"""
Flask backend application for SAM3 Agent
"""
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from models.sam3_loader import SAM3ModelManager
from agent.core import AgentCore
from agent.client_llm import MLLMClient
from agent.client_sam3 import SAM3Client

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE', 16 * 1024 * 1024))
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['OUTPUT_FOLDER'] = os.getenv('OUTPUT_FOLDER', './outputs')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

# Global model manager - keeps SAM3 in memory
sam3_manager = None
agent_core = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'sam3_loaded': sam3_manager is not None and sam3_manager.is_loaded(),
        'backend': 'running'
    })


@app.route('/api/models/load', methods=['POST'])
def load_models():
    """Load SAM3 model into memory"""
    global sam3_manager
    
    if sam3_manager is None:
        bpe_path = os.getenv('SAM3_BPE_PATH', 'essential_assets/bpe_simple_vocab_16e6.txt.gz')
        device = os.getenv('SAM3_DEVICE', 'cuda')
        sam3_manager = SAM3ModelManager(bpe_path=bpe_path, device=device)
    
    sam3_manager.load_model()
    
    return jsonify({
        'status': 'success',
        'message': 'SAM3 model loaded successfully',
        'device': sam3_manager.device
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload image file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/api/sam3/segment', methods=['POST'])
def sam3_segment():
    """Direct SAM3 segmentation endpoint"""
    global sam3_manager
    
    data = request.json
    image_path = data.get('image_path')
    text_prompt = data.get('text_prompt')
    
    if not image_path or not text_prompt:
        return jsonify({'error': 'Missing image_path or text_prompt'}), 400
    
    # Ensure model is loaded
    if sam3_manager is None or not sam3_manager.is_loaded():
        return jsonify({'error': 'SAM3 model not loaded'}), 503
    
    # Create SAM3 client
    sam3_client = SAM3Client(sam3_manager.processor)
    
    # Run segmentation
    output_dir = app.config['OUTPUT_FOLDER']
    result = sam3_client.segment(
        image_path=image_path,
        text_prompt=text_prompt,
        output_folder=output_dir
    )
    
    return jsonify({
        'status': 'success',
        'result': result
    })


@app.route('/api/agent/run', methods=['POST'])
def run_agent():
    """Run SAM3 Agent with iterative reasoning"""
    global sam3_manager, agent_core
    
    data = request.json
    image_path = data.get('image_path')
    text_prompt = data.get('text_prompt')
    mllm_config = data.get('mllm_config', {})
    debug = data.get('debug', True)
    
    if not image_path or not text_prompt:
        return jsonify({'error': 'Missing image_path or text_prompt'}), 400
    
    # Ensure SAM3 model is loaded
    if sam3_manager is None or not sam3_manager.is_loaded():
        return jsonify({'error': 'SAM3 model not loaded'}), 503
    
    # Create MLLM client with config (defaults from env, override from request)
    mllm_client = MLLMClient(
        api_base=mllm_config.get('api_base', os.getenv('MLLM_API_BASE')),
        api_key=mllm_config.get('api_key', os.getenv('MLLM_API_KEY')),
        model=mllm_config.get('model', os.getenv('MLLM_MODEL_NAME')),
        max_tokens=mllm_config.get('max_tokens', int(os.getenv('MLLM_MAX_TOKENS', 4096)))
    )
    
    # Create SAM3 client
    sam3_client = SAM3Client(sam3_manager.processor)
    
    # Initialize agent if not exists
    if agent_core is None:
        agent_core = AgentCore(
            mllm_client=mllm_client,
            sam3_client=sam3_client,
            output_dir=app.config['OUTPUT_FOLDER']
        )
    else:
        # Update clients
        agent_core.mllm_client = mllm_client
        agent_core.sam3_client = sam3_client
    
    # Run agent
    result = agent_core.run(
        image_path=image_path,
        text_prompt=text_prompt,
        debug=debug
    )
    
    return jsonify({
        'status': 'success',
        'result': result
    })


@app.route('/api/outputs/<path:filename>', methods=['GET'])
def serve_output(filename):
    """Serve output files (images, JSON)"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'mllm': {
            'api_base': os.getenv('MLLM_API_BASE'),
            'model': os.getenv('MLLM_MODEL_NAME'),
            'max_tokens': int(os.getenv('MLLM_MAX_TOKENS', 4096))
        },
        'sam3': {
            'device': os.getenv('SAM3_DEVICE', 'cuda'),
            'model_size': os.getenv('SAM3_MODEL_SIZE', 'large')
        }
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration (runtime only, not persisted)"""
    # This would update runtime config, but not .env file
    # Implementation depends on requirements
    return jsonify({'status': 'success', 'message': 'Config update not yet implemented'})


if __name__ == '__main__':
    # Auto-load SAM3 on startup
    print("🚀 Starting SAM3 Agent Backend...")
    print("📦 Loading SAM3 model into memory...")
    
    bpe_path = os.getenv('SAM3_BPE_PATH', 'essential_assets/bpe_simple_vocab_16e6.txt.gz')
    device = os.getenv('SAM3_DEVICE', 'cuda')
    
    if bpe_path:
        sam3_manager = SAM3ModelManager(bpe_path=bpe_path, device=device)
        sam3_manager.load_model()
        print(f"✅ SAM3 model loaded successfully on {device}")
    else:
        print("⚠️  SAM3_BPE_PATH not set, model will need to be loaded via API")
    
    # Start Flask server
    port = int(os.getenv('BACKEND_PORT', 5000))
    host = os.getenv('BACKEND_HOST', '0.0.0.0')
    
    print(f"🌐 Server starting on {host}:{port}")
    app.run(host=host, port=port, debug=True)
