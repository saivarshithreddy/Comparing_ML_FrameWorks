#!/usr/bin/env python3
"""
ML Frameworks Comparison Web Application
A beautiful, modern web interface for comparing TensorFlow Serving vs TorchServe performance
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import os
import subprocess
import threading
import time
from datetime import datetime
import psutil

app = Flask(__name__)

# Global variables for tracking experiments
current_experiment = None
experiment_results = []
experiment_thread = None

@app.route('/')
def index():
    """Main dashboard page with classic professional UI"""
    return render_template('index_classic.html')

@app.route('/api/system-info')
def system_info():
    """Get system resource information"""
    return jsonify({
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start-comparison', methods=['POST'])
def start_comparison():
    """Start a new comparison experiment"""
    global current_experiment, experiment_thread, experiment_results
    
    if current_experiment and current_experiment['status'] == 'running':
        return jsonify({'error': 'Experiment already running'}), 400
    
    data = request.get_json()
    rates = data.get('rates', [1, 5, 10])
    buffer_sizes = data.get('buffer_sizes', [10, 50])
    duration = data.get('duration', 15)
    worker_threads = data.get('worker_threads', 10)
    
    current_experiment = {
        'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'status': 'starting',
        'parameters': {
            'rates': rates,
            'buffer_sizes': buffer_sizes,
            'duration': duration,
            'worker_threads': worker_threads
        },
        'start_time': datetime.now().isoformat(),
        'progress': 0
    }
    
    # Start experiment in background thread
    experiment_thread = threading.Thread(target=run_comparison_experiment, args=(rates, buffer_sizes, duration, worker_threads))
    experiment_thread.daemon = True
    experiment_thread.start()
    
    return jsonify({'message': 'Experiment started', 'experiment_id': current_experiment['id']})

@app.route('/api/experiment-status')
def experiment_status():
    """Get current experiment status"""
    global current_experiment
    
    if not current_experiment:
        return jsonify({'status': 'idle'})
    
    return jsonify(current_experiment)

@app.route('/api/results')
def get_results():
    """Get all experiment results"""
    # Load existing results if available
    results_dir = "serving_comparison_results"
    results = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(results_dir, file), 'r') as f:
                        result = json.load(f)
                        results.append(result)
                except:
                    continue
    
    return jsonify(results)

@app.route('/api/results/<filename>')
def download_result(filename):
    """Download result file"""
    return send_from_directory('serving_comparison_results', filename)

def run_comparison_experiment(rates, buffer_sizes, duration, worker_threads):
    """Run the comparison experiment in background"""
    global current_experiment, experiment_results
    
    try:
        current_experiment['status'] = 'running'
        
        # Build command
        cmd = [
            'python3', 'comparison_serving.py',
            '--rates'] + [str(r) for r in rates] + [
            '--buffer-sizes'] + [str(b) for b in buffer_sizes] + [
            '--duration', str(duration),
            '--worker-threads', str(worker_threads)
        ]
        
        # Run the experiment
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Monitor progress
        while process.poll() is None:
            # Update progress (simplified)
            elapsed = time.time() - datetime.fromisoformat(current_experiment['start_time']).timestamp()
            total_time = len(rates) * len(buffer_sizes) * duration
            progress = min(100, (elapsed / total_time) * 100)
            current_experiment['progress'] = progress
            
            time.sleep(2)
        
        # Get results
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            current_experiment['status'] = 'completed'
            current_experiment['progress'] = 100
            
            # Load the latest results
            results_dir = "serving_comparison_results"
            if os.path.exists(results_dir):
                latest_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')], 
                                    key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), 
                                    reverse=True)
                if latest_files:
                    with open(os.path.join(results_dir, latest_files[0]), 'r') as f:
                        result = json.load(f)
                        current_experiment['results'] = result
        else:
            current_experiment['status'] = 'failed'
            current_experiment['error'] = stderr
            
    except Exception as e:
        current_experiment['status'] = 'failed'
        current_experiment['error'] = str(e)

if __name__ == '__main__':
    # Check if running in production (Render/Docker)
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    else:
        app.run(debug=True, host='0.0.0.0', port=8080)
