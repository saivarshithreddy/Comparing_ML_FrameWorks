#!/usr/bin/env python3
"""
TensorFlow Serving vs TorchServe Performance Comparison
------------------------------------------------------
This script implements a stochastic arrival process to compare
TensorFlow Serving and TorchServe performance.

Usage:
  python3 compare_serving.py --rates 1 5 10 --buffer-sizes 10 50 --duration 15
"""

import numpy as np
import time
import requests
import threading
import json
import argparse
import matplotlib.pyplot as plt
import os
import psutil
import logging
import io
import sys
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("serving_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TorchServe API keys
# TorchServe API keys - Updated with the latest keys
AUTH_KEYS = {
    "management": "-HFCGUzb",
    "inference": "TWd_A4fB",
    "api": "isQwa4n0"
}

class ServingComparison:
    """Test harness for comparing model serving platforms performance"""
    
    def __init__(self, 
                 tf_model_name="fashion",
                 ts_model_name="fashion_mnist",
                 tf_host="localhost",
                 ts_host="localhost"):
        """
        Initialize the performance testing framework.
        """
        # TensorFlow Serving endpoints
        self.tf_model_name = tf_model_name
        self.tf_host = tf_host
        self.tf_predict_url = f"http://{tf_host}:8501/v1/models/{tf_model_name}:predict"
        self.tf_status_url = f"http://{tf_host}:8501/v1/models/{tf_model_name}"
        
        # TorchServe endpoints
        self.ts_model_name = ts_model_name
        self.ts_host = ts_host
        self.ts_predict_url = f"http://{ts_host}:8080/predictions/{ts_model_name}"
        self.ts_management_url = f"http://{ts_host}:8081/models/{ts_model_name}"
        
        # Auth keys
        self.auth_keys = AUTH_KEYS
        
        # Results directory
        self.results_dir = "serving_comparison_results"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "temp"), exist_ok=True)
        
        # Load test images
        logger.info("Loading test images from Fashion-MNIST...")
        self.test_data = self._load_test_images()
        logger.info(f"Loaded {len(self.test_data)} test images")
        
        logger.info("ServingComparison initialized")
        logger.info(f"TensorFlow Serving predict URL: {self.tf_predict_url}")
        logger.info(f"TorchServe predict URL: {self.ts_predict_url}")
    
    def _load_test_images(self, count=20):
        """Load test images from Fashion-MNIST dataset"""
        try:
            transform = transforms.Compose([transforms.ToTensor()])
            test_dataset = datasets.FashionMNIST(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
            
            # Select a subset of images
            images = []
            indices = list(range(0, min(1000, len(test_dataset)), 50))  # Take every 50th image
            
            for i in indices:
                img, label = test_dataset[i]
                # Convert tensor to PIL Image and save to disk for TorchServe
                img_pil = transforms.ToPILImage()(img)
                img_path = os.path.join(self.results_dir, "temp", f"test_image_{i}.png")
                img_pil.save(img_path)
                images.append((img_pil, img_path, label))
            
            return images
            
        except Exception as e:
            logger.error(f"Error loading Fashion-MNIST data: {str(e)}")
            # Create fallback images
            images = []
            for i in range(count):
                img = Image.new('L', (28, 28), color=0)  # Black background
                # Add a white square
                for x in range(10, 18):
                    for y in range(10, 18):
                        img.putpixel((x, y), 255)
                
                img_path = os.path.join(self.results_dir, "temp", f"synthetic_image_{i}.png")
                img.save(img_path)
                images.append((img, img_path, i % 10))
                
            return images
    
    def preprocess_for_tensorflow(self, image):
        """Prepare an image for TensorFlow Serving"""
        # Resize to 128x128 for the model
        img = image.resize((128, 128))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape to match model input shape
        img_array = img_array.reshape(1, 128, 128, 1).astype(np.float32)
        
        # Create the request format
        tf_data = {"instances": img_array.tolist()}
        return tf_data
    
    def test_connections(self):
        """Test connection to both services"""
        logger.info("Testing connections to both services...")
        
        tf_success = self._test_tensorflow_serving()
        ts_success = self._test_torchserve()
        
        if tf_success and ts_success:
            logger.info("Both services are available")
        elif tf_success:
            logger.info("Only TensorFlow Serving is available")
        elif ts_success:
            logger.info("Only TorchServe is available")
        else:
            logger.error("Neither service is available")
        
        return tf_success, ts_success
    
    def _test_tensorflow_serving(self):
        """Test connection to TensorFlow Serving"""
        try:
            # Check if model is available
            response = requests.get(self.tf_status_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"TensorFlow Serving model status: OK")
                
                # Send a test prediction
                test_image, _, _ = self.test_data[0]
                tf_data = self.preprocess_for_tensorflow(test_image)
                
                predict_response = requests.post(
                    self.tf_predict_url,
                    json=tf_data,
                    headers={"content-type": "application/json"},
                    timeout=10
                )
                
                if predict_response.status_code == 200:
                    logger.info(f"TensorFlow Serving test prediction: SUCCESS")
                    return True
                else:
                    logger.error(f"TensorFlow Serving prediction failed with status {predict_response.status_code}")
                    logger.error(f"Response: {predict_response.text}")
                    return False
            else:
                logger.error(f"TensorFlow Serving model unavailable: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"TensorFlow Serving connection test failed: {str(e)}")
            return False
    
    def _test_torchserve(self):
        """Test connection to TorchServe"""
        try:
            # Use authentication for TorchServe
            headers = {"Authorization": f"Bearer {self.auth_keys['management']}"}
            
            response = requests.get(self.ts_management_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"TorchServe model status: OK")
                
                # Send a test prediction
                _, img_path, _ = self.test_data[0]
                
                with open(img_path, 'rb') as f:
                    files = {'data': ('test_image.png', f, 'image/png')}
                    
                    predict_headers = {"Authorization": f"Bearer {self.auth_keys['inference']}"}
                    
                    predict_response = requests.post(
                        self.ts_predict_url,
                        files=files,
                        headers=predict_headers,
                        timeout=10
                    )
                
                if predict_response.status_code == 200:
                    logger.info(f"TorchServe test prediction: SUCCESS")
                    return True
                else:
                    logger.error(f"TorchServe prediction failed with status {predict_response.status_code}")
                    logger.error(f"Response: {predict_response.text}")
                    return False
            else:
                logger.error(f"TorchServe model unavailable: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"TorchServe connection test failed: {str(e)}")
            return False
    
    def poisson_process(self, rate, duration, seed=None):
        """Generate timestamps for a Poisson process"""
        if seed is not None:
            np.random.seed(seed)
        
        # Expected time between arrivals
        expected_interval = 1.0 / rate
        
        timestamps = []
        current_time = 0
        
        while current_time < duration:
            # Generate next interval using exponential distribution
            interval = np.random.exponential(expected_interval)
            current_time += interval
            
            if current_time < duration:
                timestamps.append(current_time)
        
        logger.info(f"Generated {len(timestamps)} timestamps for rate={rate}, duration={duration}s")
        return timestamps
    
    def send_tf_request(self, request_id=None):
        """Send a request to TensorFlow Serving"""
        # Choose a test image based on the request ID
        if request_id is not None:
            idx = request_id % len(self.test_data)
        else:
            idx = np.random.randint(0, len(self.test_data))
            
        test_image, _, label = self.test_data[idx]
        
        # Prepare metrics
        metrics = {
            "start_time": time.time(),
            "request_id": request_id,
            "image_class": label
        }
        
        success = False
        response = None
        
        try:
            # Prepare data
            tf_data = self.preprocess_for_tensorflow(test_image)
            metrics["preprocessing_time"] = time.time() - metrics["start_time"]
            
            # Send request
            send_start = time.time()
            response = requests.post(
                self.tf_predict_url,
                json=tf_data,
                headers={"content-type": "application/json"},
                timeout=10
            )
            metrics["network_time"] = time.time() - send_start
            
            metrics["end_time"] = time.time()
            metrics["total_latency"] = metrics["end_time"] - metrics["start_time"]
            metrics["status_code"] = response.status_code
            
            if response.status_code == 200:
                success = True
                
                # Parse prediction
                try:
                    pred_data = response.json().get("predictions", [])
                    if pred_data:
                        pred_idx = np.argmax(pred_data[0])
                        metrics["predicted_class"] = int(pred_idx)
                        metrics["correct_prediction"] = (pred_idx == label)
                except Exception as e:
                    logger.debug(f"Error parsing prediction: {str(e)}")
                
            else:
                logger.warning(f"Request {request_id}: Error {response.status_code}, Latency: {metrics['total_latency']*1000:.2f}ms")
            
            return response, metrics["total_latency"], success, metrics
            
        except Exception as e:
            metrics["end_time"] = time.time()
            metrics["total_latency"] = metrics["end_time"] - metrics["start_time"]
            metrics["error"] = str(e)
            
            logger.error(f"Request {request_id}: Failed with exception: {str(e)}")
            return None, metrics["total_latency"], False, metrics
    
    def send_ts_request(self, request_id=None):
        """Send a request to TorchServe"""
        # Choose a test image based on the request ID
        if request_id is not None:
            idx = request_id % len(self.test_data)
        else:
            idx = np.random.randint(0, len(self.test_data))
            
        _, img_path, label = self.test_data[idx]
        
        # Prepare metrics
        metrics = {
            "start_time": time.time(),
            "request_id": request_id,
            "image_class": label
        }
        
        success = False
        response = None
        
        try:
            # Record preprocessing time (minimal since file is already prepared)
            metrics["preprocessing_time"] = time.time() - metrics["start_time"]
            
            # Send request
            send_start = time.time()
            
            with open(img_path, 'rb') as f:
                files = {'data': (f'image_{request_id}.png', f, 'image/png')}
                
                headers = {"Authorization": f"Bearer {self.auth_keys['inference']}"}
                
                response = requests.post(
                    self.ts_predict_url,
                    files=files,
                    headers=headers,
                    timeout=10
                )
            
            metrics["network_time"] = time.time() - send_start
            
            metrics["end_time"] = time.time()
            metrics["total_latency"] = metrics["end_time"] - metrics["start_time"]
            metrics["status_code"] = response.status_code
            
            if response.status_code == 200:
                success = True
                
                # Parse prediction
                try:
                    pred_data = response.json()
                    if isinstance(pred_data, dict):
                        # Get highest probability class
                        pred_class = max(pred_data.items(), key=lambda x: x[1])[0]
                        # In Fashion-MNIST, classes are named like "T-shirt/top", "Trouser", etc.
                        # Would need mapping from class names to indices
                        metrics["predicted_class_name"] = pred_class
                except Exception as e:
                    logger.debug(f"Error parsing prediction: {str(e)}")
                
            else:
                logger.warning(f"Request {request_id}: Error {response.status_code}, Latency: {metrics['total_latency']*1000:.2f}ms")
            
            return response, metrics["total_latency"], success, metrics
            
        except Exception as e:
            metrics["end_time"] = time.time()
            metrics["total_latency"] = metrics["end_time"] - metrics["start_time"]
            metrics["error"] = str(e)
            
            logger.error(f"Request {request_id}: Failed with exception: {str(e)}")
            return None, metrics["total_latency"], False, metrics
    
    def run_experiment(self, framework, rate, duration, buffer_size=None, num_workers=5):
        """Run a performance experiment for the specified framework"""
        logger.info(f"Starting experiment: framework={framework}, rate={rate}, duration={duration}, buffer_size={buffer_size}")
        
        # Generate request timestamps
        timestamps = self.poisson_process(rate, duration)
        total_requests = len(timestamps)
        
        # Metrics collection
        latencies = []
        preprocessing_times = []
        network_times = []
        successes = 0
        failures = 0
        accuracy = 0
        total_accuracy_samples = 0
        detailed_metrics = []
        
        # Request buffer and synchronization
        request_buffer = []
        buffer_full_count = 0
        max_buffer_size = buffer_size if buffer_size is not None else 1000
        buffer_lock = threading.Lock()
        metrics_lock = threading.Lock()
        is_experiment_complete = threading.Event()
        
        # Resource tracking
        cpu_usage = []
        memory_usage = []
        
        # Request rate tracking
        actual_request_times = []
        
        # Choose the appropriate send function
        send_func = self.send_tf_request if framework == "tensorflow" else self.send_ts_request
        
        # Modified buffer processing to handle multiple worker threads
        def worker_thread():
            nonlocal successes, failures, accuracy, total_accuracy_samples
            
            while not is_experiment_complete.is_set() or request_buffer:
                # Get the next request ID to process
                req_id = None
                with buffer_lock:
                    if request_buffer:
                        req_id = request_buffer.pop(0)
                    
                if req_id is None:
                    # No more requests to process right now
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging
                    continue
                
                try:
                    response, latency, success, metrics = send_func(req_id)
                    
                    with metrics_lock:
                        if success:
                            successes += 1
                            if "correct_prediction" in metrics and metrics["correct_prediction"] is not None:
                                total_accuracy_samples += 1
                                if metrics["correct_prediction"]:
                                    accuracy += 1
                        else:
                            failures += 1
                        
                        latencies.append(latency)
                        if "preprocessing_time" in metrics:
                            preprocessing_times.append(metrics["preprocessing_time"])
                        if "network_time" in metrics:
                            network_times.append(metrics["network_time"])
                        
                        detailed_metrics.append(metrics)
                        
                        # Track resource usage
                        cpu_usage.append(psutil.cpu_percent(interval=0.1))
                        memory_usage.append(psutil.virtual_memory().percent)
                
                except Exception as e:
                    logger.error(f"Error processing request {req_id} in thread: {str(e)}")
                    with metrics_lock:
                        failures += 1
                        
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
        
        # Start worker threads
        worker_threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker_thread)
            t.daemon = True
            t.start()
            worker_threads.append(t)
        
        # Start experiment
        experiment_start_time = time.time()
        
        for i, timestamp in enumerate(timestamps):
            # Wait until scheduled time
            current_time = time.time() - experiment_start_time
            wait_time = timestamp - current_time
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Record actual request time
            actual_request_times.append(time.time() - experiment_start_time)
            
            # Add request to buffer if there's space
            with buffer_lock:
                if len(request_buffer) < max_buffer_size:
                    request_buffer.append(i+1)
                else:
                    buffer_full_count += 1
                    logger.warning(f"Buffer full for request {i+1}, dropping request")
        
        # Set the flag indicating all requests have been sent
        is_experiment_complete.set()
        
        # Wait for all worker threads to finish processing
        # Give them time to process remaining requests
        wait_time = 0
        max_wait_time = 30  # Maximum wait time in seconds
        
        while any(t.is_alive() for t in worker_threads) and wait_time < max_wait_time:
            # Check if there are any remaining requests in the buffer
            with buffer_lock:
                if not request_buffer:
                    # No more requests to process, we can exit
                    break
            time.sleep(1)
            wait_time += 1
            
        if wait_time >= max_wait_time:
            logger.warning(f"Reached maximum wait time for processing. Some requests may not be processed.")
        
        # Calculate metrics
        experiment_end_time = time.time()
        total_time = experiment_end_time - experiment_start_time
        
        # Calculate actual rate
        if len(actual_request_times) >= 2:
            actual_rate = (len(actual_request_times) - 1) / (actual_request_times[-1] - actual_request_times[0])
        else:
            actual_rate = 0
            
        # Calculate throughput
        throughput = len(latencies) / total_time if total_time > 0 else 0
        
        # Process latency metrics
        if latencies:
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            std_latency = np.std(latencies)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = min_latency = max_latency = std_latency = 0
        
        # Process preprocessing and network times
        if preprocessing_times:
            avg_preprocessing = np.mean(preprocessing_times)
            max_preprocessing = np.max(preprocessing_times)
        else:
            avg_preprocessing = max_preprocessing = 0
            
        if network_times:
            avg_network = np.mean(network_times)
            max_network = np.max(network_times)
        else:
            avg_network = max_network = 0
            
        # Calculate model accuracy
        model_accuracy = (accuracy / total_accuracy_samples * 100) if total_accuracy_samples > 0 else 0
        
        # Process resource metrics
        if cpu_usage:
            avg_cpu = np.mean(cpu_usage)
            max_cpu = np.max(cpu_usage)
        else:
            avg_cpu = max_cpu = 0
            
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            max_memory = np.max(memory_usage)
        else:
            avg_memory = max_memory = 0
        
        # Compile all metrics
        metrics = {
            "framework": framework,
            "requested_rate": rate,
            "actual_rate": actual_rate,
            "buffer_size": max_buffer_size,
            "experiment_duration": total_time,
            "total_requests": total_requests,
            "processed_requests": len(latencies),
            "successful_requests": successes,
            "failed_requests": failures,
            "dropped_requests": buffer_full_count,
            "success_rate": successes / total_requests if total_requests > 0 else 0,
            # Latency metrics
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "std_latency": std_latency,
            # Preprocessing and network metrics
            "avg_preprocessing_time": avg_preprocessing,
            "max_preprocessing_time": max_preprocessing,
            "avg_network_time": avg_network,
            "max_network_time": max_network,
            # Performance metrics
            "throughput": throughput,
            "model_accuracy": model_accuracy,
            "accuracy_samples": total_accuracy_samples,
            # Resource usage
            "avg_cpu_usage": avg_cpu,
            "max_cpu_usage": max_cpu,
            "avg_memory_usage": avg_memory,
            "max_memory_usage": max_memory,
            # Timestamps
            "test_timestamp": datetime.now().isoformat(),
            "detailed_metrics": detailed_metrics,
            "latencies": latencies,
        }
        
        # Log summary
        logger.info(f"Experiment completed:")
        logger.info(f"  Requested rate: {rate} req/sec")
        logger.info(f"  Actual rate: {metrics['actual_rate']:.2f} req/sec")
        logger.info(f"  Success rate: {metrics['success_rate']*100:.2f}%")
        logger.info(f"  Avg latency: {metrics['avg_latency']*1000:.2f}ms")
        logger.info(f"  P95 latency: {metrics['p95_latency']*1000:.2f}ms")
        logger.info(f"  P99 latency: {metrics['p99_latency']*1000:.2f}ms")
        logger.info(f"  Throughput: {metrics['throughput']:.2f} req/sec")
        
        return metrics
    
    def plot_results(self, metrics_list, output_prefix):
        """Plot performance results"""
        logger.info("Generating plots...")
        
        # Organize metrics by framework
        frameworks = {}
        for m in metrics_list:
            if m["framework"] not in frameworks:
                frameworks[m["framework"]] = []
            frameworks[m["framework"]].append(m)
        
        # Prepare data for plotting
        rate_values = sorted(list(set([m["requested_rate"] for m in metrics_list])))
        buffer_values = sorted(list(set([m["buffer_size"] for m in metrics_list])))
        
        # Setup plot style
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12})
        
        # 1. Plot latency by arrival rate
        plt.figure(figsize=(12, 10))
        
        # Average latency subplot
        plt.subplot(2, 1, 1)
        for framework, metrics in frameworks.items():
            for buffer_size in buffer_values:
                data = [m for m in metrics if m["buffer_size"] == buffer_size]
                if data:
                    data.sort(key=lambda x: x["requested_rate"])
                    x = [m["requested_rate"] for m in data]
                    y = [m["avg_latency"] * 1000 for m in data]  # Convert to ms
                    plt.plot(x, y, marker='o', linewidth=2, label=f"{framework} (buffer={buffer_size})")
        
        plt.title('Average Latency by Request Rate', fontsize=16)
        plt.ylabel('Average Latency (ms)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        # P95 and P99 latencies subplot
        plt.subplot(2, 1, 2)
        markers = {'p95': 's', 'p99': '^'}
        for framework, metrics in frameworks.items():
            for buffer_size in buffer_values:
                data = [m for m in metrics if m["buffer_size"] == buffer_size]
                if data:
                    data.sort(key=lambda x: x["requested_rate"])
                    x = [m["requested_rate"] for m in data]
                    y_p95 = [m["p95_latency"] * 1000 for m in data]  # Convert to ms
                    y_p99 = [m["p99_latency"] * 1000 for m in data]  # Convert to ms
                    
                    plt.plot(x, y_p95, marker=markers['p95'], linestyle='--', 
                             label=f"{framework} P95 (buffer={buffer_size})")
                    plt.plot(x, y_p99, marker=markers['p99'], linestyle=':', 
                             label=f"{framework} P99 (buffer={buffer_size})")
        
        plt.xlabel('Request Rate (requests/second)', fontsize=14)
        plt.ylabel('Latency (ms)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "plots", f"{output_prefix}_latency_by_rate.png"), dpi=300)
        
        # 2. Plot throughput by arrival rate
        plt.figure(figsize=(10, 8))
        for framework, metrics in frameworks.items():
            for buffer_size in buffer_values:
                data = [m for m in metrics if m["buffer_size"] == buffer_size]
                if data:
                    data.sort(key=lambda x: x["requested_rate"])
                    x = [m["requested_rate"] for m in data]
                    y = [m["throughput"] for m in data]
                    plt.plot(x, y, marker='o', linewidth=2, label=f"{framework} (buffer={buffer_size})")
        
        # Add ideal throughput line (1:1)
        if rate_values:
            max_rate = max(rate_values)
            plt.plot([0, max_rate], [0, max_rate], 'k--', label='Ideal throughput', alpha=0.5)
        
        plt.title('Throughput vs. Request Rate', fontsize=16)
        plt.xlabel('Request Rate (requests/second)', fontsize=14)
        plt.ylabel('Throughput (requests/second)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "plots", f"{output_prefix}_throughput_by_rate.png"), dpi=300)
        
        # 3. Plot success rate and actual vs. requested rate
        plt.figure(figsize=(12, 10))
        
        # Success rate subplot
        plt.subplot(2, 1, 1)
        for framework, metrics in frameworks.items():
            for buffer_size in buffer_values:
                data = [m for m in metrics if m["buffer_size"] == buffer_size]
                if data:
                    data.sort(key=lambda x: x["requested_rate"])
                    x = [m["requested_rate"] for m in data]
                    y = [m["success_rate"] * 100 for m in data]
                    plt.plot(x, y, marker='o', linewidth=2, label=f"{framework} (buffer={buffer_size})")
        
        plt.title('Success Rate by Request Rate', fontsize=16)
        plt.ylabel('Success Rate (%)', fontsize=14)
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True)
        
        # Actual vs. requested rate subplot
        plt.subplot(2, 1, 2)
        for framework, metrics in frameworks.items():
            for buffer_size in buffer_values:
                data = [m for m in metrics if m["buffer_size"] == buffer_size]
                if data:
                    data.sort(key=lambda x: x["requested_rate"])
                    x = [m["requested_rate"] for m in data]
                    y = [m["actual_rate"] for m in data]
                    plt.plot(x, y, marker='o', linewidth=2, label=f"{framework} (buffer={buffer_size})")
        
        # Add ideal line (1:1)
        if rate_values:
            max_rate = max(rate_values)
            plt.plot([0, max_rate], [0, max_rate], 'k--', label='Ideal (1:1)', alpha=0.5)
        
        plt.title('Actual vs. Requested Rate', fontsize=16)
        plt.xlabel('Requested Rate (requests/second)', fontsize=14)
        plt.ylabel('Actual Rate (requests/second)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "plots", f"{output_prefix}_success_and_actual_rate.png"), dpi=300)
        
        # 4. Plot resource usage
        plt.figure(figsize=(12, 10))
        
        # CPU usage subplot
        plt.subplot(2, 1, 1)
        for framework, metrics in frameworks.items():
            data = []
            for buffer_size in buffer_values:
                buffer_data = [m for m in metrics if m["buffer_size"] == buffer_size]
                data.extend(buffer_data)
            
            if data:
                data.sort(key=lambda x: x["requested_rate"])
                x = [m["requested_rate"] for m in data]
                y_avg = [m["avg_cpu_usage"] for m in data]
                y_max = [m["max_cpu_usage"] for m in data]
                
                plt.plot(x, y_avg, marker='o', linewidth=2, label=f"{framework} Avg CPU")
                plt.plot(x, y_max, marker='s', linestyle='--', linewidth=2, label=f"{framework} Max CPU")
        
        plt.title('CPU Usage by Request Rate', fontsize=16)
        plt.ylabel('CPU Usage (%)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        # Memory usage subplot
        plt.subplot(2, 1, 2)
        for framework, metrics in frameworks.items():
            data = []
            for buffer_size in buffer_values:
                buffer_data = [m for m in metrics if m["buffer_size"] == buffer_size]
                data.extend(buffer_data)
            
            if data:
                data.sort(key=lambda x: x["requested_rate"])
                x = [m["requested_rate"] for m in data]
                y_avg = [m["avg_memory_usage"] for m in data]
                y_max = [m["max_memory_usage"] for m in data]
                
                plt.plot(x, y_avg, marker='o', linewidth=2, label=f"{framework} Avg Memory")
                plt.plot(x, y_max, marker='s', linestyle='--', linewidth=2, label=f"{framework} Max Memory")
        
        plt.title('Memory Usage by Request Rate', fontsize=16)
        plt.xlabel('Request Rate (requests/second)', fontsize=14)
        plt.ylabel('Memory Usage (%)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "plots", f"{output_prefix}_resource_usage.png"), dpi=300)
        
        # 5. Plot time breakdown (preprocessing vs. network)
        plt.figure(figsize=(12, 10))
        for framework, metrics in frameworks.items():
            for i, rate in enumerate(rate_values):
                plt.subplot(len(rate_values), 1, i+1)
                
                # Collect data for this rate
                buffer_data = []
                xlabels = []
                
                for buffer_size in buffer_values:
                    buffer_metrics = [m for m in metrics if m["requested_rate"] == rate and m["buffer_size"] == buffer_size]
                    
                    if buffer_metrics:
                        m = buffer_metrics[0]
                        
                        # Calculate time components
                        prep_time = m.get("avg_preprocessing_time", 0) * 1000  # ms
                        net_time = m.get("avg_network_time", 0) * 1000  # ms
                        total_time = m.get("avg_latency", 0) * 1000  # ms
                        service_time = max(0, total_time - prep_time - net_time)  # ms
                        
                        buffer_data.append((prep_time, net_time, service_time))
                        xlabels.append(f"buf={buffer_size}")
                
                # Create stacked bar for this rate
                if buffer_data:
                    x = np.arange(len(buffer_data))
                    width = 0.35
                    
                    prep_times = [d[0] for d in buffer_data]
                    net_times = [d[1] for d in buffer_data]
                    service_times = [d[2] for d in buffer_data]
                    
                    plt.bar(x, prep_times, width, label='Preprocessing')
                    plt.bar(x, net_times, width, bottom=prep_times, label='Network')
                    
                    # Add the service time on top
                    bottoms = [p+n for p,n in zip(prep_times, net_times)]
                    plt.bar(x, service_times, width, bottom=bottoms, label='Service')
                    
                    plt.title(f'{framework} - Time Breakdown at {rate} req/sec', fontsize=14)
                    plt.ylabel('Time (ms)', fontsize=12)
                    plt.xticks(x, xlabels)
                    
                    # Only add legend to first subplot
                    if i == 0:
                        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "plots", f"{output_prefix}_time_breakdown.png"), dpi=300)
        
        logger.info("All plots saved successfully")
    
    def save_metrics(self, metrics_list, filename):
        """Save metrics to files"""
        filepath = os.path.join(self.results_dir, filename)
        
        # Organize metrics by framework for summary
        frameworks = {}
        for m in metrics_list:
            if m["framework"] not in frameworks:
                frameworks[m["framework"]] = []
            frameworks[m["framework"]].append(m)
        
        # Get unique buffer sizes and rates for comparison
        buffer_values = sorted(list(set([m["buffer_size"] for m in metrics_list])))
        rate_values = sorted(list(set([m["requested_rate"] for m in metrics_list])))
        
        # Remove raw latency data for storage efficiency
        metrics_for_save = []
        for m in metrics_list:
            m_copy = m.copy()
            if "latencies" in m_copy:
                # Save latency statistics
                if m_copy["latencies"]:
                    m_copy["latency_stats"] = {
                        "min": min(m_copy["latencies"]),
                        "max": max(m_copy["latencies"]),
                        "mean": np.mean(m_copy["latencies"]),
                        "median": np.median(m_copy["latencies"]),
                        "p95": np.percentile(m_copy["latencies"], 95),
                        "p99": np.percentile(m_copy["latencies"], 99),
                        "std_dev": np.std(m_copy["latencies"])
                    }
                del m_copy["latencies"]
            
            # Clean up detailed metrics
            if "detailed_metrics" in m_copy:
                del m_copy["detailed_metrics"]
                
            metrics_for_save.append(m_copy)
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(metrics_for_save, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
        
        # Save summary text file
        summary_filepath = os.path.join(self.results_dir, f"{os.path.splitext(filename)[0]}_summary.txt")
        with open(summary_filepath, 'w') as f:
            f.write("=== MODEL SERVING PERFORMANCE TEST SUMMARY ===\n\n")
            f.write(f"Test conducted at: {datetime.now().isoformat()}\n\n")
            
            for framework in frameworks.keys():
                framework_metrics = [m for m in metrics_for_save if m["framework"] == framework]
                if framework_metrics:
                    f.write(f"=== {framework.upper()} RESULTS ===\n\n")
                    
                    for metric in sorted(framework_metrics, key=lambda x: (x["buffer_size"], x["requested_rate"])):
                        f.write(f"Buffer Size: {metric['buffer_size']}, Requested Rate: {metric['requested_rate']} req/sec\n")
                        f.write(f"Actual Rate: {metric.get('actual_rate', 0):.2f} req/sec\n")
                        f.write(f"Total Requests: {metric['total_requests']}\n")
                        f.write(f"Processed: {metric['processed_requests']}, "
                                f"Successful: {metric['successful_requests']}, "
                                f"Failed: {metric['failed_requests']}, "
                                f"Dropped: {metric['dropped_requests']}\n")
                        f.write(f"Success Rate: {metric['success_rate']*100:.2f}%\n")
                        f.write(f"Latency (ms): Avg={metric['avg_latency']*1000:.2f}, "
                                f"Min={metric.get('min_latency', 0)*1000:.2f}, "
                                f"Max={metric.get('max_latency', 0)*1000:.2f}, "
                                f"P50={metric['p50_latency']*1000:.2f}, "
                                f"P95={metric['p95_latency']*1000:.2f}, "
                                f"P99={metric['p99_latency']*1000:.2f}, "
                                f"StdDev={metric.get('std_latency', 0)*1000:.2f}\n")
                        
                        # Add time breakdown
                        if 'avg_preprocessing_time' in metric:
                            f.write(f"Time Breakdown (ms): "
                                    f"Preprocessing={metric['avg_preprocessing_time']*1000:.2f}, "
                                    f"Network={metric['avg_network_time']*1000:.2f}, "
                                    f"Service={metric['avg_latency']*1000 - metric['avg_preprocessing_time']*1000 - metric['avg_network_time']*1000:.2f}\n")
                            
                        f.write(f"Throughput: {metric['throughput']:.2f} req/sec\n")
                        
                        # Include model accuracy if available
                        if metric.get('model_accuracy', 0) > 0:
                            f.write(f"Model Accuracy: {metric['model_accuracy']:.2f}% "
                                    f"(from {metric['accuracy_samples']} samples)\n")
                            
                        f.write(f"CPU Usage: Avg={metric['avg_cpu_usage']:.2f}%, "
                                f"Max={metric['max_cpu_usage']:.2f}%\n")
                        f.write(f"Memory Usage: Avg={metric['avg_memory_usage']:.2f}%, "
                                f"Max={metric['max_memory_usage']:.2f}%\n\n")
            
            # Comparison section
            if len(frameworks) > 1:
                f.write("=== FRAMEWORK COMPARISON ===\n\n")
                
                # Compare for each rate and buffer size combination
                for buffer_size in buffer_values:
                    for rate in rate_values:
                        metrics_subset = [m for m in metrics_for_save 
                                          if m["buffer_size"] == buffer_size and m["requested_rate"] == rate]
                        
                        if len(metrics_subset) > 1:  # Only compare if we have multiple frameworks
                            f.write(f"Buffer Size: {buffer_size}, Rate: {rate} req/sec\n")
                            
                            # Compare latency
                            sorted_by_latency = sorted(metrics_subset, key=lambda x: x["avg_latency"])
                            if len(sorted_by_latency) >= 2 and sorted_by_latency[1]["avg_latency"] > 0:
                                latency_diff = (sorted_by_latency[1]["avg_latency"] / sorted_by_latency[0]["avg_latency"] - 1) * 100
                                f.write(f"  {sorted_by_latency[0]['framework']} has {latency_diff:.2f}% lower latency\n")
                            
                            # Compare throughput
                            sorted_by_throughput = sorted(metrics_subset, key=lambda x: x["throughput"], reverse=True)
                            if len(sorted_by_throughput) >= 2 and sorted_by_throughput[1]["throughput"] > 0:
                                throughput_diff = (sorted_by_throughput[0]["throughput"] / sorted_by_throughput[1]["throughput"] - 1) * 100
                                f.write(f"  {sorted_by_throughput[0]['framework']} has {throughput_diff:.2f}% higher throughput\n")
                            
                            # Compare success rate
                            sorted_by_success = sorted(metrics_subset, key=lambda x: x["success_rate"], reverse=True)
                            if len(sorted_by_success) >= 2 and sorted_by_success[1]["success_rate"] > 0:
                                success_diff = (sorted_by_success[0]["success_rate"] - sorted_by_success[1]["success_rate"]) * 100
                                f.write(f"  {sorted_by_success[0]['framework']} has {success_diff:.2f}% higher success rate\n")
                            
                            f.write("\n")
                
                # Overall winner determination
                f.write("=== OVERALL PERFORMANCE COMPARISON ===\n\n")
                framework_points = {framework: 0 for framework in frameworks.keys()}
                
                # Assign points for each rate/buffer combination
                for buffer_size in buffer_values:
                    for rate in rate_values:
                        metrics_subset = [m for m in metrics_for_save 
                                          if m["buffer_size"] == buffer_size and m["requested_rate"] == rate]
                        
                        if len(metrics_subset) > 1:
                            # 1 point for lowest latency
                            sorted_by_latency = sorted(metrics_subset, key=lambda x: x["avg_latency"])
                            if sorted_by_latency[0]["avg_latency"] > 0:
                                framework_points[sorted_by_latency[0]["framework"]] += 1
                            
                            # 1 point for highest throughput
                            sorted_by_throughput = sorted(metrics_subset, key=lambda x: x["throughput"], reverse=True)
                            if sorted_by_throughput[0]["throughput"] > 0:
                                framework_points[sorted_by_throughput[0]["framework"]] += 1
                            
                            # 1 point for highest success rate
                            sorted_by_success = sorted(metrics_subset, key=lambda x: x["success_rate"], reverse=True)
                            if sorted_by_success[0]["success_rate"] > 0:
                                framework_points[sorted_by_success[0]["framework"]] += 1
                
                # Determine overall winner
                if framework_points:
                    winner = max(framework_points.items(), key=lambda x: x[1])
                    f.write(f"Overall performance winner: {winner[0].upper()} with {winner[1]} points\n\n")
                    
                    # Show point breakdown
                    f.write("Points breakdown:\n")
                    for framework, points in framework_points.items():
                        f.write(f"  {framework}: {points} points\n")
            
            # CSV export
            csv_filepath = os.path.join(self.results_dir, f"{os.path.splitext(filename)[0]}_summary.csv")
            with open(csv_filepath, 'w') as csv_file:
                # Write header
                csv_file.write("Framework,BufferSize,RequestedRate,ActualRate,TotalRequests,ProcessedRequests,")
                csv_file.write("SuccessfulRequests,FailedRequests,DroppedRequests,SuccessRate,AvgLatency_ms,")
                csv_file.write("P50Latency_ms,P95Latency_ms,P99Latency_ms,Throughput,")
                csv_file.write("ModelAccuracy,AvgCpuUsage,MaxCpuUsage,AvgMemoryUsage,MaxMemoryUsage\n")
                
                # Write data rows
                for m in metrics_for_save:
                    csv_file.write(f"{m['framework']},{m['buffer_size']},{m['requested_rate']},{m.get('actual_rate', 0):.2f},")
                    csv_file.write(f"{m['total_requests']},{m['processed_requests']},")
                    csv_file.write(f"{m['successful_requests']},{m['failed_requests']},{m['dropped_requests']},")
                    csv_file.write(f"{m['success_rate']*100:.2f},{m['avg_latency']*1000:.2f},")
                    csv_file.write(f"{m['p50_latency']*1000:.2f},{m['p95_latency']*1000:.2f},{m['p99_latency']*1000:.2f},")
                    csv_file.write(f"{m['throughput']:.2f},")
                    csv_file.write(f"{m.get('model_accuracy', 0):.2f},{m['avg_cpu_usage']:.2f},{m['max_cpu_usage']:.2f},")
                    csv_file.write(f"{m['avg_memory_usage']:.2f},{m['max_memory_usage']:.2f}\n")
            
            logger.info(f"CSV summary saved to {csv_filepath}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="TensorFlow Serving vs TorchServe Performance Comparison")
    parser.add_argument("--rates", nargs="+", type=float, default=[1, 5, 10], 
                      help="List of request rates to test (requests/second)")
    parser.add_argument("--buffer-sizes", nargs="+", type=int, default=[10, 50], 
                      help="List of buffer sizes to test")
    parser.add_argument("--duration", type=int, default=15, 
                      help="Duration of each test in seconds")
    parser.add_argument("--output", type=str, default="serving_comparison", 
                      help="Prefix for output files")
    parser.add_argument("--test-only", action="store_true",
                      help="Only test connections and exit")
    parser.add_argument("--tf-model", type=str, default="fashion",
                      help="TensorFlow model name")
    parser.add_argument("--ts-model", type=str, default="fashion_mnist",
                      help="TorchServe model name")
    parser.add_argument("--tf-host", type=str, default="localhost",
                      help="TensorFlow Serving host")
    parser.add_argument("--ts-host", type=str, default="localhost",
                      help="TorchServe host")
    parser.add_argument("--tf-only", action="store_true",
                      help="Only test TensorFlow Serving")
    parser.add_argument("--ts-only", action="store_true",
                      help="Only test TorchServe")
    parser.add_argument("--worker-threads", type=int, default=10,
                      help="Number of worker threads for processing requests")
    
    args = parser.parse_args()
    
    # Initialize performance test framework
    tester = ServingComparison(
        tf_model_name=args.tf_model,
        ts_model_name=args.ts_model,
        tf_host=args.tf_host,
        ts_host=args.ts_host
    )
    
    # Test connections first
    logger.info("Testing connections to model serving platforms...")
    tf_success, ts_success = tester.test_connections()
    
    if args.test_only:
        logger.info("Connection test completed. Exiting.")
        return
    
    # Override success flags based on command line args
    if args.tf_only:
        ts_success = False
    if args.ts_only:
        tf_success = False
    
    # Check if at least one service is available
    if not (tf_success or ts_success):
        logger.error("No model serving platforms are available. Please check your services and try again.")
        return
    
    all_metrics = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{args.output}_{timestamp}"
    
    # Run tests for TensorFlow Serving
    if tf_success:
        for rate in args.rates:
            for buffer_size in args.buffer_sizes:
                logger.info(f"\n===== Testing TensorFlow Serving with rate={rate} req/sec, buffer={buffer_size} =====")
                metrics = tester.run_experiment("tensorflow", rate, args.duration, buffer_size, args.worker_threads)
                all_metrics.append(metrics)
                
                # Save intermediate results after each test
                tester.save_metrics(all_metrics, f"{output_prefix}_metrics.json")
    else:
        logger.warning("Skipping TensorFlow Serving tests.")
    
    # Run tests for TorchServe
    if ts_success:
        for rate in args.rates:
            for buffer_size in args.buffer_sizes:
                logger.info(f"\n===== Testing TorchServe with rate={rate} req/sec, buffer={buffer_size} =====")
                metrics = tester.run_experiment("torchserve", rate, args.duration, buffer_size, args.worker_threads)
                all_metrics.append(metrics)
                
                # Save intermediate results after each test
                tester.save_metrics(all_metrics, f"{output_prefix}_metrics.json")
    else:
        logger.warning("Skipping TorchServe tests.")
    
    # Generate plots if we have metrics
    if all_metrics:
        logger.info("Generating comparison plots...")
        tester.plot_results(all_metrics, output_prefix)
    
    logger.info(f"\nExperiment completed. Results saved with prefix '{output_prefix}'")
    logger.info(f"Check the '{tester.results_dir}' directory for detailed results and plots.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())