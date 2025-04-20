# TensorFlow Serving vs TorchServe: Performance Benchmark

A comprehensive performance benchmark comparing TensorFlow Serving and TorchServe for machine learning model deployment using a common MobileNetV2 model trained on Fashion-MNIST.

## Project Overview

This project implements a stochastic arrival process to compare the performance characteristics of two popular machine learning model serving frameworks. The comparison evaluates performance across various metrics including latency, throughput, success rate, and resource utilization.

## EC2 Environment Setup

1. Launch an Ubuntu EC2 instance (t3.small or larger)

2. Configure security groups:
   ```
   1. IPv4 SSH TCP 22 0.0.0.0/0
   2. IPv4 Custom TCP 8081 0.0.0.0/0  # TorchServe Management
   3. IPv4 Custom TCP 8082 0.0.0.0/0  # TorchServe Metrics
   4. IPv4 Custom TCP 8080 0.0.0.0/0  # TorchServe REST API
   5. IPv4 Custom TCP 8501 0.0.0.0/0  # TensorFlow Serving
   ```

3. Connect to your instance via SSH:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@your-ec2-public-dns.amazonaws.com
   ```

4. Install dependencies:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   
   # Install Python and pip
   sudo apt install -y python3-pip python3-dev
   
   # Install required packages
   pip3 install torch torchvision torchaudio numpy matplotlib requests psutil
   pip3 install torchserve torch-model-archiver torch-workflow-archiver
   pip3 install tensorflow tensorflow-serving-api pillow
   
   # Install TensorFlow Serving
   echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
   curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
   sudo apt update
   sudo apt install -y tensorflow-model-server
   ```

## Implementation Workflow

### Step 1: Model Training

1. Create the model training script:
   ```bash
   nano train.py
   # Copy the MobileNetV2 training code that uses Fashion-MNIST dataset
   ```

2. Run the training script:
   ```bash
   python3 train.py
   ```

3. This generates the following model files:
   - `mobilenet_fashion_mnist.pth` - PyTorch model
   - `mobilenet_fashion_mnist.pt` - TorchScript model
   - `mobilenet_fashion_mnist.onnx` - ONNX model
   - `index_to_name.json` - Class mapping

### Step 2: Model Packaging for TorchServe

1. Create the model handler:
   ```bash
   nano model_handler.py
   # Copy the TorchServe model handler implementation
   ```

2. Package the model for TorchServe:
   ```bash
   torch-model-archiver --model-name fashion_mnist \
       --version 1.0 \
       --model-file mobilenet_fashion_mnist.py \
       --serialized-file mobilenet_fashion_mnist.pt \
       --handler model_handler.py \
       --extra-files index_to_name.json \
       --export-path model_store
   ```

### Step 3: TensorFlow Model Configuration

1. Create TensorFlow model directory:
   ```bash
   mkdir -p tensorflow_model/fashion/1
   ```

2. Copy or convert the model for TensorFlow Serving:
   ```bash
   # If using TensorFlow SavedModel format
   # Copy the saved model to the appropriate directory
   ```

3. Create TensorFlow model configuration:
   ```bash
   mkdir -p tensorflow_model_config
   nano tensorflow_model_config/models.config
   # Add model configuration
   ```

### Step 4: Start Serving Frameworks

1. Start TorchServe:
   ```bash
   # Create config file
   mkdir -p config
   nano config/config.properties
   # Add TorchServe configuration including auth tokens
   
   # Start TorchServe
   torchserve --stop
   sleep 5
   torchserve --start --model-store model_store --ts-config config/config.properties
   ```

2. Verify TorchServe is running:
   ```bash
   curl -v -X GET http://localhost:8081/models/fashion_mnist \
     -H "Authorization: Bearer znzY7E0K"
   ```

3. Start TensorFlow Serving:
   ```bash
   # Start in background mode
   nohup tensorflow_model_server \
     --rest_api_port=8501 \
     --model_name=fashion \
     --model_base_path=$(pwd)/tensorflow_model/fashion > logs/tf_serving.log 2>&1 &
   ```

4. Verify TensorFlow Serving is running:
   ```bash
   curl http://localhost:8501/v1/models/fashion
   ```

### Step 5: Create Benchmark Scripts

1. Create the comparison script:
   ```bash
   nano compare_serving.py
   # Copy the benchmark implementation code
   ```

2. Create the benchmark runner script:
   ```bash
   nano run_benchmark.sh
   ```

3. Add the following content to run_benchmark.sh:
   ```bash
   #!/bin/bash
   
   # Check if TorchServe is running
   echo "Checking TorchServe status..."
   curl -s -H "Authorization: Bearer znzY7E0K" http://localhost:8081/ping > /dev/null
   if [ $? -ne 0 ]; then
       echo "TorchServe is not running. Check logs for errors."
       exit 1
   fi
   echo "TorchServe is running."
   
   # Set the auth keys as environment variables
   export TS_MANAGEMENT_KEY="znzY7E0K"
   export TS_INFERENCE_KEY="FRQNu7js"
   export TS_API_KEY="Df2GX-OB"
   
   # Run the benchmark with optimized parameters
   echo -e "\n===== Running Performance Benchmark ====="
   echo "This will compare TensorFlow Serving and TorchServe with the following parameters:"
   echo "- Request rates: 1, 5, 10 requests/second"
   echo "- Buffer sizes: 10, 50 requests"
   echo "- Test duration: 15 seconds per test"
   echo "- Worker threads: 10"
   echo -e "Total estimated duration: ~3 minutes\n"
   
   echo "Starting benchmark..."
   python3 compare_serving.py --rates 1 5 10 --buffer-sizes 10 50 --duration 15 --worker-threads 10
   
   echo -e "\nBenchmark complete! Check serving_comparison_results directory for results."
   echo "You can find the plots in serving_comparison_results/plots/"
   echo "Summary reports are available in serving_comparison_results/"
   ```

4. Make the benchmark script executable:
   ```bash
   chmod +x run_benchmark.sh
   ```

### Step 6: Run the Benchmark

1. Execute the benchmark:
   ```bash
   ./run_benchmark.sh
   ```

2. The script will:
   - Verify TorchServe is running
   - Set the necessary authorization keys
   - Run comparison tests with various request rates and buffer sizes
   - Generate results in the serving_comparison_results directory

3. View the results:
   ```bash
   cat ~/serving_comparison_results/serving_comparison_*_metrics_summary.txt
   ```

4. Visualizations are available in:
   ```
   serving_comparison_results/plots/
   ```

## Directory Structure

```
~/
├── compare_serving.py           # Main benchmark script
├── config/                      # TorchServe configuration
├── data/                        # Downloaded datasets
├── fashion_mnist_data/          # Processed Fashion-MNIST data
├── index_to_name.json           # Class name mapping
├── logs/                        # Log files
├── mobilenet_fashion_mnist.onnx # ONNX model
├── mobilenet_fashion_mnist.pt   # TorchScript model
├── mobilenet_fashion_mnist.pth  # PyTorch model
├── model_handler.py             # TorchServe handler
├── model_store/                 # TorchServe model repository
├── models/                      # Saved model files
├── run_benchmark.sh             # Benchmark runner script
├── serving_comparison.log       # Benchmark log
├── serving_comparison_results/  # Results and visualizations
├── tensorflow_model/            # TensorFlow models
├── tensorflow_model_config/     # TensorFlow configuration
└── train.py                     # Model training script
```

## Benchmark Parameters

The benchmark script supports the following parameters:

- `--rates`: Request rates to test (requests/second)
- `--buffer-sizes`: Request buffer sizes to test
- `--duration`: Duration of each test in seconds
- `--worker-threads`: Number of worker threads for processing requests
- `--tf-host`: TensorFlow Serving host (default: localhost)
- `--ts-host`: TorchServe host (default: localhost)
- `--tf-model`: TensorFlow model name (default: fashion)
- `--ts-model`: TorchServe model name (default: fashion_mnist)
- `--tf-only`: Only test TensorFlow Serving
- `--ts-only`: Only test TorchServe

## Key Findings

Our benchmark results revealed the following key performance characteristics:

1. **Latency Performance**:
   - At lower request rates (1-5 req/sec), TensorFlow Serving generally shows lower latency with buffer size 10
   - At higher request rates (10 req/sec), TorchServe achieves lower latency, particularly with buffer size 10
   - TorchServe shows a 7-24% lower latency at high request rates

2. **Throughput Performance**:
   - TensorFlow Serving performs better at lower rates with larger buffer sizes
   - TorchServe shows up to 16% higher throughput at higher request rates
   - Both frameworks approach the ideal throughput line, with TorchServe buffer=10 coming closest at high rates

3. **CPU and Memory Usage**:
   - TensorFlow has lower average CPU usage across all request rates
   - TorchServe shows higher peak CPU usage (reaching 100% at higher rates)
   - Memory usage differs: TensorFlow uses ~78% vs TorchServe's ~83%

4. **Success Rate**:
   - Both frameworks maintain near-perfect success rates at lower request levels
   - At 10 req/sec, TorchServe maintains higher success rates (95-100%) compared to TensorFlow (~87%)

5. **Overall Performance**:
   - Both frameworks tied at 9 points in the overall evaluation
   - TensorFlow excels in resource efficiency and low-load scenarios
   - TorchServe performs better under high load conditions

## Framework Selection Recommendations

Based on our findings, we recommend the following:

- **TensorFlow Serving**: Ideal for steady, predictable workloads where resource efficiency is a priority
- **TorchServe**: Better suited for high-volume, variable traffic patterns where high throughput and success rate are critical

The choice between frameworks should be based on specific deployment requirements, expected traffic patterns, and resource constraints.
