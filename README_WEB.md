# ML Frameworks Comparison Web Dashboard

A beautiful, modern web interface for comparing TensorFlow Serving vs TorchServe performance metrics in real-time.

## Features

### 🎨 **Modern UI Design**
- Gradient-based color scheme with glassmorphism effects
- Responsive design that works on all devices
- Smooth animations and transitions
- Real-time status indicators

### 📊 **Interactive Dashboard**
- **System Resource Monitoring**: Real-time CPU, memory, and disk usage
- **Experiment Configuration**: Easy-to-use form for setting up benchmarks
- **Live Progress Tracking**: Visual progress bar with experiment status
- **Performance Charts**: Interactive latency comparison charts using Chart.js
- **Results Table**: Detailed metrics in a sortable table format

### 🚀 **Performance Features**
- **Real-time Updates**: Auto-refreshing system metrics every 5 seconds
- **Background Processing**: Experiments run without blocking the UI
- **Multiple Frameworks**: Compare TensorFlow Serving and TorchServe side-by-side
- **Comprehensive Metrics**: Latency, throughput, success rate, CPU usage

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web Application
```bash
python app.py
```

### 3. Open Your Browser
Navigate to `http://localhost:5000` to access the dashboard.

## Usage Guide

### Running Experiments

1. **Configure Experiment Parameters**:
   - **Request Rates**: Enter comma-separated values (e.g., "1,5,10")
   - **Buffer Sizes**: Enter comma-separated values (e.g., "10,50")
   - **Duration**: Set experiment duration in seconds (5-300)
   - **Worker Threads**: Number of concurrent threads (1-50)

2. **Start Comparison**:
   - Click "Start Comparison" to begin the benchmark
   - Monitor progress in real-time
   - View results automatically when complete

### Understanding the Dashboard

#### System Resources Section
- **CPU Usage**: Current processor utilization
- **Memory Usage**: RAM consumption percentage
- **Disk Usage**: Storage space utilization
- **Last Updated**: Timestamp of last system update

#### Experiment Controls
- Configure all benchmark parameters
- Start/stop experiments
- View quick statistics

#### Progress Tracking
- Visual progress bar
- Experiment ID and status
- Real-time percentage completion

#### Performance Results
- **Interactive Chart**: Latency comparison over different request rates
- **Results Table**: Detailed metrics for each framework
- **Quick Stats**: Total experiments and average latency

## API Endpoints

### System Information
```
GET /api/system-info
```
Returns current system resource usage.

### Experiment Management
```
POST /api/start-comparison
```
Starts a new comparison experiment.

```
GET /api/experiment-status
```
Returns current experiment status and progress.

### Results
```
GET /api/results
```
Retrieves all experiment results.

```
GET /api/results/<filename>
```
Downloads specific result files.

## Technical Architecture

### Frontend Technologies
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive data visualization
- **Font Awesome**: Icon library
- **Vanilla JavaScript**: No heavy frontend framework dependencies

### Backend Technologies
- **Flask**: Lightweight Python web framework
- **psutil**: System monitoring
- **subprocess**: Background experiment execution
- **threading**: Non-blocking operation

### Design Patterns
- **RESTful API**: Clean endpoint structure
- **Background Processing**: Non-blocking experiments
- **Real-time Updates**: Polling-based updates
- **Responsive Design**: Mobile-first approach

## Hosting Options

### Development
```bash
python app.py
```

### Production (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Cloud Deployment
The application can be easily deployed to:
- **Heroku**: Add `Procfile` with `web: gunicorn app:app`
- **AWS Elastic Beanstalk**: Python platform
- **Google Cloud Run**: Container-based deployment
- **DigitalOcean App Platform**: Direct Git deployment

## Customization

### Adding New Metrics
1. Update the `system_info()` endpoint in `app.py`
2. Add corresponding UI elements in `index.html`
3. Update the JavaScript to display new metrics

### Modifying the UI Theme
Edit the CSS variables in `index.html`:
```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}
```

### Adding New Charts
1. Add new canvas elements in the HTML
2. Initialize new Chart.js instances in JavaScript
3. Update the API to return chart data

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   lsof -ti:5000 | xargs kill -9
   ```

2. **Dependencies Not Found**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Experiments Not Starting**:
   - Ensure `comparison_serving.py` is in the same directory
   - Check that TensorFlow Serving and TorchServe are running
   - Verify API keys and endpoints

### Performance Optimization

1. **Reduce Update Frequency**: Change polling intervals in JavaScript
2. **Optimize Chart Rendering**: Limit data points displayed
3. **Enable Caching**: Add response caching for static data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

---

**Built with ❤️ for the ML community**
