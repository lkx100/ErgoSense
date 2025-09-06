# ErgoSense 🧘‍♀️

A real-time posture monitoring application that uses computer vision to help improve your workspace ergonomics and maintain healthy posture habits.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14+-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0+-red.svg)](https://opencv.org/)

## 🌟 Features

- **Real-time Pose Detection**: Uses Google's MediaPipe for accurate body landmark detection
- **Posture Analysis**: Monitors key ergonomic metrics including:
  - Neck forward position (forward head posture)
  - Shoulder elevation
  - Face distance from screen
- **Live Visualization**: Real-time overlay of detected pose landmarks on camera feed
- **Configurable Thresholds**: Customizable sensitivity settings for different posture alerts
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### 1. Using Docker
Install docker desktop from [here](https://docs.docker.com/get-started/introduction/get-docker-desktop/), if not already installed.

Pull and run the official Docker image:

```bash
# Pull the latest image
docker pull luckyx100/ergosense

# Run the container
docker run -it -p 8501:8501 luckyx100/ergosense
```

To Access the application:
Open http://localhost:8501 in your browser

### 2. Local Setup

   - Python 3.12 or higher
   - [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

**Installation Steps**

1. **Clone the repository**
   ```bash
   git clone https://github.com/lkx100/ErgoSense.git
   cd ErgoSense
   ```

2. **Install dependencies**
   
   Using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or Using pip:
   ```bash
   pip install opencv-python mediapipe streamlit streamlit-webrtc
   ```

3. **Download MediaPipe models** (if not included)
   
   The application uses MediaPipe pose detection models. Ensure the model files are in the `models/` directory:
   - `pose_landmarker_lite.task` or
   - `pose_landmarker_full.task` or
   - `pose_landmarker_heavy.task`

   Download any one of them officially from [Mediapipe by Google](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) to ensure you have the latest versions.

4. **Running the Application**

   The Streamlit browser UI is now the primary interface. All video processing happens locally in your browser (no frames are uploaded).

   ```bash
   # Run locally with UV
   uv run streamlit run ergo_web.py

   # or directly with Python
   python -m streamlit run ergo_web.py  # use python3 for macOS or linux
   ```

## 📁 Project Structure

```
ErgoSense/
├── ergo_web.py            # Main Streamlit web interface
├── config/                # Configuration settings
│   └── defaults.py        # Default thresholds and settings
├── core/                  # Core functionality
│   ├── pose_detector.py   # MediaPipe pose detection wrapper
│   └── landmark_extractor.py # Posture analysis logic
├── models/                # Store MediaPipe model files here
├── utils/                 # Utility functions
│   └── camera.py         # Camera management and visualization
└── monitoring/            # Future monitoring features
├── main.py                # Legacy desktop demo (deprecated)
├── Dockerfile             # Container configuration
├── uv.lock                # UV dependency lock file
├── pyproject.toml         # Project configuration and dependencies
```

## ⚙️ Configuration

### Posture Thresholds

Customize posture detection sensitivity in `config/defaults.py`:
- **neck_forward_threshold**: 1.3 (30% increase triggers alert)
- **shoulder_raised_threshold**: 0.8 (20% decrease triggers alert)  
- **face_too_close_threshold**: 1.2 (20% increase triggers alert)

### Camera Settings

Configure camera options in `config/defaults.py`:
- **camera_id**: 0 (change for different camera)
- **target_fps**: 30 (adjust frame rate)

### Model Selection

Choose between different MediaPipe models for performance vs accuracy:
- **pose_landmarker_lite.task**: Fastest, lower accuracy
- **pose_landmarker_full.task**: Balanced performance (default)
- **pose_landmarker_heavy.task**: Highest accuracy, slower

## 📊 Metrics

ErgoSense analyzes the following posture metrics:

- **Ear-Shoulder Distance**: Measures forward head posture
- **Shoulder Height**: Detects shoulder elevation/hunching
- **Head Tilt**: Monitors head alignment
- **Face Size**: Estimates distance from screen

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:

1. **Report bugs** by opening an issue
2. **Suggest features** for ergonomic monitoring
3. **Submit pull requests** with improvements
4. **Improve documentation**

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests (when available)
5. Submit a pull request

## 🙏 Acknowledgments

- **Google MediaPipe** - For providing excellent pose detection models
- **OpenCV** - For computer vision utilities
- **Python Community** - For the amazing ecosystem

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Try different camera IDs in `CAMERA_SETTINGS`
- Ensure no other applications are using the camera

**Poor pose detection:**
- Ensure good lighting
- Position yourself clearly in camera view
- Try different MediaPipe models (lite/full/heavy)

**Performance issues:**
- Lower the target FPS in settings
- Use the "lite" MediaPipe model
- Ensure adequate system resources

## 📞 Support

If you encounter issues or have questions:

1. Check the [Issues](https://github.com/lkx100/ErgoSense/issues) page
2. Create a new issue with detailed information
3. Include your system specifications and error messages

You can also reach out via email [here](mailto:lk5999950@gmail.com)

---

**Stay healthy, stay productive! 💻✨**