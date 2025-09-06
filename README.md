# ⚡ ThoraxAI - Pneumonia Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Advanced AI-powered thoracic X-ray analysis for pneumonia detection using deep learning**

[🚀 Quick Start](#-quick-start) • [📋 Features](#-features) • [🔬 How It Works](#-how-it-works) • [📊 Performance](#-performance) • [🛠️ Installation](#️-installation)

</div>

---

## 🎯 Overview

ThoraxAI is a cutting-edge medical diagnostic system that leverages state-of-the-art deep learning algorithms to automatically detect pneumonia from chest X-ray images. Built with TensorFlow and deployed via Streamlit, this system provides healthcare professionals and researchers with an accurate, fast, and user-friendly tool for thoracic image analysis.

### 🌟 Key Capabilities

- **🔬 Real-time Analysis**: Instant pneumonia detection from uploaded X-ray images
- **🧠 Deep Learning**: Powered by pre-trained CNN models (DenseNet121, InceptionV3, ResNet50, Xception)
- **📈 High Accuracy**: Advanced neural network architecture for reliable diagnosis
- **🎨 Modern UI**: Cyberpunk-themed interface with intuitive user experience
- **⚡ Fast Processing**: Optimized for quick image analysis and results
- **📱 Web-based**: Accessible from any device with a web browser
- **🔒 Privacy-focused**: Local processing ensures patient data security

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone http://github.com/BrightsSolution/ThoraxAI
   cd ThoraxAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run pneumonia_detector.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a chest X-ray image
   - Get instant AI-powered analysis results

---

## 📋 Features

### 🔬 Diagnostic Capabilities

- **Pneumonia Detection**: Binary classification (Normal vs. Pneumonia)
- **Confidence Scoring**: Percentage-based confidence levels for each prediction
- **Image Preprocessing**: Automatic resizing, normalization, and optimization
- **Multiple Format Support**: JPG, JPEG, PNG image formats

### 🎨 User Interface

- **⚡ Control Panel**: System overview and technical specifications
- **🔬 Diagnostic Scanner**: Real-time image analysis interface
- **📊 Results Display**: Clear, professional diagnostic reports
- **🎯 Responsive Design**: Optimized for desktop and mobile devices

### 🧠 Technical Features

- **Transfer Learning**: Utilizes pre-trained CNN models for enhanced accuracy
- **Data Augmentation**: Advanced image preprocessing techniques
- **Model Optimization**: Fine-tuned for medical image analysis
- **Scalable Architecture**: Easy to extend with additional diagnostic capabilities

---

## 🔬 How It Works

### 1. **Image Upload** 📸
   - Users upload chest X-ray images through the web interface
   - System validates and preprocesses the input image

### 2. **Preprocessing** ⚙️
   - Image resizing to optimal dimensions (150x150 pixels)
   - Normalization to [0,1] range for neural network compatibility
   - RGB channel conversion and batch preparation

### 3. **Neural Analysis** 🧠
   - Pre-trained CNN model processes the image
   - Feature extraction and classification
   - Confidence score calculation

### 4. **Results Display** 📊
   - Binary classification result (Normal/Pneumonia)
   - Confidence percentage
   - Professional diagnostic report format

---

## 📊 Performance

### Model Specifications

- **Architecture**: Convolutional Neural Network (CNN)
- **Base Models**: DenseNet121, InceptionV3, ResNet50, Xception
- **Input Size**: 150x150x3 (RGB images)
- **Output**: Binary classification with confidence scores
- **Processing Time**: < 2 seconds per image

### Accuracy Metrics

- **Training Accuracy**: 95%+
- **Validation Accuracy**: 90%+
- **Test Accuracy**: 88%+
- **Precision**: 89%+
- **Recall**: 87%+

---

## 🛠️ Installation

### System Requirements

- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **GPU**: Optional (CUDA-compatible for faster processing)

### Detailed Setup

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv thoraxai_env
   
   # Activate environment
   # Windows
   thoraxai_env\Scripts\activate
   # macOS/Linux
   source thoraxai_env/bin/activate
   ```

2. **Dependency Installation**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Model Download**
   - Pre-trained model weights are included in the `model/` directory
   - Additional weights available in `Pretrained_model_weights/`

4. **Launch Application**
   ```bash
   streamlit run pneumonia_detector.py
   ```

---

## 📁 Project Structure

```
ThoraxAI/
├── 📄 pneumonia_detector.py      # Main Streamlit application
├── 📄 main.py                    # Entry point
├── 📄 requirements.txt           # Python dependencies
├── 📄 pyproject.toml            # Project configuration
├── 📁 model/                     # Trained model files
│   ├── pnemonia_classifier_v1.h5
│   └── pnemonia_classifier_v2.h5
├── 📁 Pretrained_model_weights/  # Pre-trained CNN weights
│   ├── densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
│   ├── inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
│   ├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
│   └── xception_weights_tf_dim_ordering_tf_kernels_notop.h5
├── 📁 assets/                    # Static assets
│   ├── loss_accuracy.png
│   ├── normal.jpeg
│   ├── pnemunia.jpeg
│   ├── snippet.txt
│   └── train_test_val.png
├── 📁 ipynb/                     # Jupyter notebooks
│   ├── Balancing_Pneumonia_Data.ipynb
│   └── Pneumonia_detection.ipynb
└── 📄 README.md                  # This file
```

---

## 🔧 Configuration

### Model Selection

The application uses `pnemonia_classifier_v2.h5` by default. To use a different model:

1. Place your model file in the `model/` directory
2. Update the model path in `pneumonia_detector.py`:
   ```python
   pneumonia_classifier_model = tf.keras.models.load_model('./model/your_model.h5')
   ```

### Customization

- **UI Theme**: Modify CSS in the `st.markdown()` section
- **Model Parameters**: Adjust image preprocessing settings
- **Confidence Thresholds**: Customize classification thresholds

---

## 🚀 Deployment

### Local Deployment

```bash
# Run with custom port
streamlit run pneumonia_detector.py --server.port 8502

# Run with custom host
streamlit run pneumonia_detector.py --server.address 0.0.0.0
```

### Cloud Deployment

1. **Streamlit Cloud**
   - Connect your GitHub repository
   - Deploy directly from the main branch
   - Automatic updates on code changes

2. **Docker Deployment**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "pneumonia_detector.py"]
   ```

---

## 📈 Future Enhancements

- **🔄 Multi-class Classification**: Detection of various lung conditions
- **📊 Batch Processing**: Multiple image analysis capabilities
- **🔍 Detailed Reports**: Comprehensive diagnostic reports with visualizations
- **🌐 API Integration**: RESTful API for third-party applications
- **📱 Mobile App**: Native mobile application development
- **☁️ Cloud Processing**: Scalable cloud-based analysis

---

## 🤝 Contributing

We welcome contributions to improve ThoraxAI! Here's how you can help:

1. **🐛 Bug Reports**: Report issues and bugs
2. **💡 Feature Requests**: Suggest new features
3. **🔧 Code Contributions**: Submit pull requests
4. **📚 Documentation**: Improve documentation and examples
5. **🧪 Testing**: Help test new features and improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**Medical Disclaimer**: This software is for research and educational purposes only. It is not intended to replace professional medical diagnosis, advice, or treatment. Always consult with qualified healthcare professionals for medical decisions.

**Accuracy Notice**: While the system demonstrates high accuracy in controlled testing, medical diagnosis should always be performed by licensed healthcare professionals.

---

## 📞 Support

- **📧 Website**: [Bright Solutions](https://brightssolution.com/)
- **🔴 Live Demo**: [ThoraxAI](https://thoraxai.streamlit.app/)
- **🐛 Issues**: [GitHub Issues](https://github.com/BrightsSolution/ThoraxAI/issues)

---

<div align="center">

**⚡ ThoraxAI - Advancing Medical AI Through Deep Learning ⚡**

*Built with ❤️ for the medical community*

[⬆️ Back to Top](#-thoraxai---pneumonia-detection-system)

</div>