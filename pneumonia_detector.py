import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Streamlit page configuration
st.set_page_config(
    page_title="Pneumonia Disease Detector",
    page_icon="🤒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Modern Cyberpunk Medical Dark Theme */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            background-attachment: fixed;
            color: #e0e6ed;
            font-family: 'Inter', sans-serif;
        }
        
        .main-container {
            background: linear-gradient(145deg, rgba(20, 20, 40, 0.95), rgba(30, 30, 60, 0.95));
            border-radius: 20px;
            padding: 40px;
            margin: 25px;
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.8),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            backdrop-filter: blur(15px);
            position: relative;
            overflow: hidden;
        }
        
        .main-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
            animation: borderGlow 3s linear infinite;
        }
        
        .main-title {
            font-family: 'Orbitron', monospace;
            font-size: 3.5em;
            font-weight: 900;
            color: #00ffff;
            text-align: center;
            margin-bottom: 40px;
            text-shadow: 
                0 0 10px #00ffff,
                0 0 20px #00ffff,
                0 0 30px #00ffff;
            animation: titlePulse 2s ease-in-out infinite;
            letter-spacing: 2px;
        }
        
        .section-title {
            font-family: 'Orbitron', monospace;
            font-size: 2.4em;
            font-weight: 700;
            color: #ff6b6b;
            margin: 45px 0 25px;
            text-shadow: 0 0 15px rgba(255, 107, 107, 0.8);
            border-left: 4px solid #ff6b6b;
            padding-left: 20px;
            animation: slideInFromLeft 0.8s ease-out;
            position: relative;
        }
        
        .section-title::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 50px;
            height: 2px;
            background: linear-gradient(90deg, #ff6b6b, transparent);
        }
        
        .content {
            font-size: 1.2em;
            color: #b8c5d1;
            line-height: 2.0;
            text-align: justify;
            font-weight: 300;
        }
        
        .highlight {
            color: #00ff88;
            font-weight: 600;
            text-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
        }
        
        .separator {
            height: 3px;
            background: linear-gradient(90deg, transparent, #00ffff, #ff00ff, #00ffff, transparent);
            margin: 30px 0;
            border-radius: 2px;
            animation: separatorFlow 4s ease-in-out infinite;
        }
        
        .prediction-text {
            font-family: 'Orbitron', monospace;
            font-size: 2.2em;
            font-weight: 700;
            text-align: center;
            text-shadow: 0 0 15px currentColor;
            animation: predictionGlow 1.5s ease-in-out infinite alternate;
        }
        
        .prediction-text.healthy {
            color: #00ff88;
        }
        
        .prediction-text.pneumonia {
            color: #ff4757;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border-radius: 15px;
            padding: 16px 35px;
            font-weight: 600;
            font-size: 1.1em;
            border: none;
            box-shadow: 
                0 8px 20px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            font-family: 'Inter', sans-serif;
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .stButton>button:hover {
            background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
            box-shadow: 
                0 12px 30px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transform: translateY(-3px) scale(1.05);
        }
        
        .stButton>button:hover::before {
            left: 100%;
        }
        
        .stFileUploader, .stImage {
            background: linear-gradient(145deg, rgba(20, 20, 40, 0.9), rgba(30, 30, 60, 0.9));
            border-radius: 15px;
            padding: 20px;
            border: 2px solid rgba(0, 255, 255, 0.3);
            box-shadow: 
                0 8px 25px rgba(0, 0, 0, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
        }
        
        .stFileUploader:hover, .stImage:hover {
            border-color: #00ffff;
            box-shadow: 
                0 12px 35px rgba(0, 255, 255, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            transform: translateY(-5px) scale(1.02);
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Inter', sans-serif;
            font-size: 1.3em;
            font-weight: 500;
            color: #b8c5d1;
            padding: 18px 35px;
            border-radius: 15px 15px 0 0;
            background: linear-gradient(145deg, rgba(20, 20, 40, 0.8), rgba(30, 30, 60, 0.8));
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(0, 255, 255, 0.2);
            margin-right: 5px;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(145deg, #667eea, #764ba2);
            color: #ffffff;
            font-weight: 600;
            box-shadow: 
                0 8px 25px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            border-color: #00ffff;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(145deg, #764ba2, #667eea);
            color: #ffffff;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }
        
        .footer {
            font-size: 1.1em;
            color: #b8c5d1;
            margin-top: 60px;
            text-align: center;
            padding: 40px;
            background: linear-gradient(145deg, rgba(20, 20, 40, 0.95), rgba(30, 30, 60, 0.95));
            border-radius: 20px;
            box-shadow: 
                0 15px 35px rgba(0, 0, 0, 0.7),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            backdrop-filter: blur(15px);
            position: relative;
        }
        
        .footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
            animation: borderGlow 3s linear infinite;
        }
        
        .footer a {
            color: #00ffff;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
            text-shadow: 0 0 8px rgba(0, 255, 255, 0.6);
        }
        
        .footer a:hover {
            color: #ff00ff;
            text-shadow: 0 0 12px rgba(255, 0, 255, 0.8);
            text-decoration: underline;
        }
        
        .stSidebar {
            background: linear-gradient(145deg, rgba(20, 20, 40, 0.95), rgba(30, 30, 60, 0.95));
            color: #b8c5d1;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(0, 255, 255, 0.3);
        }
        
        .stSidebar h3 {
            color: #00ffff;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.6);
            font-family: 'Orbitron', monospace;
        }
        
        .stSidebar p {
            color: #b8c5d1;
        }
        
        .content ul li::marker {
            color: #ff6b6b;
        }
        
        /* Advanced Animations */
        @keyframes titlePulse {
            0%, 100% { 
                text-shadow: 
                    0 0 10px #00ffff,
                    0 0 20px #00ffff,
                    0 0 30px #00ffff;
                transform: scale(1);
            }
            50% { 
                text-shadow: 
                    0 0 15px #00ffff,
                    0 0 25px #00ffff,
                    0 0 35px #00ffff;
                transform: scale(1.02);
            }
        }
        
        @keyframes slideInFromLeft {
            from { 
                transform: translateX(-50px); 
                opacity: 0; 
            }
            to { 
                transform: translateX(0); 
                opacity: 1; 
            }
        }
        
        @keyframes borderGlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes separatorFlow {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        @keyframes predictionGlow {
            from { text-shadow: 0 0 10px currentColor; }
            to { text-shadow: 0 0 20px currentColor, 0 0 30px currentColor; }
        }
        
        @keyframes scaleIn {
            from { 
                transform: scale(0.9); 
                opacity: 0; 
            }
            to { 
                transform: scale(1); 
                opacity: 1; 
            }
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(20, 20, 40, 0.8);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #00ffff, #ff00ff);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, #ff00ff, #00ffff);
        }
    </style>
""", unsafe_allow_html=True)

# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="main-title">⚡ THORAX AI - PNEUMONIA DETECTION SYSTEM ⚡</div>', unsafe_allow_html=True)

# Tab layout
tab1, tab2 = st.tabs(["⚡ CONTROL PANEL", "🔬 DIAGNOSTIC SCANNER"])

# First Tab: Home
with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ SYSTEM OPERATOR</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <span class="highlight">ThoraxAI</span> is an advanced medical diagnostic system powered by cutting-edge 
            <span class="highlight">🤖 Artificial Intelligence</span> and 🧠 Deep Learning technologies. 
            Built with state-of-the-art neural networks, this system provides healthcare professionals with 
            accurate, real-time pneumonia detection capabilities for improved patient care and diagnostic efficiency. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 MISSION BRIEFING</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <span class="highlight">ThoraxAI System Capabilities:</span>
            <ul>
                <li><span class="highlight">🔬 Medical Diagnosis:</span> 
                    Pneumonia is a life-threatening infectious disease affecting one or both lungs, commonly caused by bacteria called Streptococcus pneumoniae. 
                    According to the World Health Organization (WHO), one in three deaths worldwide is caused by pneumonia. 
                    Chest X-rays used to diagnose pneumonia require expert radiologists for evaluation. 
                    ThoraxAI provides an automatic system for detecting pneumonia, beneficial for treating the 
                    disease without delay, particularly in remote areas. Our advanced deep learning algorithms 
                    analyze medical images using Convolutional Neural Networks (CNNs) for accurate disease 
                    classification and detection.<br/>
                </li>
                <li><span class="highlight">🧠 AI-Powered Detection:</span> 
                    Built with state-of-the-art disease detection models using pre-trained Convolutional Neural Networks (CNNs). 
                    The system was trained on a comprehensive 
                    <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" target="_blank" style="color: #93c5fd; font-weight: bold;">chest X-ray dataset</a>
                    and is deployed for real-time medical predictions. 📡<br/>
                    <ul>
                        <li><span class="highlight">🔬 Advanced Processing Pipeline:</span>
                            <ul>
                                <li>High-resolution medical image analysis</li>
                                <li>Automated quality assessment and validation</li>
                                <li>Real-time diagnostic processing capabilities</li>
                            </ul>
                        </li>
                        <li><span class="highlight">⚙️ Data Preprocessing and Augmentation:</span>
                            <ul>
                                <li><span class="highlight">Image Enhancement:</span>
                                    <ul>
                                        <li>Advanced noise reduction and artifact removal</li>
                                        <li>Medical image quality optimization</li>
                                    </ul>
                                </li>
                                <li><span class="highlight">Standardization:</span>
                                    <ul>
                                        <li>Consistent image sizing for optimal processing</li>
                                        <li>Medical imaging protocol compliance</li>
                                    </ul>
                                </li>
                                <li><span class="highlight">Normalization:</span>
                                    <ul>
                                        <li>Pixel value optimization for neural network processing</li>
                                        <li>Medical imaging standard compliance</li>
                                    </ul>
                                </li>
                                <li><span class="highlight">Data Augmentation:</span>
                                    <ul>
                                        <li>Advanced training sample generation:</li>
                                        <ul>
                                            <li>Medical image rotation 🔄</li>
                                            <li>Intelligent image flipping 🔁</li>
                                            <li>Contrast and brightness adjustment</li>
                                        </ul>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ SYSTEM SPECIFICATIONS</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Core Technologies:</span> Python, NumPy, Pandas, Matplotlib, TensorFlow, Keras, and Scikit-Learn for advanced data processing and machine learning.</li>
                <li><span class="highlight">⚙️ AI Methodologies:</span> Pre-trained CNNs, Data Augmentation, Transfer Learning, and Advanced Image Preprocessing Techniques for optimal medical image analysis.</li>
                <li><span class="highlight">🌐 Deployment Platform:</span> Streamlit-powered web interface providing interactive, professional medical diagnostic capabilities.</li>
                <li><span class="highlight">🔬 Medical Standards:</span> Compliance with medical imaging protocols and healthcare data processing standards.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 NEURAL DIAGNOSTIC SCANNER</div>', unsafe_allow_html=True)
    st.markdown('''
        <div class="content">
            <span class="highlight">INITIALIZING NEURAL SCAN PROTOCOL...</span><br/><br/>
            Upload a thoracic X-ray image for AI-powered diagnostic analysis. The system will perform deep learning analysis and provide real-time classification:
            <ul>
                <li><span class="highlight">STATUS: NORMAL</span> - No anomalies detected ✅</li>
                <li><span class="highlight">STATUS: PNEUMONIA DETECTED</span> - Pathological indicators found ⚠️</li>
            </ul>
        </div><br/>
    ''', unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns([1, 2])  # 1: Image section, 2: Prediction section

    with col1:
        uploaded_file = st.file_uploader("🔬 UPLOAD THORACIC SCAN:", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            image = image.convert("RGB")  # Ensure the image has 3 channels (RGB)
            st.image(image, caption="SCAN PREVIEW", width=250)

    with col2:
        if uploaded_file is not None:
            # Preprocess the image
            image = image.resize((150, 150))  # Resize as per model input
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Load the model
            pneumonia_classifier_model = tf.keras.models.load_model('./model/pnemonia_classifier_v2.h5')
            class_names = ["Normal", "Pneumonia"]

            # Make predictions
            predictions = pneumonia_classifier_model.predict(image_batch)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            # Display the prediction result
            status_class = "healthy" if predicted_class.lower() == "normal" else "pneumonia"
            if predicted_class.lower() == "normal":
                status_message = f"SCAN COMPLETE: NORMAL THORACIC STRUCTURE DETECTED ✅<br/>CONFIDENCE: {confidence:.2%}"
            else:
                status_message = f"ALERT: PNEUMONIA INDICATORS DETECTED ⚠️<br/>CONFIDENCE: {confidence:.2%}"

            st.markdown(f'''
                <br/><br/><br/><br/><br/><br/><br/><br/><br/>
                <div class="prediction-text {status_class}">
                    {status_message}
                </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)