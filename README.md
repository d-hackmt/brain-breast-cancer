# 🧠 OncoScan AI: Brain & Breast Tumor Detection

OncoScan AI is a full-stack medical imaging application designed for detecting tumors in MRI and Breast Scan images. It leverages deep learning (CNNs) to provide real-time predictions and GradCAM (Gradient-weighted Class Activation Mapping) visualizations to highlight areas of medical interest.

## 🏗️ Project Architecture

The project is structured as a modern multi-tier application:

- **Frontend (React)**: A premium, dark-themed UI built with Vite for high performance.
- **Backend (FastAPI)**: Asynchronous Python REST API handling model inference and visualization.
- **Analytics Dashboard (Streamlit)**: Specialized portal for training metrics and TensorBoard integration.
- **AI Core (TensorFlow)**: Custom CNN implementations (including unique `TrigConv2D` layers) with weights hosted on Hugging Face.

## 📂 Folder Structure

```
├── backend/            # FastAPI REST API
├── frontend/           # React Web Application
├── dashboard/          # Streamlit Analytics Dashboard
├── models/             # Shared Model Loaders & Custom Layers
├── utils/              # Shared Utility functions (Predict, GradCAM, TB)
├── config.py           # Global Configuration
└── requirements.txt    # Backend & Dashboard Dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Brain-Breast-Cancer-Detection-final
   ```

2. **Setup Backend & Dashboard**:
   ```bash
   python -m venv bbenv
   source bbenv/bin/activate  # Windows: bbenv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup Frontend**:
   ```bash
   cd frontend
   npm install
   ```

## 🛠️ Running the Application

To run the full suite, you need three separate terminals:

1. **Backend (Port 8000)**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

2. **Frontend (Port 5173)**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Dashboard (Port 8501)**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 🔬 Key Features

- **Multi-Class Detection**: Classifies scans into categories like No Tumor, Malignant, Benign, etc.
- **GradCAM Visualization**: Generates heatmaps to explain AI decision-making (Explainable AI).
- **Interactive Analytics**: Integrated TensorBoard for monitoring model training performance.
- **Cloud Model Hosting**: Seamless model weight management via Hugging Face Hub.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
