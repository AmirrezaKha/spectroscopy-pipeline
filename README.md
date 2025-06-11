# FTIR Spectroscopy Pipeline

This project performs preprocessing, anomaly detection, and concentration estimation on FTIR spectroscopy data using Python.

## 📁 Project Structure

```
Spectroscopy-Pipeline/
│
├── data/
│   ├── components.npy
│   └── mixture_spectrum.npy
│
├── preprocessing.py
├── anomaly_detection.py
├── model.py
├── main.py
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git remote add origin https://github.com/AmirrezaKha/spectroscopy-pipeline.git
cd spectroscopy-pipeline
```

### 2. Install Dependencies

```bash
pip install numpy scipy matplotlib scikit-learn torch
```

### 3. Run the Main Pipeline

```bash
python main.py
```

## 🧪 Features

- **Preprocessing**: Savitzky-Golay filtering, SNV, and baseline correction
- **Anomaly Detection**:
  - Isolation Forest
  - Mahalanobis Distance
  - Z-score Method
- **Modeling**: Optional Autoencoder-based training for noise reduction
- **Estimation**: Least Squares solution for component concentration

## 📊 Output

- Anomalies printed in console
- Estimated concentrations
- Plots of mixture vs. components

## 🧠 Author

Developed by [Your Name]

## 📄 License

MIT License