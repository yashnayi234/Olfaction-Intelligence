# 🧬 OlfacNet Intelligence

### Giving Computers the Sense of Smell

### [Vercel App](olfaction-intelligence.vercel.app)

> An AI-powered olfactory platform that predicts odor profiles from molecular structures using deep learning — trained on 44,500+ molecules across 105 odor categories.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python scripts/train.py --epochs 20 --batch-size 64
```

### 3. Start the API Server
```bash
python scripts/api_server.py
```

### 4. Start the Web UI
```bash
cd web && npm install && npm run dev
```

Then visit **http://localhost:3000** for the web interface and **http://localhost:8000/docs** for the API documentation.

---

## 📂 Project Structure
```
Olfaction-Intelligence/
├── olfacnet/                      # Core Python package
│   ├── config.py                  # Centralized configuration & hyperparameters
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset & DataLoader
│   │   ├── preprocessing.py       # SMILES → Graph, Morgan Fingerprints
│   │   └── utils.py               # Label parsing, category mapping
│   ├── models/
│   │   ├── gnn.py                 # Graph Convolutional Network (GCN)
│   │   ├── fingerprint_net.py     # Morgan Fingerprint encoder (MLP)
│   │   ├── attention.py           # Multi-Head Attention fusion
│   │   └── olfacnet.py            # OlfacNet v2 (full model)
│   ├── training/
│   │   ├── losses.py              # Focal Loss
│   │   ├── metrics.py             # mAP, AUC-ROC, F1, Hamming Loss
│   │   └── trainer.py             # Training loop + early stopping
│   └── inference/
│       └── predictor.py           # SMILES → Odor prediction pipeline
├── scripts/
│   ├── train.py                   # CLI training entry point
│   ├── evaluate.py                # CLI evaluation entry point
│   └── api_server.py              # FastAPI REST server
├── web/                           # Next.js web application
│   └── src/
│       ├── app/
│       │   ├── page.js            # Landing page
│       │   ├── predict/page.js    # Odor prediction interface
│       │   ├── explore/page.js    # Dataset explorer
│       │   └── research/page.js   # Model architecture & metrics
│       └── components/
│           ├── Navbar.js
│           └── Footer.js
├── data/
│   ├── raw/                       # Original CSV datasets
│   ├── processed/                 # Labels, merged molecules
│   └── graphs/                    # Pre-computed molecular graphs (44K+)
├── experiments/
│   ├── checkpoints/               # Saved model weights
│   └── results/                   # Evaluation metrics
└── requirements.txt
```

---

## 🏗 Model Architecture — OlfacNet v2

```
                SMILES Input
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   Mol → Graph              Mol → Fingerprint
        │                         │
   GCN (3 layers)          Morgan FP (2048-bit)
   128-dim + BN + ReLU        MLP (512→256→128)
        │                         │
   Global Mean Pool               │
        │                         │
        └────────┬────────────────┘
                 │
        Multi-Head Attention
           Fusion (256, 4 heads)
                 │
           Classifier (256→128→105)
                 │
              Sigmoid
          (105 odor classes)
```

### Key Innovations
- **True Graph Convolution** — D⁻¹/²·A·D⁻¹/² normalized message passing (Kipf & Welling, 2017)
- **Morgan Fingerprints** — Industry-standard 2048-bit circular substructure encoding
- **Attention Fusion** — Multi-head self-attention dynamically weights graph vs. fingerprint features
- **Focal Loss** — Handles class imbalance across 105 odor categories (α=0.25, γ=2.0)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Per-Class Accuracy | **98.94%** |
| AUC-ROC (macro) | **0.805** |
| Hamming Loss | **0.0095** |
| Test Loss (Focal) | **0.004** |
| Parameters | **1.69M** |
| Training Samples | **44,543** |
| Odor Categories | **105** |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Predict odor profile from SMILES |
| GET | `/api/odors` | List all 105 odor categories |
| GET | `/api/stats` | Model statistics & metrics |
| GET | `/api/examples` | Example molecules for testing |

### Example API Call
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "top_k": 5}'
```

---

## 🛠 Technologies
- **PyTorch** — Deep learning framework
- **RDKit** — Chemical informatics & molecular processing
- **FastAPI** — REST API server
- **Next.js** — React web framework
- **scikit-learn** — Evaluation metrics

---

## 📈 Future Work
- Expand dataset with more molecular samples
- Implement Graph Attention Networks (GAT) for improved message passing
- Add ONNX export for faster inference
- Explore transformer-based architectures for sequence modeling
- Deploy with Docker containerization

---

## 📄 License
MIT License

## 🙏 Acknowledgments
- Inspired by [Osmo.ai](https://www.osmo.ai/) — "Giving computers the sense of smell"
- Molecular data from PubChem and odorant databases
