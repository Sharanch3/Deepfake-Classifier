# 🎭 Deepfake Image Classifier

## Problem Statement

The rise of AI image generation tools has made it trivially easy to produce photorealistic fake images, enabling misinformation, identity fraud, and media manipulation at scale. Manually identifying these deepfakes is unreliable and does not scale.

This project builds a binary image classifier to automatically distinguish **real photographs** from **AI-generated images**. Using transfer learning on a pre-trained ResNet-50 backbone, the model learns subtle deepfake artifacts without requiring large datasets or extensive compute — and is deployed as a simple web app so anyone can verify an image instantly.

---

## 🌐 Live Demo
**Link:** https://deepfake-classifier.streamlit.app/

---

## How It Works

The classifier uses a **ResNet-50** backbone pre-trained on ImageNet, with the final fully connected layer replaced by a binary classification head. Only the last ResNet block (`layer4`) and the new FC head are fine-tuned, keeping the rest of the network frozen. The model outputs a sigmoid probability — images scoring above 0.5 are classified as **REAL**, and below as **FAKE (AI-generated)**.

---

## Project Structure

```
Deepfake-Classifier/
├── artifacts/
│   └── model.pth          # Trained model weights
├── notebook/              # Training & experimentation notebooks
│   ├── v1_baseline_cnn.ipynb
│   ├── v2_cnn_with_regularization.ipynb
│   └── v3_resnet_transfer_learning.ipynb
├── main.py                # Streamlit app entry point
├── utils.py               # Model definition and prediction logic
├── requirements.txt       # Python dependencies
├── pyproject.toml
└── .python-version
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Sharanch3/Deepfake-Classifier.git
cd Deepfake-Classifier

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run main.py
```

Then open your browser to `http://localhost:8501`.

---

## Usage

1. Launch the app with the command above.
2. Upload a `.jpg`, `.jpeg`, or `.png` image using the file uploader.
3. The model analyzes the image and returns one of two results:
   - ✅ **Real Image** — the image appears to be an authentic photograph.
   - 🎭 **AI Generated** — the image appears to be synthetically generated.

> **Note:** Results are probabilistic and may not be 100% accurate on all image types.

---

## Model Development

The model went through three iterations, documented in the `notebook/` folder.

### V1 — Baseline Custom CNN

A simple 3-block convolutional network trained from scratch.

- **Architecture:** 3× Conv2d → ReLU → MaxPool, followed by two fully connected layers
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam (lr=0.001)
- **Training time:** ~2h 35m (9,331s) on 150,000 images over 5 epochs
- **Issue:** No regularization — prone to overfitting. Validation accuracy used `torch.max` on raw logits, which is incorrect for binary classification with a single output neuron.

### V2 — Custom CNN with Regularization

Same architecture as V1 but hardened against overfitting.

- **Added:** `BatchNorm2d` / `BatchNorm1d` after every layer, `Dropout2d` (p=0.2) in conv blocks, `Dropout1d` (p=0.4) before the final FC layer
- **Optimizer:** Adam with L2 weight decay (`weight_decay=1e-4`)
- **Training time:** ~2h 5m (7,519s) — faster than V1 due to better gradient flow from BatchNorm
- **Fix:** Validation now uses `torch.sigmoid` + 0.5 threshold — the correct approach for a single-neuron binary output

### V3 — Transfer Learning with ResNet-50 ✅ *(Final Model)*

Replaced the custom CNN with a pre-trained ResNet-50 backbone.

- **Architecture:** ResNet-50 (ImageNet weights), all layers frozen except `layer4` and a new FC head with Dropout (p=0.3)
- **Optimizer:** Adam applied only to trainable parameters (`filter(lambda x: x.requires_grad, ...)`)
- **Training time:** ~2h 45m (9,887s) — longer per epoch due to deeper architecture, but stronger accuracy
- **Evaluation:** Classification report + confusion matrix (seaborn heatmap) on the validation set
- **Output:** Weights saved to `artifacts/model.pth`

This approach converges faster and generalizes better than the custom CNN by leveraging rich ImageNet features.

---

## Data Pipeline

All three notebooks share the same data loading and preprocessing setup.

The dataset contains **200,000 images** across 2 classes (`ai` and `real`), organized using `ImageFolder` and split 75/25 into train (150,000) and validation (50,000) sets. Images are loaded in batches of 32.

**Data augmentation** (applied during training):
- Random rotation (±10°)
- Random horizontal flip
- Color jitter (brightness & contrast ±0.2)

**Preprocessing** (applied to all splits):
- Resize to 224 × 224
- ToTensor + ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

```
data/
├── ai/
│   └── *.jpg / *.png
└── real/
    └── *.jpg / *.png
```

> The `data/` directory is not included in this repository. Provide your own dataset in the structure above before training.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `torchvision` | ResNet-50 model and image transforms |
| `streamlit` | Web application UI |
| `Pillow` | Image loading and preprocessing |
| `scikit-learn` | Classification report & confusion matrix |
| `seaborn` | Confusion matrix heatmap |
| `matplotlib` | Plotting |

---

## Final Model Details

| Property | Value |
|---|---|
| Architecture | ResNet-50 (transfer learning) |
| Task | Binary classification (Real vs. Fake) |
| Input size | 224 × 224 RGB |
| Output | Sigmoid probability (threshold: 0.5) |
| Frozen layers | `layer1`, `layer2`, `layer3` |
| Trainable layers | `layer4` + FC head |
| Dropout | 0.3 (FC head) |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | Adam (lr=0.001) |
| Normalization | ImageNet mean/std |
| Weights | `artifacts/model.pth` |