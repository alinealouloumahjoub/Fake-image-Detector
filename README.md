# Real vs AI-Generated Image Detector

A deep learning project that automatically detects whether an image is a **real photograph** or an **AI-generated fake**, using a dual-model architecture with intelligent routing.

---

## Authors

| Name | Role |
|---|---|
| **Aline Aloulou Mahjoub** | Model training, EDA, Streamlit app |
| **Rahma Essaiem** | Model training, EDA, Streamlit app |

---

## Project Structure

```
AI-Image-Detector/
│
├── 📂 notebooks/
│   ├── EDA_notebook1_140k_faces.ipynb       ← EDA for Dataset 1 (faces)
│   ├── EDA_notebook2_defactify.ipynb        ← EDA for Dataset 2 (scenes)
│   ├── notebook1_140k_faces_efficientnet.ipynb   ← Final model: faces
│   └── notebook2_defactify_efficientnet.ipynb    ← Final model: scenes
│
├── 📂 drafts/
│   ├── draft-1-resnet18.ipynb               ← Draft 1: ResNet-18 on 140K faces
│   └── draft-2-mobilenetv2.ipynb            ← Draft 2: MobileNetV2 on 140K faces
│
├── 📂 models/
│   ├── efficientnet_faces.pkl               ← Face specialist (Notebook 1)
│   ├── best_efficientnet_faces.pth                ← Face specialist weights
│   ├── efficientnet_general.pkl             ← Scene specialist (Notebook 2)
│   └── best_efficientnet_general.pth        ← Scene specialist weights
│
└── app.py                                   ← Streamlit web application
```

---

##  Problem Statement

With the rise of AI image generation tools like Stable Diffusion, Midjourney, and DALL-E, it has become increasingly easy to create fake images so realistic that the human eye can no longer tell the difference. This leads to:

- **Fake News** : fabricated events and scenes
- **DeepFakes** : AI-generated human faces
- **Election Manipulation** : staged political scenes
- **Reputation Damage** : false imagery of real people
- **Social Media Manipulation** : viral AI-generated content

> *Seeing is no longer believing.*

---

## Solution : Dual-Model Architecture

Instead of one general model, we built **two specialist models** and a routing system:

```
Upload Image
      ↓
OpenCV Face Detection
   ↙              ↘
Face found       No face found
   ↓                  ↓
🧑 Face Model    🌍 Scene Model
efficientnet     efficientnet
_model.pkl       _general.pkl
   ↘              ↙
    REAL or FAKE
```

---

##  Datasets

### Dataset 1 : 140K Real and Fake Faces *(Kaggle)*
> Used in: Draft 1, Draft 2, Notebook 1

| Property | Value |
|---|---|
| Total images | 140,000 |
| Resolution | 256×256 px |
| Real source | Flickr photographs |
| Fake source | StyleGAN |
| Balance |  50% REAL / 50% FAKE |

| Split | REAL | FAKE | Total |
|---|---|---|---|
| Train | 50,000 | 50,000 | 100,000 |
| Validation | 10,000 | 10,000 | 20,000 |
| Test | 10,000 | 10,000 | 20,000 |

---

### Dataset 2 : Defactify *(HuggingFace)*
> Used in: Notebook 2

| Property | Value |
|---|---|
| Total images | 96,000 |
| Real source | MS COCO (everyday scenes) |
| Fake source | SD 2.1, SDXL, SD3, DALL-E 3, Midjourney v6 |
| Balance | 83% FAKE / 17% REAL (imbalanced) |

| Split | REAL | FAKE | Total |
|---|---|---|---|
| Train | 7,000 | 35,000 | 42,000 |
| Validation | 1,500 | 7,500 | 9,000 |
| Test | 7,500 | 37,500 | 45,000 |

>  Class imbalance handled via `CrossEntropyLoss(weight=[1.0, 5.0])` it penalizes REAL mistakes 5× more.

---

## 🏗️ Models

We trained and compared 4 architectures across all notebooks:

| Model | Params | Dataset | Notebook |
|---|---|---|---|
| Simple CNN | ~0.5M | 140K Faces | Notebook 1 (baseline) |
| ResNet-18 | 11.2M | 140K Faces | Draft 1 |
| MobileNetV2 | 2.2M | 140K Faces | Draft 2 |
| **EfficientNet-B0** | **4.0M** | **Both** | **Notebooks 1 & 2 (final)** |

All pretrained models use **Transfer Learning**:
`ImageNet weights => Replace final layer (1000 => 2 classes) => Fine-tune all layers`

---

## Training Configuration

| Setting | Value |
|---|---|
| Loss Function | CrossEntropyLoss (weighted for Notebook 2) |
| Optimizer | Adam |
| Learning Rate | 1e-4 (selected via LR finder) |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Batch Size | 32 |
| Epochs | 10 |
| Input Size | 224×224 px |
| Dropout (Simple CNN) | 0.5 |

### Learning Rate Search Results
We tested 3 values for 3 epochs each:

| LR | Result |
|---|---|
| 1e-3 | Too fast ( unstable) |
| **1e-4** | **Best stable and accurate** |
| 1e-5 | Too slow ( underfitting) |

---

## Results

### Draft 1 : ResNet-18 (140K Faces)

| Metric | Value |
|---|---|
| Test Accuracy | **99.56%** |
| F1-Score | **0.9956** |
| Misclassified | 88 / 20,000 |
| Parameters | 11.2M |
| Best LR | 1e-5 |

---

### Draft 2 : MobileNetV2 (140K Faces)

| Metric | Value |
|---|---|
| Test Accuracy | **99.77%** |
| F1-Score | **0.9977** |
| Parameters | 2.2M |
| Best LR | 1e-4 |

---

### Notebook 1 : EfficientNet-B0 (140K Faces) <= Face Specialist

| Metric | Value |
|---|---|
| Test Accuracy | **99.88%** |
| F1-Score | **0.9988** |
| AUC-ROC | **0.9991** |
| Parameters | 4.0M |
| Best LR | 1e-4 |

---

### Notebook 2 : EfficientNet-B0 (Defactify) <= Scene Specialist

| Metric | Value |
|---|---|
| Test Accuracy | **92.43%** |
| F1-Score | **0.9291** |
| AUC-ROC | **0.9912** |
| Parameters | 4.0M |
| Best LR | 1e-4 |

---

### All Models Comparison

| Model | Dataset | Accuracy | F1 | AUC |
|---|---|---|---|---|
| ResNet-18 | 140K Faces | 99.56% | 0.9956 | — |
| MobileNetV2 | 140K Faces | 99.77% | 0.9977 | — |
| EfficientNet-B0 | 140K Faces |  99.88% | 99.88% | 99.91% |
| **EfficientNet-B0** | **Defactify** | **92.43%** | **92.91%** | **99.12%** |

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Image-Detector.git
cd AI-Image-Detector
```

### 2. Install dependencies
```bash
pip install streamlit torch torchvision timm opencv-python pillow
```

### 3. Place model files
Make sure these two files are in the same folder as `app.py`:
```
app.py
efficientnet_model.pkl       <= from models/
efficientnet_general.pkl     <=> from models/
```

### 4. Run the app
```bash
python -m streamlit run app.py --server.fileWatcherType none
```

### 5. Open in browser
```
http://localhost:8501
```

---

## Interpretability : Grad-CAM

We use **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize which regions of the image the model focused on when making its decision.

- **Red/warm areas** : most influential regions
-  **Blue/cold areas** : ignored regions

For fake faces: model focuses on **skin texture, eye artifacts, hair boundaries**.
For fake scenes: model focuses on **background textures, lighting inconsistencies, edge artifacts**.

---

## Limitations

| Generator | Face Model | Scene Model |
|---|---|---|
| StyleGAN faces | Detected | — |
| Stable Diffusion 2.1/XL/3 | Not trained | Detected |
| DALL-E 3 | Not trained |  Detected |
| Midjourney v6 |  Not trained |  Detected |
| Gemini / Adobe Firefly / Flux |  Not in training data |  Not in training data |

> The models can only detect generators they were trained on. Very recent or unseen generators may not be reliably detected.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Programming language |
| PyTorch | Deep learning framework |
| timm | EfficientNet-B0 architecture |
| torchvision | Image transforms |
| HuggingFace Datasets | Defactify dataset loading |
| OpenCV | Face detection routing |
| Streamlit | Web application |
| Grad-CAM | Model interpretability |
| Kaggle GPU (T4) | Training environment |

---

## References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [140K Real and Fake Faces — xhlulu](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [Defactify Image Dataset — HuggingFace](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
