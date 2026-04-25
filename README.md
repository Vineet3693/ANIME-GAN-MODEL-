# 🎌 Anime Face GAN (DCGAN) — PyTorch

> A Deep Convolutional Generative Adversarial Network (DCGAN) trained on the Anime Face Dataset to generate realistic anime faces from random noise.

---

## 👤 Author
**Vineet Yadav**

---

## 📌 Project Overview

This project implements a **DCGAN (Deep Convolutional GAN)** from scratch using PyTorch. The model is trained on 63,565 anime face images and can generate new, diverse anime faces from random noise vectors.

The project is structured as a series of notebooks (all in one Kaggle notebook), covering:
- Data preprocessing
- Model architecture
- Training with checkpointing
- Visualization
- Evaluation
- Streamlit deployment

---

## 📁 Project Structure


text
your_project/
  app.py                                  # Streamlit app
  README.md                               # This file
  models/
    generator_epoch_50_scripted.pt        # Scripted Generator (for inference)
    discriminator_epoch_50_scripted.pt    # Scripted Discriminator (for scoring)
  checkpoints/
    training_state.pth                    # Latest training state (resume)
    generator_epoch_*.pth                 # Per-epoch generator weights
    discriminator_epoch_*.pth             # Per-epoch discriminator weights
    optimG_epoch_*.pth                    # Per-epoch generator optimizer state
    optimD_epoch_*.pth                    # Per-epoch discriminator optimizer state
  generated_images/
    epoch_001_grid.png                    # Generated samples at epoch 1
    epoch_010_grid.png                    # Generated samples at epoch 10
    ...
    epoch_050_grid.png                    # Generated samples at epoch 50
  final_samples/
    final_batch_01_grid.png               # Final evaluation batch 1
    final_batch_02_grid.png               # Final evaluation batch 2
    final_batch_03_grid.png               # Final evaluation batch 3
    final_batch_04_grid.png               # Final evaluation batch 4
    final_all_batches_grid.png            # Combined final grid
  plots/
    preprocessed_samples.png             # Preprocessed dataset samples
    augmentation_comparison.png          # Augmentation comparison
  training_progress.gif                  # GIF of training progress epoch 1→50
  loss_history.csv                       # Loss history for all 50 epochs
  dataset_info.json                      # Dataset metadata
  generated_faces_eval_epoch50.png       # Final evaluation grid (epoch 50)

---

## 📊 Dataset

- **Name:** Anime Face Dataset
- **Source:** [Kaggle — splcher/animefacedataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
- **Total Images:** 63,565
- **Valid Images:** 63,565
- **Corrupt Images:** 0
- **Image Size (after preprocessing):** 64 × 64 × 3 (RGB)
- **Normalization:** mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] → pixel range [-1, 1]

---

## 🧠 Model Architecture

### Generator
| Layer | Type | Input Shape | Output Shape |
|-------|------|-------------|--------------|
| 1 | ConvTranspose2d + BatchNorm + ReLU | (N, 100, 1, 1) | (N, 512, 4, 4) |
| 2 | ConvTranspose2d + BatchNorm + ReLU | (N, 512, 4, 4) | (N, 256, 8, 8) |
| 3 | ConvTranspose2d + BatchNorm + ReLU | (N, 256, 8, 8) | (N, 128, 16, 16) |
| 4 | ConvTranspose2d + BatchNorm + ReLU | (N, 128, 16, 16) | (N, 64, 32, 32) |
| 5 | ConvTranspose2d + Tanh | (N, 64, 32, 32) | (N, 3, 64, 64) |

- **Trainable Parameters:** 3,576,704
- **Input:** random noise vector z ∈ ℝ^100
- **Output:** RGB image (3 × 64 × 64) in [-1, 1]

### Discriminator
| Layer | Type | Input Shape | Output Shape |
|-------|------|-------------|--------------|
| 1 | Conv2d + LeakyReLU | (N, 3, 64, 64) | (N, 64, 32, 32) |
| 2 | Conv2d + BatchNorm + LeakyReLU | (N, 64, 32, 32) | (N, 128, 16, 16) |
| 3 | Conv2d + BatchNorm + LeakyReLU | (N, 128, 16, 16) | (N, 256, 8, 8) |
| 4 | Conv2d + BatchNorm + LeakyReLU | (N, 256, 8, 8) | (N, 512, 4, 4) |
| 5 | Conv2d + Sigmoid | (N, 512, 4, 4) | (N, 1) |

- **Trainable Parameters:** 2,765,568
- **Input:** RGB image (3 × 64 × 64)
- **Output:** probability in [0, 1] (real/fake score)

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 128 |
| Learning Rate | 0.0002 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Loss Function | BCELoss (Binary Cross Entropy) |
| Latent Dim | 100 |
| Image Size | 64 × 64 |
| GPU | NVIDIA T4 (Kaggle) |
| Framework | PyTorch |

---

## 📉 Training Loss

| Epoch | G Loss | D Loss |
|-------|--------|--------|
| 1 | 7.8856 | 0.7174 |
| 10 | 4.4883 | 0.5087 |
| 20 | 3.3791 | 0.5073 |
| 30 | 3.5455 | 0.4494 |
| 40 | 3.8183 | 0.3286 |
| 50 | 4.1525 | 0.3114 |

- Generator loss starts high and stabilizes around 3.5–4.2 (expected in GANs).
- Discriminator loss decreases over time, showing it gets better at distinguishing real/fake.

---

## ✅ Results

- **Face quality:** Clear and realistic anime faces ✅
- **Diversity:** Good variety of hair colors, styles, and expressions ✅
- **Distortions:** Few minor distortions (expected in DCGAN at 64×64) ✅
- **Mode collapse:** None observed ✅

---

## 🚀 How to Run (Streamlit App)

### 1) Clone the repository

bash
git clone https://github.com/your-username/anime-face-gan.git
cd anime-face-gan

### 2) Install dependencies

bash
pip install streamlit torch torchvision pillow numpy

### 3) Place the model files
Make sure these files are in the `models/` folder:

text
models/
  generator_epoch_50_scripted.pt
  discriminator_epoch_50_scripted.pt

### 4) Run the app

bash
streamlit run app.py

### 5) Open in browser

text
http://localhost:8501


---

## 🎮 Streamlit App Features

| Feature | Description |
|---------|-------------|
| 🎲 Random seed | Control randomness for reproducibility |
| 🖼️ Number of faces | Generate 4 to 64 faces at once |
| 📐 Grid columns | Adjust display layout |
| 📊 Discriminator score | See how "real" each face looks (0=fake, 1=real) |
| ✅ Generate button | Click to generate new anime faces instantly |

---

## 📦 Requirements


text
torch=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
Pillow>=9.0.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
tqdm>=4.64.0
imageio>=2.22.0



---

## 🔁 Resume Training

If training is interrupted, resume using the checkpoint:


python
import torch
state = torch.load("/kaggle/working/checkpoints/training_state.pth")
print("Last completed epoch:", state["epoch"])
# Set EPOCHS higher and re-run training cell

---

## 🔮 Future Improvements

- [ ] Train for more epochs (100–200) for sharper faces
- [ ] Use label smoothing (real=0.9, fake=0.0) for better stability
- [ ] Upgrade to StyleGAN2 for higher resolution (128×128 or 256×256)
- [ ] Add FID (Fréchet Inception Distance) metric for quantitative evaluation
- [ ] Add interpolation between two latent vectors in Streamlit
- [ ] Deploy to Hugging Face Spaces

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- Dataset: [splcher/animefacedataset](https://www.kaggle.com/datasets/splcher/animefacedataset) on Kaggle
- DCGAN Paper: [Radford et al., 2015](https://arxiv.org/abs/1511.06434)
- PyTorch DCGAN Tutorial: [pytorch.org](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

## 📬 Contact
https://github.com/Vineet3693/


