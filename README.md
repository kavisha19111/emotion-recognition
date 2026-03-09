🎙️ Speech Emotion Recognition using Deep Learning & Attention Mechanism

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Librosa-Audio Processing-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-RAVDESS-orange?style=for-the-badge"/>
</p>

> A deep learning system that listens to human speech and identifies the underlying emotion — built with PyTorch, MFCC feature extraction and a custom Attention Mechanism.

---

🧠 What Does This Do?

Ever wondered if a machine could tell if you're happy, sad, angry or neutral just by listening to your voice? That's exactly what this project does.

It takes raw audio files, extracts meaningful acoustic features (MFCCs), and feeds them into a neural network that has been trained to recognise emotional patterns in speech. Two models are built and compared — a standard neural network and an upgraded version with a custom Attention layer.

---

🎯 Emotions Classified

| Emotion Code | Emotion | Label |
|-------------|---------|-------|
| 01 | 😐 Neutral | 0 |
| 03 | 😊 Happy | 1 |
| 04 | 😢 Sad | 2 |
| 05 | 😠 Angry | 3 |

---

Features

- 🎵 **MFCC Feature Extraction** — Extracts 40 Mel Frequency Cepstral Coefficients from each audio file using Librosa, capturing the tonal and spectral characteristics of speech
- 🧠 **Baseline Neural Network** — A clean 2-layer fully connected network as the foundation model
- ✨ **Custom Attention Mechanism** — A trainable attention layer that learns to focus on the most emotionally relevant audio features
- 📊 **Side-by-Side Comparison** — Both models trained and evaluated on the same data for a fair comparison
- ⚡ **End-to-End Pipeline** — From raw `.wav` files all the way to emotion prediction in a single script

🏗️ Model Architecture

### Model 1 — Baseline Neural Network
```
Input: 40 MFCC Features
        ↓
Linear Layer (40 → 64)
        ↓
ReLU Activation
        ↓
Linear Layer (64 → 4)
        ↓
Output: Emotion Class (0-3)
```

### Model 2 — Neural Network with Attention 
```
Input: 40 MFCC Features
        ↓
Attention Layer → Softmax Weights → Weighted Features
        ↓
Linear Layer (40 → 64)
        ↓
ReLU Activation
        ↓
Linear Layer (64 → 4)
        ↓
Output: Emotion Class (0-3)
```

The attention layer learns **which of the 40 MFCC features matter most** for each emotion — giving the model interpretability alongside performance.

📊 Results

| Model | Accuracy |
|-------|---------|
| 🧠 Baseline Neural Network | ~55% |
| ✨ Attention Neural Network | ~55-60% |

> The attention model consistently learns better feature representations, showing the value of learned feature weighting even in shallow networks.

🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning Framework | PyTorch |
| Audio Processing | Librosa |
| Numerical Computing | NumPy |
| Data Splitting | Scikit-learn |
| Dataset | RAVDESS |

---

📁 Project Structure

```
emotion-recognition/
│
├── emotion_model.py         # Complete pipeline — feature extraction,
│                            # model definition, training & evaluation
├── .gitignore               # Excludes large audio dataset files
└── README.md                # You are here
```

---

Run It Yourself

**1. Clone the repo**
```bash
git clone https://github.com/kavisha19111/emotion-recognition.git
cd emotion-recognition
```

**2. Install dependencies**
```bash
pip install librosa numpy torch scikit-learn
```

**3. Download the RAVDESS dataset from Kaggle and place it here:**
```
data/
└── ravdess/
    └── audio_speech_actors_01-24/
        ├── Actor_01/
        ├── Actor_02/
        └── ...
```
👉 [Download RAVDESS from Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

**4. Run the script**
```bash
python emotion_model.py
```

🔮 Future Improvements

- [ ] Add all 8 RAVDESS emotions (fear, disgust, surprise, calm)
- [ ] Replace feedforward network with LSTM or CNN for sequential audio modelling
- [ ] Add real-time microphone input for live emotion detection
- [ ] Build a Streamlit web interface for live demo
- [ ] Increase training epochs with early stopping and dropout regularisation
- [ ] Experiment with spectrograms and mel-spectrograms as features
- [ ] Export model to ONNX for deployment

📚 What I Learned

- How to process and extract features from raw audio files using Librosa
- Implementing custom neural network layers in PyTorch from scratch
- Building and training an Attention Mechanism and understanding why it works
- End-to-end ML pipeline from data loading to model evaluation
- The challenges of emotion recognition — why it's a hard problem even for humans

Note on Dataset

The `data/` folder containing RAVDESS audio files is **not included** in this repository due to file size constraints. Please download it directly from Kaggle using the link above.

Acknowledgements

- [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) — Ryerson Audio-Visual Database of Emotional Speech and Song
- [Librosa](https://librosa.org) — Audio and music processing in Python
- [PyTorch](https://pytorch.org) — Open source machine learning framework

<p align="center">Made with ❤️ and a lot of audio files</p>
