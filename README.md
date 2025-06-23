# 🎯 Multi-Label Image Classification using CNN

This project demonstrates a practical implementation of **multi-label image classification** using **Convolutional Neural Networks (CNNs)** in TensorFlow and Keras. The model is trained to identify multiple genres (labels) for movie posters, using a custom dataset of poster images and genre annotations.

---

## 🧠 Project Benefits

- ✅ **Multi-label capability**: Unlike single-label classification, this model can assign **multiple genres** (e.g., Action, Comedy, Drama) to a single poster.
- 🎨 **Visual feature extraction**: Leverages deep convolutional layers to learn abstract image features.
- 📊 **Probabilistic outputs**: Predicts genre probabilities and ranks top genres using sigmoid activation.
- 🔄 **Real-world use case**: Demonstrates how deep learning can be applied to classify artwork, covers, or visual documents into multi-category tags.
- 💡 **Scalable**: Easily extensible to other domains such as fashion tags, product attributes, or artwork classification.

---

## 📂 Dataset

The model uses the [Movies Poster Dataset](https://github.com/laxmimerit/Movies-Poster_Dataset), which includes:
- 📷 Movie posters as image files
- 🏷 Genre labels in multi-hot encoding format in a CSV file

---

## 🧱 Architecture

The CNN model uses a sequential architecture with the following components:

- Multiple **Conv2D + MaxPool2D** blocks
- **Batch Normalization** to stabilize training
- **Dropout** layers for regularization
- **Dense layers** with sigmoid activation for multi-label output

| Layer Type     | Details                       |
|----------------|-------------------------------|
| Input          | 350x350 RGB images            |
| Conv2D         | ReLU, increasing filters      |
| MaxPooling     | 2x2 pooling                   |
| Dropout        | 30% to 50% across layers      |
| Output         | Dense layer with 25 sigmoid units |

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+
- TensorFlow 2.0 (GPU recommended)
- Keras, NumPy, pandas, matplotlib, tqdm, scikit-learn
- Google Colab (recommended for GPU support)

### 📦 Installation

```bash
pip install tensorflow-gpu==2.0.0-rc0
