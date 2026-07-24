
# UNet Architecture in PyTorch

A clean, modular implementation of the **U-Net** convolutional network architecture for semantic image segmentation, built from scratch using PyTorch.

## 🧠 Model Architecture
The UNet architecture follows an encoder-decoder structure with skip connections to retain fine spatial detail:

* **Encoder (Contracting Path):** Consecutive $3 \times 3$ convolutions followed by ReLU activations and $2 \times 2$ max pooling for deep feature extraction.
* **Bottleneck:** Captures high-level latent representations at the lowest spatial resolution.
* **Decoder (Expanding Path):** Transposed $2 \times 2$ convolutions (upsampling) combined with skip connections from the encoder to recover spatial precision.
* **Output Layer:** Final $1 \times 1$ convolution mapping feature maps to class probability channels.

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Libraries:** Torchvision, OS, Time, TQDM, NumPy

## 🚀 Quick Start
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)

# Run a test forward pass
unet_model.py
