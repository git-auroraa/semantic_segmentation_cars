# 🚗 Car Image Segmentation using U-Net

This project implements semantic segmentation on car images using a U-Net architecture in PyTorch. It focuses on extracting cars from background images in the Carvana dataset. The trained model is further integrated into a Streamlit dashboard to enable car background removal and car detail enrichment using LLaMA 3.

---

## 📌 Features

- Semantic segmentation using U-Net
- Car background removal using binary masks
- Dashboard built with Streamlit
- Integration with LLaMA 3 to fetch car details (year, make, model)
- Deployment on Hugging Face Spaces

---

## 🧠 Model Architecture

- **Base Architecture:** U-Net
- **Input Size:** 128×128 RGB images
- **Loss Function:** Binary Cross-Entropy
- **Metrics:** Pixel Accuracy, IoU, Dice Score
- **Framework:** PyTorch

---

## 🗂️ Dataset

- **Name:** [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)
- **Format:**
  - Images: `.jpg` (RGB)
  - Masks: `.gif` (grayscale, binary)

---

## 🖥️ Dashboard

A Streamlit dashboard allows users to:

- Upload car images
- Automatically segment and remove backgrounds
- Enter car year, make, and model
- Get car details via LLaMA 3 integration

🔗 **Try the app on Hugging Face Spaces**:  
[https://huggingface.co/spaces/Nupur-git-bit/car-background-remover](https://huggingface.co/spaces/Nupur-git-bit/car-background-remover)




