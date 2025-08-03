import os
os.environ["HOME"] = "/tmp"

import streamlit as st
# rest of your imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import datetime

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ========== CONFIG ==========
INPUT_IMAGE_HEIGHT = 128
INPUT_IMAGE_WIDTH = 128
MODEL_PATH = "src/unet_carvana_final.pth"
GROQ_MODEL_NAME = "llama3-70b-8192"

CAR_DATA = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot"],
    "Ford": ["F-150", "Mustang", "Escape", "Explorer"],
    "Tesla": ["Model S", "Model 3", "Model X", "Model Y"],
    "BMW": ["3 Series", "5 Series", "X3", "X5"]
}

# ========== U-NET MODEL ==========
class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64, 128)):
        super().__init__()
        self.encBlocks = nn.ModuleList([
            Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs

class Decoder(nn.Module):
    def __init__(self, channels=(128, 64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            Block(channels[i+1]*2, channels[i+1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        _, _, H, W = x.shape
        return transforms.CenterCrop([H, W])(encFeatures)

class UNet(nn.Module):
    def __init__(self, encChannels=(3, 16, 32, 64, 128),
                       decChannels=(128, 64, 32, 16),
                       nbClasses=1, retainDim=True,
                       outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        self.head = nn.Conv2d(decChannels[-1], nbClasses, kernel_size=1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[-1], encFeatures[:-1][::-1])
        mask = self.head(decFeatures)
        if self.retainDim:
            mask = F.interpolate(mask, size=self.outSize, mode='bilinear', align_corners=False)
        return mask

# ========== HELPERS ==========
@st.cache_resource
def load_model():
    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def apply_car_mask(model, image: Image.Image) -> Image.Image:
    transform = transforms.Compose([
        transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        binary_mask = (pred_mask > 0.5).float()

    mask_np = binary_mask.squeeze().numpy()
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(image.size)

    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    result = Image.composite(image, white_bg, mask_img)

    return result

def query_car_info(make, model, year):
    prompt = f"Tell me about the {year} {make} {model} car in a short paragraph."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a car expert assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Car Background Remover", layout="centered")
st.title("🚗 Car Background Remover + Info Assistant (Groq-powered)")

uploaded_file = st.file_uploader("📤 Upload a car image", type=["jpg", "jpeg", "png"])

# Dropdowns
make = st.selectbox("🔧 Select Car Make", list(CAR_DATA.keys()))
model = st.selectbox("🚘 Select Model", CAR_DATA[make])
year = st.selectbox("📅 Select Year", list(reversed(range(datetime.datetime.now().year, 2004, -1))))

if st.button("Process Car Image"):
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("Removing background..."):
            model_loaded = load_model()
            result_img = apply_car_mask(model_loaded, image)
            st.image(result_img, caption="Background Removed", use_container_width=True)

        with st.spinner("Fetching car details..."):
            try:
                info = query_car_info(make, model, year)
                st.markdown("### 🧠 Car Info")
                st.write(info)
            except Exception as e:
                st.error(f"Failed to fetch info from Groq: {str(e)}")
    else:
        st.warning("Please upload an image.")
