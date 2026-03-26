print("Script started")

import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

classes = ['high','low','medium']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)
model.load_state_dict(torch.load("vit_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

print("Loading image...")
img = Image.open("dataset/low/low_0.png").convert("RGB")
img = transform(img).unsqueeze(0).to(device)

print("Predicting...")
with torch.no_grad():
    outputs = model(img)
    _, pred = torch.max(outputs, 1)

print("Prediction:", classes[pred.item()])