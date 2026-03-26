import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn as nn

classes = ['high','low','medium']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD VIT ----------
vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)
vit.load_state_dict(torch.load("vit_model.pth", map_location=device))
vit.to(device)
vit.eval()

# ---------- LOAD CNN ----------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*54*54,128)
        self.fc2 = nn.Linear(128,3)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,32*54*54)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = CNNModel()
cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn.to(device)
cnn.eval()

# ---------- IMAGE ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("dataset/low/low_0.png").convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# ---------- PREDICTIONS ----------
with torch.no_grad():
    vit_pred = torch.argmax(vit(img),1).item()
    cnn_pred = torch.argmax(cnn(img),1).item()

# ---------- ENSEMBLE ----------
final_pred = round((vit_pred + cnn_pred)/2)

print("ViT Prediction:", classes[vit_pred])
print("CNN Prediction:", classes[cnn_pred])
print("Final Ensemble Prediction:", classes[final_pred])