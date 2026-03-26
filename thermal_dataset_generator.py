import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Thermal camera resolution (Bosch/FLIR similar)
H, W = 240, 320

# Create dataset folders
os.makedirs("dataset/low", exist_ok=True)
os.makedirs("dataset/medium", exist_ok=True)
os.makedirs("dataset/high", exist_ok=True)

# --------------------------------------------------
# REALISTIC ABDOMEN THERMAL BACKGROUND
# --------------------------------------------------
def create_abdomen():
    X,Y = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))
    
    torso = ((X/0.85)**2 + (Y/0.65)**2) < 1
    shoulders = ((X/1.1)**2 + ((Y+0.35)/0.7)**2) < 1
    hips = ((X/0.75)**2 + ((Y-0.4)/0.8)**2) < 1
    body_mask = torso | shoulders | hips
    
    center_heat = 34.8 + 0.9*np.exp(-(X**2 + Y**2)*2)
    belly = 0.6*np.exp(-((X)**2 + (Y-0.2)**2)*6)
    asym = 0.25*np.exp(-((X+0.25)**2 + (Y)**2)*4)
    
    veins = cv2.GaussianBlur(np.random.randn(H,W),(31,31),0) * 0.15
    
    body_temp = center_heat + belly + asym + veins
    
    background = 33 + np.random.normal(0,0.1,(H,W))
    body_temp[~body_mask] = background[~body_mask]
    
    return cv2.GaussianBlur(body_temp,(41,41),0)

# --------------------------------------------------
# C-SECTION INCISION GENERATOR
# --------------------------------------------------
def add_incision(img):
    mask = np.zeros((H,W))
    for x in range(90,230):
        y = int(155 + 10*np.sin(x/35))
        mask[y-2:y+2, x] = 1
    return img + mask*0.9

# Gaussian heat function
def gaussian_blob(x0,y0,amp,sigma):
    X,Y = np.meshgrid(np.arange(W), np.arange(H))
    return amp*np.exp(-((X-x0)**2+(Y-y0)**2)/(2*sigma**2))

# --------------------------------------------------
# THERMAL CAMERA SENSOR SIMULATION
# --------------------------------------------------
def camera_effect(img):
    img = cv2.GaussianBlur(img,(7,7),0)
    img += np.random.normal(0,0.04,img.shape)
    
    # dead pixels
    dead = np.random.rand(H,W) < 0.002
    img[dead] = img.mean()
    
    # vignette
    X,Y = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))
    vignette = 1 - 0.35*(X**2 + Y**2)
    img *= vignette
    
    # low resolution sensor
    low = cv2.resize(img,(80,60))
    img = cv2.resize(low,(320,240))
    
    # quantization + clipping
    img = np.round(img/0.05)*0.05
    img = np.clip(img,30,40)
    return img

# --------------------------------------------------
# DATASET GENERATION LOOP
# --------------------------------------------------
N = 100  # images per class

for i in range(N):

    abdomen = create_abdomen()
    wound = add_incision(abdomen)

    # ---------------- LOW RISK ----------------
    halo_low = gaussian_blob(160,155,
                             np.random.uniform(0.3,0.6),
                             np.random.uniform(20,30))
    low_img = camera_effect(wound + halo_low)
    plt.imsave(f"dataset/low/low_{i}.png", low_img, cmap='inferno')

    # ---------------- MEDIUM RISK ----------------
    halo1 = gaussian_blob(150,150,
                          np.random.uniform(0.7,1.0),
                          np.random.uniform(25,35))
    halo2 = gaussian_blob(185,160,
                          np.random.uniform(0.4,0.7),
                          np.random.uniform(20,30))
    med_img = camera_effect(wound + halo1 + halo2)
    plt.imsave(f"dataset/medium/medium_{i}.png", med_img, cmap='inferno')

    # ---------------- HIGH RISK ----------------
    halo_high = gaussian_blob(160,155,
                              np.random.uniform(1.3,1.7),
                              np.random.uniform(35,45))
    
    hot1 = gaussian_blob(np.random.randint(140,170),
                         np.random.randint(145,165),
                         1.3,10)
    hot2 = gaussian_blob(np.random.randint(180,210),
                         np.random.randint(150,170),
                         1.2,12)
    
    high_img = camera_effect(wound + halo_high + hot1 + hot2)
    plt.imsave(f"dataset/high/high_{i}.png", high_img, cmap='inferno')

print("✅ FULL DATASET CREATED SUCCESSFULLY")
