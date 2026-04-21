import torch
import cv2
import numpy as np

from models.gaze_model import GazeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GazeModel().to(device)
model.load_state_dict(torch.load("gaze_model.pth", map_location=device))
model.eval()

img_path = "sample.jpg"  # put any image

img = cv2.imread(img_path)
h, w = img.shape[:2]

img_resized = cv2.resize(img, (224,224)) / 255.0
img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2,0,1).unsqueeze(0)

mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
img_tensor = (img_tensor - mean) / std

img_tensor = img_tensor.to(device)

with torch.no_grad():
    pred = model(img_tensor)[0].cpu().numpy()

# convert normalized coords to pixel
x = int(pred[0] * w)
y = int(pred[1] * h)

cv2.circle(img, (x,y), 8, (0,255,0), -1)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
