import torch
from torch.utils.data import Dataset
import cv2
import os

class GazeFollowDataset(Dataset):

    def __init__(self, annotation_file, root_dir):
        self.samples = []
        self.root_dir = root_dir

        with open(annotation_file, "r") as f:
            lines = f.readlines()[1:]

            for line in lines:
                parts = line.strip().split(",")

                img_rel_path = parts[0]
                img_path = os.path.join(root_dir, img_rel_path)

                gaze_x = float(parts[7])
                gaze_y = float(parts[8])

                self.samples.append((img_path, gaze_x, gaze_y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gx, gy = self.samples[idx]

        img = cv2.imread(img_path)

        if img is None:
            img = torch.zeros((3,224,224))
            gaze = torch.tensor([0.0,0.0])
            return img, gaze

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # ✅ normalization (important)
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img = (img - mean) / std

        gaze = torch.tensor([gx, gy], dtype=torch.float32)

        return img, gaze
