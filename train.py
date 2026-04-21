import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from dataset.dataset import GazeFollowDataset
from models.gaze_model import GazeModel

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_ROOT = "data"  # put dataset here
    ANNO_FILE = os.path.join(DATA_ROOT, "test_annotation.txt")

    dataset = GazeFollowDataset(ANNO_FILE, DATA_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    model = GazeModel().to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    criterion = nn.MSELoss()

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img, gaze_gt in loader:
            img = img.to(device)
            gaze_gt = gaze_gt.to(device)

            pred = model(img)
            loss = criterion(pred, gaze_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "gaze_model.pth")

if __name__ == "__main__":
    main()
