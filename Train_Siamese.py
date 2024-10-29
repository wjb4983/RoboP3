import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Updated dimensions after pooling layers
        self.fc2 = nn.Linear(256, 128)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(label * euclidean_distance ** 2 +
                          (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2)
        return loss


class Market1501Dataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.id_to_images = self._get_id_to_images()
        self.image_pairs = self._load_image_pairs()

    def _get_id_to_images(self):
        id_to_images = {}
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    person_id = file.split('_')[0]
                    if person_id not in id_to_images:
                        id_to_images[person_id] = []
                    id_to_images[person_id].append(os.path.join(root, file))
        return id_to_images

    def _load_image_pairs(self):
        image_pairs = []
        person_ids = list(self.id_to_images.keys())

        # Create positive and negative pairs
        for person_id in person_ids:
            images = self.id_to_images[person_id]
            # Create positive pairs (same ID)
            if len(images) > 1:
                for i in range(len(images) - 1):
                    for j in range(i + 1, len(images)):
                        image_pairs.append((images[i], images[j], 1))  # Label 1 for positive pairs

        # Create negative pairs (different IDs)
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                img1 = random.choice(self.id_to_images[person_ids[i]])
                img2 = random.choice(self.id_to_images[person_ids[j]])
                image_pairs.append((img1, img2, 0))  # Label 0 for negative pairs

        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([label], dtype=torch.float32)
        return img1, img2, label


def train_siamese_network(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, label in tqdm(dataloader, desc=f'Epoch [{epoch + 1}/{epochs}]'):
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')


if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Market-1501 dataset
    dataset_path = r"..\Market-1501-v15.09.15\bounding_box_train" # Update with the correct path
    dataset = Market1501Dataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_siamese_network(model, dataloader, criterion, optimizer, epochs=10)
