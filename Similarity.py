import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)  # Input: 2x3xHxW -> 1x32xHxW
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 32xHxW -> 64xHxW
        self.fc1 = nn.Linear(64 * 1080 * 1920 // 4 // 4, 256)  # Adjust size based on pooling
        self.fc2 = nn.Linear(256, 1)  # Output: Similarity score

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)  # Concatenate images along channel dimension
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output similarity score between 0 and 1
        return x


import os
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MOTSSimilarityDataset(Dataset):
    def __init__(self, img_dir, gt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self.load_gt_file(gt_file)

    def load_gt_file(self, gt_file):
        data = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                image_num = int(parts[0])
                object_id = int(parts[1])
                
                if image_num not in data:
                    data[image_num] = []
                data[image_num].append(object_id)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_num = list(self.data.keys())[idx]
        object_ids = self.data[image_num]

        # Randomly select two object IDs from the current frame
        obj_id_1 = random.choice(object_ids)
        obj_id_2 = random.choice(object_ids)

        # Load the images
        img_path_1 = os.path.join(self.img_dir, f'{image_num:06d}.jpg')
        img_path_2 = os.path.join(self.img_dir, f'{image_num:06d}.jpg')  # Adjust according to how to get previous frames

        image1 = Image.open(img_path_1).convert("RGB")
        image2 = Image.open(img_path_2).convert("RGB")

        # Determine similarity label (1 if same object, 0 if different)
        label = 1 if obj_id_1 == obj_id_2 else 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), label

img_dir = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\img1'  # Update with the correct path
gt_file = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\gt\gt.txt'  # Update with the correct path
device = 'cuda'
# Initialize the dataset and dataloader for similarity
transform = transforms.Compose([
    transforms.Resize((540, 960)),  # Adjust size if necessary
    transforms.ToTensor()
])

similarity_dataset = MOTSSimilarityDataset(img_dir, gt_file, transform)
similarity_data_loader = torch.utils.data.DataLoader(similarity_dataset, batch_size=4, shuffle=True)

# Initialize the Similarity Network
similarity_model = SimilarityNetwork()
similarity_model.to(device)

# Set up the optimizer
optimizer = torch.optim.Adam(similarity_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Set the number of epochs you want to train
for epoch in range(num_epochs):
    similarity_model.train()
    running_loss = 0.0

    for (images, labels) in similarity_data_loader:
        img1, img2 = images
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = similarity_model(img1, img2)
        loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(similarity_data_loader):.4f}')

# Save the trained model
torch.save(similarity_model.state_dict(), 'similarity_network_mots_model.pth')

