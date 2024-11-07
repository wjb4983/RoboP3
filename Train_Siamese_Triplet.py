import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import os
import random
from tqdm import tqdm


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))
        self.fc1 = nn.Linear(524288, 256)  # Adjusted dimensions after pooling
        self.fc2 = nn.Linear(256, 128)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output


# Triplet Loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        return loss


class Market1501TripletDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.id_to_images = self._get_id_to_images()
        self.triplets = self._load_triplets()

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

    def _load_triplets(self):
        triplets = []
        person_ids = list(self.id_to_images.keys())

        for person_id in person_ids:
            images = self.id_to_images[person_id]
            if len(images) > 1:
                for img in images:
                    # Select a random positive example
                    pos_img = random.choice([x for x in images if x != img])
                    # Select a random negative example
                    neg_person_id = random.choice([x for x in person_ids if x != person_id])
                    neg_img = random.choice(self.id_to_images[neg_person_id])
                    triplets.append((img, pos_img, neg_img))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


def train_triplet_network(model, dataloader, criterion, optimizer, epochs=10, save_path='siamese_triplet_model.pth'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in tqdm(dataloader, desc=f'Epoch [{epoch + 1}/{epochs}]'):
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            optimizer.zero_grad()

            anchor_output, positive_output, negative_output = model(anchor, positive, negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Market-1501 dataset
    dataset_path = r"..\Market-1501-v15.09.15\bounding_box_train"  # Update with correct path
    dataset = Market1501TripletDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SiameseNetwork().cuda()
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_triplet_network(model, dataloader, criterion, optimizer, epochs=10)
