import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = x / x.norm(p=2, dim=1, keepdim=True)  # Normalize embeddings
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, img1, img2):
        output1 = self.embedding_net(img1)
        output2 = self.embedding_net(img2)
        return output1, output2


class Market1501PairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir

        # Create a mapping from person IDs to image paths
        self.data = {}
        for img_name in os.listdir(root_dir):
            if img_name.endswith('.jpg'):
                pid = int(img_name.split('_')[0])
                img_path = os.path.join(root_dir, img_name)
                self.data.setdefault(pid, []).append(img_path)

        self.pids = list(self.data.keys())
        self.num_pids = len(self.pids)

        # Generate pairs
        self.pairs = []
        for pid in self.pids:
            imgs = self.data[pid]
            # Positive pairs
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    self.pairs.append((imgs[i], imgs[j], 1))  # Label 1 for similar
            # Negative pairs
            neg_pids = [neg_pid for neg_pid in self.pids if neg_pid != pid]
            neg_imgs = []
            for neg_pid in random.sample(neg_pids, min(len(neg_pids), len(imgs))):
                neg_imgs.extend(self.data[neg_pid])
            for img1, img2 in zip(imgs, neg_imgs):
                self.pairs.append((img1, img2, 0))  # Label 0 for dissimilar

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        # Adjust labels: 0 for similar, 1 for dissimilar
        label = float(1 - label)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        # Compute contrastive loss
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

def main():
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128)),  # (height, width)
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Mean
                             [0.229, 0.224, 0.225])  # Std
    ])

    train_dataset = Market1501PairDataset(
        root_dir=r"..\Market-1501-v15.09.15\bounding_box_train",
        transform=data_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    import torch.optim as optim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_net = EmbeddingNet().to(device)
    model = SiameseNetwork(embedding_net).to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10  # Increased epochs due to training from scratch
    save_path = 'siamese_model.pth'

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # Wrap the DataLoader with tqdm
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for batch_idx, (img1, img2, label) in enumerate(train_loader_tqdm):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # Update tqdm description with current loss
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}')
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()
