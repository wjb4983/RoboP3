import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class MOTSDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.annotations = self.load_annotations()

    def load_annotations(self):
        annotations = []
        for gt_file in os.listdir(self.gt_dir):
            if gt_file.endswith('.txt'):
                with open(os.path.join(self.gt_dir, gt_file), 'r') as f:
                    for line in f.readlines():
                        data = line.strip().split(',')
                        frame_id = data[0]
                        x = float(data[2])
                        y = float(data[3])
                        width = float(data[4])
                        height = float(data[5])
                        
                        xmin = x
                        ymin = y
                        xmax = x + width
                        ymax = y + height
                        bbox = [xmin, ymin, xmax, ymax]

                        annotations.append({
                            'image_id': frame_id,
                            'bbox': bbox,
                            'class_id': 1  # Assuming a single class (adjust if necessary)
                        })
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, f"{annotation['image_id']}.jpg")  # Adjust for your image format
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)

        bbox = torch.tensor(annotation['bbox'], dtype=torch.float32)
        labels = torch.tensor([annotation['class_id']], dtype=torch.int64)

        target = {
            'boxes': bbox.unsqueeze(0),  # Adding a dimension for batch
            'labels': labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target

def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Replace the classifier with a new one (for fine-tuning)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train_model(model, train_loader, device, num_epochs):
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item():.4f}')

def main():
    img_dir = 'MOTS/train/MOT.../img'  # Replace with the path to your images
    gt_dir = 'MOTS/train/MOT.../gt'    # Replace with the path to your ground truth annotations

    # Load dataset
    dataset = MOTSDataset(img_dir=img_dir, gt_dir=gt_dir)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Initialize model
    num_classes = 2  # Adjust based on your dataset (1 for the class + 1 for background)
    model = get_model(num_classes)

    # Define device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Train the model
    train_model(model, train_loader, device, num_epochs=10)

if __name__ == "__main__":
    main()
