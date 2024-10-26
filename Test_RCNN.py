import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pycocotools.mask as mask_utils

def decode_rle(rle_string, height, width):
    """Decodes RLE string to a binary mask."""
    if rle_string == "":
        return np.zeros((height, width), dtype=np.uint8)

    rle_numbers = [int(x) for x in rle_string.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    mask = np.zeros(height * width, dtype=np.uint8)

    for start, length in rle_pairs:
        mask[start - 1:start + length - 1] = 1

    return mask.reshape((height, width))

class MOTSDataset(Dataset):
    def __init__(self, img_dir, gt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = self.load_gt_file(gt_file)

    def load_gt_file(self, gt_file):
        data = defaultdict(lambda: {"boxes": [], "labels": [], "height": 0, "width": 0})
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # Skip incomplete lines

                image_num = int(parts[0])     # Image number
                class_id = int(parts[2])       # Class ID
                height = int(parts[3])         # Image height
                width = int(parts[4])          # Image width
                rle = parts[5]                 # RLE for the bounding box
                # Append bounding boxes and labels
                bbox = self.rle_to_bbox(rle, width, height)
                data[image_num]["boxes"].append(bbox)
                data[image_num]["labels"].append(class_id)
                data[image_num]["height"] = height
                data[image_num]["width"] = width

        return data

    def rle_to_bbox(self, rle, width, height):
        rle = {
            'counts': rle.encode(),  # Encode the RLE string
            'size': [height, width]  # Size of the mask
        }
    
        # Decode the RLE using pycocotools
        mask = mask_utils.decode(rle)
        mask = mask.astype(np.uint8)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        if mask.sum() == 0:
            return [0, 0, 0, 0]  # No object, return empty bbox
        
        x_indices = np.where(mask.sum(axis=0) > 0)[0]
        y_indices = np.where(mask.sum(axis=1) > 0)[0]

        x_min, x_max = max(0, x_indices[0]), min(width, x_indices[-1])
        y_min, y_max = max(0, y_indices[0]), min(height, y_indices[-1])
        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_num = list(self.data.keys())[idx]
        annotations = self.data[image_num]

        img_path = os.path.join(self.img_dir, f'{image_num:06d}.jpg')  # Adjust filename pattern if necessary
        image = Image.open(img_path)#.convert("RGB")
    
        # Convert boxes and labels to tensors
        boxes_tensor = torch.tensor(annotations["boxes"], dtype=torch.float32)
        labels_tensor = torch.tensor(annotations["labels"], dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
            

        # Create target as a dictionary
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor
        }
    
        return image, target

def visualize_predictions(model, dataset, num_images=5, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    images, targets = [], []

    for i in range(num_images):
        # Get an image and its target
        img, target = dataset[i]
        images.append(img)
        targets.append(target)

    images_tensor = torch.stack(images).to(device)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(images_tensor)  # Forward pass through the model

    for i, (img, output, target) in enumerate(zip(images, outputs, targets)):
        plt.figure(figsize=(12, 8))
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        
        # Visualize bounding boxes and labels
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        # Filter out boxes with low scores
        threshold = 0.5  # You can adjust this threshold
        keep = scores >= threshold

        for box, label in zip(boxes[keep], labels[keep]):
            x_min, y_min, x_max, y_max = box
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                               fill=False, edgecolor='red', linewidth=2))
            plt.text(x_min, y_min, f'Class {label}', color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))

        plt.axis('off')
        plt.title(f'Image {i + 1}')
        plt.show()

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model and load weights
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 11  # Adjust based on your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('faster_rcnn_mots_model.pth', map_location=device))
model.to(device)

# Initialize the dataset
img_dir = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\img1'  # Update with the correct path
gt_file = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\gt\gt.txt'  # Update with the correct path
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = MOTSDataset(img_dir, gt_file, transform)

# Visualize predictions on a few images
visualize_predictions(model, dataset, num_images=5, device=device)
