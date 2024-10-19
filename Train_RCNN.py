import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import pycocotools.mask as mask_utils

def decode_rle(rle_string, height, width):
    """Decodes MOTS RLE string to a binary mask.

    Args:
        rle_string (str): The RLE string.
        height (int): The height of the mask.
        width (int): The width of the mask.

    Returns:
        np.ndarray: A binary mask representing the object.
    """

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
        data = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # Skip incomplete lines

                image_num = int(parts[0])     # Image number
                object_id = int(parts[1])      # Object ID
                class_id = int(parts[2])       # Class ID
                height = int(parts[3])         # Image height
                width = int(parts[4])          # Image width
                rle = parts[5]                 # RLE for the bounding box

                data.append((image_num, object_id, class_id, height, width, rle))
        return data

    def rle_to_bbox(self, rle, width, height):
        rle = {     
            'counts': rle.encode(),  # Encode the RLE string
            'size': [height, width]          # Size of the mask
        }
    
        # Decode the RLE using pycocotools
        mask = mask_utils.decode(rle)
        mask = mask.astype(np.uint8)
        
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
        image_num, object_id, class_id, height, width, rle = self.data[idx]
    
        img_path = os.path.join(self.img_dir, f'{image_num:06d}.jpg')  # Adjust filename pattern if necessary
        image = Image.open(img_path).convert("RGB")
    
        bbox = self.rle_to_bbox(rle, width, height)
        if self.transform:
            image = self.transform(image)
    
        # Create target as a dictionary for a single instance
        target = {
            "boxes": torch.tensor(bbox, dtype=torch.float32).unsqueeze(0),  # shape: [1, 4]
            "labels": torch.tensor([class_id], dtype=torch.int64)  # shape: [1]
        }
    
        return image, target

# Initialize the dataset and dataloader
img_dir = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\img1'  # Update with the correct path
gt_file = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\gt\gt.txt'  # Update with the correct path
transform = transforms.Compose([
    transforms.Resize((1080, 1920)),  # Adjust size if necessary
    transforms.ToTensor()
])

dataset = MOTSDataset(img_dir, gt_file, transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
print(f'Total number of samples in dataset: {len(dataset)}')
# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Replace the pre-trained head with a new one (adjust the number of classes as necessary)
num_classes = 11#len(set(d[2] for d in dataset.data)) + 1  # Assuming classes start from 1
print(num_classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to GPU
model.to(device)

# Set the model to training mode
model.train()

# Set up the optimizer
params_to_optimize = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params_to_optimize, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10  # Set the number of epochs you want to train
for epoch in range(num_epochs):
    print("Epoch", epoch + 1)
    for images, targets in data_loader:

        images = [image.to(device) for image in images]  # Move images to GPU
        
        # Convert targets to a list of dictionaries for the batch
        batch_targets = []
        num_images = len(images)
        
        # Iterate through each image and create a corresponding target dictionary
        for i in range(num_images):
            target = {
                "boxes": targets["boxes"][i].to(device),  # Move boxes for the i-th image to GPU
                "labels": targets["labels"][i].to(device)  # Move labels for the i-th image to GPU
            }
            batch_targets.append(target)

        optimizer.zero_grad()  # Zero the gradients
        loss_dict = model(images, batch_targets)  # Forward pass
        losses = sum(loss for loss in loss_dict.values())  # Sum up all the losses
        losses.backward()  # Backward pass
        optimizer.step()  # Update parameters

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'faster_rcnn_mots_model.pth')
