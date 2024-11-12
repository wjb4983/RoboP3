import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import pycocotools.mask as mask_utils
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

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

        img_path = os.path.join(self.img_dir, f'{image_num:06d}.jpg')
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

def collate_fn(batch):
    # Custom collate function to handle variable-length bounding boxes
    images, targets = zip(*batch)

    # Stack images into a tensor of shape [batch_size, channels, height, width]
    images = torch.stack(images)  # This assumes images are in [channels, height, width] format

    # Padding the boxes to have the same size
    max_num_boxes = max(len(target["boxes"]) for target in targets)
    
    # Pad boxes and labels
    padded_boxes = []
    padded_labels = []
    
    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]
        
        # Pad with zeros to ensure same length
        if len(boxes) < max_num_boxes:
            padding_size = max_num_boxes - len(boxes)
            boxes = torch.cat([boxes, torch.zeros((padding_size, boxes.shape[1]), dtype=boxes.dtype)])
            labels = torch.cat([labels, torch.zeros(padding_size, dtype=torch.int64)])  # Pad labels as well

        padded_boxes.append(boxes)
        padded_labels.append(labels)

    # Convert lists of tensors to a tensor
    return images, {"boxes": torch.stack(padded_boxes), "labels": torch.stack(padded_labels)}


# Initialize the dataset and dataloader
img_dir = r'..\MOTS\train\MOTS20-02\img1'  # Update with the correct path
gt_file = r'..\MOTS\train\MOTS20-02\gt\gt.txt'  # Update with the correct path
transform = transforms.Compose([
    # transforms.Resize((1080, 1920)),  # Adjust size if necessary
    transforms.ToTensor()
])

dataset = MOTSDataset(img_dir, gt_file, transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
print(f'Total number of samples in dataset: {len(dataset)}')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Replace the pre-trained head with a new one (adjust the number of classes as necessary)
num_classes = 11  # Adjust based on your dataset
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
        # for i in range(len(images)):
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        #     # Original Image
        #     axes[0].imshow(images[i].cpu().numpy())
        # print(targets)
        # Convert targets to a list of dictionaries for the batch
        batch_targets = []
        for target in range(4):
            boxes = targets["boxes"][target].to(device)  # Move boxes for the i-th image to GPU
            labels = targets["labels"][target].to(device)  # Move labels for the i-th image to GPU
        
            # Create a mask to filter out boxes that are [0, 0, 0, 0]
            mask = (boxes.sum(dim=1) != 0)  # This creates a mask where only boxes with non-zero sums are kept
        
            # Apply the mask to boxes and labels
            filtered_boxes = boxes[mask]
            filtered_labels = labels[mask]
        
            # Create the target dictionary with filtered boxes and labels
            targett = {
                "boxes": filtered_boxes,
                "labels": filtered_labels
            }
            
            batch_targets.append(targett)
        # print(batch_targets)
        optimizer.zero_grad()  # Zero the gradients
        loss_dict = model(images, batch_targets)  # Forward pass
        losses = sum(loss for loss in loss_dict.values())  # Sum up all the losses
        losses.backward()  # Backward pass
        optimizer.step()  # Update parameters

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'faster_rcnn_mots_model.pth')
