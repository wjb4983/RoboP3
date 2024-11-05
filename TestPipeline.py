import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import pycocotools.mask as mask_utils


resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((128, 128)),  # Resize to match Siamese model input
    transforms.ToTensor()
])

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))
        self.fc1 = nn.Linear(524288, 256)  # Updated dimensions after pooling layers
        self.fc2 = nn.Linear(256, 128)

    def forward_once(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
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

def visualize_predictions(model, dataset, num_images=100, device='cpu', siamese_model=None):
    model.eval()  # Set the model to evaluation mode
    siamese_model.eval()  # Set the Siamese model to evaluation mode

    images, targets = [], []
    previous_target = None  # Initialize previous target
    prev_img = None
    for j in range(int(num_images/5)):
        images, targets = [], []
        for i in range(5):
            # Get an image and its target
            img, target = dataset[i+j*5]
            images.append(img)
            targets.append(target)
    
        images_tensor = torch.stack(images).to(device)
    
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(images_tensor)  # Forward pass through the model
    
    
    
        for i, (img, output, target) in enumerate(zip(images, outputs, targets)):
            plt.figure(figsize=(12, 8))
            plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    
            # Get bounding boxes, scores, and labels
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
    
            # Filter out boxes with low scores
            threshold = 0.0  # You can adjust this threshold
            keep = scores #>= threshold
    
            # Create list to store slices for Siamese input
            current_bboxes = []
            current_images = []
            for box in boxes:#boxes[keep]:
                x_min, y_min, x_max, y_max = map(int, box)
                bbox_crop = img[:, y_min:y_max, x_min:x_max]  # Crop based on bounding box coordinates
                # current_bbox_resized = F.interpolate(current_bbox.unsqueeze(0), size=desired_size, mode='bilinear').to(device)
                bbox_crop = resize_transform(bbox_crop).unsqueeze(0).to(device)  # Resize and add batch dimension
                # bbox_crop = bbox_crop.unsqueeze(0).to(device)
                current_bboxes.append(box)
                current_images.append(bbox_crop)
            
            # If this is not the first frame, use Siamese network to track
            if previous_target is not None and current_bboxes:
                if isinstance(previous_target, np.ndarray):
                    previous_target = torch.tensor(previous_target).to(device)
                previous_target = previous_target.unsqueeze(0)
            
                # Process each bbox with the Siamese model
                similarities = []
                x1, y1, x2, y2 = map(int, previous_target.squeeze().cpu().numpy())
                prev_crop = prev_img[:, y1:y2, x1:x2]  # Crop based on bounding box coordinates
                # prev_crop = F.interpolate(prev_crop.unsqueeze(0), size=(128,128), mode='bilinear').to(device)
                prev_crop = resize_transform(prev_crop).unsqueeze(0).to(device)  # Resize and add batch dimension
                # prev_crop = prev_crop.unsqueeze(0).to(device)  # Resize and add batch dimension

            
                for current_image in current_images:
                    # Forward pass through Siamese network
                    sim1, sim2 = siamese_model(prev_crop, current_image)
                    similarity_score = F.cosine_similarity(sim1, sim2).item()
                    similarities.append(similarity_score)
                    plt.figure(figsize=(12, 6))  # Create a new figure with a defined size
                
                    # Display previous crop
                    tprev_crop = prev_crop.squeeze(0)
                    plt.subplot(1, 2, 1)
                    plt.imshow(tprev_crop.permute(1, 2, 0).cpu().numpy())
                    plt.title("Previous Crop")
                
                    # Display current image
                    tcurrent_image = current_image.squeeze(0)
                    plt.subplot(1, 2, 2)
                    plt.imshow(tcurrent_image.permute(1, 2, 0).cpu().numpy())
                    plt.title("Current Image")
                    
                    plt.show()
                # print(similarities)
                best_index = np.argmax(similarities)
                previous_target = torch.tensor(current_bboxes[best_index])
                prev_label = labels[best_index]
                prev_img = img
            
                best_index = np.argmax(similarities)
                plt.title(f'Best Match Index: {best_index}, Similarity: {similarities[best_index]:.4f}')
                
            else:
                # If it's the first frame, initialize previous target with the first box
                print("first frame")
                if len(current_bboxes) > 0:
                    previous_target = torch.tensor(boxes[0])  # Initialize with the first detected bbox
                    prev_label = labels[0]
                    prev_img = img
            # for box, label in zip(boxes[keep], labels[keep]):
            #     x_min, y_min, x_max, y_max = box
            #     current_bboxes.append(img[:, int(y_min):int(y_max), int(x_min):int(x_max)])
        
            #     plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
            #                                         fill=False, edgecolor='red', linewidth=2))
            #     plt.text(x_min, y_min, f'Class {prev_label}', color='white', fontsize=12,
            #               bbox=dict(facecolor='red', alpha=0.5))
            # print(previous_target)
            x_min, y_min, x_max, y_max = previous_target
            current_bboxes.append(img[:, int(y_min):int(y_max), int(x_min):int(x_max)])
    
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                fill=False, edgecolor='red', linewidth=2))
            plt.text(x_min, y_min, f'Class {prev_label}', color='white', fontsize=12,
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
siam_model = SiameseNetwork()
siam_model.load_state_dict(torch.load('siamese_model____.pth', map_location=device))
siam_model.to(device)

# Initialize the dataset
img_dir = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\img1'  # Update with the correct path
gt_file = r'C:\Users\wbott\Downloads\MOTS\MOTS\train\MOTS20-02\gt\gt.txt'  # Update with the correct path
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = MOTSDataset(img_dir, gt_file, transform)

# Visualize predictions on a few images
visualize_predictions(model, dataset, num_images=100, device=device, siamese_model=siam_model)
