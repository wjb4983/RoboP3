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
import cv2

resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((128, 128)),  # Resize to match Siamese model input
    transforms.ToTensor()
])


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

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, img1, img2):
        output1 = self.embedding_net(img1)
        output2 = self.embedding_net(img2)
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

                image_num = int(parts[0])  # Image number
                class_id = int(parts[2])  # Class ID
                height = int(parts[3])  # Image height
                width = int(parts[4])  # Image width
                rle = parts[5]  # RLE for the bounding box
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
        image = Image.open(img_path)  # .convert("RGB")

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


import imageio
import io
import matplotlib.pyplot as plt


def visualize_predictions_with_tracking_to_gif(
        model, dataset, num_images=100, device='cpu', siamese_model=None, gif_filename="tracking_output.gif",
        similarity_threshold=0.5, spatial_threshold=50
):
    model.eval()  # Set the model to evaluation mode
    siamese_model.eval()  # Set the Siamese model to evaluation mode

    frames = []  # List to store frames for the GIF
    previous_embedding = None  # Initialize the previous embedding for tracking
    prev_box = None  # Initialize the previous bounding box for spatial reference
    images, targets = [], []

    for j in range(int(num_images / 5)):
        images, targets = [], []
        for i in range(5):
            # Get an image and its target
            img, target = dataset[i + j * 5]
            images.append(img)
            targets.append(target)

        images_tensor = torch.stack(images).to(device)

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(images_tensor)  # Forward pass through the model

        for i, (img, output, target) in enumerate(zip(images, outputs, targets)):
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img.permute(1, 2, 0).cpu().numpy())

            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            # Filter boxes with a minimum score threshold
            keep = scores >= 0.5
            current_bboxes = []
            current_images = []

            for box in boxes[keep]:
                x_min, y_min, x_max, y_max = map(int, box)
                bbox_crop = img[:, y_min:y_max, x_min:x_max]
                bbox_crop = resize_transform(bbox_crop).unsqueeze(0).to(device)
                current_bboxes.append(box)
                current_images.append(bbox_crop)

            if previous_embedding is not None and current_bboxes:
                best_match_score = float("-inf")
                best_match_index = -1

                for idx, current_image in enumerate(current_images):
                    current_embedding = siamese_model.forward_once(current_image)

                    # Calculate cosine similarity with the previous embedding
                    similarity_score = F.cosine_similarity(previous_embedding, current_embedding).item()

                    # Check spatial proximity if previous bounding box is available
                    x_min, y_min, x_max, y_max = current_bboxes[idx]
                    if prev_box is not None:
                        prev_x_min, prev_y_min, prev_x_max, prev_y_max = prev_box
                        spatial_distance = ((x_min - prev_x_min) ** 2 + (y_min - prev_y_min) ** 2) ** 0.5

                        # Choose the box with the highest similarity that is spatially close and above the threshold
                        if similarity_score > best_match_score and similarity_score > similarity_threshold and spatial_distance < spatial_threshold:
                            print(similarity_score)
                            best_match_score = similarity_score
                            best_match_index = idx

                if best_match_index != -1:
                    # Update previous embedding and bounding box with the best match
                    previous_embedding = siamese_model.forward_once(current_images[best_match_index])
                    prev_box = current_bboxes[best_match_index]
                    selected_box = current_bboxes[best_match_index]

                    x_min, y_min, x_max, y_max = map(int, selected_box)
                    ax.add_patch(
                        plt.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            fill=False,
                            edgecolor="red",
                            linewidth=2,
                        )
                    )
                    ax.text(
                        x_min,
                        y_min,
                        f"Class {labels[best_match_index]}",
                        color="white",
                        fontsize=12,
                        bbox=dict(facecolor="red", alpha=0.5),
                    )

            else:
                # Initialize tracking on the first frame or when no match is found
                initial_box_index = 8
                if previous_embedding is None and len(current_bboxes) > initial_box_index:
                    previous_embedding = siamese_model.forward_once(current_images[initial_box_index])
                    prev_box = current_bboxes[initial_box_index]
                    selected_box = current_bboxes[initial_box_index]

                    x_min, y_min, x_max, y_max = map(int, selected_box)
                    ax.add_patch(
                        plt.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            fill=False,
                            edgecolor="red",
                            linewidth=2,
                        )
                    )
                    ax.text(
                        x_min,
                        y_min,
                        f"Class {labels[0]}",
                        color="white",
                        fontsize=12,
                        bbox=dict(facecolor="red", alpha=0.5),
                    )

            ax.axis("off")
            ax.set_title(f"Image {i + 1}")

            # Save the frame to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            frames.append(imageio.imread(buf))  # Add the frame to the GIF list
            buf.close()
            plt.close(fig)  # Close the figure to free up memory

    # Save frames to a GIF file
    imageio.mimsave(gif_filename, frames, fps=30)
    print(f"GIF saved as {gif_filename}")


def process_video_and_save_gif(
        video_path, model, triplet_model, device='cpu', similarity_threshold=0.7, spatial_threshold=30,
        gif_filename="tracking_output.gif"
):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Transformation to convert OpenCV frames to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    model.eval()  # Set the Faster R-CNN model to evaluation mode
    triplet_model.eval()  # Set the Triplet model to evaluation mode

    previous_embedding = None  # Initialize the previous embedding for tracking
    prev_box = None  # Initialize the previous bounding box for spatial reference
    frames = []  # List to store frames for the GIF

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if there are no frames left

        # Convert BGR to RGB and to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb).to(device)

        with torch.no_grad():
            # Forward pass through Faster R-CNN model
            output = model([frame_tensor])[0]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame_rgb)

        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        threshold = 0.5
        keep = scores >= threshold
        current_bboxes = []
        current_images = []

        # Crop each bounding box and prepare for embedding generation
        for box in boxes[keep]:
            x_min, y_min, x_max, y_max = map(int, box)
            bbox_crop = frame_tensor[:, y_min:y_max, x_min:x_max]
            bbox_crop = transforms.functional.resize(bbox_crop.unsqueeze(0), (128, 128)).to(device)
            current_bboxes.append(box)
            current_images.append(bbox_crop)

        if previous_embedding is not None and current_bboxes:
            best_match_score = float("inf")  # Initialize with high distance for Euclidean
            best_match_index = -1

            for idx, current_image in enumerate(current_images):
                current_embedding = triplet_model.forward_once(current_image)

                # Calculate Euclidean distance with the previous embedding
                distance_score = torch.dist(previous_embedding, current_embedding).item()

                # Check spatial proximity if previous bounding box is available
                x_min, y_min, x_max, y_max = current_bboxes[idx]
                if prev_box is not None:
                    prev_x_min, prev_y_min, prev_x_max, prev_y_max = prev_box
                    spatial_distance = ((x_min - prev_x_min) ** 2 + (y_min - prev_y_min) ** 2) ** 0.5

                    # Choose the box with the lowest distance that is spatially close and below the threshold
                    if distance_score < best_match_score and distance_score < similarity_threshold and spatial_distance < spatial_threshold:
                        best_match_score = distance_score
                        best_match_index = idx

            if best_match_index != -1:
                # Update previous embedding and bounding box with the best match
                previous_embedding = triplet_model.forward_once(current_images[best_match_index])
                prev_box = current_bboxes[best_match_index]
                selected_box = current_bboxes[best_match_index]

                x_min, y_min, x_max, y_max = map(int, selected_box)
                ax.add_patch(
                    plt.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
                ax.text(
                    x_min,
                    y_min,
                    f"Class {labels[best_match_index]}",
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5),
                )

        else:
            # Initialize tracking on the first frame or when no match is found
            if len(current_bboxes) > 0:
                previous_embedding = triplet_model.forward_once(current_images[0])
                prev_box = current_bboxes[0]
                selected_box = current_bboxes[0]

                x_min, y_min, x_max, y_max = map(int, selected_box)
                ax.add_patch(
                    plt.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
                ax.text(
                    x_min,
                    y_min,
                    f"Class {labels[0]}",
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5),
                )

        ax.axis("off")
        ax.set_title(f"Frame {frame_count + 1}")

        # Save the frame to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        frames.append(imageio.imread(buf))  # Add the frame to the GIF list
        buf.close()
        plt.close(fig)  # Close the figure to free up memory

        frame_count += 1

    # Release video capture and save frames to a GIF file
    cap.release()
    imageio.mimsave(gif_filename, frames, fps=30)
    print(f"GIF saved as {gif_filename}")


# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model and load weights
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 11  # Adjust based on your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('faster_rcnn_mots_model.pth', map_location=device))
model.to(device)
embedding_net = EmbeddingNet().to(device)
siam_model = SiameseNetwork(embedding_net).to(device)
siam_model.load_state_dict(torch.load('siamese_model.pth', map_location=device))
siam_model.to(device)

# Initialize the dataset
img_dir = r'..\MOTS\train\MOTS20-02\img1'  # Update with the correct path
gt_file = r'..\MOTS\train\MOTS20-02\gt\gt.txt'  # Update with the correct path
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = MOTSDataset(img_dir, gt_file, transform)

video_path = r"../MOT16-01-raw.webm"

# Visualize predictions on a few images
# visualize_predictions_with_tracking_to_gif(model, dataset, num_images=600, device=device, siamese_model=siam_model)
process_video_and_save_gif(video_path, model, siam_model, device=device, similarity_threshold=0.7, spatial_threshold=30,
                           gif_filename="tracking_output.gif")
