import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from Train_Siamese import Market1501Dataset, SiameseNetwork
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Subset



def create_subset_dataloader(dataset, subset_size=100000):
    # Randomly select a subset of indices from the dataset
    indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=1, shuffle=False)


def test_siamese_network(model, dataloader, threshold=0.5):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    similarities = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            output1, output2 = model(img1, img2)

            # Calculate Euclidean distance between the embeddings
            euclidean_distance = F.pairwise_distance(output1, output2)

            # Determine if distance indicates a "same person" (below threshold) or "different person" (above threshold)
            is_same = (euclidean_distance < threshold).float()
            if is_same.sum() > 0:
                print("True")
            y_true.extend(label.cpu().numpy().flatten())  # Ground truth labels
            y_pred.extend(is_same.cpu().numpy().flatten())  # Predicted labels
            similarities.extend(euclidean_distance.cpu().numpy().flatten())

    # Convert results to numpy arrays for metric calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    similarities = np.array(similarities)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return similarities, y_true, y_pred


# Assuming you have already defined and loaded `Market1501Dataset`
# Create a subset DataLoader with a sample size of 100 pairs
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

subset_size = 10000
test_dataset = Market1501Dataset(folder_path=r"..\Market-1501-v15.09.15\bounding_box_test", transform=transform)
test_dataloader = create_subset_dataloader(test_dataset, subset_size=subset_size)

# Load the trained model
model = SiameseNetwork().cuda()
model.load_state_dict(torch.load("siamese_model.pth"))
model.eval()

# Run the test function
similarities, y_true, y_pred = test_siamese_network(model, test_dataloader, threshold=0.5)
