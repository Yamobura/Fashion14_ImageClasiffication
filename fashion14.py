def predict_image(model, image_path, transform, device, idx2label):
	model.eval()
	image = Image.open(image_path).convert('RGB')
	image = transform(image).unsqueeze(0).to(device)
	with torch.no_grad():
		outputs = model(image)
		_, predicted = outputs.max(1)
	return idx2label[predicted.item()]


# Imports
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm


# ...existing code...


# Paths
DATA_DIR = 'FashionStyle14_v1'
CSV_DIR = os.path.join(DATA_DIR)
IMG_DIR = os.path.join(DATA_DIR, 'dataset')

# Read CSVs
def read_csv(csv_name):
	csv_path = os.path.join(CSV_DIR, csv_name)
	with open(csv_path, 'r', encoding='utf-8') as f:
		lines = [line.strip() for line in f.readlines() if line.strip()]
	return lines




# Extract labels from paths
def get_label_from_path(path):
	return path.split('/')[1]

train_list = read_csv('train.csv')
val_list = read_csv('val.csv')
test_list = read_csv('test.csv')

all_labels = sorted({get_label_from_path(p) for p in train_list + val_list + test_list})
label2idx = {label: idx for idx, label in enumerate(all_labels)}

class FashionDataset(Dataset):
	def __init__(self, file_list, img_dir, label2idx, transform=None):
		self.file_list = file_list
		self.img_dir = img_dir
		self.label2idx = label2idx
		self.transform = transform

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		rel_path = self.file_list[idx]
		label_name = get_label_from_path(rel_path)
		label = self.label2idx[label_name]
		img_path = os.path.join(self.img_dir, label_name, os.path.basename(rel_path))
		try:
			image = Image.open(img_path).convert('RGB')
		except FileNotFoundError:
			# Skip missing file by trying the next index (wrap around if at end)
			next_idx = (idx + 1) % len(self.file_list)
			if next_idx == idx:
				raise RuntimeError("No images found in dataset!")
			return self.__getitem__(next_idx)
		if self.transform:
			image = self.transform(image)
		return image, label

# Image transforms
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
train_dataset = FashionDataset(train_list, IMG_DIR, label2idx, transform)
val_dataset = FashionDataset(val_list, IMG_DIR, label2idx, transform)
test_dataset = FashionDataset(test_list, IMG_DIR, label2idx, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Build classification model (ResNet18)
num_classes = len(label2idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
print(f"Model ready: ResNet18 with {num_classes} classes. Using device: {device}")

# Training and validation loop
def train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for images, labels in tqdm(loader, desc='Training', leave=False):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item() * images.size(0)
		_, predicted = outputs.max(1)
		correct += predicted.eq(labels).sum().item()
		total += labels.size(0)
	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in tqdm(loader, desc='Validation', leave=False):
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * images.size(0)
			_, predicted = outputs.max(1)
			correct += predicted.eq(labels).sum().item()
			total += labels.size(0)
	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc

if __name__ == "__main__":
	import sys
	model_path = "fashion_resnet18.pth"
	idx2label = {v: k for k, v in label2idx.items()}

	if len(sys.argv) > 1 and sys.argv[1] == "predict":
		# Load model and predict custom image
		if not os.path.isfile(model_path):
			print(f"Saved model '{model_path}' not found. Please train and save the model first.")
			sys.exit(1)
		model.load_state_dict(torch.load(model_path, map_location=device))
		custom_image_path = sys.argv[2] if len(sys.argv) > 2 else "my_test.jpg"
		if os.path.isfile(custom_image_path):
			pred_class = predict_image(model, custom_image_path, transform, device, idx2label)
			print(f"Predicted class for {custom_image_path}: {pred_class}")
		else:
			print(f"Custom image file '{custom_image_path}' not found.")
		sys.exit(0)

	# Training and evaluation as before
	num_epochs = 5
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	for epoch in range(num_epochs):
		print(f"Epoch {epoch+1}/{num_epochs}")
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = validate(model, val_loader, criterion, device)
		print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
		print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

	# Save the trained model
	torch.save(model.state_dict(), model_path)
	print(f"Model saved to {model_path}")

	# Test set evaluation
	def test(model, loader, device):
		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for images, labels in tqdm(loader, desc='Testing', leave=False):
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				_, predicted = outputs.max(1)
				correct += predicted.eq(labels).sum().item()
				total += labels.size(0)
		return correct / total if total > 0 else 0.0

	test_acc = test(model, test_loader, device)
	print(f"Test Accuracy: {test_acc:.4f}")

# Read CSVs
def read_csv(csv_name):
	csv_path = os.path.join(CSV_DIR, csv_name)
	with open(csv_path, 'r', encoding='utf-8') as f:
		lines = [line.strip() for line in f.readlines() if line.strip()]
	return lines

train_list = read_csv('train.csv')
val_list = read_csv('val.csv')
test_list = read_csv('test.csv')

# Extract labels from paths
def get_label_from_path(path):
	return path.split('/')[1]

all_labels = sorted({get_label_from_path(p) for p in train_list + val_list + test_list})
label2idx = {label: idx for idx, label in enumerate(all_labels)}

class FashionDataset(Dataset):
	def __init__(self, file_list, img_dir, label2idx, transform=None):
		self.file_list = file_list
		self.img_dir = img_dir
		self.label2idx = label2idx
		self.transform = transform

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		rel_path = self.file_list[idx]
		label_name = get_label_from_path(rel_path)
		label = self.label2idx[label_name]
		img_path = os.path.join(self.img_dir, label_name, os.path.basename(rel_path))
		image = Image.open(img_path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image, label

# Image transforms
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
train_dataset = FashionDataset(train_list, IMG_DIR, label2idx, transform)
val_dataset = FashionDataset(val_list, IMG_DIR, label2idx, transform)
test_dataset = FashionDataset(test_list, IMG_DIR, label2idx, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
