import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


# Define the transforms to apply to the images
test_transforms = transforms.Compose([
 transforms.Resize((299, 299)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cistest_path = '~/content/segregation/trans_test/full/'

# Load the test set
test_data = ImageFolder(cistest_path, transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# Load the pre-trained Inception-v3 model
model = inception_v3(pretrained=True)

# Replace the last layer with a new layer with 10 units
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 16)

# Load the trained model weights
model.load_state_dict(torch.load('/home2/vishwakarma/trans_train_inception.pt'))

# Set the model to evaluation mode
model.eval()


# Calculate the accuracy on the test set
correct = 0
total = 0
with torch.no_grad():
	for images, labels in test_loader:
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the model on the {} test images: {:.2f}%'.format(total, 100 * correct / total))

