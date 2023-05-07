import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms, models, datasets
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_tests = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_path = '~/content/segregation/train/'
cistest_path = '~/content/segregation/cis_test/'
cisval_path = '~/content/segregation/cis_val/'

transtest_path = '~/content/segregation/trans_test/'
transval_path = '/home/content/segregation/trans_val/'

train_full_directory = train_path+'full/'
cistest_full_directory = cistest_path+'full/'
cisval_full_directory = cisval_path+'full/'

# Define dataset and dataloader
train_full_data = datasets.ImageFolder(root=train_full_directory, transform = transform)
cistest_full_data = datasets.ImageFolder(root = cistest_full_directory, transform = transform_tests)                                 
cisval_full_data = datasets.ImageFolder(root = cisval_full_directory, transform = transform_tests)

train_full_dataloader = DataLoader(train_full_data, batch_size= 256)
cistest_full_dataloader = DataLoader(cistest_full_data, batch_size= 128)
cisval_full_dataloader = DataLoader(cisval_full_data, batch_size= 128)


print("Data is loaded")
# Define Inception-v3 model
"""
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=2),
    nn.Conv2d(32, 32, kernel_size=3),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 80, kernel_size=1),
    nn.BatchNorm2d(80),
    nn.ReLU(inplace=True),
    nn.Conv2d(80, 192, kernel_size=3),
    nn.BatchNorm2d(192),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(192, 288, kernel_size=3, padding=1),
    nn.BatchNorm2d(288),
    nn.ReLU(inplace=True),
    nn.Conv2d(288, 288, kernel_size=3, padding=1),
    nn.BatchNorm2d(288),
    nn.ReLU(inplace=True),
    nn.Conv2d(288, 192, kernel_size=3, padding=1),
    nn.BatchNorm2d(192),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(192, 768, kernel_size=1),
    nn.BatchNorm2d(768),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(768, 16)
)
"""
model = models.inception_v3(pretrained=True)

# Freeze all parameters in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 16)
#model.load_state_dict(torch.load(args.load_weight))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# Define optimizer and loss function
optimizer = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Define early stopping parameters
best_val_loss = float('inf')

# Train the model
best_model = None
for epoch in range(20):
    print("processing epoch: ",epoch)
    # Training loop
    running_loss = []
    model.train()
    for i, (inputs, labels) in tqdm(enumerate(train_full_dataloader)):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs.cpu(), labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    if epoch%2 == 0:
        print("Average training loss is : ",sum(running_loss)/(len(running_loss)))
    
    # Validation loop
    with torch.no_grad():
        val_loss = []
        model.eval()
        for i, (inputs, labels) in tqdm(enumerate(cisval_full_dataloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.cpu(), labels)
            val_loss.append(loss.item())
        #val_loss /= len
    if epoch%2 == 0:
        print("Average validation loss is : ",sum(val_loss)/(len(val_loss)))
    val_loss_epoch = sum(val_loss)/(len(val_loss))
    if epoch == 0 :
        val_loss_epoch = np.inf
        best_loss = val_loss_epoch
    if val_loss_epoch <= best_loss:
        best_loss = val_loss_epoch
        best_model = model
        print('[{:2d},  save, {}]'.format(epoch, '/home2/vishwakarma/cis_train_inception2.pt'))
        if torch.cuda.device_count() > 1:    
            torch.save(model.module.state_dict(), '/home2/vishwakarma/cis_train_inception2.pt')
        else:
            torch.save(model.state_dict(), '~/cis_train_inception.pt')

