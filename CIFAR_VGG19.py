import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Training a CIFAR10 classifier Using VGG19 with BN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define a transform to normalize the train data
transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(30),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Define a transform to normalize the validation data
transform_vali = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_vali)

#Dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

#use pretrained VGG19
model = torchvision.models.vgg19_bn(pretrained=True)

#change the last layer
model.classifier[6] = nn.Linear(4096,10)

#move the model to GPU
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#CIFAR10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Training
print("-------------------Training start--------------------")
epochs = 100
batch_size = 128

train_acc_list, valid_acc_list = [], []
train_loss_list, valid_loss_list = [], []

for epoch in range(epochs):
    model.train()
    epoch_train_loss, epoch_accuracy = [], []
    for batch_index, (features, target) in enumerate(trainloader):
        # Mini-batch
        features = features.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(features)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        #Calculate the accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (target == argmax).sum().item() / features.size(0)
        epoch_train_loss.append(loss.item())
        epoch_accuracy.append(accuracy)

        if batch_index % 100 == 0:
            print(f"Epoch: {epoch+1:03d}/{epochs:03d} | "
                  f"Batch {batch_index:03d}/{len(trainloader):03d} | "
                  f"Loss: {loss:.4f} | "
                  f"Accuracy: {accuracy:.4f}")

    train_acc = np.mean(epoch_accuracy)
    train_loss = np.mean(epoch_train_loss)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # Validation
    model.eval()
    epoch_val_loss, epoch_val_accuracy = [], []
    with torch.no_grad():
        for batch_index, (features, target) in enumerate(testloader):
           # Mini-batch
            features = features.to(device)
            target = target.to(device)
            outputs = model(features)
            loss = criterion(outputs, target)

            epoch_val_loss.append(loss.item())
            _, argmax = torch.max(outputs, 1)
            accuracy = (target == argmax).sum().item() / features.size(0)
            epoch_val_accuracy.append(accuracy)
            if batch_index % 100 == 0:
                print(f"Epoch: {epoch+1:03d}/{epochs:03d} | "
                  f"Valid Loss: {loss:.4f} | "
                  f"Valid Accuracy: {accuracy:.4f}") 
    valid_acc = np.mean(epoch_val_accuracy)
    valid_loss = np.mean(epoch_val_loss)
    valid_acc_list.append(valid_acc)
    valid_loss_list.append(valid_loss)

   #save model
torch.save(model.state_dict(), 'CIFAR_VGG19.pth')
print("-------------------Training finished--------------------")


# Plot the training and validation loss
epochs_array = np.arange(1, epochs+1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_array, train_acc_list, label='Train Accuracy')
plt.plot(epochs_array, valid_acc_list, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_array, train_loss_list, label='Train Loss')
plt.plot(epochs_array, valid_loss_list, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()


plt.tight_layout()
plt.savefig('VGG19.png')
plt.show()

