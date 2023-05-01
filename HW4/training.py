# Import Libraries
import numpy as np
import torch
import torchvision.transforms as tvt
import torch.utils.data 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
from PIL import Image
import os
import seaborn as sns

# GLOBAL VARIABLES
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def get_images(root, category):
    category_path = os.path.join(root, category)
    image_files = [image for image in os.listdir(category_path) if image != ".DS_Store"]

    images_pil = [Image.open(os.path.join(category_path, image)).convert("RGB") for image in image_files]
    return images_pil

class GenerateDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_list, transform=None):
        super().__init__()
        self.root = root
        self.class_list = class_list
        self.transform = transform
        self.data = []

        for idx, category in enumerate(self.class_list):
            images = get_images(self.root, category)
            for image in images:
                self.data.append([image, idx])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(self.data[idx][0]) if self.transform else self.data[idx][0]
        label = torch.tensor(self.data[idx][1])

        return image, label

class HW4Net(nn.Module):
    def __init__(self, task):
        super(HW4Net, self).__init__()
        self.task = task

        if(self.task == "task1"):
            """
            ==========================================================================================
            Layer (type:depth-idx)                   Output Shape              Param #
            ==========================================================================================
            HW4Net                                   [10, 5]                   --
            ├─Conv2d: 1-1                            [10, 16, 62, 62]          448
            ├─MaxPool2d: 1-2                         [10, 16, 31, 31]          --
            ├─Conv2d: 1-3                            [10, 32, 29, 29]          4,640
            ├─MaxPool2d: 1-4                         [10, 32, 14, 14]          --
            ├─Linear: 1-5                            [10, 64]                  401,472
            ├─Linear: 1-6                            [10, 5]                   325
            ==========================================================================================
            Total params: 406,885
            Trainable params: 406,885
            Non-trainable params: 0
            Total mult-adds (M): 60.26
            ==========================================================================================
            Input size (MB): 0.49
            Forward/backward pass size (MB): 7.08
            Params size (MB): 1.63
            Estimated Total Size (MB): 9.20
            ==========================================================================================
            """
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(32*14*14, 64)
            self.fc2 = nn.Linear(64, 5) # x = 5 because there are 5 classes
        
        elif(self.task == "task2"):
            """
            ==========================================================================================
            Layer (type:depth-idx)                   Output Shape              Param #
            ==========================================================================================
            HW4Net                                   [10, 5]                   --
            ├─Conv2d: 1-1                            [10, 16, 64, 64]          448
            ├─MaxPool2d: 1-2                         [10, 16, 32, 32]          --
            ├─Conv2d: 1-3                            [10, 32, 32, 32]          4,640
            ├─MaxPool2d: 1-4                         [10, 32, 16, 16]          --
            ├─Linear: 1-5                            [10, 64]                  524,352
            ├─Linear: 1-6                            [10, 5]                   325
            ==========================================================================================
            Total params: 529,765
            Trainable params: 529,765
            Non-trainable params: 0
            Total mult-adds (M): 71.11
            ==========================================================================================
            Input size (MB): 0.49
            Forward/backward pass size (MB): 7.87
            Params size (MB): 2.12
            Estimated Total Size (MB): 10.48
            ==========================================================================================
            """
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32*16*16, 64)
            self.fc2 = nn.Linear(64, 5) # x = 5 because there are 5 classes

        elif(self.task == "task3"):
            """
            ==========================================================================================
            Layer (type:depth-idx)                   Output Shape              Param #
            ==========================================================================================
            HW4Net                                   [10, 5]                   --
            ├─Conv2d: 1-1                            [10, 16, 62, 62]          448
            ├─MaxPool2d: 1-2                         [10, 16, 31, 31]          --
            ├─Conv2d: 1-3                            [10, 32, 29, 29]          4,640
            ├─MaxPool2d: 1-4                         [10, 32, 14, 14]          --
            ├─Conv2d: 1-5                            [10, 32, 14, 14]          9,248
            ├─Conv2d: 1-6                            [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-7                            [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-8                            [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-9                            [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-10                           [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-11                           [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-12                           [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-13                           [10, 32, 14, 14]          (recursive)
            ├─Conv2d: 1-14                           [10, 32, 14, 14]          (recursive)
            ├─Linear: 1-15                           [10, 64]                  401,472
            ├─Linear: 1-16                           [10, 5]                   325
            ==========================================================================================
            Total params: 416,133
            Trainable params: 416,133
            Non-trainable params: 0
            Total mult-adds (M): 241.52
            ==========================================================================================
            Input size (MB): 0.49
            Forward/backward pass size (MB): 12.10
            Params size (MB): 1.66
            Estimated Total Size (MB): 14.25
            ==========================================================================================
            """
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
            self.fc1 = nn.Linear(32*14*14, 64)
            self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        if(self.task == "task3"):
            x = self.pool(F.relu(self.conv1(x))) 
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(net, epochs, lr, betas, dataloader, save_to_path):
    # summary(net, input_size=(10, 3, 64, 64)); quit()
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
    loss_per_iteration = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            print(outputs.shape)
            print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if((batch_idx + 1) % 100 == 0):
                print("[epoch: %d, batch: %5d] loss: %.3f" % (epoch, batch_idx + 1, running_loss / 100))
                loss_per_iteration.append(running_loss / 100)
                running_loss = 0.0

    if(save_to_path):
        torch.save(net.state_dict(), save_to_path)
    return loss_per_iteration

def test(net, path_to_network, dataloader, num_classes):
    net.load_state_dict(torch.load(path_to_network))
    net = net.to(device)
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, dim=1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1
            
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return confusion_matrix, accuracy

def plot_losses(loss1, loss2, loss3, epochs):
    plt.plot(range(len(loss1)), loss1, label="Net1 Loss")
    plt.plot(range(len(loss2)), loss2, label="Net2 Loss")
    plt.plot(range(len(loss3)), loss3, label="Net3 Loss")

    plt.title(f"Loss per Iteration")
    plt.xlabel(f"Iterations over {epochs} epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.show()

def display_confusion_matrix(conf, class_list, accuracy):
    sns.heatmap(conf, xticklabels=class_list, yticklabels=class_list, annot=True)
    plt.xlabel(f"True Label \n Accuracy: {accuracy}")
    plt.ylabel("Predicted Label")
    plt.show()

def task1(train_loader, test_loader, epochs, num_classes, path_to_network=None):
    net1 = HW4Net(task="task1")
    net1_loss = train(net1, epochs=epochs, lr=1e-3, betas=(0.9, 0.99), dataloader=train_loader, save_to_path=path_to_network)
    confusion_matrix, accuracy = test(net1, path_to_network, test_loader, num_classes)
    return net1_loss, confusion_matrix, accuracy

def task2(train_loader, test_loader, epochs, num_classes, path_to_network=None):
    net2 = HW4Net(task="task2")
    net2_loss = train(net2, epochs=epochs, lr=1e-3, betas=(0.9, 0.99), dataloader=train_loader, save_to_path=path_to_network)
    confusion_matrix, accuracy = test(net2, path_to_network, test_loader, num_classes)
    return net2_loss, confusion_matrix, accuracy

def task3(train_loader, test_loader, epochs, num_classes, path_to_network=None):
    net3 = HW4Net(task="task3")
    net3_loss = train(net3, epochs=epochs, lr=1e-3, betas=(0.9, 0.99), dataloader=train_loader, save_to_path=path_to_network)
    confusion_matrix, accuracy = test(net3, path_to_network, test_loader, num_classes)
    return net3_loss, confusion_matrix, accuracy


if __name__ == "__main__":
    train_root = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/Train"
    val_root = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/Val"
    class_list = ["airplane", "bus", "cat", "dog", "pizza"]
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    epochs = 7

    train_dataset = GenerateDataset(train_root, class_list, transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=2, shuffle=True)

    test_dataset = GenerateDataset(val_root, class_list, transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, num_workers=2, shuffle=True)

    # Task 1
    net1_loss, conf1, acc1 = task1(train_dataloader, test_dataloader, epochs, len(class_list), path_to_network="/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/networks/net1.pth")
    print(f"The accuracy of this network is {acc1}")
    display_confusion_matrix(conf1, class_list, acc1)
    
    # Task 2
    # net2_loss, conf2, acc2 = task2(train_dataloader, test_dataloader, epochs, len(class_list), path_to_network="/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/networks/net2.pth")
    # print(f"The accuracy of this network is {acc2}")
    # display_confusion_matrix(conf2, class_list, acc2)

    # # Task 3
    # net3_loss, conf3, acc3 = task3(train_dataloader, test_dataloader, epochs, len(class_list), path_to_network="/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/networks/net3.pth")
    # print(f"The accuracy of this network is {acc3}")
    # display_confusion_matrix(conf3, class_list, acc3)

    # Display training loss for all networks
    # plot_losses(net1_loss, net2_loss, net3_loss, epochs=epochs)
