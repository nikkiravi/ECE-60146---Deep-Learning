# Import Libraries
import numpy as np
import torch
import torchvision.transforms as tvt
import torch.utils.data 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
from torchvision.ops import complete_box_iou_loss
import pandas as pd
import cv2
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# GLOBAL VARIABLES
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
path_to_model = r"/content/drive/MyDrive/ECE 60146/HW5/model/"

################################# CREATING DATASETS ##############################
class GenerateDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform

    def __return_integer_encoding(self, category):
        categories = {"pizza": 0, "bus": 1, "cat": 2}
        return categories[category]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_info = self.df.iloc[idx]
        path_to_image = os.path.join(r"/content/drive/MyDrive/ECE 60146/HW5/", image_info["path_to_image"])
        image = Image.open(path_to_image)
        image = self.transform(image) if self.transform else image
        label = int(self.__return_integer_encoding(image_info["category"]))

        bbox = [image_info["x1"], image_info["y1"], image_info["x2"], image_info["y2"]]
        # bbox_with_pixel_coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float)

        return image, label, bbox_tensor
    
def get_training_dataloader():
    # Constants
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create PyTorch Datasets and Dataloader
    df = pd.read_csv(r"/content/drive/MyDrive/ECE 60146/HW5/train_data.csv")
    train_dataset = GenerateDataset(df, transform)
    #print(train_dataset[0][0].shape, train_dataset[0][1], train_dataset[0][2]); quit()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=2, shuffle=True)
    return trainloader

def get_testing_dataloader():
    # Constants
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create PyTorch Datasets and Dataloader
    df = pd.read_csv(r"/content/drive/MyDrive/ECE 60146/HW5/test_data.csv")
    test_dataset = GenerateDataset(df, transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, num_workers=2, shuffle=True)
    return testloader

############################## NETWORK ##############################
class ResnetBlock(nn.Module):
    # Inspired by Professor Kak's SkipBlock class
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(ResnetBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.downsample = downsample
        self.skip_connections = skip_connections

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if(self.downsample):
            self.downsampler = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        
        if(self.in_ch == self.out_ch):
            out = self.conv2(out)
            out = self.bn2(out)
            out = torch.nn.functional.relu(out)
        if(self.downsample):
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if(self.skip_connections):
            if(self.in_ch == self.out_ch):
                out = out + identity
            else:
                out = torch.cat((out[:,:self.in_ch,:,:] + identity, out[:, self.in_ch:, :, :] + identity), dim=1)
        return out

class HW5Net(nn.Module):
    # Inspired by Professor Kak's Loadnet2 
    def __init__(self, skip_connections=True, depth=16):
        super(HW5Net, self).__init__()
        self.skip_connections = skip_connections
        self.depth = depth // 2
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.skip64_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip64_arr.append(ResnetBlock(in_ch=64, out_ch=64, skip_connections=self.skip_connections))
        self.skip64ds = ResnetBlock(in_ch=64, out_ch=64, downsample=True, skip_connections=self.skip_connections)
        self.skip64to128 = ResnetBlock(in_ch=64, out_ch=128, skip_connections=self.skip_connections)
        self.skip128_arr = nn.ModuleList()
        for idx in range(self.depth):
            self.skip128_arr.append(ResnetBlock(in_ch=128, out_ch=128, skip_connections=self.skip_connections))
        self.skip128ds = ResnetBlock(in_ch=128, out_ch=128, downsample=True, skip_connections=self.skip_connections)

        self.fc1 = nn.Linear(in_features=32*32*128, out_features=3)
        # self.fc2 = nn.Linear(in_features=1000, out_features=3) # Outputs probability of three classes

        # Regression
        self.conv_seqn = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features=64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True))
        
        self.fc_seqn = nn.Sequential(nn.Linear(in_features=128*128*64, out_features=4)) # Outputs [x, y, x+w, y+h]
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))

        # Classification
        cls = x.clone()
        for idx, skip64 in enumerate(self.skip64_arr[:self.depth//4]):
            cls = skip64(cls)
        cls = self.skip64ds(cls)
        for idx, skip64 in enumerate(self.skip64_arr[self.depth//4:]):
            cls = skip64(cls)
        cls = self.bn1(cls)
        cls = self.skip64to128(cls)
        for idx, skip128 in enumerate(self.skip128_arr[:self.depth//4]):
            cls = skip128(cls)
        cls = self.bn2(cls)
        cls = self.skip128ds(cls)
        for idx, skip128 in enumerate(self.skip128_arr[self.depth//4:]):
            cls = skip128(cls)
        # print(cls.shape); quit()
        cls = cls.view(-1, 32 * 32 * 128)
        cls = self.fc1(cls)
        #cls = self.fc2(cls)

        # Regression for Bbox
        bbox = self.conv_seqn(x)
        # print(bbox.shape); quit()
        bbox = bbox.view(x.size(0), -1)
        bbox = self.fc_seqn(bbox)

        return cls, bbox

############################## TRAINING ##############################
def train(model, trainloader, path_to_model=path_to_model, mse=True, lr=1e-4, betas=(0.9, 0.99), epochs=7): 
    # Train function inspired by Professor Kak's run_code_for_training_with_CrossEntropy_and_MSE_Losses function
    print("Training Started")
    model = model.to(device)
    num_layers = len(list(model.parameters()))
    assert num_layers >= 50, f"number of layers greater that or equal to 50 expected, got: {num_layers}"
    print(f"Number of layers: {num_layers}")

    model_name = "model_mse.pth" if mse else "model_ciou.pth"
    path_to_model = os.path.join(path_to_model, model_name)

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss() if mse else complete_box_iou_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    labeling_loss_tally = []   
    regression_loss_tally = [] 

    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}")
        running_loss_labeling = 0.0
        running_loss_regression = 0.0
        for batch_idx, (inputs, labels, bbox) in enumerate(trainloader):
            print(f"Batch idx: {batch_idx}")
            inputs = inputs.to(device)
            labels = labels.to(device)
            bbox = bbox.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            output_label, output_bbox = outputs[0], outputs[1]

            label_loss = cls_criterion(output_label, labels)
            label_loss.backward(retain_graph=True) # Preserve this loss while calculating bbox loss
            if(mse):
                bbox_loss = reg_criterion(output_bbox, bbox)
            else:
                bbox_loss = reg_criterion(output_bbox, bbox, reduction="mean")
            bbox_loss.backward()
            optimizer.step()

            running_loss_labeling += label_loss.item()
            running_loss_regression += bbox_loss.item()

            if(batch_idx % 50 == 49):
                avg_loss_labeling = running_loss_labeling / float(50)
                avg_loss_regression = running_loss_regression / float(50)
                labeling_loss_tally.append(avg_loss_labeling)  
                regression_loss_tally.append(avg_loss_regression)    
                print(f"Epoch: {epoch}/{epochs}, Iteration: {batch_idx + 1}: Labeling Loss: {avg_loss_labeling}, Regression Loss:{avg_loss_regression}")
                print(f"-----Saving model-----")
                torch.save(model.state_dict(), path_to_model)
                
                running_loss_labeling = 0.0
                running_loss_regression = 0.0

    return labeling_loss_tally, regression_loss_tally

def test(model, testloader, path_to_model=path_to_model, mse=True, num_classes=3):
    model_name = "model_mse.pth" if mse else "model_ciou.pth"
    path_to_model = os.path.join(path_to_model, model_name)
    model.load_state_dict(torch.load(path_to_model))
    model = model.to(device)
    confusion_matrix = np.zeros((num_classes, num_classes))
    image_info = []

    with torch.no_grad():
        for inputs, labels, bbox in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            bbox = bbox.to(device)

            outputs = model(inputs)
            output_label = outputs[0]
            output_bbox = outputs[1].tolist()

            _, predicted = torch.max(output_label, dim=1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1
            
            for img, original_label, predicted_label, original_bbox, predicted_bbox in zip(inputs, labels, predicted, bbox, output_bbox):
                image_info.append({"Image": img, "Original Label": original_label, "Predicted Label": predicted_label, 
                                   "Original Bbox": original_bbox, "Predicted Bbox": predicted_bbox})
            
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return confusion_matrix, accuracy, image_info

def plot_labeling_loss(labeling, epochs):
    iterations = range(len(labeling))
    figure = plt.figure(1)
    plt.plot(iterations, labeling, label="Labeling Loss")

    plt.title(f"Loss per Iteration")
    plt.xlabel(f"Iterations over {epochs} epochs")
    plt.ylabel("Loss")
    
    filename = "label_loss.jpg"
    plt.savefig(os.path.join("/content/drive/MyDrive/ECE 60146/HW5/Results", filename))

def plot_regression_loss(regression, epochs, mse):
    iterations = range(len(regression))
    figure = plt.figure(2)
    plt.plot(iterations, regression, label="Regression Loss")

    plt.title(f"Loss per Iteration")
    plt.xlabel(f"Iterations over {epochs} epochs")
    plt.ylabel("Loss")
    
    filename = "mse_loss.jpg" if mse else "ciou_loss.jpg"
    plt.savefig(os.path.join("/content/drive/MyDrive/ECE 60146/HW5/Results", filename))

def display_confusion_matrix(conf, accuracy, class_list=["pizza", "bus", "cat"], mse=True):
    figure = plt.figure(3)
    sns.heatmap(conf, xticklabels=class_list, yticklabels=class_list, annot=True)
    plt.xlabel(f"True Label \n Accuracy: {accuracy}")
    plt.ylabel("Predicted Label")

    filename = "conf_mse.jpg" if mse else "conf_ciou.jpg"
    plt.savefig(os.path.join("/content/drive/MyDrive/ECE 60146/HW5/Results", filename))

def draw_rectangles(image, bbox, predicted_bbox, label, predicted_label):
    # categories = {"pizza": 0, "bus": 1, "cat": 2}
    inverse_categories = {0: "pizza", 1: "bus", 2: "cat"}

    label = inverse_categories[int(label.item())]
    predicted_label = inverse_categories[int(predicted_label.item())]

    image = np.asarray(tvt.ToPILImage()(image))
    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(36, 255, 12), thickness=2)
    image = cv2.putText(image, label, (int(bbox[0]), int(bbox[1] - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)
    
    image = cv2.rectangle(image, (int(predicted_bbox[0]), int(predicted_bbox[1])), (int(predicted_bbox[2]), int(predicted_bbox[3])), color=(255, 45, 12), thickness=2)
    image = cv2.putText(image, predicted_label, (int(predicted_bbox[0]), int(predicted_bbox[1] - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 45, 12), thickness=2)
    return image

def display_bbox(image_info, mse=True):
    fig, ax = plt.subplots(3, 3)
    row, col = 0, 0
    for idx, details in enumerate(image_info):
        image = details["Image"]
        original_label = details["Original Label"]
        predicted_label = details["Predicted Label"]
        original_bbox = details["Original Bbox"]
        predicted_bbox = details["Predicted Bbox"]

        image = draw_rectangles(image, original_bbox, predicted_bbox, original_label, predicted_label)
        ax[row, col].imshow(image)
        row = row+1 if not ((idx + 1) % 3) else row
        col = 0 if not((idx + 1) % 3) else col+1
        
        if(row == 3 and col == 0):
            break
    
    filename = "bbox_mse.jpg" if mse else "bbox_ciou.jpg"
    plt.savefig(os.path.join("/content/drive/MyDrive/ECE 60146/HW5/Results", filename))

def display_input_bbox(dataloader):
    inverse_categories = {0: "pizza", 1: "bus", 2: "cat"}
    fig, ax = plt.subplots(3, 3)
    row, col = 0, 0
    for batch_idx, (images, labels, bboxs) in enumerate(dataloader):
        for idx in range(len(labels)):
            if(col == 3):
                row += 1
                col = 0
            if(row == 3):
                break
            image, label, bbox = images[idx], inverse_categories[int(labels[idx])], bboxs[idx]
            image = np.asarray(tvt.ToPILImage()(image))
            image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(36, 255, 12), thickness=2)
            image = cv2.putText(image, label, (int(bbox[0]), int(bbox[1] - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(36, 255, 12), thickness=2)
            ax[row, col].imshow(image)
            col += 1
        if(row == 3):
            break
    
    plt.savefig(os.path.join("/content/drive/MyDrive/ECE 60146/HW5/Results", "inputs.jpg"))

def parser():
    parser = argparse.ArgumentParser(description="Object Detection and Localization")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=8)
    parser.add_argument("--mse", action="store_true", help="Choosing MSE Loss")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    epochs = args.epochs
    regression = True if args.mse else False

    trainloader = get_training_dataloader()
    testloader = get_testing_dataloader()
    display_input_bbox(trainloader)

    model = HW5Net()
    label_loss, regression_loss = train(model, trainloader, epochs=epochs, mse=regression)
    plot_labeling_loss(label_loss, epochs=epochs)
    plot_regression_loss(regression_loss, epochs=epochs, mse=regression)

    confusion_matrix, accuracy, image_info = test(model, testloader, mse=regression)
    display_confusion_matrix(confusion_matrix, accuracy, mse=regression)
    display_bbox(image_info, mse=regression)
