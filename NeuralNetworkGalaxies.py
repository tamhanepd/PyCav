
import torchvision
import torch
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from xml.dom.minidom import parse
import numpy as np 
import utils
from engine import train_one_epoch, evaluate

os.environ["CUDA_VISIBLE_DEVICES"]=""

num_classes = 3  #2 classes (Galaxy, Cavity) + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'Galaxy', 'Cavity']

class MarkDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "NNData"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "NNData", self.imgs[idx])
        #print(img_path)
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        #print(bbox_xml_path)
        img = Image.open(img_path).convert("RGB")        
        
        dom = parse(bbox_xml_path)                        #collecting data from XMLs
        data = dom.documentElement
        objects = data.getElementsByTagName('object')        

        boxes = []
        labels = []
        for object_ in objects:
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue
            if name == "Galaxy":
                labeller = 1        #needed to switch the label names into numerical values to work with pytorch 
            elif name == "Cavity":
                labeller = 2
            else:
                labeller = 0 
            labels.append(labeller)  
            
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)        #Seperating xml regions into x,y coords
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)        
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)  #iscrowd = 0 for polygons of object instances, 1 for Run length encoding
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        img = T.functional.to_tensor(img)

        return img, target
 
    def __len__(self):
        return len(self.imgs)


root = os.getcwd()   #This has to be the path to the folder where NNData and Annotations are stored
#testroot = r'Testing Dataset Path'

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

dataset = MarkDataset(root)             #combines the images with the xml files
dataset_test = MarkDataset(root)
indices = torch.randperm(len(dataset)).tolist()                   #Randomizes the list of data to then split it into train and test data
dataset = torch.utils.data.Subset(dataset, indices[:-5])          #Training dataset includes every image and xml file except for 5      
dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

data_loader = torch.utils.data.DataLoader(                        #Sets up batch size of how the data is read through the system, alter for different results
    dataset, batch_size=2, shuffle=True, 
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False,
    collate_fn=utils.collate_fn)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]

# Learning Rate and Stochastic Gradient Descent is set and altered here
optimizer = torch.optim.SGD(params, lr=0.005, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

num_epochs = 20                       #Alter this value for varying results

for epoch in range(num_epochs):
    #train_one_epoch function takes both images and targets to device
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

    #update the learning rate
    lr_scheduler.step()

    #evaluate on the test dataset    
    evaluate(model, data_loader_test, device=torch.device('cpu'))    
    
    print('')
    print('==================================================')
    print('')

print("That's it!")
torch.save(model.state_dict(), 'galrcnn.pth')
#print(model)