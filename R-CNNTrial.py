# Contributor: Neo Dizdar

import torchvision
import torch
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import os
import PIL
import cv2
from xml.dom.minidom import parse
import numpy as np 
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
print(model)
checkpoint = torch.load('galrcnn.pth')
model.load_state_dict(checkpoint)
#num_classes = 3  # 1 class (person) + background
#in_features = model.roi_heads.box_predictor.cls_score.in_features
#model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.eval()

Categories = ['__background__', 'Galaxy', 'Cavity']




def get_prediction(img_path, threshold):
    img = PIL.Image.open(img_path).convert('RGB') # Load the image
    transform = T.Compose([T.ToTensor()]) 
    img = transform(img) # Apply the transform to the image
    print(img)
    pred = model([img]) # Pass the image to the model
    print(pred)
    pred_class = [Categories[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    #print(pred_boxes)
    pred_score = list(pred[0]['scores'].detach().numpy())
    #print(pred_score)
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    #print("Here")
    #print(pred_score)
    return pred_boxes, pred_class,pred_score



def object_detection_api_boxes(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls, pred_score = get_prediction(img_path, threshold) 
    
    img = cv2.imread(img_path) 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        pred_score[i] = str(round(pred_score[i]*100,2)) + '%'
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i] , boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        cv2.putText(img,pred_score[i], boxes[i][1],  cv2.FONT_HERSHEY_SIMPLEX, text_size - 0.5, (255,0,0),thickness=1)
        print("Theres is a ",pred_score[i]," chance that a", pred_cls[i],"is located between ", boxes[i][0], "and ", boxes[i][1])
    plt.figure(figsize=(12.80,7.20)) # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('RCNNTrialTest.png', bbox_inches= 'tight', dpi=200, pad_inches = -0.1)
    plt.show()

def object_detection_api_circles(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls, pred_score = get_prediction(img_path, threshold) 
    
    img = cv2.imread(img_path) 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        pred_score[i] = str(round(pred_score[i]*100,2)) + '%'

        topcorner = boxes[i][0]                                #Ellipse altering code
        bottomcorner = boxes[i][1]
        centerx = (topcorner[0] + bottomcorner[0])/2
        centery = (topcorner[1] + bottomcorner[1])/2
        center = (int(centerx),int(centery))
        radii = (int(bottomcorner[0]), int(bottomcorner[1]))
        axislength = tuple(np.subtract(radii, center))

        cv2.ellipse(img, center, axislength,0, 0, 360, (0, 255, 0), thickness=rect_th) # Draw ellipse with the coordinates
        cv2.putText(img,pred_cls[i] , boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
        cv2.putText(img,pred_score[i], boxes[i][1],  cv2.FONT_HERSHEY_SIMPLEX, text_size - 0.5, (255,0,0),thickness=1) 
        print("Theres is a ",pred_score[i]," chance that a", pred_cls[i],"is located between ", boxes[i][0], "and ", boxes[i][1])
    plt.figure(figsize=(12.80,7.20)) # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('RCNNTrialTest.png', bbox_inches= 'tight', dpi=200, pad_inches = -0.1)
    plt.show()

object_detection_api_circles('./data/Galaxies/NNData/NGC5044ASINH0.png',threshold=0.4, rect_th=1, text_th=2, text_size=0.8)
