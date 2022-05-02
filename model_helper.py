import torch
import torchvision
import os
from torchvision import transforms
import utils
import label_utils

from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import engine 

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # get filenames
        key = list(self.dictionary.keys())[idx]
        # get all bounding boxes
        boxes = self.dictionary[key]
        boxes[:,1:3] = boxes[:,2:0:-1]
        labels = boxes[:,4]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = boxes[:,0:4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:,1])*(boxes[:,2]-boxes[:,0])

        image_id = torch.tensor([idx])
        # open the file as a PIL image
        img = Image.open(key)

        #faster r-cnn
        res = {}
        res['boxes'] = boxes
        res['labels'] = labels
        res['iscrowd'] = iscrowd
        res['area'] = area
        res['image_id'] = image_id
        
        if self.transform:
            img = self.transform(img)
        
        return img, res

def segment(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    #instance segmentation code snippet from Github
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
