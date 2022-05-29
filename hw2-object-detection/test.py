from download import download_data
from download import download_trained
import os
from model_helper import segment
from model_helper import ImageDataset

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import engine
import utils
import label_utils

def main():

    if not os.path.exists("drinks"):
        download_data()

    #for Google Colab (change path if local)
    #labels_test_path = "/content/drive/My Drive/drinks/labels_test.csv"
    #labels_train_path = "/content/drive/My Drive/drinks/labels_train.csv"
    labels_test_path = "drinks/labels_test.csv"
    labels_train_path = "drinks/labels_train.csv"
    test_dict, test_classes = label_utils.build_label_dictionary(labels_test_path)
    train_dict, train_classes = label_utils.build_label_dictionary(labels_train_path)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 4
    test_split = ImageDataset(test_dict, transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(
        test_split, 
        batch_size = 1, 
        shuffle = False, 
        num_workers = 2,
        collate_fn = utils.collate_fn)

    model = segment(num_classes)

    #pth file output from train.py
    if not os.path.exists("drinks-trained-weights.pth"):
        download_trained()
    if torch.cuda.is_available():
      #for google colab
      #model.load_state_dict(torch.load("/content/drive/MyDrive/coe-197/drinks-trained-weights.pth"))
      model.load_state_dict(torch.load("drinks-trained-weights.pth"))
    else:
      #for google colab
      #model.load_state_dict(torch.load("/content/drive/MyDrive/coe-197/drinks-trained-weights.pth", map_location=torch.device("cpu")))
      model.load_state_dict(torch.load("drinks-trained-weights.pth", map_location=torch.device("cpu")))

    #use either gpu or cpu
    model.to(device)

    #test model performance evaluation
    engine.evaluate(model, test_loader, device = device)
    
if __name__ == "__main__":
    main()
