from download import download_data
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
    train_split = ImageDataset(train_dict, transforms.ToTensor())
    test_split = ImageDataset(test_dict, transforms.ToTensor())

    # train and test data loaders
    train_loader = torch.utils.data.DataLoader(
        train_split, 
        batch_size = 8, 
        shuffle = True, 
        num_workers = 2,
        collate_fn = utils.collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
        test_split, 
        batch_size = 1, 
        shuffle = False, 
        num_workers = 2,
        collate_fn = utils.collate_fn)

    model = segment(num_classes)

    #use either gpu or cpu
    model.to(device)

    # optimizer and lr params
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr = 0.01, momentum = 0.9, weight_decay = 0.0001)
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1, verbose = True)

    #60 epochs
    for epoch in range(60):
        engine.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq = 100)
        learning_rate_scheduler.step()

    engine.evaluate(model, test_loader, device = device)
    torch.save(model.state_dict(), "drinks-trained-weights.pth")
    
if __name__ == "__main__":
    main()
