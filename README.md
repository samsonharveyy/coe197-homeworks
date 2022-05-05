# CoE 197 - Object Detection using Faster RCNN
Samson, Harvey S.

An object detection model using the drinks dataset from https://bit.ly/adl2-ssd.

The model used to perform the object detection is Faster R-CNN MobileNetV3-Large 320 FPN. It works similarly to Faster R-CNN with ResNet-50 FPN backbone and is good for real
mobile-use cases.

## Paper and References
* [Papers with Code](https://paperswithcode.com/lib/torchvision/faster-r-cnn)
* [Arxiv](https://arxiv.org/abs/1506.01497v3)
* [PyTorch](https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html#fasterrcnn-mobilenet-v3-large-fpn)


## Installation requirements
```
pip3 install -r requirements.txt
```

## Model Training
The model will download automatically the drinks dataset and will proceed with the model training by running:
```
python3 train.py
```
At the end of training, a drinks-trained-weights.pth file will be returned which will be used for the testing part.

## Model Testing
The model will download again the drinks dataset if it is not found within the file directory. The drinks-trained-weights will also be 
downloaded if it is does not exist yet, which will be automatically fed to the model. To run:
```
python3 test.py
```

Both training and testing modules are run in Google Colab. From multiple tests, the model is found to perform considerably good as exhibited by accuracy
and precision values.

## Demo
A file called video_demo.ipynb is implemented using Google Colab and automatically downloads the pretrained weights to feed to the model for evaluation. A video is taken as an input which is spliced into frames to do the object detection, and converts it back to a .mp4 file. A sample video called video_demo.py is available in this repository for reference.

## Other References
Some references I found useful when building the model:
* [Deep Learning Helper - Supervised Learning: Object Detection](https://github.com/izzajalandoni/Deep-Learning-Helper/blob/main/SupervisedLearning/object_detection_mmdetection.ipynb)
* [Deep Learning Experiments - MLP](https://github.com/roatienza/Deep-Learning-Experiments/tree/master/versions/2022/datasets/python)
* [Torchvision](https://github.com/pytorch/vision)
* [OpenCV in Google Colab](https://github.com/oyyarko/opencv_arko/blob/master/OpenColab.ipynb)



