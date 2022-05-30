# CoE 197 - Keyword Spotting Using Transformer
Samson, Harvey S.

A functioning keyword spotting (KWS) system using a transformer based model.


## Installation requirements
```
pip install -r requirements.txt
```

## Model Training
The model will download automatically the drinks dataset and will proceed with the model training by running:
```
python train.py
```
At the end of training, a kws-best-acc.pt file will be generated where the highest test_acc value is stored. 

## Model Testing
The model will download the kws-best-acc.pt generated from the training and use it in this phase to make inferences. To run:
```
python kws-infer.py
```

Training module was executed in Google Colab while the inference part (GUI) is implemented in the local machine. The model is found to have a test_acc of 92.2 and test_loss of 0.36.

## Demo
The file kws-infer.py is GUI-based and can recognize words by using the computer microphone as an input. It can now make inferences from the words said by the user and terminates the program if the user says "stop".

## Other References
Some references I found useful when building the model:
* [Transformer Model](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/transformer/python/transformer_demo.ipynb)
* [KWS using PyTorch Lightning](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws_demo.ipynb)
* [KWS Inference](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws-infer.py)
* [Einops](https://github.com/arogozhnikov/einops)