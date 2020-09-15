# Male-Female Classification
![alt text](https://sun9-13.userapi.com/UGl901HP7Nha_3CXlFycZVMcVPXI603KwEOQPA/aGT9oNZPAsY.jpg)

## Table of Contents (Optional)

> If your `README` has a lot of info, section headers might be nice.

- [Installation](#installation)
- [Setup](#Setup)
- [Usage](#Usage)
- [Description of the solution](#Description)
- [Result](#Result)

## Installation

- `model/21_56.pth` and `run.py` required to get started

### Setup


Script requires the following to run:

  * absl-py==0.9.0
  * tqdm==4.48.2
  * torch==1.2.0
  * torchvision==0.4.0
  * Pillow==7.2.0

## Usage

It's **required** to specify path to image folder:
```bash
python run.py -f = "Path to image folder"
```
Also you should specify the path to the model:
```bash
python run.py -f = "Path to image folder" -m = "path to model"
```
Script will generate `process_results.json` file with the results of prediction
```json
{"000074.jpg": "female", "000083.jpg": "female"}
```

## Description
Load and split the data
```python
data = ImageFolder(root = directory , transform = transform)
targets = data.targets
train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size = 0.2, shuffle=True, stratify=targets)
```


This solution uses a pretrained network `ResNet34` with all layers frozen except Batch normalization layers because they are trained on mathematical expectation and variance from ImageNet.
```python
for name, param in model_resnet34.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
```
We are using fully conected head to predict two classes
```python
model_resnet34.fc = nn.Sequential(nn.Linear(model_resnet34.fc.in_features, 512),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(512, num_classes))
```

Train for 2 epoch
```python
optimizer = optim.Adam(model_resnet34.parameters(), lr=0.001)
train(model_resnet34, optimizer, torch.nn.CrossEntropyLoss(), train_loader, epochs=2, device=device)
```
##Result
* accuracy = 0.974 on the validation set
