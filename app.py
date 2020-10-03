
from __future__ import print_function, division
import streamlit as st
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
import os
import cv2
from PIL import Image
import pdb
import time
import copy
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip,VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90,Transpose,RandomBrightnessContrast,RandomCrop)
from albumentations.pytorch import ToTensor
import albumentations as albu
import matplotlib.image as mpi
#import segmentation_models_pytorch as smp
from pathlib import Path
from sklearn.metrics import recall_score
import gc
warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import torchvision.models as models
#from efficientnet_pytorch import EfficientNet
import pretrainedmodels
st.set_option('deprecation.showfileUploaderEncoding', False)

def get_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    list_transforms = []
    list_transforms.extend(
        [ 
            Resize(137,236,interpolation = 2),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

'''model = EfficientNet.from_name('efficientnet-b3')
in_features = model._fc.in_features
model._fc = nn.Linear(in_features,186)
'''

ckpt_path = "/content/drive/My Drive/resnet50/resnet50bengai.pth"
model = pretrainedmodels.__dict__['resnet50'](num_classes=1000,pretrained=None)
in_features = model.last_linear.in_features
model.last_linear = nn.Linear(in_features,186)
device = torch.device("cpu")
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

model.load_state_dict(state["state_dict"])

tfms = get_transforms()

data = pd.read_csv("/content/drive/My Drive/class_map.csv")

def predict(img):

    aug = tfms(image=img)
    img = aug["image"]

    img = torch.tensor(img, device=torch.device("cpu"), dtype=torch.float32)
    img = img[None,:,:,:]

    batch_preds = model(img.to(device)).detach().cpu().numpy()
    b1 = batch_preds[:,:168]
    b2 = batch_preds[:,168:179]
    b3 = batch_preds[:,179:]

    return [np.argmax(b1, axis=1), np.argmax(b2, axis=1), np.argmax(b3, axis=1)]


st.title("Bengali Handwritten Grapheme Classification")

st.sidebar.subheader("Visualize the Classes")
if(st.sidebar.checkbox(" Grapheme Root ")):
  st.write(data[data['component_type']=='grapheme_root'].component.unique())
if(st.sidebar.checkbox(" Vowel Diacritic ")):
  st.write(data[data['component_type']=='vowel_diacritic'].component.unique())
if(st.sidebar.checkbox(" Consonant Diacritic ")):
  st.write(data[data['component_type']=='consonant_diacritic'].component.unique())

st.sidebar.subheader("Visualize Sample Images From Dataset")
if(st.sidebar.checkbox("View Samples")):
  number = st.sidebar.slider("Number of samples", 1, 20, 1)
  lst = []
  imgs = os.listdir("/content/drive/My Drive/bengali_images")
  for i in range(number):
    img = cv2.imread("/content/drive/My Drive/bengali_images/"+imgs[i])
    lst.append(img)
  st.image(lst)

st.header("Classify the Bengali Handwritten Image into its grapheme, consonant and vowel")

uploaded_person = st.file_uploader("Upload an image", type="jpg")

if uploaded_person is not None:
    img = Image.open(uploaded_person)
    st.image(img, caption="User Input", width=100, use_column_width=False)
    img.save("image.jpg")

if st.button('Execute'):
    img = cv2.imread("./image.jpg")
    st.write("Classifying...")
    [g, v, c] = predict(img)
    execute_bar = st.progress(0)
    status_text = st.empty()
    for percent_complete in range(100):
        time.sleep(0.05)
        execute_bar.progress(percent_complete + 1)
        status_text.text("%d %s"%(percent_complete+1, ' % '))
    status_text.text(' Done Classifying! ') 
    root = data.loc[(data['component_type']=="grapheme_root")&(data['label']==int(g))].iloc[0, 2]
    vowel = data.loc[(data['component_type']=="vowel_diacritic")&(data['label']==int(v))].iloc[0, 2]
    consonant = data.loc[(data['component_type']=="consonant_diacritic")&(data['label']==int(c))].iloc[0, 2]
    st.write('**Root**    :  ', root)
    st.write('**Vowel**   :  ', vowel)
    st.write('**Consonant**  : ', consonant)

