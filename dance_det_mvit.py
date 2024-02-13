#!/usr/bin/python3

import skvideo.io
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
from torch import nn
import torch.nn.functional as nnf
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report

import matplotlib.pyplot as plt



np.float = float
np.int = int




class My_Model(nn.Module):

     def __init__(self):
           super(My_Model, self).__init__()
        
           # Pick a pretrained model and load the pretrained weights
           model_name = "mvit_base_32x3"
           self.model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)


        
     def forward(self, inputs):
           x = self.model(inputs)
           x = self.classifier_layer(x)
           return x





# Function that returns shuffled train labels and data name folders
def data_names():

    # Parent folder
    dire = "/home/alexandros/Sxolh/Diplwmatikh/test_dataset/video"

    # Construct Data Namefiles
    data_names = []

    for i in range(1, 3):
        data_dir = dire + str(i) + ".mp4"
        data_names.append(data_dir)

    return data_names





# Function that applys dance detection in non-labeled data.
def dance_detection(video_data, model):
 
    # Initialize Predicted Labels
    y_pred = []

    # Regularization layers (Dropout etc.)
    model.eval()

    # Get model's device ID
    device = next(model.parameters()).device

    # Because in evaluation mode we don't need the gradients, we do everything with torch.no_grad()
    with torch.no_grad():
        # for i in range(0, len(video_data)):

            # a) Move the inputs to the desired device
            inputs = video_data["video"]
            inputs = inputs.to(device)
            

            # b) Forward pass (model(input))
            y_forward = model(inputs[None, ...])
            
            
            # Find the probability of most likely category
            prob = nnf.softmax(y_forward, dim=1)
            top_p, top_class = prob.topk(1, dim = 1)
            
            
            # c) Predictions (argmax{posteriors})
            if (top_p < 0.99):
                y_forward_argmax = torch.tensor([0]).to(device)       
            else:
                y_forward_argmax = torch.argmax(y_forward, dim=1)
            
            # Print probability of first class  
          #  print(top_p)
          #  print(top_class)
          #  print("")
            

            # d) Store Predictions, Gold Labels and Batch Loss
            y_pred.append(y_forward_argmax.cpu().numpy())

    return y_pred
    

# Get data filenames and labels
data_names = data_names()


test_indices = [0,1]



# Load trained Model
model = torch.load('/home/alexandros/Sxolh/detection_model_mvit')


# Device on which to run the model
# Set to cuda to load on GPU
device = "cuda"


# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()




####################
# MViT transform
####################

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 64
num_frames = 32         # Model takes as input [1, 3, x, side_size, side_size] pictures, where x=num_frames/4
sampling_rate = 2
frames_per_second = 64  # I can change only this parameter, in order to fix duration of the clip. At this moment clip_duration=3sec
crop_size = 224
side_size = 256



# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(
                crop_size=(crop_size, crop_size)
            )
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second


#########



#true_labels = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1])



# Initialize parameters
test_predictions = []
start = 0
i = 0
w = 0
num_video = 1
while(w == 0):

   # Select the duration of the clip to load by specifying the start and end duration
   # The start_sec should correspond to where the action occurs in the video
   start_sec = start
   end_sec = start_sec + clip_duration


   # Initialize an EncodedVideo helper class
   test_videos = EncodedVideo.from_path(data_names[test_indices[num_video]])
              
   # Load the desired clip
   test_video_data = test_videos.get_clip(start_sec=start_sec, end_sec=end_sec)

   
   if (test_video_data['video'] == None):
      w = 1
   else:           
      
      # Apply a transform to normalize the video input
      test_video_data = transform(test_video_data)
         
         
      # Apply our Model for Dance Detection in unlabeled data    
      t_preds = dance_detection(test_video_data, model)
   
     
      # Put results into a list
      test_predictions.extend(t_preds)


      # Start from where the previous clip ended
      start = end_sec 
      i += 1


# Concatanate Predicted and True Results for Test Set
test_predicted_values = np.concatenate(test_predictions, axis=0)

 
# Print predicted values    
print(test_predicted_values)
#print(true_labels)



# Print results
#print(classification_report(true_labels, test_predicted_values))
#print("")
#print("")
#print("Accuracy in test set = ", accuracy_score(true_labels, test_predicted_values))



filename = "results_of_video" + str(test_indices[num_video]+1) + ".txt"


# Write results in a .txt file
f = open(filename, "a")



start = 0

for j in range(0, i):
   start_sec = start
   end_sec = start_sec + clip_duration

   if (test_predicted_values[j] == 0):
       f.write(repr(start_sec) + " - " + repr(end_sec) + " -> " + repr(0) + "\n")   # NO DANCING
   
   else:
       f.write(repr(start_sec) + " - " + repr(end_sec) + " -> " + repr(1) + "\n")   # DANCING
   
   start = end_sec    

f.close()
