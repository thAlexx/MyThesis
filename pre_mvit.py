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
    dire = "/home/alexandros/Sxolh/Diplwmatikh/test/test_data/video"

    # Read Test Labels
    file = open('/home/alexandros/Sxolh/Diplwmatikh/test/test_labels.txt','r')
    content = file.readlines()
    file.close()

    # Convert to a list of integers
    labels = []
    for i in range(0, len(content)):
        labels.append(int(content[i]))

    # Construct Data Namefiles
    data_names = []

    for i in range(1, 231):
        data_dir = dire + str(i) + ".mp4"
        data_names.append(data_dir)

    return data_names, labels


def split_data(data_names, labels):
    # Initialize dataset parameters
    dataset_size = 230
    val_size = 40
    test_size = 30
    

    # kanw shuffle exontas arxikopoisei to seed, settarismeno sto debug prokeimenou na exw idio apotelesma
    indices = list(range(1, 1 + dataset_size))
    np.random.seed(0)
    np.random.shuffle(indices)  
    # indices px [1,4,0,2,5 ...]

    # Define train, test and val indices
    train_indices = indices[test_size + val_size:]
    val_indices = indices[test_size : test_size + val_size]
    test_indices = indices[:test_size]

    return train_indices, val_indices, test_indices    


# Function that trains the model for an epoch.
def train_dataset(_epoch, video_data, labels, model, loss_function, optimizer):

        # Regularization layers (Dropout etc.)
        model.train()
        loss_now = 0.0

        # Get model's device ID
        device = next(model.parameters()).device

        # Create a 1D tensor for labels
        label = torch.tensor([labels]).to(device)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = inputs.to(device)

        # a) Zero the gradients, before each batch.
        optimizer.zero_grad()

        # b) Forward pass (model(input))
        y_preds = model(inputs[None, ...])

        # c) Compute loss (loss_function(y,model(input)))
        loss = loss_function(y_preds, label)

        # d) Backward pass (compute gradient wrt model parameters)
        loss.backward()

        # e) Update weights
        optimizer.step()
             
        # Store loss
        loss_now += loss.data.item()
    
        
        return loss_now




# Function that evaluates the model in an epoch.
def evaluate_dataset(video_data, labels, model, loss_function):
 
    # Initialize Predicted Labels
    y_pred = []
    # Initialize Gold Labels
    y_gold = [] 

    # Regularization layers (Dropout etc.)
    model.eval()
    loss_now = 0.0

    # Get model's device ID
    device = next(model.parameters()).device

    # Because in evaluation mode we don't need the gradients, we do everything with torch.no_grad()
    with torch.no_grad():

            # Create a 1D tensor for labels
            label = torch.tensor([labels]).to(device)

            # a) Move the inputs to the desired device
            inputs = video_data["video"]
            inputs = inputs.to(device)

            # b) Forward pass (model(input))
            y_forward = model(inputs[None, ...])
                       
            
            # c) Compute loss (loss_function(y, model(input))), to compare train/test loss.
            loss = loss_function(y_forward, label)
            
            
            # d) Predictions (argmax{posteriors})
            y_forward_argmax = torch.argmax(y_forward, dim=1)
            

            # e) Store Predictions, Gold Labels and Batch Loss
            y_pred.append(y_forward_argmax.cpu().numpy())
            y_gold.append(label.cpu().numpy())

            # Store loss
            loss_now += loss.data.item()

    return loss_now, (y_gold, y_pred)




# Get data filenames and labels
data_names, labels = data_names()

# Split data in train, validation and test
train_indices, val_indices, test_indices = split_data(data_names, labels)





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



"""


# First time that you want to run the code, then
# Create an object of "My_Model" Class
# Then put comments until line: "# Load trained Model", and load the trained model from the path given below
model = My_Model()


# Freeze all layers of Pretrained Network, except...
for param in model.parameters():
    param.requires_grad = False


# Unfreeze last layer of Pretrained Network
for param in model.model.head.parameters():
    param.requires_grad = True



# Add one last layer in order to match our goals of dance detection
model.classifier_layer = nn.Sequential(
            nn.Linear(400 , 32),
            nn.Dropout(0.1),
            nn.Linear(32 , 2)
        )
"""

### If you run code for first time or run for test set put comments
# Load trained Model
model = torch.load('/home/alexandros/Sxolh/best_detection_model_mvit')


# Device on which to run the model
# Set to cuda to load on GPU
device = "cuda"


# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()





# Compute the Cross Entropy Loss between Input and Target (Loss-Function)
loss_function = nn.CrossEntropyLoss().to(device)

# Define Adam optimizer in order to implement adaptive learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)





# Train and Evaluate our Model "epochs"-times
### TOTAL Train Loss =   0.27192644921452525   (TRAINING, epoch=1)
### TOTAL Train Loss =   0.14928535074382254   (TRAINING, epoch=2)
### TOTAL Train Loss =   0.1090476236419477    (TRAINING, epoch=3)
### TOTAL Train Loss =   0.10075352325707594   (TRAINING, epoch=4)
### TOTAL Train Loss =   0.14605680147551017   (TRAINING, epoch=5)
### TOTAL Train Loss =   0.11531605078479965   (TRAINING, epoch=6)
### TOTAL Train Loss =   0.15515278676536115   (TRAINING, epoch=7)
### TOTAL Train Loss =   0.19657474763998034   (TRAINING, epoch=8)
### TOTAL Train Loss =   0.1376471291403253    (TRAINING, epoch=9)
### TOTAL Train Loss =   0.12319425453786513   (TRAINING, epoch=10)



### TOTAL Train Loss =   0.07741701918024332  (EVALUATION, epoch=1),    ACCURACY = 0.9790794979079498
### TOTAL Train Loss =   0.12438409973843895  (EVALUATION, epoch=2),    ACCURACY = 0.9728033472803347    
### TOTAL Train Loss =   0.0920453671548579   (EVALUATION, epoch=3),    ACCURACY = 0.9811715481171548
### TOTAL Train Loss =   0.06201470699527654  (EVALUATION, epoch=4),    ACCURACY = 0.9895397489539749
### TOTAL Train Loss =   0.1261029197855288   (EVALUATION, epoch=5),    ACCURACY = 0.9790794979079498
### TOTAL Train Loss =   0.0814140102625763   (EVALUATION, epoch=6),    ACCURACY = 0.9874476987447699
### TOTAL Train Loss =   0.10221867261269901  (EVALUATION, epoch=7),    ACCURACY = 0.9853556485355649
### TOTAL Train Loss =   0.06201470699527654  (EVALUATION, epoch=8),    ACCURACY = 0.9895397489539749
### TOTAL Train Loss =   0.06201470699527654  (EVALUATION, epoch=9),    ACCURACY = 0.9895397489539749
### TOTAL Train Loss =   0.06201470699527654  (EVALUATION, epoch=10),   ACCURACY = 0.9895397489539749


### TOTAL VALIDATION LOSS =  0.10595020302785355  (epoch=1)   ("IMPROVEMENT"),       ACCURACY = 0.9495798319327731
### TOTAL VALIDATION LOSS =  0.10944642517210212  (epoch=2)   ("NO IMPROVEMENT"),    ACCURACY = 0.9747899159663865
### TOTAL VALIDATION LOSS =  0.1587094763697353   (epoch=3)   ("NO IMPROVEMENT"),    ACCURACY = 0.957983193277311
### TOTAL VALIDATION LOSS =  0.08742544097525014  (epoch=4)   ("IMPROVEMENT"),       ACCURACY = 0.9831932773109243
### TOTAL VALIDATION LOSS =  0.11240626214881178  (epoch=5)   ("NO IMPROVEMENT"),    ACCURACY = 0.9747899159663865
### TOTAL VALIDATION LOSS =  0.11230779968026364  (epoch=6)   ("NO IMPROVEMENT"),    ACCURACY = 0.9831932773109243
### TOTAL VALIDATION LOSS =  0.09239331068036563  (epoch=7)   ("NO IMPROVEMENT"),    ACCURACY = 0.9831932773109243
### TOTAL VALIDATION LOSS =  0.08742544097525014  (epoch=8)   ("NO IMPROVEMENT"),    ACCURACY = 0.9831932773109243
### TOTAL VALIDATION LOSS =  0.08742544097525014  (epoch=9)   ("NO IMPROVEMENT"),    ACCURACY = 0.9831932773109243
### TOTAL VALIDATION LOSS =  0.08742544097525014  (epoch=10)  ("NO IMPROVEMENT"),    ACCURACY = 0.9831932773109243


### TOTAL TEST LOSS =  0.0029960994074660027 (epoch=7)       ACCURACY = 1.0
### TOTAL TEST LOSS =  0.0029960994074660027 (BEST_MODEL)    ACCURACY = 1.0




# Decide what you want to do
# training                  -> w = 0, z = 1
# evaluate training set     -> w = 1, z = 1
# evaluate validation set   -> w = 2, z = 1
# evaluate test set         -> w = 3, z = 0
# do nothing of the above   -> w = 3, z = 1  

## Accepted combinations: (w, z) = (0-2, 1),  (3, 0)
w = 2
z = 1


# Store in variable "minimum_val_loss" the minimum total loss found in evaluation of training set in an epoch where there was improvement.
# If there was no improvement then: epochs_with_no_improve += 1, and keep the same "minimum_val_loss"


# Parameters for Early Stopping
n_epochs_stop = 3
minimum_val_loss = 0.08742544097525014
epochs_with_no_improve = 1



epochs = 1
for epoch in range(epochs):

    # TRAIN TRAINING SET
    if (w == 0):
       print("TRAINING in EPOCH = ", epoch+5)
       print("")
       print("")
       
       
       # Initialize parameters
       t_loss = 0
       counter = 0
       for j in range(0, len(train_indices)):
       
           # Decide how many clips of each video you are going to take (2 from the first category, and 3 from the second one)
           if (train_indices[j] == 95) or (train_indices[j] == 12) or (train_indices[j] == 97)  or (train_indices[j] == 147):
               k = 2
               counter += 1
           else:
               k = 3
               
               
           # Initialize the starting of the first clip
           start = 0
           
           # Initialize an EncodedVideo helper class
           train_videos = EncodedVideo.from_path(data_names[train_indices[j]-1])
           
           # Store Train Labels
           train_labels = labels[train_indices[j]-1]
           
           for i in range(0, k):
               
               # Initialize the starting and ending (secs) of clip taken from a video
               start_sec = start
               end_sec = start_sec + clip_duration

               # Load the desired clip
               train_video_data = train_videos.get_clip(start_sec=start_sec, end_sec=end_sec)
           
               if (train_video_data['video'] == 'None'):
                   break
               else:
               
                   # Apply a transform to normalize the video input
                   train_video_data = transform(train_video_data)


                   # Train the model for one epoch
                   train_loss = train_dataset(epoch, train_video_data, train_labels, model, loss_function, optimizer)
           
           
                   # Add loss found in last batch
                   t_loss = t_loss + train_loss
           
                   print("Arxeio No = ", j+1)
                   print("Clip No = ", i+1)
                   print("")
                   print("Train Loss Until Now = ", t_loss)
                   print("")
                   print("")
                   print("")
           
                   # Store my model in each iteration
                   torch.save(model, "/home/alexandros/Sxolh/detection_model_mvit")
               
               
                   # Start from where the previous clip ended
                   start = end_sec
           
           
       # Compute total train loss
       t_loss = t_loss / ((len(train_indices)-counter) * 3 + counter * 2)
       print("TOTAL Train Loss = ", t_loss)
       print("")
       print("")
       print("")
       print("Counter of 2-clips of a video is = ", counter)
           
           
    # EVALUATE TRAINING SET
    elif (w == 1):
         print("EVALUATION of TRAINING SET in EPOCH = ", epoch+5)
         print("")
         print("")
         print("")   
         
         
         # Initialize parameters
         counter = 0
         t_loss = 0
         gold = []
         predictions = []
         for j in range(0, len(train_indices)): 
         
         
             # Decide how many clips of each video you are going to take (2 from the first category, and 3 from the second one)
             if (train_indices[j] == 95) or (train_indices[j] == 12) or (train_indices[j] == 97)  or (train_indices[j] == 147):
                 k = 2
                 counter += 1
             else:
                 k = 3
               
             # Initialize the starting of the first clip
             start = 0
             
             # Initialize an EncodedVideo helper class
             train_videos = EncodedVideo.from_path(data_names[train_indices[j]-1])
             
             
             # Store Train Labels
             train_labels = labels[train_indices[j]-1] 
             
         
             for i in range(0, k):

                 # Initialize the starting and ending (secs) of clip taken from a video
                 start_sec = start
                 end_sec = start_sec + clip_duration


                 # Load the desired clip
                 train_video_data = train_videos.get_clip(start_sec=start_sec, end_sec=end_sec)
  
                     
                 # Apply a transform to normalize the video input
                 train_video_data = transform(train_video_data)
                     
              
                 # Evaluate performance of the Model on Trainning Set
                 train_loss, (y_train_gold, y_train_pred) = evaluate_dataset(train_video_data, train_labels, model, loss_function)
                 
                 
                 # Put results into a list
                 gold.extend(y_train_gold)
                 predictions.extend(y_train_pred)
             
                 # Add loss found in last batch
                 t_loss = t_loss + train_loss
                 
                 print("Arxeio No = ", j+1)
                 print("Clip No = ", i+1)
                 print("")
                 print("Total Train Loss Until Now = ", t_loss)
                 print("")
                 print("")
                 print("")
                 
                 # Start from where the previous clip ended
                 start = end_sec
                 

         # Compute total train loss        
         t_loss = t_loss / ((len(train_indices)-counter) * 3 + counter * 2)  

         print("TOTAL TRAIN LOSS = ", t_loss)
         print("")
         print("")
         print("Counter of 2-clips of a video is = ", counter)
         print("")
         print("")
  
             
         # Concatanate Predicted and True Results for Test Set
         predicted_values = np.concatenate(predictions, axis=0)
         true_values = np.concatenate(gold, axis=0)
             
             
         # Print true and predicted values    
         #print(true_values)
         #print(predicted_values)
         
         
         # Print results
         print(classification_report(true_values, predicted_values))
         print("")
         print("")
         print("Accuracy in training set = ", accuracy_score(true_values, predicted_values))
             
             
    # UPLOAD AND EVALUATE VALIDATION SET
    elif (w == 2):
         print("EVALUATION of VALIDATION SET in EPOCH = ", epoch+5)
         print("")
         print("")
         print("")
         
         
         # Initialize parameters
         counter = 0
         v_loss = 0
         gold = []
         predictions = []
         for j in range(0, len(val_indices)):



             # Decide how many clips of each video you are going to take (2 from the first category, and 3 from the second one)
             if (val_indices[j] == 95) or (val_indices[j] == 12) or (val_indices[j] == 97)  or (val_indices[j] == 147):
                 k = 2
                 counter += 1
             else:
                 k = 3
               
             # Initialize the starting of the first clip           
             start = 0


             # Initialize an EncodedVideo helper class
             val_videos = EncodedVideo.from_path(data_names[val_indices[j]-1])
             
             # Store Validation Labels
             val_labels = labels[val_indices[j]-1]                 
             
             for i in range(0, k):   
             
                 # Initialize the starting and ending (secs) of clip taken from a video
                 start_sec = start
                 end_sec = start_sec + clip_duration
             
                        
                 # Load the desired clip
                 val_video_data = val_videos.get_clip(start_sec=start_sec, end_sec=end_sec)

		         
                 # Apply a transform to normalize the video input
                 val_video_data = transform(val_video_data)
                     

                     
                 # Evaluate performance of the Model on Validation Set    
                 val_loss, (y_val_gold, y_val_pred) = evaluate_dataset(val_video_data, val_labels, model, loss_function)
             
                 # Put results into a list
                 gold.extend(y_val_gold)
                 predictions.extend(y_val_pred)
             
             
             
                 # Add loss found in last batch
                 v_loss = v_loss + val_loss
                 
                 print("Arxeio No = ", j+1)
                 print("Clip No = ", i+1)
                 print("")
                 print("Total Validation Loss Until Now = ", v_loss)
                 print("")
                 
                 # Start from where the previous clip ended
                 start = end_sec
                 
                 
         # Compute total validation loss
         v_loss = v_loss / ((len(val_indices)-counter) * 3 + counter * 2)
         
         print("TOTAL VALIDATION LOSS = ", v_loss)
         print("")
         print("Counter of 2-clips of a video is = ", counter)
         print("")
         print("")
         
         
         
         # Concatanate Predicted and True Results for Test Set
         predicted_values = np.concatenate(predictions, axis=0)
         true_values = np.concatenate(gold, axis=0)
             
             
         # Print true and predicted values    
         #print(true_values)
         #print(predicted_values)
         
         
         # Print results
         print(classification_report(true_values, predicted_values))
         print("")
         print("")
         print("Accuracy in validation set = ", accuracy_score(true_values, predicted_values))
         
         
         
         
         
         
         # Early Stopping
         if v_loss < minimum_val_loss:
             # Save the model
             torch.save(model, "/home/alexandros/Sxolh/best_detection_model_mvit")
             epochs_with_no_improve = 0
             minimum_val_loss = val_loss
             print("IMPROVEMENT")
             
         else:
             epochs_with_no_improve += 1
             print("NO IMPROVEMENT")
              
         if epochs_with_no_improve == n_epochs_stop:
             print('Early Stopping')
             break 
             
    else:
         print("")     
                 
# UPLOAD AND EVALUATE TEST SET
if (z == 0):     
         
         # Find accuracy in our model
         
         print("EVALUATION of TEST SET")
         print("")
         print("")
         print("")
         
         
         # Initialize parameters
         counter = 0   
         test_loss = 0
         test = []
         test_predictions = []
         for j in range(0, len(test_indices)):


             # Initialize an EncodedVideo helper class
             test_videos = EncodedVideo.from_path(data_names[test_indices[j]-1])

             # Store Validation Labels
             test_labels = labels[test_indices[j]-1]


             # Decide how many clips of each video you are going to take (2 from the first category, and 3 from the second one)
             if (test_indices[j] == 95) or (test_indices[j] == 12) or (test_indices[j] == 97)  or (test_indices[j] == 147):
                 k = 2
                 counter += 1
             else:
                 k = 3
               
             # Initialize the starting of the first clip           
             start = 0

             for i in range(0, k):  
                                  
                # Initialize the starting and ending (secs) of clip taken from a video  
                start_sec = start
                end_sec = start_sec + clip_duration                          
                
                
                # Load the desired clip
                test_video_data = test_videos.get_clip(start_sec=start_sec, end_sec=end_sec)
                        
                # Apply a transform to normalize the video input
                test_video_data = transform(test_video_data)
                 

                 
                # Evaluate performance of the Model on Validation Set    
                t_loss, (t, t_preds) = evaluate_dataset(test_video_data, test_labels, model, loss_function)
             
                # Put results into a list
                test.extend(t)
                test_predictions.extend(t_preds)
             
                # Add loss found in last batch
                test_loss = test_loss + t_loss
             
              
                print("Total TEST Loss Until Now = ", test_loss)
                print("")

                
                # Start from where the previous clip ended
                start = end_sec
                
             
         # Compute total test loss
         test_loss = test_loss / ((len(test_indices)-counter) * 3 + counter * 2)
         
         print("TOTAL TEST LOSS = ", test_loss)
         print("")
         print("Counter of 2-clips of a video is = ", counter)
         print("")
         print("")
         
        
         # Concatanate Predicted and True Results for Test Set
         test_predicted_values = np.concatenate(test_predictions, axis=0)
         test_true_values = np.concatenate(test, axis=0)
        
         
         # Print true and predicted values    
         print(test_true_values)
         print(test_predicted_values)
         
         
         # Print results
         print(classification_report(test_true_values, test_predicted_values))
         print("")
         print("")
         print("Accuracy in test set = ", accuracy_score(test_true_values, test_predicted_values))

else:
         print("")
