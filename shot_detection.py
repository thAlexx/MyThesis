#!/usr/bin/python3


import cv2
from matplotlib import pyplot as plt
import numpy as np


# Initialize useful list
myFrames = []


class Frame:
    distance= 0
    similarity= 0
    img= None
    s= None
    hist=[]
    index= -1



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



# Function that finds the similarity between frames
def FindSimilarity(hist, last_hist):
    # Initialize parameters
    s = 0
    h = 0
    x2=0


    # Compute s,h,x2
    for i in range(0, 256):
        s += min(hist[i], last_hist[i])      # s: keeps the sum of minimum(hist[i],last_frame_hist[i]) for each histogram value (0,..,256)
        h += last_hist[i]                    # h: keeps the sum of last_frame_hist[i] for each histogram value (0,..,256)
        temp_max= max(hist[i], last_hist[i]) # temp_max: keeps the maximum value between hist[i], last_frame_hist[i]
        if temp_max >0:
            x2+= ((hist[i]-last_hist[i])*(hist[i]-last_hist[i]))/ max(hist[i], last_hist[i])   # x2 = [(hist[i]-last_hist[i])^2] / temp_max => normalized diastance


    similarity = s / h
    distance = x2
    
    return similarity, distance



# Function that corresponds frames to a camera
def FramesToCameras(list_indices, common_camera):
    
    # Initialize parameters
   # camera_frame = np.zeros(len(myFrames))
    camera_frame = []
    cam_counter = 1
    j = 0
    k = 0
    
    # Fill an array of length = len(frames_of_video) with the corresponding camera
    for i in range(0, len(myFrames)):
    
        # If there are frames from both "new" and already found cameras then:
        if (k < len(common_camera)) and (j < len(list_indices)):
    
    
           # If this frame corresponds to a "new" camera then fill it with values from "list_indices" - list
           if (list_indices[j] < common_camera[k][0]):           
               if (i < list_indices[j]):
                  #camera_frame[i] = cam_counter
                  camera_frame.append(cam_counter)
               elif (i == list_indices[j]):
                  #camera_frame[i] = cam_counter
                  camera_frame.append(cam_counter)
                  cam_counter += 1
                  j += 1     
               
           # If this frame corresponds to a camera already found, then fill it with values from "common_camera" - tuple
           else:
               if (i < common_camera[k][0]):
                  #camera_frame[i] = camera_frame[common_camera[k][1]]
                  camera_frame.append(camera_frame[common_camera[k][1]])
               elif (i == common_camera[k][0]):
                  #camera_frame[i] = camera_frame[common_camera[k][1]]
                  camera_frame.append(camera_frame[common_camera[k][1]])
                  k += 1

        # If there are frames only from "new" or already found cameras then:
        else:
           
           # If this frame corresponds to a "new" camera then fill it with values from "list_indices" - list
           if (j < len(list_indices)):
               if (i < list_indices[j]):
                  #camera_frame[i] = cam_counter
                  camera_frame.append(cam_counter)
               elif (i == list_indices[j]):
                  #camera_frame[i] = cam_counter
                  camera_frame.append(cam_counter)
                  cam_counter += 1
                  j += 1 
                  
           # If this frame corresponds to a camera already found, then fill it with values from "common_camera" - tuple           
           if (k < len(common_camera)):
               if (i < common_camera[k][0]):
                  #camera_frame[i] = camera_frame[common_camera[k][1]]
                  camera_frame.append(camera_frame[common_camera[k][1]])
               elif (i == common_camera[k][0]):
                  #camera_frame[i] = camera_frame[common_camera[k][1]]
                  camera_frame.append(camera_frame[common_camera[k][1]])
                  k += 1

           # This case corresponds to last "new" camera
           if (k == len(common_camera)) and (j == len(list_indices)):
               
               camera_frame.append(cam_counter)
               

    print(camera_frame)
    print(len(camera_frame))


    return camera_frame


# Function that determines when there is shot change
def similarThresholdWay(result_path):
   
    # Initialize parameters
    threshold = 0.795
    threshold2 = 0.75
    num_camera = 0
    common_camera = []
    list_indices = []
    for i in range(0, len(myFrames) - 1):
        
        # Consider shot change if Frame with similarity <= threshold
        if myFrames[i].similarity <= threshold:
           
            # If it's the first time a shot change found then increase the number of cameras and store the index of last frame 1st camera
            if num_camera == 0:
               num_camera += 1
               list_indices.append(i)

            else:

               list_similarity = []
               for j in range(0, len(list_indices)):
                   
                   # Find similarity between first frame of new camera shot and last frames from cameras that we have already found
                   sim, dist = FindSimilarity(myFrames[i+1].hist, myFrames[list_indices[j]].hist)
                   
                   # Keep a list with the similarities of Frame 'i' and the last frame of each shot change
                   list_similarity.append(sim)
                   
               # Find the maximum similarity
              # max_similarity = max(list_similarity)             
               max_similarity = list_similarity[0]
               index = list_indices[0]
               for j in range(1, len(list_similarity)):
                   if max_similarity < list_similarity[j]:
                      max_similarity = list_similarity[j]
                      index = list_indices[j]
               
               
               # If maximum similarity <= threshold then there is a new camera and we store the last frame
               if max_similarity <= threshold2:
                  num_camera += 1
                  list_indices.append(i)
               
               else:
                  common_camera.append((i, index))



    # Print (last) frames of each camera found
    #for i in range(0, len(list_indices)):
    #     plt.imshow(myFrames[list_indices[i]].img)
    #     plt.show()


                      
           
    print("TOTAL NUMBER OF CAMERAS FOUND = ", num_camera)

    camera_frame = FramesToCameras(list_indices, common_camera)
    
    #for i in range(0, len(myFrames)):
    #    if camera_frame[i] == 4:
    #      plt.imshow(myFrames[i].img)
    #      plt.show()
    
    
    print(list_indices)
    print(common_camera)
    


# Function that finds similarity between two consecutive frames through histograms
def ColorHistDetect(VideoName):

    # Path where results are stored
    result_path = "/home/alexandros/Sxolh/Diplwmatikh/similarResult"

    # Read one frame at a time
    cap = cv2.VideoCapture(VideoName)  
    index = -1

    while (cap.isOpened()):

        ret, frame = cap.read()
        
        # If there aren't any more frames, stop reading
        if ret == False:
            break
        index += 1
        if index >= 1800:
            break

        img = frame
        temp_frame = Frame()          # Object of class Frame
        temp_frame.index = index      # Store its index (f.e. for 1st frame index = 0)
        temp_frame.img = img.copy()   # Store the frame
        myFrames.append(temp_frame)   # Keep a list of Frame objects 


        # Use gaussian filter for each frame 
        img = cv2.GaussianBlur(img, (9, 9), 0.0)  

        
        # Convert frame from BGR to HSV
        img= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Get h,s,v components of frame
        h, s,v = cv2.split(img)
        
        # Calculate Histogram of h,s,v components
        hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])  
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])  
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])  
        
        # Calculate total Histogram of each frames using weights for each component
        weight= [0.5,0.3,0.2]
        hist = weight[0] * hist_h + weight[1] * hist_s + weight[2] * hist_v


        # Store histogram in Frame object
        temp_frame.hist = hist

        # If there 2 or more Frame objects stored in list, then check if shot has changed
        if len(myFrames) >= 2:
        
            # Keep last frame and its histogram
            last_Frame = myFrames[len(myFrames) - 2]
            last_hist = last_Frame.hist


            # Find similarity and distance
            last_Frame.similarity, last_Frame.distance = FindSimilarity(hist, last_hist)              
            
    cap.release()


    # Call this function in order to determine when there is shot change, from similarity between two conecutive frames
    similarThresholdWay(result_path)    
    



# Get data filenames and labels
data_names = data_names()


# Run the shot change detection from here for video given in "data_names[1]" path
ColorHistDetect(data_names[1])
