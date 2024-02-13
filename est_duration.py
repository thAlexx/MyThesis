#!/usr/bin/python3


# Import useful libraries
import pandas as pd
import math
import numpy as np
np.float = float
np.int = int



# Read information from github
url = 'https://github.com/pxaris/lyra-dataset/blob/main/data/raw.tsv?raw=true'
data = pd.read_csv(url, sep='\t', header=0)


# URLS not available in youtube
url_not_used = ["vLTBEHw011s", "00SbCAkyXhQ", "_mUIVNdIJFU", "kJl_glSyJMo", "0cj8BNcAhg4", "EoJoqvZukgw",
                "Uijl65qxrrg", "Aws0Y3aLaIs", "CPLOoAZonTI", "U6oWHEVNvfY", "4yPS8r0qVTw", "V-8YVU5FnTo",
                "cYE9_TXCmHs", "q9e6UQh6d0c", ]


# Get which videos are danced
is_danced = data["is-danced"] == 1



def vid_duration_av(is_danced):

   # Find total duration of videos that are danced and are available
   tot_duration_available = 0
   count = 0
   for i in range(0,1570):
       if is_danced[i] == True:
           if data["youtube-id"][i] not in url_not_used:
               tot_duration_available += data["end-ts"][i] - data["start-ts"][i]
               count += 1


   print(count)
   
   
   # Estimate clips with only dancing in all available danced videos
   total_dancing_duration_available = ((tot_duration_available * 0.4)/60, (tot_duration_available * 0.45)/60)
   #print(total_dancing_duration_available)


   # Compute minutes and seconds
   frac, whole = math.modf(total_dancing_duration_available[0])
   mins1 = int(whole)
   secs1 = round(frac*60, 1)


   frac, whole = math.modf(total_dancing_duration_available[1])
   mins2 = int(whole)
   secs2 = round(frac*60, 1)

   return mins1, secs1, mins2, secs2


def vid_duration(is_danced):

   # Find total duration of videos that are danced
   tot_duration = 0
   count = 0
   for i in range(0,1570):
       if is_danced[i] == True:
          tot_duration += data["end-ts"][i] - data["start-ts"][i]
          count += 1


   print(count)
   
   # Estimate clips with only dancing in all danced videos
   total_dancing_duration = ((tot_duration * 0.4)/60, (tot_duration * 0.45)/60)
   #print(total_dancing_duration)



   # Compute minutes and seconds
   frac, whole = math.modf(total_dancing_duration[0])
   mins1 = int(whole)
   secs1 = round(frac*60, 1)


   frac, whole = math.modf(total_dancing_duration[1])
   mins2 = int(whole)
   secs2 = round(frac*60, 1)

   return mins1, secs1, mins2, secs2



# Estimate the total duration of dancing parts in available videos
mins1, secs1, mins2, secs2 = vid_duration_av(is_danced)


print("TOTAL ESTIMATION OF DANCING PARTS IN AVAILABLE VIDEOS:")
print("From: ", mins1, " minutes and ", secs1, " seconds", " to ", mins2, " minutes and ", secs2, " seconds")
print("")
print("")


# Estimate the total duration of dancing parts in all videos
mins1, secs1, mins2, secs2 = vid_duration(is_danced)

print("TOTAL ESTIMATION OF DANCING PARTS IN ALL VIDEOS:")
print("From: ", mins1, " minutes and ", secs1, " seconds", " to ", mins2, " minutes and ", secs2, " seconds")
