# eva4

#Assignment-2/Session4
#Tril1: 1 lac params, No Batch Normalization
1. Reduced the params to one lac. Kernel count from 4,8,16,32
2. Got the accuracy  of 99.38%

#Trial2: Used Batch Normalization 
1. This time reduced the params to 27000
2. Used 4,8,16,32 kernel counts
3. Used batch normalization .. conv->relu->batchNorm
2. Got the accuracy of 99.35%
#Trial 3: With Batch Normalization
1. This time tried with less than 16000
2. used kernel counts 8,16,32
3. Got 99.35% accuracy
#Trial4: With Droputs
1. Accuracy dropped to 98 here. Something went wrong here.
