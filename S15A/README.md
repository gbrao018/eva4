Project Scope Description: We are  going to solve the multi object training as a supervised learning problem. In short it is Monocular depth estimation and mask detection. We will be preparing our own dataset required for training.  We will discuss the step by step thought process right from dataset preperation, creating ground truths , optimal data storage strategy, data augmentation , loss functions, modal architecture, and performance.  In the end, we put more mphasis on various loss functions we use and why? And their performance.

What are we going to predict?. Given a background image and a overlayed image (foreground placed on the given background), we are going to predict the depth map for the overlayed image and mask for the foreground image. As we are attempting this as a supervised learning , given background and overlayed images, we will create the depth map and mask required for ground truths which are used in training phase.  

Hence, we are not classifying the objects present in the scene. We are not doing instance segmentation. We are only doing pixel-wise segmentation. We will assign the labels to each pixel whether it belongs to background or foreground 

Dataset Prepration:
	We are simulating a still camera environment which sees a background and then some portion of it is occluded when foreground objects come in between the camera and background. We are going to create 400000 overlayed scene images with 100 such backgrounds and 200 fore grounds (100 foreground + 100 of their flips).
	Strategy to place the foreground on background: What we are doing here is Monocular depth estimation. Monocular cues are the ones that are obtained from the 2D image of only one eye. These include the following. 1. Occlusion: When one object is hidden fully or partially, this background/hidden (occluded) object is considered to be farther away, behind fore ground object which is covering it (occluding object). 2. Relative Cues: The second cue is that of relative height. The objects that are farther away have higher bases in the image as shown in below Fig1.0. The couple in this figure are perceived at different depths
	.
Assuming camera position, focal length and aperture are fixed, we can observe that an object placed near the camera looks bigger and occupy more pixels on image than the object which is placed far from camera. So while positioning a foreground image on various places of background image, we will simulate and resize fore ground image such that, foreground sizes becomes smaller moving in height direction away from camera.

Example..Look at the below two images. Both are 150 * 150 
                                                             
Original back ground image:![image](https://github.com/gbrao018/eva4/blob/master/S15A/images/img1.jpg)                                   

Original Foreground image:![image(https://github.com/gbrao018/eva4/blob/master/S15A/images/img2.png)

Overlayed image, Foreground (90*90):![image(https://github.com/gbrao018/eva4/blob/master/S15A/images/img3.jpg)

Placed NEAR the camera and looks bigger

Overlayed image, fore ground placed FAR:![image(https://github.com/gbrao018/eva4/blob/master/S15A/images/img4.jpg)

(60*60). Size reduction and base elevation. 
                         
                                                                       Fig 1.0

  Let’s call backgound image as ‘bg’, foreground image as ‘fg’ and overlayed image as ‘fg_bg’ from now on interchangibly.
The fore ground positions are placed into 4 rows * 5 columns obtaining 40 fg_bg images for one bg and 1fg. Fore ground image is cloned and resized for each row. 1st row fg image size is 90*90, second row fg size is 80*80, third row fg size is 70*70, fourth row fg size is 60*60.  We did not go beyond 60*60 because, too much the small image means difficult to train for accuracy.
Basically I consider these rows as depth layers.By doing this my perception of depth is that, all these fg images in row1 share the same depth. Fg images placed in row2 share relatively at higher depth than row1. The last row fg image which occupies higher ground will share relatively higher depth comparing its previous rows.   
Code to overlay foreground on background: The background image format is choosen as jpg, where as foreground object image is choosen as png.
                                                             
            bg.jpg                                                                                                 fg.png
                    
            Fg_resized                        fg_bg                                   mask
Step#1: Resize the fg (h,w) and identify the location(x,y) to place on bg, as per below . Say w,h are the width and height of the fg 
                           
H,0)    
NOW THE OVERLAYED AREA BOUNDING BOX IS bg[y:y+h, x:x+w] is our interested area where we have to put the fg object

Step#2: Create an image from fg (b,g,r) channels. And normalize by deviding with 255.
b,g,r,a = cv2.split(fg)
fg3 = np.dstack((b,g,r)). 
fg_mask = bgr / 255.0
fg_mask[fg_mask>0] = 1
                    
 fg3(3 channels)            fg_mask(looks black but it has 0 or 1 values)

Step#3: Override the bg[y:y+h, x:x+w] area with fg3 pixels
fg_bg = clone(bg)
fg_bg[y:y+h, x:x+w] = (1.0 – fg_mask) * fg_bg[y:y+h, x:x+w]+fg_mask*fg3
                          
fg_mask*fg3 (1.0-fg_mask)*bg[y:y+h,x:x+w]    fg_bg 

We store the resultant fg_bg image in jpg format to save the storage.
fg_bg mask generation for foreground: We need the alpha channel in foreground image for transparency information which will help us in creating the mask for the fg. If we have alpha channel 
b,g,r,a = cv2.split(fg)
mask = np.dstack((a,a,a)) -> This will give the mask
But we do not have the alpha channel in our fg_bg.
So, we do it diffeently.
               
1.	fg_bg-bg                           2. gray image (3 channels)       3.1D Black/White Mask

1.	diff = fg_bg-bg
2.	gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
3.	gray_image[gray_image>0] = 255

cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) is the open cv function to convert to gray scale
gray_image[gray_image>0] = 255, will create 1D mono color Mask either black or white pixel

    
**By storingt 1D grayscale Mask, we are actually reducing storage.
               











Storage Optimization:
                                              
Jpg (10kb)                                                                                             png(40 kb)

These two fg_bg images one in jpg format the other in png format. Visually there is not much difference. But png is 40kb and jpg is 10 kb. So we will store the fg_bg overlayed image in jpg format.
fg_bg is the input for our modal to predict depth map and mask.  400k such images we need to store. This way we can reduce the storage from 400000*40kb to 400000*10kb i.e., reduced to 4 GB from otherwise 16 GB.
 Also, the ground truth images i.e., mask and depth maps are stored in 1D. This will also save the sorage. 
Storage Structure: As we are using Google cloud storage and colab, We do not save all the images into one folder, which will create problem for access as it will be too huge to search. We will save the images in a structured format.
Image naming conventions we follwed are bg1,b2,bg3 ….bg100 for back ground image
fg
Dataset150/
	fg150/
                         fg_1
	           fg_2
	           …….
                         fg_200
	bg150/
                       bg1
                       bg2
                      ------
                      bg100
	fgbg_1/
		bg1_0_0_fg_1.jpg
		
	fgbg_2/
	------
	------
	fgbg_100/
	mask_fgbg_1/
	mask_fgbg_2/
	…………
mask_fgbg_100/
	depth_fgbg_1/
	depth_fgbg_2/
	…………
depth_fgbg_100/
root_dir: /content/gdrive/My Drive/eva-04/S15A/Dataset150/

sub folders: fgbg150_0, fgbg150_1, …, fgbg150_100

filenameing convension: bg5_0_2_flip_fg_73.jpg. Corresponding mask name is mask_bg5_0_2_flip_fg_73.jpg, corresponding depth name is depth_bg5_0_2_flip_fg_73.jpg
bg5 -> backgroundimage name
0 -> first row
2 -> 3rd column
flip_fg_73 -> fore ground image name(flipped version)
***For all bg combinations, only the bgX number will change.Remaining all repeat for all background images. This way, by knowing just the fg_bg image name, we can get the right mask and depth file names
  
Colab efficiency: By runing multiple sessions same time. I am able to do multi tasking, thereby saving the time	
Link for the Dataset150: https://drive.google.com/open?id=1IucmmNUapKK1i_ORIdxYSMtA72qcxTa9
