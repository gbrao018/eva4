# Project Scope Description: 

	We are  going to solve the multi objective training as a supervised learning problem. 
	In short, it is Monocular depth estimation and mask detection. My journey through this challenge has been 
	detailed out below.  

What are we going to predict?. Given a background image and a overlayed image (foreground placed on the given background), 
we are going to predict the  depth map for the overlayed image and mask for the foreground image.

	To recap, We created the bg, fg, fgbg and corresponding ground truth depth and mask images in 15A 
	(Please look for the README.md of 15A for dataset preperation strategy). 

This writeup has 5 major sections.

#### SECTION#1: My Initial Thoughts

#### SECTION#2: Modal Creation

#### SECTION#3: Training Strategy (This section will be interesting with results shown with images)

#### SECTION#4: Experiments with Losses and Analysis (Important, but might be boring to read)

#### SECTION#5: Custom Dataset class and index based strategy & Timeit of dataloading and loss calcualations.



### SECTION#1: My Initial Thoughts

Many thoughts came into my mind. Creating a modal is one thing which I would talk in section#3, but running the modal
and verifying its accuracy on huge dataset of 4 lac is another challenge. I devided the total 4 lac samples in to 
3 lac for training, and 1 lac for test. 
	
	Now I have 3 lac data for training. With Image size is (224,224), colab can take a batch size of 16, anything,more 
	than that it throws "Cuda out of Memory" error. So, with 16 batch size, total batches per epoch= 300000/16 = 18750. 
	
	As per the observation, 1 batch takes, 1.4 second time, and the whole 18750 batches takes almost 7.3 hrs. My first 
	task is to understand whether modal is working or not and to understand different loss functions.
	
	So, I did reduce the image size to 32*32, so that I can use batch size of 1024 in colab. This worked. With this we 
	only have to run 293 batches which will take less than 7 minutes per epoch, for the whole 300k training dataset.
	
### SECTION#2: Model Creation

Initially, I went through the internet literature. Most of the solutions are depth prediction, by using dynamic image changes from t to t + delta_t time difference. State of the art curently is Unet model which uses encoders and decoder with Fully connected layers.

	I want to create my own model and I do not want to use fully connected layer. I avoided gap layer also, as 
	both FC and gap layers would destruct the spacial features.  In our case, spatial features like back ground 
	etc are very important.

	To achive the receptive field, CNN widely uses maxpooling. but max pooling shrinks the image size. 
	I see two possibilities.
	
	Case#1: Start with 1024->MP->512->MP->256->MP->128 and finally evaluate the loss with (128,128) ground truths. 
	But we cant use good batch size due to GPU limitations.
	
	Case#2: Start with any size, after 3 max pools, upscale the image till we arrive at initial size. Till x15, 
	I attained receptive field of 74 before start upscaling.
	
	I choosen the case#2. May be I can try with case#1 also in the future for comparison.
	

#### Basic modal structure: 

We concatinate both input fg + fgbg which will become 6 channels. Modal takes these 6 channels as input.
The modal has 3 maxpools. After 3rd maxpool , image size will become W/8,H/8.

I followed a specific channel output between each maxpool
	![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/model_arch.png)

	For, x15, I have concatenated the x12 maxpool also. 
	x15 = self.x15(torch.cat([x12,x13,x14],dim=1)) # 512 channels. From here on we go on upscaling till 
	we arrive at the input image size.
	
#### while in maxpool/encoder stage I used concatenations(concate). While in upscaling/decoder stage, I used addition of channels(+). 
	
### Model Parameters:

	----------------------------------------------------------------
		Layer (type)               Output Shape         Param #
	================================================================
		    Conv2d-1         [-1, 64, 256, 256]           3,520
		      ReLU-2         [-1, 64, 256, 256]               0
	       BatchNorm2d-3         [-1, 64, 256, 256]             128
		    Conv2d-4         [-1, 64, 256, 256]          36,928
		      ReLU-5         [-1, 64, 256, 256]               0
	       BatchNorm2d-6         [-1, 64, 256, 256]             128
		 MaxPool2d-7        [-1, 128, 128, 128]               0
		    Conv2d-8        [-1, 128, 128, 128]         147,584
		      ReLU-9        [-1, 128, 128, 128]               0
	      BatchNorm2d-10        [-1, 128, 128, 128]             256
		   Conv2d-11        [-1, 128, 128, 128]         295,040
		     ReLU-12        [-1, 128, 128, 128]               0
	      BatchNorm2d-13        [-1, 128, 128, 128]             256
		   Conv2d-14        [-1, 128, 128, 128]         442,496
		     ReLU-15        [-1, 128, 128, 128]               0
	      BatchNorm2d-16        [-1, 128, 128, 128]             256
		MaxPool2d-17          [-1, 384, 64, 64]               0
		   Conv2d-18          [-1, 256, 64, 64]         884,992
		     ReLU-19          [-1, 256, 64, 64]               0
	      BatchNorm2d-20          [-1, 256, 64, 64]             512
		   Conv2d-21          [-1, 256, 64, 64]       1,474,816
		     ReLU-22          [-1, 256, 64, 64]               0
	      BatchNorm2d-23          [-1, 256, 64, 64]             512
		   Conv2d-24          [-1, 256, 64, 64]       2,064,640
		     ReLU-25          [-1, 256, 64, 64]               0
	      BatchNorm2d-26          [-1, 256, 64, 64]             512
		MaxPool2d-27          [-1, 768, 32, 32]               0
		   Conv2d-28          [-1, 512, 32, 32]       3,539,456
		     ReLU-29          [-1, 512, 32, 32]               0
	      BatchNorm2d-30          [-1, 512, 32, 32]           1,024
		   Conv2d-31          [-1, 512, 32, 32]       5,898,752
		     ReLU-32          [-1, 512, 32, 32]               0
	      BatchNorm2d-33          [-1, 512, 32, 32]           1,024
		   Conv2d-34          [-1, 512, 32, 32]       8,258,048
		     ReLU-35          [-1, 512, 32, 32]               0
	      BatchNorm2d-36          [-1, 512, 32, 32]           1,024
		   Conv2d-37          [-1, 256, 64, 64]         131,328
	      BatchNorm2d-38          [-1, 256, 64, 64]             512
		     ReLU-39          [-1, 256, 64, 64]               0
		   Conv2d-40          [-1, 256, 64, 64]         590,080
	      BatchNorm2d-41          [-1, 256, 64, 64]             512
		     ReLU-42          [-1, 256, 64, 64]               0
		   Conv2d-43        [-1, 128, 128, 128]          32,896
	      BatchNorm2d-44        [-1, 128, 128, 128]             256
		     ReLU-45        [-1, 128, 128, 128]               0
		   Conv2d-46        [-1, 128, 128, 128]         147,584
	      BatchNorm2d-47        [-1, 128, 128, 128]             256
		     ReLU-48        [-1, 128, 128, 128]               0
		   Conv2d-49         [-1, 64, 256, 256]           8,256
	      BatchNorm2d-50         [-1, 64, 256, 256]             128
		     ReLU-51         [-1, 64, 256, 256]               0
		   Conv2d-52         [-1, 64, 256, 256]          36,928
	      BatchNorm2d-53         [-1, 64, 256, 256]             128
		     ReLU-54         [-1, 64, 256, 256]               0
		   Conv2d-55         [-1, 32, 256, 256]          18,464
	      BatchNorm2d-56         [-1, 32, 256, 256]              64
		     ReLU-57         [-1, 32, 256, 256]               0
		   Conv2d-58          [-1, 1, 256, 256]              33
		   Conv2d-59          [-1, 1, 256, 256]             289
	================================================================
	Total params: 24,019,618
	Trainable params: 24,019,618
	Non-trainable params: 0
	----------------------------------------------------------------
	Input size (MB): 1.50
	Forward/backward pass size (MB): 863.00
	Params size (MB): 91.63
	Estimated Total Size (MB): 956.13
	----------------------------------------------------------------

Modal forward is shown below, with input,output. All uses 3,3 kernel otherwise mentioned in comments. 
maxpool uses 2,2 with stride=2.

	def forward(self, x1):
      
        x2 = self.x2(x1) #input 6*W*H, output=W*H*64 , RF =2                          
        x3 = self.x3(x2) #input W*H*64, output=W*H*64, RF=4  
        
        #Maxpool    
        x4 = self.x4(torch.cat([x2,x3],dim=1))#input W*H*128, output=W/2*H/2*128, RF=5
        x5 = self.x5(x4)   #input W/2*H/2*128, output=W/2*H/2*128, RF=10
        x6 = self.x6(torch.cat([x4,x5],dim=1))  #input W/2*H/2*256, output=W/2*H/2*128, RF=12
        
        x7 = self.x7(torch.cat([x4,x5,x6],dim=1)) #input W/2*H/2*384, output=W/2*H/2*128, RF=14
        x8 = self.x8(torch.cat([x5,x6,x7],dim=1)) #input W/2*H/2*384, output_size = (W/4*H/4*384), RF=15
        x9 = self.x9(x8) #input W/4*W/4*384, output_size = (W/4*H/4*256), RF=30
        x10 = self.x10(torch.cat([x8,x9],dim=1)) #input W/4*H/4*640, output_size = (W/4*H/4*256), RF=32
        x11 = self.x11(torch.cat([x8,x9,x10],dim=1)) #input W/4*H/4*896, output_size = (W/4*H/4*256), RF=34

        #Maxpool
        x12 = self.x12(torch.cat([x9,x10,x11],dim=1))#input W/4*H/4*768, output_size = (W/8*H/8*768), RF=35 
        
        x13 = self.x13(x12) #input W/4*H/4*768, output_size = (W/4*H/4*512), RF=70
        x14 = self.x14(torch.cat([x12,x13],dim=1)) #input W/8*H/8*1280, output_size = (W/8*H/8*512), RF=72
        x15 = self.x15(torch.cat([x12,x13,x14],dim=1)) #input W/8*H/8*1792, output_size = (W/8*H/8*512), RF=74
        
        # Now do upsampling
        up = Upsample(2).cuda() 
        xf1 = up.forward(x15) #input W/8*H/8*512, output_size = (W/4*H/4*512), RF=74
        xf1_invconv_512_256 = self.xf1_invconv_512_256(xf1) #input W/4*H/4*512, output_size = (W/4*H/4*256), RF=76
        xf1_inconv_256 = self.xf1_inconv_256(xf1_invconv_512_256.add(x11)) #input W/4*H/4*256, output_size = (W/4*H/4*256), RF=78
        
        xf2 = up.forward(xf1_inconv_256) #upscale #input W/4*H/4*256, output_size = (W/2*H/2*256), RF=78
        xf2_invconv_256_128 = self.xf2_invconv_256_128(xf2) #input W/2*H/2*256, output_size = (W/2*H/2*128), RF=80
        xf2_inconv_128 = self.xf2_inconv_128(xf2_invconv_256_128.add(x7)) #input W/2*H/2*128, output_size = (W/2*H/2*128), RF=82
        
        xf3 = up.forward(xf2_inconv_128) #upscale. #input W/2*H/2*128, output_size = (W*H*128), RF=82
        xf3_invconv_128_64 = self.xf3_invconv_128_64(xf3) #input W*H*128, output_size = (W*H*64), RF=84
        xf3_inconv_64 = self.xf3_inconv_64(xf3_invconv_128_64.add(x3)) #input W*H*64, output_size = (W*H*64), RF=86

        x64_inconv_32 = self.x64_inconv_32(xf3_inconv_64) #input W*H*64, output_size = (W*H*32), RF=88
        #{{finished upscaling}}
        
        #Use 1*1 kernel and create final output channel for depth. 
        xdepth_conv_1_1 = self.x32_conv_1_1(x64_inconv_32) #input W*H*32, output_size = (W*H*1), RF=88. Uses 1*1 kernel.
		
		#Use 3*3 kernel and create final output channel for mask by doing sigmoid. 
        predict_mask = torch.nn.functional.sigmoid(self.predict_mask(x64_inconv_32)) #input W*H*32, output_size = (W*H*1), RF=90. Uses 3*3 kernel.
        out_array = []
        out_array.append(xdepth_conv_1_1)
        out_array.append(predict_mask)
        return out_array

	
### SECTION#3: Training Strategy (This section will be interesting with results shown with images)

	IMAGE SIZE VS BATCH SIZE USED:
	32 * 32 		1024
	64 * 64 		256
	128 * 128 		64
	256 * 256 		16

Based on my analysis, the above is the relastionship between maximum batch size of the colab GPU(I got) vs the image size.
With (32,32) image size, I can give 1032 batch size and was able to finish one epoch in 7 minutes. With (256,256) 
image size I can barely give 16 as batch size, and it takes 7 hours to complete one epoch. If we try to give bugger batch size,
it throws CUDA out of memory error. To save ourselves from Cuda error, below is the recommonded batch size.

IMAGE SIZE VS BATCH SIZE RECOMMONDED:

	32 * 32 		1024
	64 * 64 		256
	128 * 128 		64
	256 * 256 		16

### Understand the loss functions quickly by experimenting with 32*32 size and apply the resulted weights on bigger images and retrain for few epochs.

It is similar to transfer learning. loaded zip into colab vm as reading from drive is time taking.

So, with 32*32 size, I could able to try out all combinations of losses and their behavior as mentioned in earlier section.

### 3.1 Encounter with Interpolation effect. 

In this case, choosen is is (32,32) but, I then interpolate the output to (64,64) size and sent to the loss function and backprop. With this approach, it was fast, but looks few spatial features compromised. We can see the roundedness of interpolation effect in masks. And brightness of the depth is also not that good. Tried to subtract the bright ness (-0.005) before sending to loss, but that actually is whitening the image , nothing more. 

Interpolation effect (Observed roundedness with (32,32) input size and interpolated to (64,64):
![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/interpolation-effect.jpg)

		Conclusion: When we tried with small size, background quality is compromised, and rounded ness appeared due to interpolation. 
		When the input size = size (minimum 64*64) used for loss calculation without interpolation, roundeness issue disappeared.
		MSE depth Loss converged with (32,32) is 0.01. 

### 3.2. Understand the impact of input size.

So tried with (64,64) as input size and the output as (64,64) but interpolate to (128,128) that is sent to loss function.
	-> This has good predictability and better image quality on fore ground. First checked on 10000 images and later
	applied for whole dataset. But I still see the roundedness.

#### No interpolation means, pixel perfect masks. (64,64) interpolated to (128,128). Increased depth clarity , but roundedness present still.
![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/64_interpolate_128_7_2.jpg)

		Conclusion: When the input size is 64*64, interpolated to (128,128) for loss calculation , 
		depth quality is good but roundedness still exist. Without interpolation we observed the mask came up accurately pixelwise.
	
### 3.3. Arriving at a conclusion on minimum input size vs depth quality: 

When did with, (128,128) input size, and output (128,128) and NO MORE interpolation, before sending to loss and backprop. 
Initially, applied the same weights we got from initial (64,64) image on (128,128), and the results are good and improved without training.

#### Transfer learning worked: Applied weights from (64,64) to 128*128:
![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/64weights_128_image.jpg)

Now we trained the above weights on sample of 10000 images for 1 epoch. I checked the image quality, the background depth quality is clearly improved.

#### Bigger the input image size, better the depth maps. Transfer learning: Applied weihgts from (64,64) to 128*128 and trained for one epoch:
![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/64weights_128_image_with1epoch_transfertraining.jpg)

	Conclusion: With increased initial image size of 128*128, now the loss convergencence is around 0.007 
	which better than earlier 0.01. So I conclude the minimum image size required is 128*128, and bigger the image 
	size, quality of depth map is better. 
	
### 3.4. Mask converges faster than Depth loss. Playing with learning rates.

With (128,128) size, now ran one more epoch on 30000 samples. Due to homogenity of either black or white, mask quickly shows 
converges in its loss. But Depth across many images, every pixel has diffferent intensity.  Different learning rates of e-03,e-04,e-05 have been used. 

![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/128_test_good.jpg)

	Conclusion: Increased initial image size to (128,128). Depth maps are good. Mask always converges faster than depth. 

#### 3.4.1 (128,128) input, Different losses & times at convergence: lr = e-05 Loss = K(Depth-MSE + Mask-MSE + Depth_SSIM). With 100k training samples,
	
	L2-D=0.003739 L2-M=0.003631 SSIM-D=0.000475 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.006 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.121 GRAD-D=0.022899 SMOOTH-D=0.001869 SMOOTH-L1-M=0.001816 RMSE=0.097306 Meanlog10=nan Acc_D1=0.642869 Acc_D2=0.812467 Acc_D3=0.875512
	
		L2-D=0.003739 L2-M=0.003631 SSIM-D=0.000475 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.010 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.119 GRAD-D=0.022899 SMOOTH-D=0.001870 SMOOTH-L1-M=0.001816 RMSE=0.097306 Meanlog10=nan Acc_D1=0.642861 Acc_D2=0.812471 Acc_D3=0.875517
		
#### 3.4.2 Increased the sample size to 100k. Trained for 1 epoch. Background shows good improvement
		
		L2-D=0.008466 L2-M=0.004555 SSIM-D=0.001136 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.010 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.077 GRAD-D=0.026557 SMOOTH-D=0.004233 SMOOTH-L1-M=0.002278 RMSE=0.084072 Meanlog10=nan Acc_D1=0.377704 Acc_D2=0.528144 Acc_D3=0.604948:
		
		L2-D=0.008146 L2-M=0.004495 SSIM-D=0.001089 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.010 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.078 GRAD-D=0.026855 SMOOTH-D=0.004073 SMOOTH-L1-M=0.002248 RMSE=0.081618 Meanlog10=nan Acc_D1=0.381430 Acc_D2=0.537099 Acc_D3=0.613397:   0%|          | 0/1563 [25:32<?, ?it/s]
		
		L2-D=0.007948 L2-M=0.004446 SSIM-D=0.001059 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.076 GRAD-D=0.027045 SMOOTH-D=0.003974 SMOOTH-L1-M=0.002223 RMSE=0.079838 Meanlog10=nan Acc_D1=0.394439 Acc_D2=0.547774 Acc_D3=0.620838:   0%|          | 0/1563 [25:24<?, ?it/s]

##### We observed , sharp identification of background edges.

#### 3.4.3 (128,128) input, With 200k training samples, lr - e-05. Loss = K(Depth-MSE + Mask-MSE + Depth_SSIM.

	L2-D=0.005239 L2-M=0.002627 SSIM-D=0.000660 MODAL-EXEC-TIME=0.011 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.024337 SMOOTH-D=0.002619 SMOOTH-L1-M=0.001314 RMSE=0.097605 Meanlog10=nan Acc_D1=0.537048 Acc_D2=0.757224 Acc_D3=0.852888:
	
		L2-D=0.004989 L2-M=0.002641 SSIM-D=0.000626 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.158 GRAD-D=0.024597 SMOOTH-D=0.002494 SMOOTH-L1-M=0.001321 RMSE=0.093947 Meanlog10=nan Acc_D1=0.549465 Acc_D2=0.768925 Acc_D3=0.861968:
		
		L2-D=0.004856 L2-M=0.002629 SSIM-D=0.000608 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.012 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.024753 SMOOTH-D=0.002428 SMOOTH-L1-M=0.001314 RMSE=0.091208 Meanlog10=nan Acc_D1=0.565269 Acc_D2=0.780983 Acc_D3=0.870636:

##### The background white increased. We are able to see clarity in depth.

### 3.4.4 Final output with input size as 256 * 256.

So, Finally, I tried with 256*256, loss calculation with 256*256, Loss = K(L2_Depth + L2_Mask), K = 10, 20, 30 and 70 in various trials. 
Naturally depth loss start increasing epoch by epoch. Now I start seeing sharp background objects depth also. They start appearing.

	Conclusion: INCREASED IMAGE INPUT SIZE ACTUALLY PREDICTED BACKGROUND OBJECTS MORE ACCURATELY, 
	WHICH ARE IMPORTANT FOR DEPTH CALCULATION.

#### 3.4.4.1 (256,256) input, 30k training samples. REDUCED THE BATCH SIZE TO 16. Loss = K(Depth-MSE + Mask-MSE + Depth_SSIM)

	L2-D=0.005351 L2-M=0.003557 SSIM-D=0.000727 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.017597 SMOOTH-D=0.002676 SMOOTH-L1-M=0.001778 RMSE=0.098037 Meanlog10=nan Acc_D1=0.657427 Acc_D2=0.833152 Acc_D3=0.893974:
	
		L2-D=0.005074 L2-M=0.003510 SSIM-D=0.000689 MODAL-EXEC-TIME=0.007 BACKPROP-EXEC-TIME=0.008 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.017226 SMOOTH-D=0.002537 SMOOTH-L1-M=0.001755 RMSE=0.090943 Meanlog10=nan Acc_D1=0.678247 Acc_D2=0.845829 Acc_D3=0.901804:
		
		L2-D=0.004956 L2-M=0.003466 SSIM-D=0.000673 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.016984 SMOOTH-D=0.002478 SMOOTH-L1-M=0.001733 RMSE=0.088309 Meanlog10=nan Acc_D1=0.688958 Acc_D2=0.851394 Acc_D3=0.904802:
		
		L2-D=0.004888 L2-M=0.003419 SSIM-D=0.000665 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.008 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.016844 SMOOTH-D=0.002444 SMOOTH-L1-M=0.001709 RMSE=0.086521 Meanlog10=nan Acc_D1=0.697304 Acc_D2=0.855762 Acc_D3=0.907248:
	
	Conclusion: MORE THE TRAINING SAMPLES, BETTER THE DEPTH MAPS.

### 3.4.4.2 BCE loss vs MSE for mask. Loss = K(Depth-MSE + Mask-MSE + Depth_SSIM + MASK_BCE).

	L2-D=0.004944 L2-M=0.002657 SSIM-D=0.000672 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.016993 SMOOTH-D=0.002472 SMOOTH-L1-M=0.001329 RMSE=0.086277 Meanlog10=nan Acc_D1=0.696232 Acc_D2=0.858464 Acc_D3=0.909742: 
		L2-D=0.004911 L2-M=0.002531 BCE-M=0.018495 SSIM-D=0.000669 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.016789 SMOOTH-D=0.002456 SMOOTH-L1-M=0.001266 RMSE=0.085526 Meanlog10=nan Acc_D1=0.701390 Acc_D2=0.860296 Acc_D3=0.910286:
		
	Conclusion: AS FAR AS MASK IS CONCERNED. IT IS ALREADY output of sigmod. THERE IS NO EFFECT OF MSE VS BCE ON MASK. MASK AUTOMATICALLY CONVERGED WITH DEPTH MSE.

#### 3.4.4.3 Now removed L2_Mask. Will see how other losses converge?. Now loss = K(D_L2 + M_BCE + SSIM-D).

	L2-D=0.004893 L2-M=0.002537 BCE-M=0.018170 SSIM-D=0.000667 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.007 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.157 GRAD-D=0.016626 SMOOTH-D=0.002447 SMOOTH-L1-M=0.001268 RMSE=0.084866 Meanlog10=nan Acc_D1=0.705272 Acc_D2=0.861931 Acc_D3=0.911125: 
	Again Mask is pixel perfect. Now , we will make shuffle = true and removed SSIM-D. Now Loss = K(D_L2 + M_BCE). But we still watch what happens to other loss values..
	L2-D=0.006709 L2-M=0.001982 BCE-M=0.016659 SSIM-D=0.000940 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.005 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.015992 SMOOTH-D=0.003354 SMOOTH-L1-M=0.000991 RMSE=0.187009 Meanlog10=nan Acc_D1=0.420672 Acc_D2=0.639257 Acc_D3=0.771012:
	You see, L2 Mask has reduced, though it is not part of loss back prop. We can see Mask are pixel predicted and depth has improvement.
	Now we will increase the lr to e-04, as BCE still hovers around. Lets see if we can have quick convergence with BCE. 
	L2-D=0.006843 L2-M=0.001839 BCE-M=0.016846 SSIM-D=0.000947 MODAL-EXEC-TIME=0.005 BACKPROP-EXEC-TIME=0.006 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.157 GRAD-D=0.014221 SMOOTH-D=0.003422 SMOOTH-L1-M=0.000919 RMSE=0.181703 Meanlog10=nan Acc_D1=0.370108 Acc_D2=0.562902 Acc_D3=0.677281: 

We had good improvement in the depth. Now Lets increase the sample size to 100k . We had experience of good depth prediction after inreasing the sample size from 30k to 100k. This improvement is due to sample size increment. 

	L2-D=0.007237 L2-M=0.001173 BCE-M=0.013984 SSIM-D=0.001036 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.007 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.013809 SMOOTH-D=0.003618 SMOOTH-L1-M=0.000586 RMSE=0.258638 Meanlog10=nan Acc_D1=0.228543 Acc_D2=0.401711 Acc_D3=0.542491:

Next, trained on 100k dataset. This gave good improvement in depths. Every time we increase the resolution, we observed the initial epochs shows depth is not well developed. masks are ok. This is good sign because, There is more to learning
for the bigger resolution on top of transfered weights, which can give fine depths in subsequent epochs. It is slowly increasing its background prediction capability.
Mask is perfect almost.

#### FINAL OUTPUT:
![image](https://github.com/gbrao018/eva4/blob/master/S15B/logs/final_12.png)

### SECTION#4: Experiments with Losses and Analysis (Important, but might be boring to read)

I tried many loss functions.L1(MAE), L2(MSE), SmoothL1, SSIM, Gradient, BCE. For depth MSE outperformed. For mask actually it does not matter, because we are already doing sigmoid.
	
	But I would like to discuss few combinations here. What ever the combination is, I calculated many loss values and logged them to understand their change behavior. only L1 loss has the convergence issue. It became very difficult to achive the global minima.
	
#### 4.1: Loss = aD + bM is not a Good equation. a and b constants and D is MSE for depth and M is MSE for mask. NOT a good combination.
	Lets say a > b, We are punishing the convergence of mask. Actually mask has faster convergence behavior due to its black and white nature.
	
	(dense32_mask_dw30_loss8_tr1.log)
	INFO:root:Loss=0.3421425223350525 Epoch=0 Batch_id=1167  L2-D=0.010764 L2-M=0.017790 SSIM-D=0.001445 MODAL-EXEC-TIME=0.011 BACKPROP-EXEC-TIME=0.011 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.195 GRAD-D=0.024859 SMOOTH-D=0.005382 SMOOTH-L1-M=0.008895 RMSE=0.103927 Meanlog10=nan Acc_D1=0.528682 Acc_D2=0.762998 Acc_D3=0.859739
	
	INFO:root:Loss=0.3681124448776245 Epoch=0 Batch_id=1168  L2-D=0.011715 L2-M=0.015081 SSIM-D=0.001575 MODAL-EXEC-TIME=0.011 BACKPROP-EXEC-TIME=0.010 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.195 GRAD-D=0.025249 SMOOTH-D=0.005858 SMOOTH-L1-M=0.007540 RMSE=0.103981 Meanlog10=nan Acc_D1=0.529125 Acc_D2=0.763691 Acc_D3=0.860523
	
	INFO:root:Loss=0.31658148765563965 Epoch=0 Batch_id=1169  L2-D=0.009973 L2-M=0.016063 SSIM-D=0.001328 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.010 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.197 GRAD-D=0.025467 SMOOTH-D=0.004987 SMOOTH-L1-M=0.008031 RMSE=0.104027 Meanlog10=nan Acc_D1=0.529635 Acc_D2=0.764413 Acc_D3=0.861320
	
	INFO:root:Loss=0.282012015581131 Epoch=0 Batch_id=1170  L2-D=0.008887 L2-M=0.014239 SSIM-D=0.001173 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.197 GRAD-D=0.024732 SMOOTH-D=0.004443 SMOOTH-L1-M=0.007120 RMSE=0.104027 Meanlog10=nan Acc_D1=0.529635 Acc_D2=0.764413 Acc_D3=0.861320
	
	INFO:root:Loss=0.2731016278266907 Epoch=0 Batch_id=1171  L2-D=0.008591 L2-M=0.014246 SSIM-D=0.001127 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.173 GRAD-D=0.024574 SMOOTH-D=0.004295 SMOOTH-L1-M=0.007123 RMSE=0.104027 Meanlog10=nan Acc_D1=0.529635 Acc_D2=0.764413 Acc_D3=0.861320
	
We can observe above, with this combination of aD+bM (a>b), Both can not converge 100% due to the equation itself. 
Mask has not convereged at par with the depth due to higher weight given to depth. 
	
	
#### 4.2: SSIM has No effect. Smooth Loss has no effect. MSE (Mean Square Error) out performs.

Now, I would like to understand the effect of SSIM. Applied SSIM on depth. Image size 256, Loss = K(D + M + SSIM_D). K=70 is just a constant.  

My observation is that , Even if we don't use use SSIM in the loss function, actually due to MSE loss, SSIM also has seen convergence.
 	
	(256_dw70_sum_30k_lre5_15.log)
	INFO:root:Loss=0.5496146082878113 Epoch=0 Batch_id=1871  L2-D=0.005465 L2-M=0.001630 SSIM-D=0.000756 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.008 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.017681 SMOOTH-D=0.002733 SMOOTH-L1-M=0.000815 RMSE=0.086486 Meanlog10=nan Acc_D1=0.696401 Acc_D2=0.854470 Acc_D3=0.905863
	
	INFO:root:Loss=0.5341195464134216 Epoch=0 Batch_id=1872  L2-D=0.004071 L2-M=0.003008 SSIM-D=0.000552 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.008 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.016819 SMOOTH-D=0.002035 SMOOTH-L1-M=0.001504 RMSE=0.086496 Meanlog10=nan Acc_D1=0.696711 Acc_D2=0.854910 Acc_D3=0.906326
	
	INFO:root:Loss=0.6898210644721985 Epoch=0 Batch_id=1873  L2-D=0.005640 L2-M=0.003443 SSIM-D=0.000772 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.008 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.016914 SMOOTH-D=0.002820 SMOOTH-L1-M=0.001721 RMSE=0.086511 Meanlog10=nan Acc_D1=0.696994 Acc_D2=0.855317 Acc_D3=0.906776
	
	INFO:root:Loss=0.6279988288879395 Epoch=0 Batch_id=1874  L2-D=0.004888 L2-M=0.003419 SSIM-D=0.000665 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.008 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.016844 SMOOTH-D=0.002444 SMOOTH-L1-M=0.001709 RMSE=0.086521 Meanlog10=nan Acc_D1=0.697304 Acc_D2=0.855762 Acc_D3=0.907248
	
We can see the variation in the loss is due to depth and mask losses. SSIM always 3 decimal loss. Lets look at Smooth loss.
	
	SMOOTH-D=0.002035 when depth loss L2-D=0.004071 	
    	SMOOTH-D=0.002444 when depth loss L2-D=0.004888
	SMOOTH-D=0.002733 when depth loss L2-D=0.005465
	SMOOTH-D=0.002820 when depth loss L2-D=0.005640
    
	This also, we observed, Smooth loss is tuning itself to the value of MSE->Mean Square error loss.
	So, we conclude that both SSIM and Smooth lossed are NOT needed to be part of the loss function.
	
	Coming to Gradient loss change, based on below values, we observed there is not a much change in gradient loss.
	
	GRAD-D=0.024859
	GRAD-D=0.025249
	GRAD-D=0.02546
	GRAD-D=0.024732 
	GRAD-D=0.024574
	
#### 4.3 Gradient loss do not wok for depth and does not converge.

In one experiement did with gradient loss, I observed it is actually trying to average the gradients that is like uniformly spreading brihgtness in the the pixels. So, I decided not to use gradient loss also.
	
#### 4.4: So, I removed the SSIM and Smooth Loss. Now the loss function became Loss = K(L2_Depth + L2_Mask). L2 = MSE.	
	
	Example, This log is the output of K(L2_Depth + L2_Mask). K is 70 used for initial loss convergence. This is very equalent to we give equal weights to both mask and depth.
	 
	 With 100k samples,(verygood_dw70_sum_100k_lre5_10.log) 
	INFO:root:Loss=0.5529605150222778 Epoch=0 Batch_id=1558  L2-D=0.005579 L2-M=0.001584 SSIM-D=0.000736 MODAL-EXEC-TIME=0.010 BACKPROP-EXEC-TIME=0.018 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.024652 SMOOTH-D=0.002790 SMOOTH-L1-M=0.000792 RMSE=0.079838 Meanlog10=nan Acc_D1=0.394439 Acc_D2=0.547774 Acc_D3=0.620838
	INFO:root:Loss=0.6402487754821777 Epoch=0 Batch_id=1559  L2-D=0.006424 L2-M=0.001872 SSIM-D=0.000851 MODAL-EXEC-TIME=0.014 BACKPROP-EXEC-TIME=0.012 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.152 GRAD-D=0.026086 SMOOTH-D=0.003212 SMOOTH-L1-M=0.000936 RMSE=0.079838 Meanlog10=nan Acc_D1=0.394439 Acc_D2=0.547774 Acc_D3=0.620838
	INFO:root:Loss=0.561633825302124 Epoch=0 Batch_id=1560  L2-D=0.005637 L2-M=0.001635 SSIM-D=0.000751 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.010 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.157 GRAD-D=0.024538 SMOOTH-D=0.002818 SMOOTH-L1-M=0.000818 RMSE=0.079838 Meanlog10=nan Acc_D1=0.394439 Acc_D2=0.547774 Acc_D3=0.620838
	INFO:root:Loss=0.5946778655052185 Epoch=0 Batch_id=1561  L2-D=0.005760 L2-M=0.001976 SSIM-D=0.000759 MODAL-EXEC-TIME=0.008 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.159 GRAD-D=0.025065 SMOOTH-D=0.002880 SMOOTH-L1-M=0.000988 RMSE=0.079838 Meanlog10=nan Acc_D1=0.394439 Acc_D2=0.547774 Acc_D3=0.620838
	INFO:root:Loss=0.9416714906692505 Epoch=0 Batch_id=1562  L2-D=0.007948 L2-M=0.004446 SSIM-D=0.001059 MODAL-EXEC-TIME=0.009 BACKPROP-EXEC-TIME=0.009 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.076 GRAD-D=0.027045 SMOOTH-D=0.003974 SMOOTH-L1-M=0.002223 RMSE=0.079838 Meanlog10=nan Acc_D1=0.394439 Acc_D2=0.547774 Acc_D3=0.620838
	
Not much of a difference in converence pattern between Trial#2 and Trial#3. It shows MSE for both depth and mask is doing good due to equal weights. I observed the output of the mask quality is very good.
	
#### 4.5: Testing with BCE. Depth loss is the king. Mask loss function is immaterial.

I replaced the MSE for mask with BCE for mask. Loss = K(L2_depth + BCE_Mask)

	(bce_no_ml2_nodssim_256_dw70_sum_100k_lre4_20.log)	
	INFO:root:Loss=1.877028465270996 Epoch=0 Batch_id=6244  L2-D=0.008568 L2-M=0.002378 BCE-M=0.018247 SSIM-D=0.001217 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.005 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.014723 SMOOTH-D=0.004284 SMOOTH-L1-M=0.001189 RMSE=0.257441 Meanlog10=nan Acc_D1=0.228618 Acc_D2=0.404446 Acc_D3=0.547375
	
	INFO:root:Loss=2.103707790374756 Epoch=0 Batch_id=6245  L2-D=0.008927 L2-M=0.002807 BCE-M=0.021126 SSIM-D=0.001240 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.006 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.157 GRAD-D=0.014268 SMOOTH-D=0.004463 SMOOTH-L1-M=0.001403 RMSE=0.257482 Meanlog10=nan Acc_D1=0.228622 Acc_D2=0.404455 Acc_D3=0.547390
	
	INFO:root:Loss=1.9479937553405762 Epoch=0 Batch_id=6246  L2-D=0.009098 L2-M=0.002014 BCE-M=0.018730 SSIM-D=0.001290 MODAL-EXEC-TIME=0.007 BACKPROP-EXEC-TIME=0.005 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.155 GRAD-D=0.014262 SMOOTH-D=0.004549 SMOOTH-L1-M=0.001007 RMSE=0.257504 Meanlog10=nan Acc_D1=0.228639 Acc_D2=0.404488 Acc_D3=0.547439
	
	INFO:root:Loss=1.6639208793640137 Epoch=0 Batch_id=6247  L2-D=0.007568 L2-M=0.001588 BCE-M=0.016202 SSIM-D=0.001053 MODAL-EXEC-TIME=0.007 BACKPROP-EXEC-TIME=0.006 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.013985 SMOOTH-D=0.003784 SMOOTH-L1-M=0.000794 RMSE=0.257542 Meanlog10=nan Acc_D1=0.228686 Acc_D2=0.404553 Acc_D3=0.547517
	
	INFO:root:Loss=1.8382396697998047 Epoch=0 Batch_id=6248  L2-D=0.007530 L2-M=0.002196 BCE-M=0.018730 SSIM-D=0.001051 MODAL-EXEC-TIME=0.006 BACKPROP-EXEC-TIME=0.005 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.156 GRAD-D=0.013767 SMOOTH-D=0.003765 SMOOTH-L1-M=0.001098 RMSE=0.257556 Meanlog10=nan Acc_D1=0.228725 Acc_D2=0.404627 Acc_D3=0.547619
	
	INFO:root:Loss=1.5033087730407715 Epoch=0 Batch_id=6249  L2-D=0.006372 L2-M=0.001553 BCE-M=0.015103 SSIM-D=0.000897 MODAL-EXEC-TIME=0.005 BACKPROP-EXEC-TIME=0.005 L2-DEPTH-TIME=0.000 L2-MASK-TIME=0.000 SSIM-DEPTH-TIME=0.157 GRAD-D=0.013098 SMOOTH-D=0.003186 SMOOTH-L1-M=0.000776 RMSE=0.257570 Meanlog10=nan Acc_D1=0.228770 Acc_D2=0.404714 Acc_D3=0.547733

BCE is dancing according to MSE of mask. But seems for mask it does not matter wheather we use BCE or MSE, both are doing good.

#### SECTION#5: Custom Dataset class and index based strategy & Timeit of dataloading and loss calcualations.

I created a custom Dataset class which takes input -> root, size, test = False, start= 1,  transform=ToTensor())
root -> path for the Dataset directory. The same class is used for both training and testing. while training test =  False, else True. 

if test == True:
	index = 300001

Given an index, Custom Dataset class can identify the path of the respective files and provide them in an array. array contains bg,fgbg,depth and masks.
We have timed it also. The Dataloading time is around 0.07 seconds.

The algorithm goes below:

	bg_index = index // 4000 # each fg can hold till 4000
    	if index % 4000 != 0:
      	   bg_index = bg_index + 1

        # each fg can hold till 40
        fg_index = index // 40
        if index % 40 != 0:
          fg_index = fg_index + 1

        fg_index = fg_index % 100 # max cap for fg_index
        if fg_index == 0:
          fg_index =  100 

        file_index = index % 40
        if file_index == 0:
	  file_index = 40 # This is 1 to 40

Inside the root directory below is the file structure.

	root/
		bg1/
			/fg_1
				 /depth
					1.jpg
					.....
					.....
					40.jpg  
				 /mask
					1.jpg
					2.jpg
					.....
					.....
					40.jpg
				 /overlay
					1.jpg
					2.jpg
					.....
					.....
					40.jpg
			/fg_2
				   /depth
				   /mask_
				   /overlay
			 .........
			 .........
			 .........
			 /fg_200/
					/depth
					/mask_
					/overlay	
		bg2/
			/fg_1
				/depth
					 1.jpg
					 .....
					 .....
					 40.jpg  
				/mask_
					1.jpg
					.....
					.....
					40.jpg  
				/overlay
					1.jpg
					.....
					.....
					40.jpg  
			..............
			..............
			..............
			 /fg_200
			   /depth
			   /mask_
			   /overlay

		..................
		..................
		..................
		/bg100
			  /fg_1
				/depth
					 1.jpg
					 .....
					 .....
					 40.jpg  
				/mask_
					1.jpg
					.....
					.....
					40.jpg  
				/overlay
					1.jpg
					.....
					.....
					40.jpg  
			..............
			..............
			..............
			 /fg_200
			   /depth
			   /mask_
			   /overlay
			
Data Augmentation: I did not use any Data Augmentation for this project, as initially I am not sure what effects these will 
create on changes of ground truths.

	I believe, I can do some data augmentation with additional brightness (adding some factor to tensor), 
	hue, saturations. This I will try in future.

	
