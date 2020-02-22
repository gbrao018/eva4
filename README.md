#Session5/Assignment5
Goal: Achieve 99.4% accuracy on test data consistently with less than 10000 parameters and below 15 epochs and with minimum 5 steps

Step#1: (S5_BaseModel.ipynb)
   1. Target
      . Create a base model which is working with in less than 10000 parameters
   2. Result
      . Parameters 9790
      . Best Accuracy on training set 98.17
      . Best Accuracy on Test set 98.92
   3. Analysis
      . Seems a good Model.
      . No overfitting as of now
      . Still there is a scope to improve
      . In the next step we will apply batch normalization to improve the accuracy  
      
Step#2: (S5_step2_batchnorm.ipynb)
   1. Target
       . Apply Batch Normalization and observe the model fit and accuracy improvement
   2. Result
       . Parameters 9950
       . Best Training accuracy 99.43
       . Best Test Accuracy 99.11
   3. Analysis
       . This shows overfitting
       . Training accuracy reached 99.43 , but test accuracy has not reached. We need to solve this overfitting issue. 
       Next step we will add regularization/drop out to remove overfitting
       
Step#3: (S5_step3_regularization.ipynb)
   1. Target
      . Apply Regularization/dropout to remove the overfitting
   2. Result
      . parameters 9950
      . Best Training set accuracy 98.79
      . Best Test set accuracy 99.12
   3. Analysis
      . Regularization worked and solved the overfitting issue.
      . But still last big kernel exist and Gap layer has not been used yet. We will add gap layer in next step
      
Step#4: (S5_step4_MaxPool5_Gaplayer_LastLayer.ipynb)
   1. Target
      . Added gap layer and removed big kernel in the last.
      . As MNIST data shows features after receiptive field 5. Maxpool applied at receptive field 5
      . Added additional layer after GAP layer
   2. Result
      . Parameters 9932
      . Best Training set accuracy 98.98
      . Best Test set accuracy 99.40
      . We are able to see the 99.4 accuracy but not consistently
   3. Analysis
      . Test accuracy improved.
      . Model is NOT overfit and is good.
      . But there exist difference in the traing and test accuracy.Model is good and underfit
      . As MNIST shows some images shows some orientation, we will add orientation 7 degree in the next step.
      
Step#5: (S5_9k_noLR_9940.ipynb, S5_step5_final.ipynb)
   1. Target  
      . Image Augmentation - Added 7 degree orientation as some images are oriented
      . Test with different models for consistency
   2. Result
      . Paramters 9918
      . Best Training Accuracy 98.9
      . Best Test Accuracy 99.45 showed consistency in multiple models 
      . At this stage tried with multiple models with below 10000 params. Got the consistency. (S5_9k_noLR_9940.ipynb, S5_step5_final.ipynb)
   3. Analysis
      . Test data has few images which are oriented.
      . Model is underfitting
      . By adding few more paramaters and few additional epochs we can achieve the the consistent accuracy of more than 99.4 
      . Tried adding lr scheduler but could not gain more accuracy.
