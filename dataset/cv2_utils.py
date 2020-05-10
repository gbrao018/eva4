import sys
#sys.path.append('/content/gdrive/My Drive/eva-04/S15A')
import cv2
import numpy as np
import glob
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image
import matplotlib.pyplot as plt

def resize_images(resize, input_path, save_path):
    #images = []
    for file in glob.glob(input_path):
      img = cv2.imread(file)
      filepath = os.path.basename(file)
      width = resize # int(img.shape[1] * scale_percent / 100)
      height = resize# int(img.shape[0] * scale_percent / 100)
      dim = (width, height)
      # resize image
      resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)#.convert('BGR')
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()
      height, width, channels = resized.shape
      #save back the image
      #save_path = "/content/gdrive/My Drive/eva-04/S15A/room/bg150/"+filepath
      cv2.imwrite(save_path,resized)
      #images.append(resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from IPython.display import display, Image
from IPython.display import display

def show_images_in_row(images_array):
    plt.figure(figsize=(150,150))
    #columns = 100
    columns = len(images_array)
    for i, image in enumerate(images_array):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

#(x,y) is the position of the image to be placed    
def overlay_transparent3(bg, overlay, x, y, clone = True):
    if clone == True:
        bg = np.array(bg, copy=True)
        
    background_width = bg.shape[1]
    background_height = bg.shape[0]

    if x >= background_width or y >= background_height:
        return bg

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]
    
    print("overlay shape=",overlay.shape)
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )
    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    
    #mask creation
    #overlay_image = overlay[..., :3]
    #w,h = overlay.shape[:2]
    #b,g,r = cv2.split(overlay)
    #mask = np.dstack((a,a,a))
    #mask = overlay[..., 3:] / 255.0
    #overlay_image = np.dstack((b,g,r))
    print("mask shape=",mask.shape,"overlay_image shape=",overlay_image.shape)
    print(x,y,w,h)

    #print("bg shape before=", bg.shape)
    canvasbg = bg[y:y+h, x:x+w] = (1.0 - mask) * bg[y:y+h, x:x+w] + mask * overlay_image
    
    imask = mask>0
    #canvasbg[imask] = overlay_image[imask] # overlay the foreground on background
    
    #bg[y:y+h, x:x+w] = (1.0 - mask) * bg[y:y+h, x:x+w] + mask * overlay_image
    #bg1 = (1.0 - mask) * bg[y:y+h, x:x+w]
    #print("bg shape after=", bg.shape)
    #mask_overlay = mask * overlay_image
    #print("mask_overlay shape=", mask_overlay.shape)
    #bg1 = bg1 + mask_overlay

    #background = cv2.addWeighted(background,0.7,overlay,0.3,0.1)   
    return bg,canvasbg

def crop_image_from_path(image_path,y,x,h,w, clone = True):
    img = cv2.imread(image_path)
    if clone == True:
        img = np.array(img, copy=True) 
    canvas = img[y:y+h, x:x+w] # from (x,y) -> (h,w) region will be cropped
    cv2.waitKey(0)
    return canvas


def crop_image(image,y_bg,x_bg,h,w, clone = True):
    img = image
    if clone == True:
        img = np.array(image, copy=True)
    canvas = img[y_bg:y_bg+h, x_bg:x_bg+w] # from (x,y) -> (h,w) region will be cropped
    cv2.waitKey(0)
    return canvas

# This returns the bgfg,bgfg_mask,
def create_fgbg_overlay_and_mask(bg,fg,x_roi,y_roi,clone = True):
    #Step1 Create mask from fg
    bw,bh = bg.shape[:2]
    w,h = fg.shape[:2]
    b,g,r,a = cv2.split(fg)
    mask = np.dstack((a,a,a))
    
    #Step2 . Create overlay image
    fgbg_image, fgbg_canvas = overlay_transparent3(bg, fg, x_roi,y_roi,clone)
    #step2. create ,ask for the image
    blackimage = np.zeros([bh,bw,3],dtype=np.uint8)
    fgbg_mask, fgbg_mask_canvas = overlay_transparent3(blackimage, mask, x_roi,y_roi,clone)
    return fgbg_image,fgbg_mask

import os

#save folder should end with /
def flip_images_and_store(input_path,save_folder,resize):
    #imread,cv2.flip(),imwrite,
    #input_path = "/content/gdrive/My Drive/eva-04/S15A/room/bg/*.jpg"
    #output_path = "/content/gdrive/My Drive/eva-04/S15A/room/bg/"
    images = []
    for file in glob.glob(input_path):
        save_path = ""
        img = cv2.imread(file,-1)
        filepath = os.path.basename(file)
        #save_path = save_folder+filepath
        #cv2.imwrite(save_path,resized)
        width = resize # int(img.shape[1] * scale_percent / 100)
        height = resize# int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)#.convert('BGR')
        resized_flip = cv2.flip(resized,flipCode=1)
        save_flip_path = save_folder+"flip_"+filepath
        cv2.imwrite(save_flip_path,resized_flip)
        save_path = save_folder+filepath
        cv2.imwrite(save_path,resized)
