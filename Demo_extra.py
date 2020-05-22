# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:18:24 2019
Image \Demo
@author: Shahadate2
"""

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

#import argparse
import os
import cv2

from PIL import Image
#import json
import logging
import torch
import numpy as np
#import skimage.draw as draw
#import tempfile
from pycocotools.coco import COCO

import sys
import random
import math
import re
import time

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
#from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
#from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils import cv2_util

from torchvision import transforms as T
from torchvision.transforms import functional as F
# In[8]:

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image
# In[8]:

#Contents from prediction.py
class COCODemo(object):
  
    def __init__(
        self,
        cfg,
        confidence_threshold=0.5,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform
    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        print(top_predictions)
        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox
        print(boxes)

        #colors = self.compute_colors_for_labels(labels).tolist()

        for box, label in zip(boxes, labels):
            
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            
            if  label == "BE":
              image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255,255,255), 1)
            elif  label == "suspicious":
              image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (0,222,0), 2)
              
              
            elif  label == "HGD":
              image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (120,20,120), 2)
              
            elif  label == "cancer":
              image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255, 0, 0), 4)
              
              
            elif  label == "polyp":
              image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255, 255, 0), 4)
            
           

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        #masker = Masker(threshold=0.5, padding=1)
        masks = predictions.get_field("mask").numpy()
        #print(masks.shape) 
       
       
       
#https://code.ihub.org.cn/projects/578/repository/revisions/master/entry/maskrcnn_benchmark/engine/inference.py
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
      

        for mask, label in zip(masks,labels):
            thresh = mask[0, :, :, None]
            
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            
                     
            if  label == "BE":
              #image = cv2.drawContours(image, contours, -1, (255,255,255), 2)
              mask2 = np.zeros_like(image)
              mask2 = cv2.drawContours(mask2, contours, -1, (255,255,255), -1)
              BE = np.zeros_like(image) # Extract out the object and place into output image
              BE[mask2 == 255] = image[mask2 == 255]
              BE_gray = cv2.cvtColor(BE, cv2.COLOR_BGR2GRAY)
              (thresh, BE) = cv2.threshold(BE_gray, 155, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
              cv2.imwrite(save_path+name+'_BE.tif', BE)
                         
              
                        
            elif  label == "suspicious":
              mask2 = np.zeros_like(image)
              mask2 = cv2.drawContours(mask2, contours, -1, (255,255,255), -1)
              S = np.zeros_like(image) # Extract out the object and place into output image
              S[mask2 == 255] = image[mask2 == 255]
              S_gray = cv2.cvtColor(S, cv2.COLOR_BGR2GRAY)
              #(thresh, S) = cv2.threshold(S_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
              (thresh, S) = cv2.threshold(S_gray, 155, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
              cv2.imwrite(save_path+name+'_suspicious.tif', S)
            elif  label == "HGD":
              mask2 = np.zeros_like(image)
              mask2 = cv2.drawContours(mask2, contours, -1, (255,255,255), -1)
              H = np.zeros_like(image) # Extract out the object and place into output image
              H[mask2 == 255] = image[mask2 == 255]
              H_gray = cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)
              (thresh, H) = cv2.threshold(H_gray, 155, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
              
              cv2.imwrite(save_path+name+'_HGD.tif', H)
              
            elif  label == "cancer":
              mask2 = np.zeros_like(image)
              mask2 = cv2.drawContours(mask2, contours, -1, (255,255,255), -1)
              C = np.zeros_like(image) # Extract out the object and place into output image
              C[mask2 == 255] = image[mask2 == 255]
              C_gray = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
              (thresh, C) = cv2.threshold(C_gray, 155, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
              #C=C.convertTo(img8bit, CV_32FC3)
              cv2.imwrite(save_path+name+'_cancer.tif', C)
              
            elif  label == "polyp":
              mask2 = np.zeros_like(image)
              mask2 = cv2.drawContours(mask2, contours, -1, (255,255,255), -1)
              P = np.zeros_like(image) # Extract out the object and place into output image
              P[mask2 == 255] = image[mask2 == 255]
              P_gray = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY)
              (thresh, P) = cv2.threshold(P_gray,155, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU )
              cv2.imwrite(save_path+name+'_polyp.tif', P)
            
             
            
            #cv2.imwrite('/home/zst19phu/benchmark/benchmark-code/datasets/edd2020/image.jpg', image) 
        composite = image

        return composite

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        print(scores)
        labels = predictions.get_field("labels").tolist()
        print(labels)

        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            if  label == "BE":
              cv2.putText(image, s, (x+25, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,200,0), 2)
              
            elif  label == "suspicious":
              cv2.putText(image, s, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX,  0.5,(0,222,0), 2)
            elif  label == "HGD":
              cv2.putText(image, s, (x+9, y+9), cv2.FONT_HERSHEY_SIMPLEX,  0.5,(120,20,120), 2)
            elif  label == "cancer":
              cv2.putText(image, s, (x+13, y+13), cv2.FONT_HERSHEY_SIMPLEX,  0.5,(255, 0, 0), 2)
            elif  label == "polyp":
              cv2.putText(image, s, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX,  0.5,(255, 255, 0), 2)
            
         

        return image

        
#
# In[ ]:
# # Visualise
# 
# Another important part of validating your model is visualising the results. This is done below

COCODemo.CATEGORIES = [
    "__background",
        "BE",
        "suspicious",
        "HGD",
        "cancer",
        "polyp",      
        
]
config_file = "/home/zst19phu/benchmark/benchmark-code/eddcode/etest.yaml"

cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

vis_demo = COCODemo(
    cfg, 
    min_image_size=800,
    confidence_threshold=0.5)

#imgfile='/home/shahadate2/benchmark_model/maskrcnn-benchmark2/datasets/coco/val2014/114D_23B56_1.bmp'
#pil_image = Image.open(imgfile).convert("RGB")
#image = np.array(pil_image)[:, :, [2, 1, 0]]
#predictions = vis_demo.run_on_opencv_image(image)

# In[ ]:#
#matplotlib.use('QT5Agg')
#plt.subplot(1, 2, 1)
#plt.imshow(image[:,:,::-1])
#plt.axis('off')
#plt.subplot(1, 2, 2)
#plt.imshow(predictions[:,:,::-1])
#plt.axis('off')
#plt.show()

# In[ ]:#
#matplotlib.use('QT5Agg')
# #In[ ]:#
val_path="/home/zst19phu/benchmark/benchmark-code/datasets/edd2020/eddval2020/2/" #this is the validation image data
import matplotlib.pylab as pylab
import cv2
pylab.rcParams['figure.figsize'] = 10, 10
#val_path='/home/shahadate2/benchmark_model/maskrcnn-benchmark/myEndoData/TestAug3img/'
 #this is the validation image data
imglistval = os.listdir(val_path) 
for name in imglistval:
    #save_path='/home/shahadate2/benchmark_model/maskrcnn-benchmark/myEndoData/savedAug1res100/'
    save_path='/home/zst19phu/benchmark/benchmark-code/datasets/edd2020/eddvaloutput/2/'
    #save_path='/home/shahadate2/benchmark_model/maskrcnn-benchmark/myEndoData/savedAug4res100/'
    imgfile = val_path + name
    print(imgfile)
    pil_image = Image.open(imgfile).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
 
    predictions = vis_demo.run_on_opencv_image(image) # forward predict
    #vis_demo.write_mask(image,predictions)
    
    plt.imshow(predictions[:,:,::-1])
    plt.axis('off')
    
    plt.savefig(save_path+name,dpi=500)
    




#imglistval = os.listdir(val_path) 
#for name in imglistval:
   # imgfile = val_path + name
   # pil_image = Image.open(imgfile).convert("RGB")
    #image = np.array(pil_image)[:, :, [2, 1, 0]]
 
    #predictions = vis_demo.run_on_opencv_image(image) # forward predict
    #plt.subplot(1, 2, 1)
   # plt.imshow(image[:,:,::-1])
    #plt.axis('off')
 
    #plt.subplot(1, 2, 2)
    #plt.imshow(predictions[:,:,::-1])
    #plt.axis('off')
    #plt.show()
#







# In[ ]:#
