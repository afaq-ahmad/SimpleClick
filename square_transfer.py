import sys
import pickle
from pathlib import Path

import cv2
import torch
import numpy as np

import torch

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from isegm.inference.clicker import Clicker

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        
class Square_Predictor:
  def __init__(self, checkpoint = './weights/simpleclick_models/cocolvis_vit_huge.pth', gpus = '0',cpu = False,
               target_iou = 0.90,thresh = 0.49,object_detect_model = 'yolov5s'):
    self.checkpoint = checkpoint
    self.gpus = gpus #ID of used GPU.
    self.cpu = cpu #True #Use only CPU for inference.
    self.target_iou = target_iou #Target IoU threshold for the NoC metric. (min possible value = 0.8)
    self.thresh = thresh #The segmentation mask is obtained from the probability outputs using this threshold.

    eval_ritm = False
    # from isegm.utils.exp import load_config_file
    # config_path = './config.yml'
    # cfg = load_config_file(config_path, return_edict=True)
    
    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f"cuda:{gpus.split(',')[0]}")

    model = utils.load_is_model(checkpoint, device, eval_ritm)
    predictor_params = {}
    zoomin_params = {'skip_clicks': -1,'target_size': (448, 448)}
    interpolate_pos_embed_inference(model.backbone, zoomin_params['target_size'], device)

    self.predictor = get_predictor(model, 'NoBRS', device,
                              prob_thresh=self.thresh,
                              predictor_params=predictor_params,
                              zoom_in_params=zoomin_params)
                              
    self.detector = torch.hub.load('ultralytics/yolov5', object_detect_model)  # yolov5n - yolov5x6 or custom
    self.colors = Colors()
    
  def predict_rectangle(self,image):
    """
    image: RGB input image
    object_detection_mask: Yolo object mask image <box(0,1)>
    
    """
    detection_results = self.detector(image)
    
    return detection_results
    
    
  def predict_mask_all(self,image,detection_results,margin_per = 0.0):
    """
    image: RGB input image
    object_detection_mask: Yolo object mask image <box(0,1)>
    
    """
    detection_results = detection_results.xyxy[0].cpu().numpy().astype(int)
    Results_masks = []
    for object_crd in detection_results:
        
        coordinates_height = object_crd[3]-object_crd[1]
        coordinates_width = object_crd[2]-object_crd[0]
        margin_per_h = int(coordinates_height * margin_per)
        margin_per_w = int(coordinates_width * margin_per)

        object_detection_mask = np.zeros((image.shape[0],image.shape[1]),'int32')
        object_detection_mask[object_crd[1]+ margin_per_h:object_crd[3] - margin_per_h,object_crd[0] + margin_per_w:object_crd[2] - margin_per_w]=1
    
        pred_mask = self.predict_mask(image,object_detection_mask)
        
        contours, hierarchy = cv2.findContours(pred_mask.astype('uint8'), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        Results_masks.append(contours)
    return Results_masks
    
    
  def predict_mask(self,image,object_detection_mask):
    """
    image: RGB input image
    object_detection_mask: Yolo object mask image <box(0,1)>
    
    """
    clicker = Clicker(gt_mask=object_detection_mask)
    pred_mask = np.zeros((image.shape[0],image.shape[1]))
    with torch.no_grad():
        self.predictor.set_input_image(image)
        clicker.make_next_click(pred_mask)
        pred_probs = self.predictor.get_prediction(clicker)
        pred_mask = pred_probs > self.thresh

    return pred_mask
    
  def result_draw(self,image,yolo_results,masks):
    """
    Drawing masks and rectangle on image
    """
    detected_classes = yolo_results.xyxy[0].cpu().numpy().astype(int)[:,-1]
    for index in range(len(detected_classes)):
        image = cv2.drawContours(image, masks, index, self.colors(detected_classes[index]), 3)
    yolo_results.show()
    
    return image
        
    
    