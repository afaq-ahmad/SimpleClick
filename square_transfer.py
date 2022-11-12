import sys
import pickle
from pathlib import Path

import cv2
import torch
import numpy as np


sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
from isegm.inference.clicker import Clicker

class Square_Predictor:
  def __init__(self, checkpoint = './weights/simpleclick_models/cocolvis_vit_huge.pth', gpus = '0',cpu = False,
               target_iou = 0.90,thresh = 0.49):
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
        pred_mask = pred_probs > 0.49

    return pred_mask