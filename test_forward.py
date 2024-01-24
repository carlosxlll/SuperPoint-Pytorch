#-*-coding:utf8-*-
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import yaml
from model.superpoint_bn import SuperPointBNNet
import os
from copy import deepcopy

import yaml
import os
import torch
from tqdm import tqdm
from math import pi
from kornia.geometry.transform import warp_perspective as warp_perspective_kornia
import cv2
import numpy as np
from utils.params import dict_update
from solver.nms import box_nms
from utils.tensor_op import erosion2d
from dataset.utils.homographic_augmentation import sample_homography,ratio_preserving_resize
from model.magic_point import MagicPoint
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


def load_model(config, device='cpu'):
    model = SuperPointBNNet(config['model'],input_channel=1, grid_size=8, device=device, using_bn=True)
    if os.path.exists(config['model']['pretrained_model']):
        pre_model_dict = torch.load(config['model']['pretrained_model'])
        model_dict = model.state_dict()
        for k,v in pre_model_dict.items():
            if k in model_dict.keys() and v.shape==model_dict[k].shape:
                model_dict[k] = v
        model.load_state_dict(model_dict)
    model.to(device).eval()
    return model


with open('./config/superpoint_train.yaml', 'r', encoding='utf8') as fin:
    config = yaml.safe_load(fin)
device = 'cuda:0' #'cuda:2' if torch.cuda.is_available() else 'cpu'
model = load_model(config, device=device)


with torch.no_grad():

        img_path = '/root/workspace/code/mine/SuperPoint-Pytorch/data/coco/images/test2017/000000210916.jpg'
        img = cv2.imread(img_path, 0)#Gray image
        img = cv2.resize(img, (config['data']['resize'])[::-1])
        img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=device)
        valid_mask = torch.ones(img.shape, device=device)
        img_tensor = img_tensor/255.
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        ret = model(img_tensor)

        img = (img_tensor * 255).cpu().numpy().squeeze().astype(int).astype(np.uint8)
        img = cv2.merge((img, img, img))
        prob = ret['det_info']['prob_nms'].cpu().numpy().squeeze()
        keypoints = np.where(prob > 0.015)
        keypoints = np.stack(keypoints).T
        for kp in keypoints:
            cv2.circle(img, (int(kp[1]), int(kp[0])), radius=1, color=(0, 255, 0))
        cv2.imwrite('./data/sample/aa1.jpg', img)

print('Done')