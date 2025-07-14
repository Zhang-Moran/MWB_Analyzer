import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils_cam import GradCAM, show_cam_on_image, center_crop_img
from models.yolo import VideoClassificationModel,VideoClassificationModel3,VideoClassificationModel4,VideoClassificationModel5,VideoClassificationModel6,VideoClassificationModel7,VideoClassificationModel2,VideoClassificationModel8,VideoClassificationModel9

# 1-1 3-2 6-3 7-4 2-5
def main():
    model = VideoClassificationModel8() 
    model_weight_path =  r'./runs/train-cls/exp6/weights/best.pt'
    loaded_model = torch.load(model_weight_path, map_location='cpu')
    model.load_state_dict(loaded_model['model'].state_dict() ,strict=False)
    model.eval()

    target_layers = [model.conv9.conv]

    # load image
    img_path = "/home/lsh/yolov5_7.0/yolov5_img_classify/gradcam_data"
    img_list = os.listdir(img_path)
    
    for v in img_list:
      path = os.path.join(img_path, v)
      assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
      print(v)      
      img = cv2.imread(path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img, dsize=(640,640))
      img = Image.fromarray(np.uint8(img))
      img1 = np.array(img)
      img = np.transpose(img1, (2, 0, 1))
      
      img = torch.tensor(img)
      img = img.float()
      img = (img - 127.5) / 127.5
      
      img = torch.unsqueeze(img, dim=0)
      
      cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

      grayscale_cam = cam(input_tensor=img,target_category=None)

      grayscale_cam = grayscale_cam[0, :]
      
      visualization = show_cam_on_image(img1.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
      plt.imshow(visualization)
      plt.axis('off')
      plt.xticks([])
      plt.yticks([])
      plt.savefig('res3/{}_5.png'.format(v), bbox_inches='tight')


if __name__ == '__main__':
    main()
