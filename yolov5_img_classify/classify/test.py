# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import create_classification_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_img_size, check_requirements, colorstr,
                           increment_path, print_args)
from collections import defaultdict
import sklearn.metrics as mtc
from sklearn.metrics import classification_report
from mutils import read_split_data, train_one_epoch, evaluate, read_video_data
from video_dataset import VideoDataSet
from utils.torch_utils import (ModelEMA, model_info, reshape_classifier_output, select_device, smart_DDP,
                               smart_optimizer, smartCrossEntropyLoss, torch_distributed_zero_first)
from models.yolo import ClassificationModel, DetectionModel,VideoClassificationModel,VideoClassificationModel2,VideoClassificationModel3,VideoClassificationModel4,VideoClassificationModel5,VideoClassificationModel6,VideoClassificationModel7,VideoClassificationModel8,VideoClassificationModel9
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    data=ROOT / '../datasets/mnist',  # dataset dir
    weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
    batch_size=128,  # batch size
    imgsz=224,  # inference size (pixels)
    device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    project=ROOT / 'runs/val-cls',  # save to project/name
    name='exp',  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    criterion=None,
    pbar=None,
):
    # device = next(model.parameters()).device
    device = select_device(device)
    print(device)
    pt = True  # get model device, PyTorch model
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()
 

    # Dataloaderdata_dir'
    data_dir = r"/home/lsh/yolov5_7.0/datasets/output_dataset20250102/"
    val_images_path, val_images_label = read_video_data(os.path.join(data_dir, "test2"))
    val_dataset = VideoDataSet(video_path=val_images_path,
                             video_class=val_images_label)
                             
    dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2,
                                             collate_fn=val_dataset.collate_fn)

    model.eval()
    # print(model)
    pred, targets, loss, dt = [], [], 0, (Profile(), Profile(), Profile())
    n = len(dataloader)  # number of batches
    action = 'validating'
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    
    val_metrics=defaultdict(float)  
    pred_macro = [] 
    bar = tqdm(dataloader, desc, n, False, bar_format=TQDM_BAR_FORMAT, position=0)
    with torch.amp.autocast('cuda'):
        for images, labels in bar:
            with dt[0]:
                images, labels = images.to(device, non_blocking=True), labels.to(device)

            with dt[1]:
                y = model(images)

            with dt[2]:
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                
                ts = y.argsort(1, descending=True)[:, :1]
                ts = torch.transpose(ts, 0, 1)
                pred_macro.append(ts[0])
                
                if criterion:
                    loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    pred_macro = torch.cat(pred_macro)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()
    print("top1: ", top1)

    targets_array = targets.cpu()
    pred_array = pred.cpu()
    pred_macro = pred_macro.cpu()
    
    micro_precision=mtc.precision_score(targets_array, pred_macro, average="micro")     
    micro_recall=mtc.recall_score(targets_array, pred_macro, average="micro")
    micro_f1=mtc.f1_score(targets_array, pred_macro, average="micro")  
    val_metrics['micro_precision']=micro_precision
    val_metrics['micro_recall']=micro_recall
    val_metrics['micro_f1']=micro_f1
    macro_precision=mtc.precision_score(targets_array,pred_macro, average="macro")     
    macro_recall=mtc.recall_score(targets_array, pred_macro, average="macro")
    macro_f1=mtc.f1_score(targets_array, pred_macro, average="macro")      
    mcc=mtc.matthews_corrcoef(targets_array, pred_macro)
    val_metrics['macro_precision']=macro_precision
    val_metrics['macro_recall']=macro_recall
    val_metrics['macro_f1']=macro_f1
    val_metrics['mcc']=mcc

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    
    print(classification_report(targets_array, pred_macro, target_names= list(map(str, [1, 2, 3, 4, 5, 6, 7]))))

    return val_metrics


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '../datasets/mnist', help='dataset path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--verbose', nargs='?', const=True, default=True, help='verbose output')
    parser.add_argument('--project', default=ROOT / 'runs/val-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VideoClassificationModel9().cuda()
    model0 = torch.load(r'../runs/train-cls/exp7/weights/best.pt',map_location=device)
    model_dict = model0['model'].state_dict()
    model.load_state_dict(model_dict,strict=False)
    
    model.eval()
    
    val_metrics = run(model=model) 
    outputs=[]
    for k in val_metrics.keys():
      if k=='dice_coeff' or k=='dice' or k=='bce':
        outputs.append('{}:{:4f}'.format(k,val_metrics[k]/(16*50)))
      else:
        outputs.append('{}:{:2f}'.format(k,val_metrics[k]))
      print('{}'.format(','.join(outputs)))
