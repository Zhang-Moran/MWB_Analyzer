# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel,VideoClassificationModel,VideoClassificationModel2,VideoClassificationModel3,VideoClassificationModel4
from utils.dataloaders import create_classification_dataloader
from utils.general import (DATASETS_DIR, LOGGER, TQDM_BAR_FORMAT, WorkingDirectory, check_git_info, check_git_status,
                           check_requirements, colorstr, download, increment_path, init_seeds, print_args, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (ModelEMA, model_info, reshape_classifier_output, select_device, smart_DDP,
                               smart_optimizer, smartCrossEntropyLoss, torch_distributed_zero_first)
from mutils import read_split_data, train_one_epoch, evaluate, read_video_data
from video_dataset import VideoDataSet
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Save run settings
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    data_dir = r"/home/lsh/yolov5_7.0/yolov5/datasets/20230106TSM4/output_dataset/"
    # Dataloaders
    nc = 7

    train_images_path, train_images_label = read_video_data(os.path.join(data_dir, "train"))
    val_images_path, val_images_label = read_video_data(os.path.join(data_dir, "val"))

    train_dataset = VideoDataSet(video_path=train_images_path,
                              video_class=train_images_label)
    val_dataset = VideoDataSet(video_path=val_images_path,
                             video_class=val_images_label)
    
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=bs // WORLD_SIZE,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2,
                                               collate_fn=train_dataset.collate_fn)

    testloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=bs // WORLD_SIZE,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2,
                                             collate_fn=val_dataset.collate_fn)


    # Model
    
    model = VideoClassificationModel4()  # convert to classification model

    pretrained_model = torch.load(r'/home/lsh/yolov5_7.0/yolov5/yolov5s-cls.pt')
    pretrained_dict = pretrained_model['model'].state_dict()

    for m in model.modules():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # for training

    # load pre-train
    model_dict = model.state_dict()
    for name, param in model_dict.items():
        print(name)
    for name, param in pretrained_dict.items():
        n0 = name[6:]
        n = "conv"+n0
        n1 = "conv"+n0[0]+".0"+n0[1:]

        n2 = "conv" + n0[0] + ".1" + n0[1:4]+"0"+n0[5:]
        n3 = "conv" + n0[0] + ".2" + n0[1:4]+"0"+n0[5:]
        # if name in model_dict:
        #      if model_dict[name].size() == param.size():
        #          print(f'Loading weight for layer: {name}')
        #          model_dict[name] = param
        if n in model_dict:
          if model_dict[n].size() == param.size():
                 print(f'Loading weight for layer: {name}')
                 model_dict[n] = param
          # else:
          #      print(
          #           f'Skipping layer (size mismatch): {name}, pretrained: {param.size()}, model: {model_dict[n].size()}')
        elif n1 in model_dict:
          if model_dict[n1].size() == param.size():
                 print(f'Loading weight for layer: {name}')
                 model_dict[n1] = param
          else:
                 print(
                     f'Skipping layer (size mismatch): {name}, pretrained: {param.size()}, model: {model_dict[n1].size()}')
        elif n2 in model_dict and n0[4]=="1":
          if model_dict[n2].size() == param.size():
                 print(f'Loading weight for layer: {name}')
                 model_dict[n2] = param
          else:
                 print(
                     f'Skipping layer (size mismatch): {name}, pretrained: {param.size()}, model: {model_dict[n2].size()}')
        elif n3 in model_dict and n0[4]=="2":
          if model_dict[n3].size() == param.size():
                 print(f'Loading weight for layer: {name}')
                 model_dict[n3] = param
          else:
                 print(
                     f'Skipping layer (size mismatch): {name}, pretrained: {param.size()}, model: {model_dict[n3].size()}')

        else:
            print(f'Skipping layer (not found in model): {name}')
    
    model.load_state_dict(model_dict)
    # end load pre-train

    if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model, device_ids=[2])

    model = model.to(device)
    print(model)
    
    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # Schedulers
    # lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.8, verbose=True)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Train
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = torch.amp.GradScaler('cuda')
    val = 'val'
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()

        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, data in  pbar:  # progress bar
            images, labels = data
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with torch.amp.autocast('cuda'):  # stability issues when enabled
                loss = criterion(model(images), labels)

            # Backward
            torch.use_deterministic_algorithms(False)
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                # Test
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss, val_metrics = validate.run(model=ema.ema,
                                                     dataloader=testloader,
                                                     criterion=criterion,
                                                     pbar=pbar)  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy

        # Scheduler
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr}")
        
        outputs=[]
        for k in val_metrics.keys():
          if k=='dice_coeff' or k=='dice' or k=='bce':
            outputs.append('{}:{:4f}'.format(k,val_metrics[k]/(16*50)))
          else:
            outputs.append('{}:{:2f}'.format(k,val_metrics[k]))
        print('{}'.format(','.join(outputs)))
        
        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]['lr']}  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                    f"\nResults saved to {colorstr('bold', save_dir)}"
                    f"\nPredict:         python classify/predict.py --weights {best} --source im.jpg"
                    f"\nValidate:        python classify/val.py --weights {best} --data {data_dir}"
                    f"\nExport:          python export.py --weights {best} --include onnx"
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')"
                    f"\nVisualize:       https://netron.app\n")

        # Plot examples
        images, labels = (x[:25] for x in next(iter(testloader)))  # first 25 images and labels
        pred = torch.max(ema.ema(images.to(device)), 1)[1]
        file = imshow_cls(images, labels, pred, model.names, verbose=False, f=save_dir / 'test_images.jpg')

        # Log results
        meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        logger.log_images(file, name='Test Examples (true-predicted)', epoch=epoch)
        logger.log_model(best, epochs, metadata=meta)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s-cls.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='imagenette160', help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--epochs', type=int, default=155, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True, help='start from i.e. --pretrained False')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default= 0.0001, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout (fraction)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)


def run(**kwargs):
    # Usage: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
