import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import sys
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_
from collections import defaultdict
import json, re
from torchsummary import summary
from thop import profile

OUTPUT_DIR="./dump_scores/THUMOS2014/train"

# Stats global variables
STATS_TOT_WINDOWS=0

def main():
    global args
    args = parser.parse_args()

    print("------------------------------------")
    print("Environment Versions:")
    print("- Python: {}".format(sys.version))
    print("- PyTorch: {}".format(torch.__version__))
    print("- TorchVison: {}".format(torchvision.__version__))

    args_dict = args.__dict__
    print("------------------------------------")
    print(args.arch+" Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    if args.dataset == 'ucf101':
        num_class = 101
        rgb_read_format = "{:06d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'kinetics':
        num_class = 400
        rgb_read_format = "{:04d}.jpg"
    elif args.dataset == 'something':
        num_class = 174
        rgb_read_format = "{:04d}.jpg"
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, args.num_segments, args.pretrained_parts, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    print_model(model)
    # model = torch.nn.DataParallel(model) # CPU

    print("pretrained_parts: ", args.pretrained_parts)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            # checkpoint = torch.load(args.resume, map_location='cpu') # CPU
            # if not checkpoint['lr']:
            if "lr" not in checkpoint.keys():
                args.lr = input("No 'lr' attribute found in resume model, please input the 'lr' manually: ")
                args.lr = float(args.lr)
            else:
                args.lr = checkpoint['lr']
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch: {}, lr: {})"
                  .format(args.resume, checkpoint['epoch'], args.lr)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    else:
        print("Please specify the checkpoint to pretrained model")
        return

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        #input_mean = [0,0,0] #for debugging
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    end = time.time()
    # data_loader = torch.utils.data.DataLoader(
    dataset = TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       #Stack(roll=(args.arch == 'C3DRes18') or (args.arch == 'ECO') or (args.arch == 'ECOfull') or (args.arch == 'ECO_2FC')),
                       #ToTorchFormatTensor(div=(args.arch != 'C3DRes18') and (args.arch != 'ECO') and (args.arch != 'ECOfull') and (args.arch != 'ECO_2FC')),
                       normalize,
                   ]),
                   test_mode=True,
                   window_size=64, window_stride=16);
    data_loader = torch.utils.data.DataLoader(dataset,
                      batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True,
                      collate_fn=collate_fn)

    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # predict(data_loader, model, criterion, 0)
    predict(dataset, model, criterion=None, iter=0)
    # profile_model(model)
    elapsed_time = time.time() - end    
    print("STATS_TOT_WINDOWS={0}, Total prediction time={1}".format(STATS_TOT_WINDOWS, elapsed_time))
    return

def predict(data_loader, model, criterion, iter, logger=None):
    global STATS_TOT_WINDOWS
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)

    # switch to evaluate mode
    model.eval()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i, (windows, video) in enumerate(data_loader):
        # discard final batch
        if i == len(data_loader)-1:
            break
        print(('Window length {0} {1}'.format(video, len(windows))))
        output_dict = defaultdict(lambda: defaultdict(list))
        for j, (input, target) in enumerate(windows):
            # target = target.cuda(async=True)
            input_var = input 
            target_var = target

            end = time.time()
            # compute output
            output = model(input_var)
            output_dict["window_" + str(j)]["rgb_scores"] = output.cpu().numpy().flatten().tolist()
            # loss = criterion(output, target_var)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            # losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))
            # top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            STATS_TOT_WINDOWS += 1

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(data_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5)))
        # write to json file
        video_name = process_video_string(video)
        with open(OUTPUT_DIR + "/" + video_name, 'w') as outfile:
            json.dump(output_dict, outfile)

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          ' Prediction time {pred_time.avg:.3f}'
          .format(top1=top1, top5=top5, loss=losses, pred_time=batch_time)))
    
    return top1.avg

def process_video_string(line):
    words = re.split('/', line)
    video = words[-1]
    video = re.split('\.', video)
    return video[0].strip()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def collate_fn(batch):
    windows, video = zip(*batch)
    return windows[0], video[0]

def print_model(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable model paramters: {0}'.format(total_params))
    print("=================Summary==================")
    summary(model, input_size=(12, 224, 224))

def profile_model(model):
    input = torch.randn(12, 224, 224)
    flops, params = profile(model, inputs=(input, ))
    print("Flops:", flops)
    
if __name__ == '__main__':
    main()
