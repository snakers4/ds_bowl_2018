#============ Custom tensorboard logging ============#
from utils.TbLogger import Logger

#============ Basic imports ============#
import argparse
import os
import shutil
import time
import tqdm
import glob
from skimage.io import imsave,imread_collection,imread
import pandas as pd
from PIL import Image
import pickle
import gc
import cv2
import copy

cv2.setNumThreads(0)

#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn import Sigmoid

#============ Metrics ============#
from sklearn.metrics import f1_score

#============ Models with presets ============#
from models.model_params import model_presets

#============ Loss ============#
from Loss import AVDiceLoss

#============ Utils and augs ============#
from utils.BDataset import resolution_dict,BDatasetPad
# exclude 1024x1024 as they spoil the training process
from utils.BDataset import resolution_list as resolution_list
from utils.RAugs import *
from utils.BAugs import BAugsNoResize,BAugsValNoResize,BAugsNoResizeCrop
from utils.LRScheduler import CyclicLR
from utils.metric import calculate_ap
from utils.watershed import label_baseline as wt_baseline
from utils.watershed import energy_baseline as wt_seeds
from utils.utils import str2bool,restricted_float,to_np,rle_encode,rle_encoding

parser = argparse.ArgumentParser(description='Kaggle DS Bowl 2018')

#============ basic params ============#
parser.add_argument('--arch', '-a', metavar='ARCH', default='linknet34',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-nch', '--channels', default=2, type=int,
                    metavar='CN', help='number of mask channels to use (default: 2)')
parser.add_argument('--fold_num', '-fld', default=0, type=int,
                    metavar='FLD', help='fold number 0 - 3 (default: 0)')

#============ optimization params ============#
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--freeze', default=False, type=str2bool,
                    metavar='FR', help='whether to freeze the encoder')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    help='model optimizer')
parser.add_argument('-bcew', '--bce_weight', default=1, type=float,
                    help='weight for BCE part of the loss')
parser.add_argument('-dicew', '--dice_weight', default=1, type=float,
                    help='weight for DICE part of the loss')
parser.add_argument("--ths", default=0.3, type=restricted_float,
                    help='threshold applied before watershed transform')

#============ logging params and utilities ============#
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lognumber', '-log', default='test_model', type=str,
                    metavar='LN', help='text id for saving logs')
parser.add_argument('--tensorboard', default=False, type=str2bool,
                    help='Use tensorboard to for loss visualization')
parser.add_argument('--tensorboard_images', default=False, type=str2bool,
                    help='Use tensorboard to see images')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--is_img_augs', default=False, type=str2bool,
                    help='Use heavier imgaugs augs')
parser.add_argument('--is_distance_transform', default=False, type=str2bool,
                    help='Predict distance to the boundary')
parser.add_argument('--is_boundaries', default=False, type=str2bool,
                    help='Also predict cell boundaries')
parser.add_argument('--is_vectors', default=False, type=str2bool,
                    help='Also predict unit vector coordinates coorresponding to distance to boundary')

#============ other params ============#
parser.add_argument('-pr', '--predict', dest='predict', action='store_true',
                    help='generate prediction masks')
parser.add_argument('-pr_train', '--predict_train', dest='predict_train', action='store_true',
                    help='generate prediction masks')
parser.add_argument('-eval', '--evaluate', dest='evaluate', action='store_true',
                    help='just evaluate')


best_val_f1_score = -1
best_map_score = -1

train_minib_counter = 0
valid_minib_counter = 0
pred_minib_counter = 0

args = parser.parse_args()

print(args)
# time.sleep(3)

if not (args.predict or args.predict_train):
    args.lognumber = args.lognumber + '_fold' + str(args.fold_num)

# remove the log file if it exists if we run the script in the training mode
"""
if not (args.predict or args.predict_train):
    print('Folder {} delete triggered'.format(args.lognumber))
    try:
        shutil.rmtree('tb_logs/{}/'.format(args.lognumber))
    except:
        pass
"""

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    if not (args.predict or args.predict_train):
        logger = Logger('./tb_logs/{}'.format(args.lognumber))
    else:
        logger = Logger('./tb_logs/{}'.format(args.lognumber + '_predictions'))        

def main():
    global logger, args, best_val_f1_score,best_map_score
    
    print('Using parameter preset {}'.format(args.arch))
    print('Model parameters:\n {}'.format(model_presets[args.arch][1]))
    model = model_presets[args.arch][0](**model_presets[args.arch][1])

    # train on all GPUs for speed
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    # freeze the encoder if required
    if args.freeze:
        print('Encoder frozen')        
        model.module.require_encoder_grad(False)
    else:
        print('Encoder unfrozen')
        model.module.require_encoder_grad(True)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_map_score = checkpoint['best_map_score']
            # best_map_score = checkpoint['best_map_score']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(False, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
  
    # predict loops
    if (args.predict or args.predict_train):
        # read the test df    
        test_df = pd.read_pickle('../data/test_df_stage1_meta')
        test_df = test_df.reset_index() 

        test_resl_list = list(test_df.w_h.unique())
        submit_df = pd.DataFrame(columns = ['ImageId','EncodedPixels'])
        
        model1 = copy.deepcopy(model)
        model2 = copy.deepcopy(model)
        model3 = copy.deepcopy(model)   
        
        # only one fold
        # for mdl,fold in zip([model,model1,model2,model3],range(0,4)):
        for mdl,fold in zip([model],[0]):
            print('weights/' + args.lognumber + '_fold' + str(fold) + '_best.pth.tar')
            checkpoint = torch.load('weights/' + args.lognumber + '_fold' + str(args.fold_num) + '_best.pth.tar')
            mdl.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' for fold {}"
                  .format(checkpoint['epoch'], fold))

        for test_resl in test_resl_list:

            print('Predicting for the resolution {} ...' .format(test_resl))        

            predict_augs = BAugsValNoResize(mean=model.module.mean,
                                        std=model.module.std)        

            predict_dataset = BDatasetPad(df = test_df,
                             transforms = predict_augs,
                             fold_num = 0,
                             mode = 'test',
                             dset_resl = test_resl,
                             is_crop = False,
                             is_distance_transform = False                                                      
                             )
            
            predict_loader = torch.utils.data.DataLoader(
                predict_dataset,
                batch_size=args.batch_size,        
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                drop_last=False)

            rle_nuclei = predict(predict_loader,
                                 model,model1,model2,model3)
            
            del predict_dataset,predict_loader,predict_augs

            submit_df = submit_df.append(rle_nuclei)

        print('Saving submission df ...')
        submit_df = submit_df.set_index('ImageId')
        submit_df = submit_df.sort_index()
        submit_df.to_csv('../submissions/{}.csv'.format(args.lognumber))
    
    # evaluate loop
    elif args.evaluate:

        # read the train df    
        train_df = pd.read_pickle('../data/train_df_stage1_meta')
        train_df = train_df.reset_index()
 
        val_augs = BAugsValNoResize(mean=model.module.mean,
                            std=model.module.std)    
    
        criterion = AVDiceLoss(bce_weight=float(args.bce_weight),
                               dice_weight=float(args.dice_weight),
                               is_vectors=args.is_vectors).cuda()
        
        if args.optimizer.startswith('adam'):           
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                        lr = args.lr)
        elif args.optimizer.startswith('rmsprop'):
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                        lr = args.lr)
        elif args.optimizer.startswith('sgd'):
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                        lr = args.lr)        
        else:
            raise ValueError('Optimizer not supported')           

        scheduler = MultiStepLR(optimizer, milestones=[100,250], gamma=0.1)            
            
        val_dset_lengths = []
        val_losses = []
        val_f1_scores = []
        val_map_scores_wt = []
        val_map_scores_wt_seed = []

        # resolution order is shuffled each time
        # train loop
        train_datasets = [] 

        # val loop
        for resl_key,source_resl,target_resl in resolution_list:

            # for resl_key,source_resl,target_resl in resolution_list:
            # we will need to resize back to calculate metrics                

            val_dataset = BDatasetPad(df = train_df,
                             transforms = val_augs,
                             fold_num = args.fold_num,
                             mode = 'val',
                             dset_resl = resl_key,
                             is_crop = False,
                             is_img_augs = args.is_img_augs,
                             is_distance_transform = args.is_distance_transform,
                             is_boundaries = args.is_boundaries,
                             is_vectors = args.is_vectors
                             )

            print('Validating on {} resl\tVal dataset length {}'.format(resl_key,len(val_dataset)))

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=2,        
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False)

            if len(val_dataset)>0:
                # evaluate on validation set
                val_loss, val_f1_score, val_map_score_wt, val_map_score_wt_seed = validate(val_loader,
                                                                 model,
                                                                 criterion,
                                                                 scheduler,
                                                                 source_resl,
                                                                 target_resl)
                val_dset_lengths.append(len(val_dataset))
                val_losses.append(val_loss)
                val_f1_scores.append(val_f1_score)
                val_map_scores_wt.append(val_map_score_wt)
                val_map_scores_wt_seed.append(val_map_score_wt_seed)

            torch.cuda.empty_cache()
            del val_dataset
            del val_loader
            gc.collect


        val_loss_avg = np.inner(np.asarray(val_losses),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()
        val_f1_avg = np.inner(np.asarray(val_f1_scores),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()
        val_map_avg_wt = np.inner(np.asarray(val_map_scores_wt),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()
        val_map_avg_wt_seed = np.inner(np.asarray(val_map_scores_wt_seed),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()            

        print(' * OVERALL Avg Val  Loss {loss:.4f}'.format(loss=val_loss_avg))
        print(' * OVERALL Avg F1   Score {f1_scores:.4f}'.format(f1_scores=val_f1_avg))
        print(' * OVERALL Avg MAP1 Score {map_scores_wt:.4f}'.format(map_scores_wt=val_map_avg_wt)) 
        print(' * OVERALL Avg MAP2 Score {map_scores_wt_seed:.4f}'.format(map_scores_wt_seed=val_map_avg_wt_seed)) 
                
    # train loop
    else:
        # read the train df    
        train_df = pd.read_pickle('../data/train_df_stage1_meta')
        train_df = train_df.reset_index()
        
        criterion = AVDiceLoss(bce_weight=float(args.bce_weight),
                               dice_weight=float(args.dice_weight),
                               is_vectors=args.is_vectors).cuda()

        if args.optimizer.startswith('adam'):           
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                        lr = args.lr)
        elif args.optimizer.startswith('rmsprop'):
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                        lr = args.lr)
        elif args.optimizer.startswith('sgd'):
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                        lr = args.lr)        
        else:
            raise ValueError('Optimizer not supported')        


        # scheduler = ReduceLROnPlateau(optimizer = optimizer,
        #                                           mode = 'max',
        #                                          factor = 0.1,
        #                                          patience = 5,
        #                                          verbose = True,
        #                                          threshold = 1e-3,
        #                                          min_lr = 1e-6
        #                                          )

        scheduler = MultiStepLR(optimizer, milestones=[20,150], gamma=0.1)  
        # scheduler = MultiStepLR(optimizer, milestones=[150], gamma=0.1)  


        
        # resolution is embedded into the augment call itself
        train_augs = BAugsNoResizeCrop(prob=0.5,
                           mean=model.module.mean,
                           std=model.module.std)

        val_augs = BAugsValNoResize(mean=model.module.mean,
                            std=model.module.std)    

        # scheduler = MultiStepLR(optimizer,
        #                         milestones=[2,15],
        #                        gamma=1e-1)

        # scheduler = CyclicLR(optimizer = optimizer,
        #                                 base_lr = 1e-6,
        #                                 max_lr = 1e-4,
        #                                 step_size = 1200,
        #                                 mode = 'triangular'                                         
        #                                )
    
        for epoch in range(args.start_epoch, args.epochs):
            
            # unfreeze the encoder
            if epoch==20:
                print('Encoder unfrozen')
                model.module.require_encoder_grad(True)
                
            scheduler.step()
            # loop over each resolution in the dataset
            
            train_dset_lengths = []
            val_dset_lengths = []
            
            train_losses = []
            val_losses = []
            
            train_f1_scores = []
            val_f1_scores = []
            
            val_map_scores_wt = []
            val_map_scores_wt_seed = []

            # resolution order is shuffled each time
            # train loop
            train_datasets = [] 
            
            for resl_key,source_resl,target_resl in resolution_list:

                # for resl_key,source_resl,target_resl in resolution_list:
                # we will need to resize back to calculate metrics                

                # ready for k-fold training
                train_dataset = BDatasetPad(df = train_df,
                                 transforms = train_augs,
                                 fold_num = args.fold_num,
                                 mode = 'train',
                                 dset_resl = resl_key,
                                 is_crop = True,
                                 is_img_augs = args.is_img_augs,
                                 is_distance_transform = args.is_distance_transform,
                                 is_boundaries = args.is_boundaries,
                                 is_vectors = args.is_vectors
                                 )

                print('Training on {} resl\tTrain dataset length {}'.format(resl_key,len(train_dataset)))       
                
                if len(train_dataset)>0:
                    train_datasets.append(train_dataset)

            # concat the datasets
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,        
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False)
            

            train_loss, train_f1_score  = train(train_loader,
                                                model,
                                                criterion,
                                                optimizer,
                                                epoch,
                                                scheduler,
                                                source_resl,
                                                target_resl)

            train_dset_lengths.append(len(train_dataset))
            train_losses.append(train_loss)
            train_f1_scores.append(train_f1_score)

            # log metrics with naive wt and seed based wt

            torch.cuda.empty_cache()
            del train_dataset
            del train_loader
            gc.collect
            
            # val loop
            for resl_key,source_resl,target_resl in resolution_list:

                # for resl_key,source_resl,target_resl in resolution_list:
                # we will need to resize back to calculate metrics                

                val_dataset = BDatasetPad(df = train_df,
                                 transforms = val_augs,
                                 fold_num = args.fold_num,
                                 mode = 'val',
                                 dset_resl = resl_key,
                                 is_crop = False,
                                 is_distance_transform = args.is_distance_transform,
                                 is_boundaries = args.is_boundaries,
                                 is_vectors = args.is_vectors
                                 )

                print('Training on {} resl\tVal dataset length {}'.format(resl_key,len(val_dataset)))
                
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=2,        
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=True,
                    drop_last=False)
                
                if len(val_dataset)>0:
                    # evaluate on validation set
                    val_loss, val_f1_score, val_map_score_wt, val_map_score_wt_seed = validate(val_loader,
                                                                     model,
                                                                     criterion,
                                                                     scheduler,
                                                                     source_resl,
                                                                     target_resl)
                    val_dset_lengths.append(len(val_dataset))
                    val_losses.append(val_loss)
                    
                    val_f1_scores.append(val_f1_score)
                    
                    # log metrics with naive wt and seed based wt
                    val_map_scores_wt.append(val_map_score_wt)
                    val_map_scores_wt_seed.append(val_map_score_wt_seed)
                    
                torch.cuda.empty_cache()
                del val_dataset
                del val_loader
                gc.collect
            
            # add code for early stopping here 
            # 
            #

            # calculate averages for the epoch across the resolutions
            # weight by the number of training samples
            
            train_loss_avg = np.inner(np.asarray(train_losses),np.asarray(train_dset_lengths)) / np.asarray(train_dset_lengths).sum()
            val_loss_avg = np.inner(np.asarray(val_losses),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()
                                            
            train_f1_avg = np.inner(np.asarray(train_f1_scores),np.asarray(train_dset_lengths)) / np.asarray(train_dset_lengths).sum()
            val_f1_avg = np.inner(np.asarray(val_f1_scores),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()
                 
            # log metrics with naive wt and seed based wt                
            val_map_avg_wt = np.inner(np.asarray(val_map_scores_wt),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()
            val_map_avg_wt_seed = np.inner(np.asarray(val_map_scores_wt_seed),np.asarray(val_dset_lengths)) / np.asarray(val_dset_lengths).sum()            
            
                                            
            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                info = {
                    'train_epoch_loss': train_loss_avg,
                    'valid_epoch_loss': val_loss_avg,
                    'train_epoch_f1': train_f1_avg,
                    'valid_epoch_f1': val_f1_avg,
                    'valid_epoch_map_wt': val_map_avg_wt, 
                    'valid_epoch_map_wt_seed': val_map_avg_wt_seed,                      
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)                     

            # at first optimize by f1 score                                            
            # scheduler.step(val_f1_avg)   
            
            # remember best model
            is_best = val_map_avg_wt_seed > best_map_score
            best_map_score = max(val_map_avg_wt_seed, best_map_score)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_map_score': best_map_score,
            },
            is_best,
            'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
            'weights/{}_best.pth.tar'.format(str(args.lognumber))
            )
                                         
def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          scheduler,
          source_resl,
          target_resl):
                                            
    global train_minib_counter
    global logger
        
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1_scores = AverageMeter()


    # switch to train mode
    model.train()

    # sigmoid for f1 calculation and illustrations
    m = nn.Sigmoid()    

    end = time.time()
    
    for i, (input, target, or_resl, target_resl,img_sample) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # permute to pytorch format
        input = input.permute(0,3,1,2).contiguous().float().cuda(async=True)
        # take only mask and boundary at first
        target = target[:,:,:,0:args.channels].permute(0,3,1,2).contiguous().float().cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
                                            
        loss = criterion(output, target_var)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        
        
 
        # calcuale f1 scores only on cell masks
        # weird pytorch numerical issue when converting to float
        
        target_f1 = (target_var.data[:,0:1,:,:]>args.ths)*1
        f1_scores_batch = batch_f1_score(output = m(output.data[:,0:1,:,:]),
                                   target = target_f1,
                                   threshold=args.ths)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        f1_scores.update(f1_scores_batch, input.size(0))

        # log the current lr
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                                                          
                                            
        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_loss': losses.val,
                'f1_score_train': f1_scores.val,
                'train_lr': current_lr,                
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter)                

        train_minib_counter += 1

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F1    {f1_scores.val:.4f} ({f1_scores.avg:.4f})\t'.format(
                   epoch,i, len(train_loader),
                   batch_time=batch_time,data_time=data_time,
                   loss=losses,f1_scores=f1_scores))

    print(' * Avg Train Loss  {loss.avg:.4f}'.format(loss=losses))
    print(' * Avg F1    Score {f1_scores.avg:.4f}'.format(f1_scores=f1_scores))
            
    return losses.avg, f1_scores.avg

def validate(val_loader,
             model,
             criterion,
             scheduler,
             source_resl,
             target_resl):
                                
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    f1_scores = AverageMeter()
    map_scores_wt = AverageMeter()
    map_scores_wt_seed = AverageMeter()    
    
    # switch to evaluate mode
    model.eval()

    # sigmoid for f1 calculation and illustrations
    m = nn.Sigmoid()      
    
    end = time.time()
    for i, (input, target, or_resl, target_resl,img_sample) in enumerate(val_loader):
        
        # permute to pytorch format
        input = input.permute(0,3,1,2).contiguous().float().cuda(async=True)
        # take only mask and boundary at first
        target = target[:,:,:,0:args.channels].permute(0,3,1,2).contiguous().float().cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
                                            
        loss = criterion(output, target_var)
        
        # go over all of the predictions
        # apply the transformation to each mask
        # calculate score for each of the images
        
        averaged_maps_wt = []
        averaged_maps_wt_seed = []
        y_preds_wt = []
        y_preds_wt_seed = []
        energy_levels = []
            
        for j,pred_output in enumerate(output):
            or_w = or_resl[0][j]
            or_h = or_resl[1][j]
            
            # I keep only the latest preset
            
            pred_mask = m(pred_output[0,:,:]).data.cpu().numpy()
            pred_mask1 = m(pred_output[1,:,:]).data.cpu().numpy()
            pred_mask2 = m(pred_output[2,:,:]).data.cpu().numpy()
            pred_mask3 = m(pred_output[3,:,:]).data.cpu().numpy()
            pred_mask0 = m(pred_output[4,:,:]).data.cpu().numpy()
            pred_border = m(pred_output[5,:,:]).data.cpu().numpy()
            # pred_distance = m(pred_output[5,:,:]).data.cpu().numpy()            
            pred_vector0 = pred_output[6,:,:].data.cpu().numpy()
            pred_vector1 = pred_output[7,:,:].data.cpu().numpy()             

            add_x = (pred_output.size(1)-or_w) // 2
            add_y = (pred_output.size(2)-or_h) // 2
            
            if (pred_output.size(1)-or_w)%2>0:
                add_x_1 = 1
            else:
                add_x_1 = 0
            if (pred_output.size(2)-or_h)%2>0:
                add_y_1 = 1
            else:
                add_y_1 = 0            

            if (-add_x-add_x_1) == 0:
                add_x = -pred_output.size(1)
                add_x_1 = 0
                
            if (-add_y-add_y_1) == 0:
                add_y = -pred_output.size(2)
                add_y_1 = 0
                
            pred_mask = pred_mask[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            pred_mask1 = pred_mask1[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            pred_mask2 = pred_mask2[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            pred_mask3 = pred_mask3[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            pred_mask0 = pred_mask0[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            
            pred_border = pred_border[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            pred_vector0 = pred_vector0[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            pred_vector1 = pred_vector1[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
            # pred_distance = cv2.resize(pred_distance, (or_h,or_w), interpolation=cv2.INTER_LINEAR)
         
            
            # predict average energy by summing all the masks up 
            pred_energy = (pred_mask+pred_mask1+pred_mask2+pred_mask3+pred_mask0)/5*255
            pred_mask_255 = np.copy(pred_mask) * 255
            
            # read the original masks for metric evaluation
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(img_sample[j]))
            gt_masks = imread_collection(mask_glob).concatenate()

            # simple wt
            y_pred_wt = wt_baseline(pred_mask_255, args.ths)
            
            # wt with seeds
            y_pred_wt_seed = wt_seeds(pred_mask_255,pred_energy,args.ths)
            
            map_wt = calculate_ap(y_pred_wt, gt_masks)
            map_wt_seed = calculate_ap(y_pred_wt_seed, gt_masks)
            
            averaged_maps_wt.append(map_wt[1])
            averaged_maps_wt_seed.append(map_wt_seed[1])

            # apply colormap for easier tracking
            y_pred_wt = cv2.applyColorMap((y_pred_wt / y_pred_wt.max() * 255).astype('uint8'), cv2.COLORMAP_JET) 
            y_pred_wt_seed = cv2.applyColorMap((y_pred_wt_seed / y_pred_wt_seed.max() * 255).astype('uint8'), cv2.COLORMAP_JET)  
            
            y_preds_wt.append(y_pred_wt)
            y_preds_wt_seed.append(y_pred_wt_seed)
            energy_levels.append(pred_energy)
            
            # print('MAP for sample {} is {}'.format(img_sample[j],m_ap))
        
        y_preds_wt = np.asarray(y_preds_wt)
        y_preds_wt_seed = np.asarray(y_preds_wt_seed)
        energy_levels = np.asarray(energy_levels)
        
        averaged_maps_wt = np.asarray(averaged_maps_wt).mean()
        averaged_maps_wt_seed = np.asarray(averaged_maps_wt_seed).mean()

        #============ TensorBoard logging ============#                                            
        if args.tensorboard_images:
            if i == 0:
                if args.channels == 5:
                    info = {
                        'images': to_np(input[:2,:,:,:]),
                        'gt_mask': to_np(target[:2,0,:,:]),
                        'gt_mask1': to_np(target[:2,1,:,:]),
                        'gt_mask2': to_np(target[:2,2,:,:]),
                        'gt_mask3': to_np(target[:2,3,:,:]), 
                        'gt_mask0': to_np(target[:2,4,:,:]),
                        'pred_mask': to_np(m(output.data[:2,0,:,:])),
                        'pred_mask1': to_np(m(output.data[:2,1,:,:])),
                        'pred_mask2': to_np(m(output.data[:2,2,:,:])),
                        'pred_mask3': to_np(m(output.data[:2,3,:,:])),
                        'pred_mask0': to_np(m(output.data[:2,4,:,:])),
                        'pred_energy': energy_levels[:2,:,:], 
                        'pred_wt': y_preds_wt[:2,:,:],
                        'pred_wt_seed': y_preds_wt_seed[:2,:,:,:],
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, valid_minib_counter)                   
                elif args.channels == 6:
                    info = {
                        'images': to_np(input[:2,:,:,:]),
                        'gt_mask': to_np(target[:2,0,:,:]),
                        'gt_mask1': to_np(target[:2,1,:,:]),
                        'gt_mask2': to_np(target[:2,2,:,:]),
                        'gt_mask3': to_np(target[:2,3,:,:]), 
                        'gt_mask0': to_np(target[:2,4,:,:]),
                        'gt_mask_distance': to_np(target[:2,5,:,:]),
                        'pred_mask': to_np(m(output.data[:2,0,:,:])),
                        'pred_mask1': to_np(m(output.data[:2,1,:,:])),
                        'pred_mask2': to_np(m(output.data[:2,2,:,:])),
                        'pred_mask3': to_np(m(output.data[:2,3,:,:])),
                        'pred_mask0': to_np(m(output.data[:2,4,:,:])),
                        'pred_distance': to_np(m(output.data[:2,5,:,:])),
                        'pred_energy': energy_levels[:2,:,:], 
                        'pred_wt': y_preds_wt[:2,:,:],
                        'pred_wt_seed': y_preds_wt_seed[:2,:,:,:],
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, valid_minib_counter)
                elif args.channels == 7:
                    info = {
                        'images': to_np(input[:2,:,:,:]),
                        'gt_mask': to_np(target[:2,0,:,:]),
                        'gt_mask1': to_np(target[:2,1,:,:]),
                        'gt_mask2': to_np(target[:2,2,:,:]),
                        'gt_mask3': to_np(target[:2,3,:,:]), 
                        'gt_mask0': to_np(target[:2,4,:,:]),
                        'gt_mask_distance': to_np(target[:2,5,:,:]),
                        'gt_border': to_np(target[:2,6,:,:]),                        
                        'pred_mask': to_np(m(output.data[:2,0,:,:])),
                        'pred_mask1': to_np(m(output.data[:2,1,:,:])),
                        'pred_mask2': to_np(m(output.data[:2,2,:,:])),
                        'pred_mask3': to_np(m(output.data[:2,3,:,:])),
                        'pred_mask0': to_np(m(output.data[:2,4,:,:])),
                        'pred_distance': to_np(m(output.data[:2,5,:,:])),
                        'pred_border': to_np(m(output.data[:2,6,:,:])),                        
                        'pred_energy': energy_levels[:2,:,:], 
                        'pred_wt': y_preds_wt[:2,:,:],
                        'pred_wt_seed': y_preds_wt_seed[:2,:,:,:],
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, valid_minib_counter)
                elif args.channels == 8:
                    info = {
                        'images': to_np(input[:2,:,:,:]),
                        'gt_mask': to_np(target[:2,0,:,:]),
                        'gt_mask1': to_np(target[:2,1,:,:]),
                        'gt_mask2': to_np(target[:2,2,:,:]),
                        'gt_mask3': to_np(target[:2,3,:,:]), 
                        'gt_mask0': to_np(target[:2,4,:,:]),
                        'gt_border': to_np(target[:2,5,:,:]),   
                        'gt_vectors': to_np(target[:2,6,:,:]+target[:2,7,:,:]), # simple hack - just sum the vectors
                        'pred_mask': to_np(m(output.data[:2,0,:,:])),
                        'pred_mask1': to_np(m(output.data[:2,1,:,:])),
                        'pred_mask2': to_np(m(output.data[:2,2,:,:])),
                        'pred_mask3': to_np(m(output.data[:2,3,:,:])),
                        'pred_mask0': to_np(m(output.data[:2,4,:,:])),
                        'pred_border': to_np(m(output.data[:2,5,:,:])),
                        'pred_vectors': to_np(output.data[:2,6,:,:]+output.data[:2,7,:,:]),                         
                        'pred_energy': energy_levels[:2,:,:], 
                        'pred_wt': y_preds_wt[:2,:,:],
                        'pred_wt_seed': y_preds_wt_seed[:2,:,:,:],
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, valid_minib_counter)                          
                        

                        
        # calcuale f1 scores only on inner cell masks
        # weird pytorch numerical issue when converting to float
        target_f1 = (target_var.data[:,0:1,:,:]>args.ths)*1        
        f1_scores_batch = batch_f1_score(output = m(output.data[:,0:1,:,:]),
                                   target = target_f1,
                                   threshold=args.ths)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        f1_scores.update(f1_scores_batch, input.size(0))
        map_scores_wt.update(averaged_maps_wt, input.size(0))  
        map_scores_wt_seed.update(averaged_maps_wt_seed, input.size(0)) 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'valid_loss': losses.val,
                'f1_score_val': f1_scores.val, 
                'map_wt': averaged_maps_wt,
                'map_wt_seed': averaged_maps_wt_seed,
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time  {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss  {loss.val:.4f} ({loss.avg:.4f})\t'
                  'F1    {f1_scores.val:.4f} ({f1_scores.avg:.4f})\t'
                  'MAP1  {map_scores_wt.val:.4f} ({map_scores_wt.avg:.4f})\t'
                  'MAP2  {map_scores_wt_seed.val:.4f} ({map_scores_wt_seed.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time,
                      loss=losses,
                      f1_scores=f1_scores,
                      map_scores_wt=map_scores_wt,map_scores_wt_seed=map_scores_wt_seed))

    print(' * Avg Val  Loss {loss.avg:.4f}'.format(loss=losses))
    print(' * Avg F1   Score {f1_scores.avg:.4f}'.format(f1_scores=f1_scores))
    print(' * Avg MAP1 Score {map_scores_wt.avg:.4f}'.format(map_scores_wt=map_scores_wt)) 
    print(' * Avg MAP2 Score {map_scores_wt_seed.avg:.4f}'.format(map_scores_wt_seed=map_scores_wt_seed)) 

    return losses.avg, f1_scores.avg, map_scores_wt.avg,map_scores_wt_seed.avg

def predict(predict_loader,
            model,model1,model2,model3):
    
    global logger
    global pred_minib_counter
    
    m = nn.Sigmoid()
    model.eval()
    model1.eval()
    model2.eval()
    model3.eval()

    temp_df = pd.DataFrame(columns = ['ImageId','EncodedPixels'])
    
    with tqdm.tqdm(total=len(predict_loader)) as pbar:
        for i, (input, target, or_resl, target_resl, img_ids) in enumerate(predict_loader):
            
            # reshape to PyTorch format
            input = input.permute(0,3,1,2).contiguous().float().cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            
            # compute output
            output = model(input_var)
            output1 = model1(input_var)
            output2 = model2(input_var)
            output3 = model3(input_var)
            
            for k,(pred_mask,pred_mask1,pred_mask2,pred_mask3) in enumerate(zip(output,output1,output2,output3)):

                or_w = or_resl[0][k]
                or_h = or_resl[1][k]
                
                print(or_w,or_h)
                
                mask_predictions = []
                energy_predictions = []
                
                # for pred_msk in [pred_mask,pred_mask1,pred_mask2,pred_mask3]:
                for pred_msk in [pred_mask]:
                    print(pred_msk.shape)
                    _,__ = calculate_energy(pred_msk,or_h,or_w)
                    mask_predictions.append(_)
                    energy_predictions.append(__)
                   
                avg_mask = np.asarray(mask_predictions).mean(axis=0)
                avg_energy = np.asarray(energy_predictions).mean(axis=0)
                
                print(avg_mask.shape)
                print(avg_energy.shape)
                
                imsave('../examples/mask_{}.png'.format(img_ids[k]),avg_mask.astype('uint8'))
                imsave('../examples/energy_{}.png'.format(img_ids[k]),avg_energy.astype('uint8'))
                
                labels = wt_seeds(avg_mask,
                                  avg_energy,
                                  args.ths)  
                
                labels_seed = cv2.applyColorMap((labels / labels.max() * 255).astype('uint8'), cv2.COLORMAP_JET)                  
                imsave('../examples/labels_{}.png'.format(img_ids[k]),labels_seed)

                if args.tensorboard_images:
                    info = {
                        'images': to_np(input),
                        'labels_wt': np.expand_dims(labels_seed,axis=0),
                        'pred_mask_fold0': np.expand_dims(mask_predictions[0],axis=0),
                        'pred_mask_fold1': np.expand_dims(mask_predictions[1],axis=0),
                        'pred_mask_fold2': np.expand_dims(mask_predictions[2],axis=0),
                        'pred_mask_fold3': np.expand_dims(mask_predictions[3],axis=0),
                        'pred_energy_fold0': np.expand_dims(energy_predictions[0],axis=0),
                        'pred_energy_fold1': np.expand_dims(energy_predictions[1],axis=0),
                        'pred_energy_fold2': np.expand_dims(energy_predictions[2],axis=0),
                        'pred_energy_fold3': np.expand_dims(energy_predictions[3],axis=0),
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, pred_minib_counter)

                pred_minib_counter += 1
                
                wt_areas = []
                for label in np.unique(labels):
                    if label == 0:
                        # pass the background
                        pass
                    else:
                        wt_areas.append((labels == label) * 1)
               
                for wt_area in wt_areas:
                    append_df = pd.DataFrame(columns = ['ImageId','EncodedPixels'])
                    append_df['ImageId'] = [img_ids[k]]
                    append_df['EncodedPixels'] = [' '.join(map(str, rle_encoding(wt_area))) ]
                    
                    temp_df = temp_df.append(append_df)
            
            pbar.update(1)            

    return temp_df

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def calculate_energy(pred_output,or_h,or_w):
    m = nn.Sigmoid() 
    
    add_x = (pred_output.size(1)-or_w) // 2
    add_y = (pred_output.size(2)-or_h) // 2

    if (pred_output.size(1)-or_w)%2>0:
        add_x_1 = 1
    else:
        add_x_1 = 0
    if (pred_output.size(2)-or_h)%2>0:
        add_y_1 = 1
    else:
        add_y_1 = 0            
    
    if (-add_x-add_x_1) == 0:
        add_x = -pred_output.size(1)
        add_x_1 = 0

    if (-add_y-add_y_1) == 0:
        add_y = -pred_output.size(2)
        add_y_1 = 0    
    
    pred_mask = m(pred_output[0,:,:]).data.cpu().numpy()
    pred_mask1 = m(pred_output[1,:,:]).data.cpu().numpy()
    pred_mask2 = m(pred_output[2,:,:]).data.cpu().numpy()
    pred_mask3 = m(pred_output[3,:,:]).data.cpu().numpy()
    pred_mask0 = m(pred_output[4,:,:]).data.cpu().numpy()

    pred_mask = pred_mask[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1]
    pred_mask1 = pred_mask1[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1] 
    pred_mask2 = pred_mask2[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1] 
    pred_mask3 = pred_mask3[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1] 
    pred_mask0 = pred_mask0[add_x:-add_x-add_x_1,add_y:-add_y-add_y_1] 

    # predict average energy by summing all the masks up 
    pred_energy = (pred_mask+pred_mask1+pred_mask2+pred_mask3+pred_mask0)/5*255
    pred_mask_255 = np.copy(pred_mask) * 255
    
    return pred_mask_255,pred_energy
        
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def batch_f1_score(output,
                   target,
                   threshold = 0.3):
    if (target.size() != output.size()):
        raise ValueError('Preds shape <> output shape')        
    batch_size = target.size(0)
    res = []
    for i in range(0,batch_size):
        if (((output[i,:,:,:].view(-1).cpu().numpy()>threshold)*1).sum() == 0)\
            and (target[i,:,:,:].view(-1).cpu().numpy().sum() == 0):
            res.append(1)
        elif (((output[i,:,:,:].view(-1).cpu().numpy()>threshold)*1).sum() > 0)\
            and (target[i,:,:,:].view(-1).cpu().numpy().sum() == 0):
            res.append(0)            
        elif (((output[i,:,:,:].view(-1).cpu().numpy()>threshold)*1).sum() == 0)\
            and (target[i,:,:,:].view(-1).cpu().numpy().sum() > 0):
            res.append(0)
        else:
            y_true = target[i,:,:,:].view(-1).cpu().numpy()
            y_pred = (output[i,:,:,:].view(-1).cpu().numpy()>threshold)*1
            res.append(f1_score(y_true, y_pred))

    return sum(res) / float(len(res))       
        
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

if __name__ == '__main__':
    main()