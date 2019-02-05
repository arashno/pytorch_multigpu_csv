import argparse
import os
import random
import shutil
import time
import warnings
import sys
from datetime import datetime
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.functional import softmax

import utils
import data_loader
from model import model



"""
  This method trains a deep neural network using the provided configuration.
  For more details on the different steps of training, please read inline comments below.

  Args:
    sess: a tensorflow session to build and run the computational graph.
    args: augmented command line arguments determiing the details of training.
  Returns:
    nothing.
"""
def do_train(args):
    # create model
    dnn_model = model(args.architecture, args.num_classes)

    if args.num_gpus == 1:
        dnn_model = dnn_model.cuda()
    else:
        dnn_model = torch.nn.DataParallel(dnn_model, device_ids = range(0, args.num_gpus)).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    lr = utils.get_policy(args.LR_policy, args.LR_details)
    wd = utils.get_policy(args.WD_policy, args.WD_details)
    optimizer = utils.get_optimizer(args.optimizer, dnn_model.parameters(), 0.01)
    train_loader = data_loader.CSVDataset(args.train_info, args.delimiter, args.raw_size, args.processed_size, args.batch_size, args.num_workers, args.path_prefix, True, shuffle = True).load()
    start_epoch = 0
    if args.retrain_from is not None:
        checkpoint = torch.load(utils.smart_load(args.retrain_from))
        dnn_model.module.load_state_dict(checkpoint['model'])
        if args.transfer_mode[0] == 0:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        if args.transfer_mode[0] == 1 or args.transfer_mode[0] == 3:
            dnn_model.freeze()
    

    if args.run_validation:
        val_loader= data_loader.CSVDataset(args.val_info, args.delimiter, args.raw_size, args.processed_size, args.batch_size, args.num_workers, args.path_prefix, False, shuffle = False).load()
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if args.optimizer not in ['adam', 'adadelta']:
          utils.adjust_param(optimizer, 'lr', lr, epoch)
        utils.adjust_param(optimizer, 'weight_decay', wd, epoch)
        if args.transfer_mode[0] == 3 and epoch==args.transfer_mode[1]:
            dnn_model.unfreeze()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        topn = utils.AverageMeter()

        # switch to train mode
        dnn_model.train()
        end = time.time()
        for step, (input, target, _) in islice(enumerate(train_loader), args.num_batches):
            # measure data loading time
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = dnn_model(input)
            #print(output,target)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, precn = utils.accuracy(output, target, topk=(1, args.top_n))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            topn.update(precn[0], input.size(0))
            #print(loss.item(), prec1[0], precn[0])
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % 10 == 0:
                format_str = ('%s: epoch %d, step %d, loss = %.2f, Top-1 = %.2f Top-' + str(args.top_n) + ' = %.2f')
                print(format_str % (datetime.now(), epoch, step, losses.val, top1.val, topn.val))
                sys.stdout.flush()
        state= {'epoch': epoch + 1,
              'arch': args.architecture,
              'num_classes': args.num_classes,
              'model': dnn_model.module.state_dict(),
              'optimizer': optimizer.state_dict()
               }
        torch.save(state, utils.smart_save(os.path.join(args.log_dir, 'checkpoint%04d.pth.tar'%(epoch)), max_to_keep = args.max_to_keep))
            # if validation data are provided, evaluate accuracy on the validation set after the end of each epoch
        if args.run_validation:
            valbatch_time = utils.AverageMeter()
            vallosses = utils.AverageMeter()
            valtop1 = utils.AverageMeter()
            valtop5 = utils.AverageMeter()

            # switch to evaluate mode
            dnn_model.eval()

            with torch.no_grad():
                end = time.time()
                for i, (input, target, _) in enumerate(val_loader):
                    input = input.cuda(non_blocking = True)
                    target = target.cuda(non_blocking = True)

                    # compute output
                    output = dnn_model(input)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    prec1, prec5 = utils.accuracy(output, target, topk = (1, 5))
                    vallosses.update(loss.item(), input.size(0))
                    valtop1.update(prec1[0], input.size(0))
                    valtop5.update(prec5[0], input.size(0))

                    # measure elapsed time
                    valbatch_time.update(time.time() - end)
                    end = time.time()

                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time = valbatch_time, loss = vallosses,
                        top1 = valtop1, top5 = valtop5))
                    sys.stdout.flush()
    print('Training finished')

"""
 This method evaluates (or just run) a trained deep neural network using the provided configuration.
 For more details on the different steps of training, please read inline comments below.

 Args:
     sess: a tensorflow session to build and run the computational graph.
     args: augmented command line arguments determiing the details of evaluation.
 Returns:
     nothing.
"""

def do_evaluate(args):
    # if we want to do inference only (i.e. no label is provided) we only load images and their paths

    val_loader = data_loader.CSVDataset(args.val_info, args.delimiter, args.raw_size, args.processed_size, args.batch_size, args.num_threads, args.path_prefix, False,
            shuffle = False, inference_only= args.inference_only).load()

    checkpoint = torch.load(args.log_dir)
    dnn_model = model(checkpoint['arch'],checkpoint['num_classes'])
    
    if args.num_gpus == 1:
        dnn_model = dnn_model.cuda()
    else:
        dnn_model = torch.nn.DataParallel(dnn_model, device_ids = range(0, args.num_gpus))

    # Load pretrained parameters from disk
    dnn_model.load_state_dict(checkpoint['model'])

    criterion = nn.CrossEntropyLoss().cuda()
    # evaluation 
    if not args.inference_only:
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        topn = utils.AverageMeter()

        # Open an output file to write predictions
        out_file = open(args.save_predictions, 'w')
        predictions_format_str = ('%d, %s, %d, %s, %s\n')
        for step,(input, target, info) in enumerate(val_loader):
            input = input.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)

            # Load a batch of data
            output= softmax(dnn_model(input))
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(output, target, topk=(1, args.top_n))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            topn.update(prec5[0], input.size(0))

            print('Batch Number: %d, Top-1 Hit: %d, Top-%d Hit: %d, Loss %.2f, Top-1 Accuracy: %.3f, Top-%d Accuracy: %.3f'%
                (step, top1.val, args.top_n, topn.val, losses.avg, top1.avg, args.top_n, topn.avg))

            # log results into an output file
            topnconf, topnguesses = output.topk(args.top_n, 1, True, True)
            for i in range(0, len(info)):
                out_file.write(predictions_format_str % (step * args.batch_size + i + 1, str(info[i]).encode('utf-8'), target[i],
                    ', '.join('%d' % item for item in topnguesses[i]),
                    ', '.join('%.4f' % item for item in topnconf[i])))
            out_file.flush()
            sys.stdout.flush()
        out_file.close()
    #inference
    else:
        
        # Open an output file to write predictions
        out_file = open(args.save_predictions, 'w')
        predictions_format_str = ('%d, %s, %s, %s\n')

        for step,(input, info) in enumerate(val_loader):
            # Load a batch of data
            input = input.cuda(non_blocking = True)
            # Load a batch of data
            output= softmax(dnn_model(input))

            # Run the network on the loaded batch
            topnconf,topnguesses = output.topk(args.top_n,1, True, True)
            print('Batch Number: %d of %d is done'%(step, args.num_val_batches))

            # Log to an output file
            for i in range(0, len(info)):
                out_file.write(predictions_format_str % (step * args.batch_size + i + 1, str(info[i]).encode('utf-8'),
                    ', '.join('%d' % item for item in topnguesses[i]),
                    ', '.join('%.4f' % item for item in topnconf[i])))
            out_file.flush()

        out_file.close()

def main():  # pylint: disable=unused-argument
    parser = argparse.ArgumentParser(description='Process Command-line Arguments')
    parser.add_argument('command', action= 'store', help= 'Determines what to do: train, evaluate, or inference')
    parser.add_argument('--raw_size', nargs= 3, default= [256,256,3], type= int, action= 'store', help= 'The width, height and number of channels of images for loading from disk')
    parser.add_argument('--processed_size', nargs= 3, default= [224,224,3], type= int, action= 'store', help= 'The width and height of images after preprocessing')
    parser.add_argument('--batch_size', default= 128, type= int, action= 'store', help= 'The batch size for training, evaluating, or inference')
    parser.add_argument('--num_classes', default= 1000 , type=int, action='store', help= 'The number of classes')
    parser.add_argument('--num_epochs', default= 55, type= int, action= 'store', help= 'The number of training epochs')
    parser.add_argument('--path_prefix', default= './', action='store', help= 'the prefix address for images')
    parser.add_argument('--train_info', default= None, action= 'store', help= 'Name of the file containing addresses and labels of training images')
    parser.add_argument('--val_info', default= None, action= 'store', help= 'Name of the file containing addresses and labels of validation images')
    parser.add_argument('--shuffle', default= True, type= bool, action= 'store',help= 'Shuffle training data or not')
    parser.add_argument('--num_workers', default= 5, type= int, action='store', help= 'The number of threads for loading data')
    parser.add_argument('--log_dir', default= None, action= 'store', help= 'Path for saving Tensorboard info and checkpoints')
    parser.add_argument('--snapshot_prefix', default= 'snapshot', action= 'store', help= 'Prefix for checkpoint files')
    parser.add_argument('--architecture', default= 'resnet', help= 'The DNN architecture')
    parser.add_argument('--run_name', default= 'Run'+str(time.strftime('-%d-%m-%Y_%H-%M-%S')), action= 'store', help= 'Name of the experiment')
    parser.add_argument('--num_gpus', default= 1, type= int, action= 'store', help= 'Number of GPUs (Only for training)')
    parser.add_argument('--delimiter', default= ' ', action= 'store', help= 'Delimiter of the input files')
    parser.add_argument('--retrain_from', default= None, action= 'store', help= 'Continue Training from a snapshot file')
    parser.add_argument('--num_batches', default= -1, type= int, action= 'store', help= 'The number of batches per epoch')
    parser.add_argument('--transfer_mode', default = [0], nargs='+', type= int, help= 'Transfer mode 0=None , 1=Tune last layer only , 2= Tune all the layers, 3= Tune the last layer at early epochs     (it could be specified with the second number of this argument) and then tune all the layers')
    parser.add_argument('--LR_policy', default='piecewise_linear', help='LR change policy type (piecewise_linear, constant, exponential)')
    parser.add_argument('--WD_policy', default='piecewise_linear', help='WD change policy type (piecewise_linear, constant, exponential)')
    parser.add_argument('--LR_details', default= '19, 30, 44, 53, 0.01, 0.005, 0.001, 0.0005, 0.0001', help='LR change details')
    parser.add_argument('--WD_details', default='30, 0.0005, 0.0', help='WD change details') 
    parser.add_argument('--optimizer', default= 'momentum', help= 'The optimization algorithm (SGD, Momentum, Adam, RMSprop, ...)')
    parser.add_argument('--top_n', default= 5, type= int, action= 'store', help= 'Specify the top-N accuracy')
    parser.add_argument('--max_to_keep', default= 5, type= int, action= 'store', help= 'Maximum number of snapshot files to keep')
    parser.add_argument('--save_predictions', default= 'predictions.csv', action= 'store', help= 'Save top-n predictions of the networks along with their confidence in the specified file')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.command.lower() == 'train':
        #Assert input args
        assert args.train_info is not None, 'No training dataset is provided, please provide an input file using --train_info option'

        # Counting number of training examples
        if args.num_batches == -1:
            args.num_samples, args.num_batches = utils.count_input_records(args.train_info, args.batch_size)
        else:
            args.num_samples, _ = utils.count_input_records(args.train_info, args.batch_size)

        # Counting number of validation examples
        if args.val_info is not None:
            args.num_val_samples, args.num_val_batches = utils.count_input_records(args.val_info, args.batch_size)
            args.run_validation = True
        else:
            args.run_validation = False

        # creating the logging directory
        if args.log_dir is None:
            args.log_dir = args.architecture + '_' + args.run_name
  
        if os.path.exists(args.log_dir):
            os.rmdir(args.log_dir)
        os.mkdir(args.log_dir)
        print('Saving everything in ' + args.log_dir)

        print(args)
        # do training
        do_train(args)

    elif args.command.lower() == 'eval':
        # set config
        args.inference_only = False

        # Counting number of training examples
        assert args.val_info is not None, 'No evaluation dataset is provided, please provide an input file using val_info option'
        args.num_val_samples, args.num_val_batches = utils.count_input_records(args.val_info, args.batch_size)
      
        print(args)
        # do evaluation
        do_evaluate(args)

    elif args.command.lower() == 'inference':
        # set config
        args.inference_only = True

        # Counting number of test examples
        assert args.val_info is not None, 'No inference dataset is provided, please provide an input file using --val_info option'
        args.num_val_samples, args.num_val_batches = utils.count_input_records(args.val_info, args.batch_size)

        print(args)
        # do testing
        do_evaluate(args)

if __name__ == '__main__':
    main()
