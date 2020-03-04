"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset, Flickr30kFeatureDataset
import base_model
import utils
from utils import trim_collate
from dataset import tfidf_from_questions
import random #------!!!!!
from torch.utils.data.sampler import SubsetRandomSampler#----!!!!
#from data.sampler import SubsetSequentialSampler
import pdb

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt', help='ban / simple / baseline0 / baseline0_newatt') #-----add simple
    parser.add_argument('--op', type=str, default='')#----originally 'c'
    parser.add_argument('--gamma', type=int, default=8, help='glimpse')
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', action='store_false', help='tfidf word embedding?')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args




def get_uncertainty(model, unlabeled_loader,SUBSET):#-------------!!!!!!!
    print("Find samples to label....")
    #uncertainty = torch.tensor(range(SUBSET)).cuda()
    uncertainty = torch.tensor(SUBSET).cuda()
    '''
    models.eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    '''
    return uncertainty.cpu()

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    
    if args.task == 'vqa':
        from train import train
        dict_path = 'data/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        train_dset = VQAFeatureDataset('train', dictionary, adaptive=True)
        val_dset = VQAFeatureDataset('val', dictionary, adaptive=True)
        w_emb_path = 'data/glove6b_init_300d.npy'

    elif args.task == 'flickr':
        from train_flickr import train
        dict_path = 'data/flickr30k/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        train_dset = Flickr30kFeatureDataset('train', dictionary)
        val_dset = Flickr30kFeatureDataset('val', dictionary)
        w_emb_path = 'data/flickr30k/glove6b_init_300d.npy'
        args.op = ''
        args.gamma = 1
        args.tfidf = False
    
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'args.txt'))
    logger.write(args.__repr__())

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args.op, args.gamma, args.task).cuda()


    tfidf = None
    weights = None

    if args.tfidf:#------!!!!!
        dict = Dictionary.load_from_file(dict_path)
        tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dict)

    model.w_emb.init_embedding(w_emb_path, tfidf, weights)

    model = nn.DataParallel(model).cuda()

    optim = None
    epoch = 0

    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

    
    ############################################
    # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
    SUBSET=80000#30000
    ADDENDUM=80000#3000
    CYCLE=1#10
    
    indices = list(range(82743))#list(range(NUM_TRAIN))
    random.shuffle(indices)
    labeled_set = indices[:ADDENDUM]
    unlabeled_set = indices[ADDENDUM:]
    #############################################
    
    
    if args.task == 'vqa':
        if args.use_both: # use train & val splits to optimize
            if args.use_vg: # use a portion of Visual Genome dataset
                vg_dsets = [
                    VisualGenomeFeatureDataset('train', \
                        train_dset.features, train_dset.spatials, dictionary, adaptive=True, pos_boxes=train_dset.pos_boxes),
                    VisualGenomeFeatureDataset('val', \
                        val_dset.features, val_dset.spatials, dictionary, adaptive=True, pos_boxes=val_dset.pos_boxes)]
                trainval_dset = ConcatDataset([train_dset, val_dset]+vg_dsets)
            else:
                trainval_dset = ConcatDataset([train_dset, val_dset])
            train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate)
            eval_loader = None
        else:
            train_loader = DataLoader(train_dset, batch_size, num_workers=0, collate_fn=utils.trim_collate, sampler=SubsetRandomSampler(labeled_set))#------!!!!!
            #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate)
            eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    elif args.task == 'flickr':
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate)

    
    ######################################### -------- Modified for active learning setup ------- ########################
    logger = utils.Logger(os.path.join(args.output, 'scores.txt'))

    for cycle in range(CYCLE): 
      print("##########CYCLE (%d/%d)###########"%(cycle,CYCLE))
      score = train(model, train_loader, eval_loader, args.epochs, args.output,cycle, optim, epoch)
      logger.write('\teval score [CYCLE %d] : %.2f' % (cycle, 100 * score))
 
      ##
      #  Update the labeled dataset via loss prediction-based uncertainty measurement

      # Randomly sample 10000 unlabeled data points
      random.shuffle(unlabeled_set)
      subset = unlabeled_set[:SUBSET]

      # Create unlabeled dataloader for the unlabeled subset
      unlabeled_loader = DataLoader(train_dset, batch_size, num_workers=0, collate_fn=utils.trim_collate, sampler=SubsetSequentialSampler(subset))
      

      # Measure uncertainty of each data points in the subset
      uncertainty = get_uncertainty(model, unlabeled_loader,SUBSET)

      # Index in ascending order
      arg = np.argsort(uncertainty)
            
      # Update the labeled dataset and the unlabeled dataset, respectively
      labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
      unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

      # Create a new dataloader for the updated labeled dataset
      train_loader = DataLoader(train_dset, batch_size, num_workers=0, collate_fn=utils.trim_collate, sampler=SubsetRandomSampler(labeled_set))
      
      
      
      
      
      
      
      
      
