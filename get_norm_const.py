# Calculate the normalization constant for discriminator rejection
import torch
import torch.nn as nn

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import ConditionGenerator, load_checkpoint, define_D

from utils import *


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="./data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs_zalando.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--D_checkpoint', type=str, default='', help='tocg checkpoint')
    parser.add_argument('--tocg_checkpoint', type=str, default='', help='tocg checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # Condition generator
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    # network structure
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")

    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="./data/zalando-hd-resize")
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    
    opt = parser.parse_args()
    return opt

def D_logit(pred):
    score = 0
    for i in pred:
        score += i[-1].mean((1,2,3)) / 2
    return score
def get_const(opt, train_loader, tocg, D, length):
    # Model
    D.cuda()
    D.eval()
    tocg.cuda()
    tocg.eval()

    logit_list = []
    i = 0
    for step in range(length // opt.batch_size):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image']
        with torch.no_grad():
            # inputs
            input1 = torch.cat([c_paired, cm_paired], 1)
            input2 = torch.cat([parse_agnostic, densepose], 1)

            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)
            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                    cloth_mask = torch.ones_like(fake_segmap.detach())
                    cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask
                    
                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap.detach())
                    cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
            
            
            fake_segmap_softmax = F.softmax(fake_segmap, dim=1)
            
            real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
            fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax),dim=1))
            
            print("real:", D_logit(real_segmap_pred), "fake:", D_logit(fake_segmap_pred))
            # print(fake_segmap_pred)
            logit_real = D_logit(real_segmap_pred)
            logit_fake = D_logit(fake_segmap_pred)
            for l in logit_real:
                l = l / (1-l)
                logit_list.append(l.item())
            for l in logit_fake:
                l = l / (1-l)
                logit_list.append(l.item())
                
        # i += logit_real.shape[0]+logit_fake.shape[0]
        print("i:", i)
    logit_list.sort()
    
    return logit_list[-1]
        

def main():
    opt = get_opt()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create train dataset & loader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    
    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    # Load Checkpoint
    load_checkpoint(D, opt.D_checkpoint)
    load_checkpoint(tocg, opt.tocg_checkpoint)
    
    
    M = get_const(opt, train_loader, tocg, D, length = len(train_dataset))
    print("M:", M)


if __name__ == "__main__":
    main()