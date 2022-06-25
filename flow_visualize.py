import torch
import torch.nn as nn

from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image

import argparse
import os
import time
from cp_dataset import CPDatasetTest, CPDataLoader
from networks import ImprovedMTVITON, load_checkpoint, make_grid
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="/home/nas2_userF/gyojunggu/WUTON/data/zalando-hd-resize")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--datasetting", default="unpaired")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--mtviton_checkpoint', type=str, default='', help='mtviton checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    
    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        

    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])


    opt = parser.parse_args()
    return opt


def test(opt, test_loader, board, mtviton):
    """
        Test MTVITON
    """
    # Model
    mtviton.cuda()
    mtviton.eval()
    
    output_dir = os.path.join('./output', opt.mtviton_checkpoint.split('/')[-2], opt.mtviton_checkpoint.split('/')[-1],
                             opt.datamode, opt.datasetting, 'flow_visualize')
    os.makedirs(output_dir, exist_ok=True)
    num = 0
    iter_start_time = time.time()
    for inputs in test_loader.data_loader:
        
        # input1
        c_paired = inputs['cloth'][opt.datasetting].cuda()
        cm_paired = inputs['cloth_mask'][opt.datasetting].cuda()
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

            # forward
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = mtviton(input1, input2)
            
            # warped cloth mask one hot 
            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
            
            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask
                    
                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
                    
            N, _, iH, iW = c_paired.shape
            grid = make_grid(N, iH, iW)
            
            warped_clothes = []
            warped_clothes_mask = []
            
            for i in range(len(flow_list)):
                flow = F.interpolate(flow_list[i].permute(0, 3, 1, 2), scale_factor=2**(5-i), mode='bilinear').permute(0, 2, 3, 1)
                # flow_norm = torch.cat([flow[:, :, :, 0:1] / ((6*(2**i) - 1.0) / 2.0), flow[:, :, :, 1:2] / ((8*(2**i) - 1.0) / 2.0)], 3)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                print("i:",i,"torch.mean(flow**2):", torch.mean(flow**2))
                warped_grid = grid + flow_norm
                warped_clothes.append(F.grid_sample(c_paired, warped_grid, padding_mode='border').cpu() / 2 + 0.5)
                warped_clothes_mask.append(F.grid_sample(cm_paired, warped_grid, padding_mode='border').cpu().expand(-1, 3, -1, -1))

            # generated fake cloth mask & misalign mask
            fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalign = fake_clothmask - warped_cm_onehot
            misalign[misalign < 0.0] = 0.0
        
        for i in range(c_paired.shape[0]):
            grid = make_image_grid([warped_c[i] for warped_c in warped_clothes] +
                                   [warped_cm[i] for warped_cm in warped_clothes_mask] +
                                   [(im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5)],
                                    nrow=5)
            #board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
            save_image(grid, os.path.join(output_dir,
                             (inputs['c_name']['paired'][i].split('.')[0] + '_' +
                              inputs['c_name']['unpaired'][i].split('.')[0] + '.png')))
        num += c_paired.shape[0]
        print(num)

    print(f"Test time {time.time() - iter_start_time}")


def main():
    opt = get_opt()
    print(opt)
    print("Start to test!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)
    
    # visualization
    # if not os.path.exists(opt.tensorboard_dir):
    #     os.makedirs(opt.tensorboard_dir)
    # board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.mtviton_checkpoint.split('/')[-2], opt.mtviton_checkpoint.split('/')[-1], opt.datamode, opt.datasetting))
    board = None

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    mtviton = ImprovedMTVITON(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
       
    # Load Checkpoint
    load_checkpoint(mtviton, opt.mtviton_checkpoint)

    # Train
    test(opt, test_loader, board, mtviton)

    print("Finished testing!")


if __name__ == "__main__":
    main()