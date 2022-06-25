import torch
import torch.nn as nn

from torchvision.utils import make_grid as make_image_grid

import argparse
import os
import time
from cp_dataset import CPDataset, CPDatasetTest, CPDataLoader
from networks import ClothFlow, VGGLoss, load_checkpoint, save_checkpoint, make_grid
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="/home/nas1_userB/dataset/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs_zalando.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of input label classes without unknown class')

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard/clothflow_pose_input', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/clothflow_pose_input', help='save checkpoint infos')
    parser.add_argument('--clothflow_checkpoint', type=str, default='', help='clothflow checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    # test visualize
    parser.add_argument("--no_test_visualize", action='store_true')    
    parser.add_argument("--num_test_visualize", type=int, default=3)
    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="/home/nas2_userF/gyojunggu/WUTON/data/zalando-hd-resize")
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    

    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])


    opt = parser.parse_args()
    return opt


def train(opt, train_loader, test_loader, board, clothflow):
    """
        Train ClothFlow
    """

    # Model
    clothflow.cuda()
    clothflow.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    # optimizer
    optimizer_gen = torch.optim.Adam(clothflow.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in tqdm(range(opt.load_step, opt.keep_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1        
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        # target GT
        im_c = inputs['parse_cloth'].cuda()
        parse_cloth_mask = inputs['pcm'].cuda()
        # visualization
        im = inputs['image']

        N, _, iH, iW = c_paired.shape
        grid = make_grid(N, iH, iW)

        # -----------------------------------------------------------------------------------------------
        #                                    Calculate loss for paired data
        # -----------------------------------------------------------------------------------------------

        # forward
        flow_list = clothflow(torch.cat((c_paired, cm_paired), 1), torch.cat([parse_agnostic, densepose], 1), upsample='bilinear')
        # warping
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
        warped_grid = grid + flow_norm

        warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border')
        warped_clothmask_paired = F.grid_sample(cm_paired, warped_grid, padding_mode='border')

        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0
        for flow in flow_list:
            y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
            x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
            loss_tv = loss_tv + y_tv + x_tv

        loss_gen = 10 * loss_l1_cloth + loss_vgg + 2 * loss_tv

        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # tensorboard
        if (step + 1) % opt.tensorboard_count == 0:
            # loss G
            board.add_scalar('Loss/l1_cloth', loss_l1_cloth.item(), step + 1)
            board.add_scalar('Loss/vgg', loss_vgg.item(), step + 1)
            board.add_scalar('Loss/tv', loss_tv.item(), step + 1)
            
            grid = make_image_grid([(c_paired[0].cpu() / 2 + 0.5), (cm_paired[0].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu()), ((densepose.cpu()[0]+1)/2),
                                    (im_c[0].cpu() / 2 + 0.5), parse_cloth_mask[0].cpu().expand(3, -1, -1), (warped_cloth_paired[0].cpu().detach() / 2 + 0.5), (warped_clothmask_paired[0].cpu().detach()).expand(3, -1, -1),
                                    (im[0]/2 +0.5)],
                                    nrow=4)
            
            board.add_images('train_images', grid.unsqueeze(0), step + 1)
            
            if not opt.no_test_visualize:
                inputs = test_loader.next_batch()
                # input1
                c_paired = inputs['cloth'][opt.test_datasetting].cuda()
                cm_paired = inputs['cloth_mask'][opt.test_datasetting].cuda()
                # input2
                parse_agnostic = inputs['parse_agnostic'].cuda()
                densepose = inputs['densepose'].cuda()
                # GT
                parse_cloth_mask = inputs['pcm'].cuda()  # L1
                im_c = inputs['parse_cloth'].cuda()  # VGG
                # visualization
                im = inputs['image']
                
                N, _, iH, iW = c_paired.shape
                grid = make_grid(N, iH, iW)

                clothflow.eval()
                with torch.no_grad():
                    # forward
                    flow_list = clothflow(torch.cat((c_paired, cm_paired), 1), torch.cat([parse_agnostic, densepose], 1), upsample='bilinear')
                    # warping
                    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear').permute(0, 2, 3, 1)
                    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                    warped_grid = grid + flow_norm

                    warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border')
                    warped_clothmask_paired = F.grid_sample(cm_paired, warped_grid, padding_mode='border')
                
                for i in range(opt.num_test_visualize):
                    grid = make_image_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                                            (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_clothmask_paired[i].cpu().detach()).expand(3, -1, -1),
                                            (im[i]/2 +0.5)],
                                            nrow=4)
                    board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                clothflow.train()

        # save
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(clothflow, os.path.join(opt.checkpoint_dir, opt.name, 'clothflow_step_%06d.pth' % (step + 1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create train dataset & loader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    
    # create test dataset & loader
    test_loader = None
    if not opt.no_test_visualize:
        opt.batch_size = opt.num_test_visualize
        opt.dataroot = opt.test_dataroot
        opt.datamode = 'test'
        opt.data_list = opt.test_data_list
        test_dataset = CPDatasetTest(opt)
        test_loader = CPDataLoader(opt, test_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # Model
    input1_nc = 4  # cloth + clothmask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    clothflow = ClothFlow(input1_nc=input1_nc, input2_nc=input2_nc, ngf=96, norm_layer=nn.BatchNorm2d)

    # Load Checkpoint
    if not opt.clothflow_checkpoint == '' and os.path.exists(opt.clothflow_checkpoint):
        load_checkpoint(clothflow, opt.clothflow_checkpoint)

    # Train
    train(opt, train_loader, test_loader, board, clothflow)

    # Save Checkpoint
    save_checkpoint(clothflow, os.path.join(opt.checkpoint_dir, opt.name, 'clothflow_final.pth'))

    print("Finished training %s!" % opt.name)


if __name__ == "__main__":
    main()