import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid as make_image_grid

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from cp_dataset_test import CPDatasetTest
from networks import ConditionGenerator, VGGLoss, load_checkpoint, save_checkpoint, make_grid
from network_generator import SPADEGenerator, MultiscaleDiscriminator, GANLoss

from sync_batchnorm import DataParallelWithCallback
from tensorboardX import SummaryWriter
from utils import create_network, visualize_segmap
import sys
from tqdm import tqdm

import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import transforms
import eval_models as models
import torchgeometry as tgm

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda',default=False, help='cuda or cpu')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, help='condition generator checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='', help='gen checkpoint')
    parser.add_argument('--dis_checkpoint', type=str, default='', help='dis checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    # test
    parser.add_argument("--lpips_count", type=int, default=1000)
    parser.add_argument("--test_datasetting", default="paired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0004, help='initial learning rate for adam')

    # SEAN-related hyper-parameters
    parser.add_argument('--GMM_const', type=float, default=None, help='constraint for GMM module')
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of input label classes without unknown class')
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                    help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                            'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
    
    # D
    parser.add_argument('--n_layers_D', type=int, default=3, help='# layers in each discriminator')
    parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
    
    # Training
    parser.add_argument('--GT', action='store_true')
    parser.add_argument('--occlusion', action='store_true')
    # tocg
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    # visualize
    parser.add_argument("--num_test_visualize", type=int, default=3)

    opt = parser.parse_args()

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
        "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        % (opt.batch_size, len(opt.gpu_ids))

    return opt


def train(opt, train_loader, test_loader, test_vis_loader, board, tocg, generator, discriminator, model):
    """
        Train Generator
    """

    # Model
    if not opt.GT:
        tocg.cuda()
        tocg.eval()
    generator.train()
    discriminator.train()
    model.eval()

    # criterion
    if opt.fp16:
        criterionGAN = GANLoss('hinge', tensor=torch.cuda.HalfTensor)
    else:
        criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor)
    # criterionL1 = nn.L1Loss()
    criterionFeat = nn.L1Loss()
    criterionVGG = VGGLoss(opt)

    # optimizer
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=opt.G_lr, betas=(0, 0.9))
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.D_lr, betas=(0, 0.9))
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, lr_lambda=lambda step: 1.0 -
            max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))

    if opt.fp16:
        if not opt.GT:
            from apex import amp
            [tocg, generator, discriminator], [optimizer_gen, optimizer_dis] = amp.initialize(
                [tocg, generator, discriminator], [optimizer_gen, optimizer_dis], opt_level='O1', num_losses=2)
        else:
            from apex import amp
            [generator, discriminator], [optimizer_gen, optimizer_dis] = amp.initialize(
                [generator, discriminator], [optimizer_gen, optimizer_dis], opt_level='O1', num_losses=2)

    if len(opt.gpu_ids) > 0:
        if not opt.GT:
            tocg = DataParallelWithCallback(tocg, device_ids=opt.gpu_ids)
        generator = DataParallelWithCallback(generator, device_ids=opt.gpu_ids)
        discriminator = DataParallelWithCallback(discriminator, device_ids=opt.gpu_ids)
        criterionGAN = DataParallelWithCallback(criterionGAN, device_ids=opt.gpu_ids)
        criterionFeat = DataParallelWithCallback(criterionFeat, device_ids=opt.gpu_ids)
        criterionVGG = DataParallelWithCallback(criterionVGG, device_ids=opt.gpu_ids)
        
    upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()

    for step in tqdm(range(opt.load_step, opt.keep_step + opt.decay_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input
        agnostic = inputs['agnostic'].cuda()
        parse_GT = inputs['parse'].cuda()
        pose = inputs['densepose'].cuda()
        parse_cloth = inputs['parse_cloth'].cuda()
        parse_agnostic = inputs['parse_agnostic'].cuda()
        pcm = inputs['pcm'].cuda()
        cm = inputs['cloth_mask']['paired'].cuda()
        c_paired = inputs['cloth']['paired'].cuda()
        
        # target
        im = inputs['image'].cuda()

        with torch.no_grad():
            if not opt.GT:
                # Warping Cloth
                # down
                pre_clothes_mask_down = F.interpolate(cm, size=(256, 192), mode='nearest')
                input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                densepose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                
                # multi-task inputs
                input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                # forward
                flow_list, fake_segmap, _, warped_clothmask_paired = tocg(input1, input2)
                
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
                        
                # warped cloth
                N, _, iH, iW = c_paired.shape
                grid = make_grid(N, iH, iW,opt)
                flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                warped_grid = grid + flow_norm
                warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border').detach()
                warped_clothmask = F.grid_sample(cm, warped_grid, padding_mode='border')

                # make generator input parse map
                fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                # occlusion
                if opt.occlusion:
                    warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                    warped_cloth_paired = warped_cloth_paired * warped_clothmask + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask)
                    warped_cloth_paired = warped_cloth_paired.detach()
                # region_mask = parse[:, 2:3] - warped_cm
                # region_mask[region_mask < 0.0] = 0.0
                # parse_rn = torch.cat((parse, region_mask), dim=1)
                # parse_rn[:, 2:3] -= region_mask
            else:
                # parse pre-process
                fake_parse = parse_GT.argmax(dim=1)[:, None]
                warped_cloth_paired = parse_cloth
                
            old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
            old_parse.scatter_(1, fake_parse, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse[:, i] += old_parse[:, label]
                    
            parse = parse.detach()
        # --------------------------------------------------------------------------------------------------------------
        #                                              Train the generator
        # --------------------------------------------------------------------------------------------------------------
        output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)

        fake_concat = torch.cat((parse, output_paired), dim=1)
        real_concat = torch.cat((parse, im), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        G_losses = {}
        G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)

        if not opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not opt.no_vgg_loss:
            G_losses['VGG'] = criterionVGG(output_paired, im) * opt.lambda_vgg

        loss_gen = sum(G_losses.values()).mean()

        optimizer_gen.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_gen, optimizer_gen, loss_id=0) as loss_gen_scaled:
                loss_gen_scaled.backward()
        else:
            loss_gen.backward()
        optimizer_gen.step()

        # --------------------------------------------------------------------------------------------------------------
        #                                            Train the discriminator
        # --------------------------------------------------------------------------------------------------------------
        with torch.no_grad():
            output = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
            output = output.detach()
            output.requires_grad_()

        fake_concat = torch.cat((parse, output), dim=1)
        real_concat = torch.cat((parse, im), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        D_losses = {}
        D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)

        loss_dis = sum(D_losses.values()).mean()

        optimizer_dis.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_dis, optimizer_dis, loss_id=1) as loss_dis_scaled:
                loss_dis_scaled.backward()
        else:
            loss_dis.backward()
        optimizer_dis.step()
        # --------------------------------------------------------------------------------------------------------------
        #                                            recording
        # --------------------------------------------------------------------------------------------------------------
        if (step + 1) % opt.tensorboard_count == 0:
            i = 0
            grid = make_image_grid([(c_paired[0].cpu() / 2 + 0.5), (cm[0].cpu()).expand(3, -1, -1), ((pose.cpu()[0]+1)/2), visualize_segmap(parse_agnostic.cpu(), batch=i),
                                    (warped_cloth_paired[i].cpu() / 2 + 0.5), (agnostic[i].cpu() / 2 + 0.5), (pose[i].cpu() / 2 + 0.5), visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                                    (output[i].cpu() / 2 + 0.5), (im[i].cpu() / 2 + 0.5)],
                                    nrow=4)
            board.add_images('train_images', grid.unsqueeze(0), step + 1)
            board.add_scalar('Loss/gen', loss_gen.item(), step + 1)
            board.add_scalar('Loss/gen/adv', G_losses['GAN'].mean().item(), step + 1)
            #board.add_scalar('Loss/gen/l1', G_losses['L1'].mean().item(), step + 1)
            board.add_scalar('Loss/gen/feat', G_losses['GAN_Feat'].mean().item(), step + 1)
            board.add_scalar('Loss/gen/vgg', G_losses['VGG'].mean().item(), step + 1)
            board.add_scalar('Loss/dis', loss_dis.item(), step + 1)
            board.add_scalar('Loss/dis/adv_fake', D_losses['D_Fake'].mean().item(), step + 1)
            board.add_scalar('Loss/dis/adv_real', D_losses['D_Real'].mean().item(), step + 1)
            
            # unpaired visualize
            generator.eval()
            
            inputs = test_vis_loader.next_batch()
            # input
            agnostic = inputs['agnostic'].cuda()
            parse_GT = inputs['parse'].cuda()
            pose = inputs['densepose'].cuda()
            parse_cloth = inputs['parse_cloth'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pcm = inputs['pcm'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()
            c_paired = inputs['cloth']['unpaired'].cuda()
            
            # target
            im = inputs['image'].cuda()
                        
            with torch.no_grad():
                if not opt.GT:
                    # Warping Cloth
                    # down
                    pre_clothes_mask_down = F.interpolate(cm, size=(256, 192), mode='nearest')
                    input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                    clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                    densepose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                    
                    # multi-task inputs
                    input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                    input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                    # forward
                    flow_list, fake_segmap, _, warped_clothmask_paired = tocg(input1, input2)
                    
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
                            
                    # warped cloth
                    N, _, iH, iW = c_paired.shape
                    grid = make_grid(N, iH, iW,opt)
                    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                    warped_grid = grid + flow_norm
                    warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border').detach()
                    warped_clothmask = F.grid_sample(cm, warped_grid, padding_mode='border')

                    # make generator input parse map
                    fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                    if opt.occlusion:
                        warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                        warped_cloth_paired = warped_cloth_paired * warped_clothmask + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask)
                        warped_cloth_paired = warped_cloth_paired.detach()

                else:
                    # parse pre-process
                    fake_parse = parse_GT.argmax(dim=1)[:, None]
                    warped_cloth_paired = parse_cloth
                    
                old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
                old_parse.scatter_(1, fake_parse, 1.0)

                labels = {
                    0:  ['background',  [0]],
                    1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                    2:  ['upper',       [3]],
                    3:  ['hair',        [1]],
                    4:  ['left_arm',    [5]],
                    5:  ['right_arm',   [6]],
                    6:  ['noise',       [12]]
                }
                parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                for i in range(len(labels)):
                    for label in labels[i][1]:
                        parse[:, i] += old_parse[:, label]
                        
                parse = parse.detach()
            
                output = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
                
                for i in range(opt.num_test_visualize):
                    grid = make_image_grid([(c_paired[i].cpu() / 2 + 0.5), (cm[i].cpu()).expand(3, -1, -1), ((pose.cpu()[i]+1)/2), visualize_segmap(parse_agnostic.cpu(), batch=i),
                        (warped_cloth_paired[i].cpu() / 2 + 0.5), (agnostic[i].cpu() / 2 + 0.5), (pose[i].cpu() / 2 + 0.5), visualize_segmap(fake_parse_gauss.cpu(), batch=i),
                        (output[i].cpu() / 2 + 0.5), (im[i].cpu() / 2 + 0.5)],
                        nrow=4)
                    board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                
            generator.train()

        if (step + 1) % opt.lpips_count == 0:
            generator.eval()
            T2 = transforms.Compose([transforms.Resize((128, 128))])
            lpips_list = []
            avg_distance = 0.0
            
            with torch.no_grad():
                print("LPIPS")
                for i in tqdm(range(500)):
                    inputs = test_loader.next_batch()
                    # input
                    agnostic = inputs['agnostic'].cuda()
                    parse_GT = inputs['parse'].cuda()
                    pose = inputs['densepose'].cuda()
                    parse_cloth = inputs['parse_cloth'].cuda()
                    parse_agnostic = inputs['parse_agnostic'].cuda()
                    pcm = inputs['pcm'].cuda()
                    cm = inputs['cloth_mask']['paired'].cuda()
                    c_paired = inputs['cloth']['paired'].cuda()
                    
                    # target
                    im = inputs['image'].cuda()
                                
                    with torch.no_grad():
                        if not opt.GT:
                            # Warping Cloth
                            # down
                            pre_clothes_mask_down = F.interpolate(cm, size=(256, 192), mode='nearest')
                            input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
                            clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
                            densepose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                            
                            # multi-task inputs
                            input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)

                            # forward
                            flow_list, fake_segmap, _, warped_clothmask_paired = tocg(input1, input2)
                            
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
                                    
                            # warped cloth
                            N, _, iH, iW = c_paired.shape
                            flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                            
                            grid = make_grid(N, iH, iW,opt)
                            warped_grid = grid + flow_norm
                            warped_cloth_paired = F.grid_sample(c_paired, warped_grid, padding_mode='border').detach()
                            warped_clothmask = F.grid_sample(cm, warped_grid, padding_mode='border')

                            # make generator input parse map
                            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear'))
                            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

                            if opt.occlusion:
                                warped_clothmask = remove_overlap(F.softmax(fake_parse_gauss, dim=1), warped_clothmask)
                                warped_cloth_paired = warped_cloth_paired * warped_clothmask + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask)
                                warped_cloth_paired = warped_cloth_paired.detach()

                        else:
                            # parse pre-process
                            fake_parse = parse_GT.argmax(dim=1)[:, None]
                            warped_cloth_paired = parse_cloth
                            
                        old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_().cuda()
                        old_parse.scatter_(1, fake_parse, 1.0)

                        labels = {
                            0:  ['background',  [0]],
                            1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                            2:  ['upper',       [3]],
                            3:  ['hair',        [1]],
                            4:  ['left_arm',    [5]],
                            5:  ['right_arm',   [6]],
                            6:  ['noise',       [12]]
                        }
                        parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_().cuda()
                        for i in range(len(labels)):
                            for label in labels[i][1]:
                                parse[:, i] += old_parse[:, label]
                                
                        parse = parse.detach()
                    
                    output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
                    
                    avg_distance += model.forward(T2(im), T2(output_paired))
                    
            avg_distance = avg_distance / 500
            print(f"LPIPS{avg_distance}")
            board.add_scalar('test/LPIPS', avg_distance, step + 1)
                
            generator.train()

        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print("step: %8d, time: %.3f, G_loss: %.4f, G_adv_loss: %.4f, D_loss: %.4f, D_fake_loss: %.4f, D_real_loss: %.4f"
                  % (step + 1, t, loss_gen.item(), G_losses['GAN'].mean().item(), loss_dis.item(),
                     D_losses['D_Fake'].mean().item(), D_losses['D_Real'].mean().item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(generator.module, os.path.join(opt.checkpoint_dir, opt.name, 'gen_step_%06d.pth' % (step + 1)),opt)
            save_checkpoint(discriminator.module, os.path.join(opt.checkpoint_dir, opt.name, 'dis_step_%06d.pth' % (step + 1)),opt)

        if (step + 1) % 1000 == 0:
            scheduler_gen.step()
            scheduler_dis.step()


def main():
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    
    # test dataloader
    opt.batch_size = 1
    opt.dataroot = opt.test_dataroot
    opt.datamode = 'test'
    opt.data_list = opt.test_data_list
    test_dataset = CPDatasetTest(opt)
    test_dataset = Subset(test_dataset, np.arange(500))
    test_loader = CPDataLoader(opt, test_dataset)
    
    # test vis loader
    opt.batch_size = opt.num_test_visualize
    test_vis_dataset = CPDatasetTest(opt)
    test_vis_loader = CPDataLoader(opt, test_vis_dataset)
    
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    
    # warping-seg Model
    tocg = None
    
    if not opt.GT:
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
        tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=13, ngf=96, norm_layer=nn.BatchNorm2d)
        # Load Checkpoint
        load_checkpoint(tocg, opt.tocg_checkpoint)

    # Generator model
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        generator.cuda()
    generator.init_weights(opt.init_type, opt.init_variance)
    discriminator = create_network(MultiscaleDiscriminator, opt)

    # lpips
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)

    # Load Checkpoint
    if not opt.gen_checkpoint == '' and os.path.exists(opt.gen_checkpoint):
        load_checkpoint(generator, opt.gen_checkpoint)
        load_checkpoint(discriminator, opt.dis_checkpoint)

    # Train
    train(opt, train_loader, test_loader, test_vis_loader, board, tocg, generator, discriminator, model)

    # Save Checkpoint
    save_checkpoint(generator, os.path.join(opt.checkpoint_dir, opt.name, 'gen_model_final.pth'),opt)
    save_checkpoint(discriminator, os.path.join(opt.checkpoint_dir, opt.name, 'dis_model_final.pth'),opt)

    print("Finished training %s!" % opt.name)


if __name__ == "__main__":
    main()
