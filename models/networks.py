import imp
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from models.normalize import RAIN,RAINV2
from torch.nn.utils import spectral_norm

from models.drconv import DRConv2d,SepDRConv2d,SepNoRefDRConv2d
from models.att import AdaptiveFusion,DVTATT
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.startswith('rain'):
        norm_layer = RAIN
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG,att,use_MGD, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    """load a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: rainnet
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'rhnet':
        net = RHNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=True,att=att,use_MGD=use_MGD)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    # print('gpu_ids~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',gpu_ids)

    return init_net(net, init_type, init_gain, gpu_ids)

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'target_decay':
        def target_decay(epoch):
            lr_l = 1
            if epoch >= 100 and epoch < 110:
                lr_l = 0.1
            elif epoch >= 110:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=target_decay)        
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    return net

def get_act_conv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.Conv2d(dims_in, dims_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)



def get_act_dconv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.ConvTranspose2d(dims_in, dims_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)

class RHNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=RAIN, 
                 norm_type_indicator=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                 use_dropout=False, use_attention=True,att='dvt',use_MGD=True):
        super(RHNet, self).__init__()
        self.input_nc = input_nc
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        self.att = att
        self.MGD = use_MGD
        # print(self.LD,self.MGD)
        norm_type_indicator=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        self.norm_namebuffer = ['RAIN','RAINV2']
        norm_type_list = [get_norm_layer('instance'), RAIN,RAINV2]
        # -------------------------------Network Settings-------------------------------------
        self.model_layer0         = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=False)
        self.model_layer1         = get_act_conv(nn.LeakyReLU(0.2, True), ngf, ngf*2, 4, 2, 1, False)
        self.model_layer1norm     = norm_type_list[norm_type_indicator[0]](ngf*2)

        self.model_layer2         = get_act_conv(nn.LeakyReLU(0.2, True), ngf*2, ngf*4, 3, 1, 1, False)
        self.model_layer2norm     = norm_type_list[norm_type_indicator[1]](ngf*4)

        self.model_layer3         = get_act_conv(nn.LeakyReLU(0.2, True), ngf*4, ngf*8, 4, 2, 1, False)
        self.model_layer3norm     = norm_type_list[norm_type_indicator[2]](ngf*8)
        
        self.model_layer4 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 3, 1, 1, False)
        self.model_layer4norm     = norm_type_list[norm_type_indicator[3]](ngf*8)
        
        self.model_layer5 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 4, 2, 1, False)
        self.model_layer5norm  = norm_type_list[norm_type_indicator[4]](ngf*8)

        self.model_layer6 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 3, 1, 1, False)
        self.model_layer6norm  = norm_type_list[norm_type_indicator[5]](ngf*8)
        
        self.model_layer71 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 3, 1, 1, False)
        self.model_layer72 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 3, 1, 1, False)
        self.model_layer72norm    = norm_type_list[norm_type_indicator[7]](ngf*8)
        
        self.model_layer8 = get_act_dconv(nn.ReLU(True), ngf*16, ngf*8, 3, 1, 1, False)
        self.model_layer8norm    = norm_type_list[norm_type_indicator[8]](ngf*8)
        
        self.model_layer9 = get_act_dconv(nn.ReLU(True), ngf*16, ngf*8, 4, 2, 1, False)
        self.model_layer9norm    = norm_type_list[norm_type_indicator[9]](ngf*8)
        
        self.model_layer10 = get_act_dconv(nn.ReLU(True), ngf*16, ngf*8, 3, 1, 1, False)
        self.model_layer10norm    = norm_type_list[norm_type_indicator[10]](ngf*8)
        
        self.model_layer11        = get_act_dconv(nn.ReLU(True), ngf*16, ngf*4, 4, 2, 1, False)
        self.model_layer11norm    = norm_type_list[norm_type_indicator[11]](ngf*4)
        print('using MGD')

        if use_attention:
            self.model_layer11att = DRConv2d(ngf*8, ngf*8, 1, 2)
            
        self.model_layer12        = get_act_dconv(nn.ReLU(True), ngf*8, ngf*2, 3, 1, 1, False)
        self.model_layer12norm    = norm_type_list[norm_type_indicator[12]](ngf*2)

        if use_attention:
            self.model_layer12att = DRConv2d(ngf*4, ngf*4, 1, 2)
            
        self.model_layer13        = get_act_dconv(nn.ReLU(True), ngf*4, ngf, 4, 2, 1, False)
        self.model_layer13norm    = norm_type_list[norm_type_indicator[13]](ngf)
        if use_attention:
            self.model_layer13att = DRConv2d(ngf*2, ngf*2, 1, 2)
        self.model_out = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1), nn.Tanh())
# 

        self.DVTATT = DVTATT(ngf*8*2)
        print('using dvtatt')

    def forward(self, x, mask,dvtfeats=None):
        x0 = self.model_layer0(x)
        x1 = self.model_layer1(x0)
        # print('x1',x1.device)
        if self.model_layer1norm._get_name() in self.norm_namebuffer:
            x1 = self.model_layer1norm(x1, mask)
        else:
            x1 = self.model_layer1norm(x1)
        x2 = self.model_layer2(x1)
        if self.model_layer2norm._get_name() in self.norm_namebuffer:
            x2 = self.model_layer2norm(x2, mask)
        else:
            x2 = self.model_layer2norm(x2)
        x3 = self.model_layer3(x2)
        if self.model_layer3norm._get_name() in self.norm_namebuffer:
            x3 = self.model_layer3norm(x3, mask)
        else:
            x3 = self.model_layer3norm(x3)

        x4 = self.model_layer4(x3)
        if self.model_layer4norm._get_name() in self.norm_namebuffer:
            x4 = self.model_layer4norm(x4, mask)
        else:
            x4 = self.model_layer4norm(x4)

        x5 = self.model_layer5(x4)
        if self.model_layer5norm._get_name() in self.norm_namebuffer:
            x5 = self.model_layer5norm(x5, mask)
        else:
            x5 = self.model_layer5norm(x5)

        x6 = self.model_layer6(x5)
        if self.model_layer6norm._get_name() in self.norm_namebuffer:
            x6 = self.model_layer6norm(x6, mask)
        else:
            x6 = self.model_layer6norm(x6)


        x71 = self.model_layer71(x6)     

        x72 = self.model_layer72(x71)
        if self.model_layer72norm._get_name() in self.norm_namebuffer:
            x72 = self.model_layer72norm(x72, mask)
        else:
            x72 = self.model_layer72norm(x72)
        x72 = torch.cat([x6,x72],1)

        ox5 = self.model_layer8(x72)
        if self.model_layer8norm._get_name() in self.norm_namebuffer:
            ox5 = self.model_layer8norm(ox5, mask)
        else:
            ox5 = self.model_layer8norm(ox5)      
        ox5 = torch.cat([x5,ox5],1)
        if self.att==1:
            ox5,attmap = self.DVTATT(ox5,mask,dvtfeats) 
        else:
            ox5,attmap = self.NODVTATT(ox5,mask) 

        ox4 = self.model_layer9(ox5)
        if self.model_layer9norm._get_name() in self.norm_namebuffer:
            ox4 = self.model_layer9norm(ox4, mask)
        else:
            ox4 = self.model_layer9norm(ox4)        
        ox4 = torch.cat([x4,ox4],1)


        ox3 = self.model_layer10(ox4)
        if self.model_layer10norm._get_name() in self.norm_namebuffer:
            ox3 = self.model_layer10norm(ox3, mask)
        else:
            ox3 = self.model_layer10norm(ox3)        
        ox3 = torch.cat([x3,ox3],1)


        ox2 = self.model_layer11(ox3)
        if self.model_layer11norm._get_name() in self.norm_namebuffer:
            ox2 = self.model_layer11norm(ox2, mask)
        else:
            ox2 = self.model_layer11norm(ox2)
        ox2 = torch.cat([x2,ox2],1)

        if self.use_attention:
            ox2 = self.model_layer11att(ox2, mask)
        
        ox1 = self.model_layer12(ox2)
        if self.model_layer12norm._get_name() in self.norm_namebuffer:
            ox1 = self.model_layer12norm(ox1, mask)
        else:
            ox1 = self.model_layer12norm(ox1)
        ox1 = torch.cat([x1,ox1],1)
        if self.use_attention:
            ox1 = self.model_layer12att(ox1, mask) 
            
        ox0 = self.model_layer13(ox1)
        if self.model_layer13norm._get_name() in self.norm_namebuffer:
            ox0 = self.model_layer13norm(ox0, mask)
        else:
            ox0 = self.model_layer13norm(ox0)
        ox0 = torch.cat([x0,ox0],1)
        if self.use_attention:
            ox0 = self.model_layer13att(ox0, mask) 

        out = self.model_out(ox0)
        return out,attmap


    
    def processImage(self, x, mask, background=None):
        if background is not None:
            x = x*mask + background * (1 - mask)
        if self.input_nc == 4:
            x = torch.cat([x, mask], dim=1) # (bs, 4, 256, 256)
        pred = self.forward(x, mask)

        return pred * mask + x[:,:3,:,:] * (1 - mask)

