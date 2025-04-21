
import torch
from .base_model import BaseModel
from . import networks


import torch.nn.functional as F
import random
from torch import nn, cuda
from torch.autograd import Variable
from .fMSE import MaskWeightedMSE
from models.image_generation import ImageGenerator

class RAIHANetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1','G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'output', 'mask', 'real_f', 'fake_f', 'bg', 'attentioned','reattentioned','ref','attmap']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.dvt,opt.use_MGD, opt.normG,
                                      not opt.no_dropout,opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:
            self.criterionL1 = MaskWeightedMSE(100)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        self.dvt = opt.dvt
        if self.dvt==1:

            self.extractor = ImageGenerator()
        self.maxpool = nn.MaxPool2d(8,8)
        self.softmax  = nn.Softmax(dim=-1)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = self.comp
        if self.opt.input_nc == 4:
            self.inputs = torch.cat([self.inputs, self.mask], 1)  # channel-wise concatenation
        self.real = self.real * self.mask + self.comp * (1-self.mask)
        self.real_f = self.real
        self.extet_comp = input['extet_comp'].to(self.device)
        self.externalmask = input['externalmask'].to(self.device)
        self.externalreal = input['externalreal'].to(self.device)
        self.extet_comp_dvt = input['extet_comp_dvt'].to(self.device)
        self.comp_dvt = input['comp_dvt'].to(self.device)
        self.ref_dvt = input['ref_dvt'].to(self.device)
        self.ref = input['ref'].to(self.device)
        self.externalreal = self.externalreal * self.externalmask + self.extet_comp * (1-self.externalmask)
        self.bg = self.real * (1 - self.mask)

    def forward(self):
        if self.dvt==1:
            with torch.no_grad():
                self.reffeats = self.extractor.inference(self.ref_dvt).permute([0,3,1,2])
                self.feats = self.extractor.inference(self.comp_dvt).permute([0,3,1,2])
                self.reffeats = torch.cat([self.feats,self.reffeats],-1)
        if self.dvt==1:
            self.output,_ = self.netG(self.inputs, self.mask, self.feats)
        else:
            self.output,_ = self.netG(self.inputs, self.mask)
        self.fake_f = self.output * self.mask
        self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        self.extendmask = torch.zeros_like(self.ref).to(self.mask.device)
        self.extendmask = self.extendmask[:,0:1,:,:]
        if self.dvt==1:

            self.refoutput,self.attmap = self.netG(torch.cat([self.extet_comp,self.ref],-1), torch.cat([self.externalmask,self.extendmask],-1),self.reffeats)
        else:
            self.refoutput,self.attmap = self.netG(torch.cat([self.extet_comp,self.ref],-1), torch.cat([self.externalmask,self.extendmask],-1))
        self.refoutput = self.refoutput[:,:,:,0:256]
        self.reattentioned = self.refoutput * self.externalmask + self.extet_comp[:,:3,:,:] * (1 - self.externalmask)

    def optimize_parameters(self):
        p_or_n = random.randint(0,1)
        self.optimizer_G.zero_grad() 
        if p_or_n==0:
            if self.dvt==1:
                with torch.no_grad():
                    self.feats =  self.extractor.inference(self.comp_dvt).permute([0,3,1,2])
                self.output,_ = self.netG(self.inputs, self.mask, self.feats)
            else:
                self.output,_ = self.netG(self.inputs, self.mask)
            self.fake_f = self.output * self.mask
            self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
            self.harmonized = self.attentioned
            self.loss_G_L1 = self.criterionL1(self.attentioned, self.real, self.mask) * self.opt.lambda_L1 
            self.loss_G_L1.backward()
            self.loss_G_L2 = 0.0
            self.optimizer_G.step()  # udpate G's weights
        else:
            self.extendmask = torch.zeros_like(self.ref).to(self.mask.device)
            self.extendmask = self.extendmask[:,0:1,:,:]
            if self.dvt==1:
                with torch.no_grad():
                    self.feats = self.extractor.inference(self.comp_dvt).permute([0,3,1,2])
                    self.reffeats = self.extractor.inference(self.ref_dvt).permute([0,3,1,2])
                    self.reffeats = torch.cat([self.feats,self.reffeats],-1)
                self.refoutput,self.attmap = self.netG(torch.cat([self.extet_comp,self.ref],-1), torch.cat([self.externalmask,self.extendmask],-1),self.reffeats)
            else:
                self.refoutput,self.attmap = self.netG(torch.cat([self.extet_comp,self.ref],-1), torch.cat([self.externalmask,self.extendmask],-1))
            self.refoutput = self.refoutput[:,:,:,0:256]
            self.reattentioned = self.refoutput * self.externalmask + self.extet_comp[:,:3,:,:] * (1 - self.externalmask)
            self.optimizer_G.zero_grad()  
            self.loss_G_L2 = self.criterionL1(self.reattentioned, self.externalreal, self.externalmask) * 1
            self.loss_G_L2.backward()
            self.loss_G_L1 = 0.0
            self.optimizer_G.step()  

