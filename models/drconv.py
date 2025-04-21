import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable, Function

def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk,  **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk,  **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out

class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                # print('here')

                return xcorr_fast(x, kernel, kwargs)
        else:
            # print('??????????????????')
            return Corr.apply(x, kernel, 1, kwargs)

class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = 2

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs)
        
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.act = nn.Sigmoid()
    def forward(self, input, mask):
        kernel = self.conv_kernel(input)
        # print('kernel',kernel.shape)

        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H

        # print('kernel view',kernel.shape)
        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H
        # print(output.shape)
        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')
        mask = mask.unsqueeze(1) 
        inv_msak = 1 - mask
        guide_mask = torch.cat((mask, inv_msak), 1)
        # print(guide_mask.shape)
        output = torch.sum(output * guide_mask, dim=1)
        return output



class DRV2Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, **kwargs):
        super(DRV2Conv2d, self).__init__()
        self.region_num = 2

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs)
        
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.act = nn.Sigmoid()
    def forward(self, input, mask,refmask):
        kernel = self.conv_kernel(input)
        # print('kernel',kernel.shape)

        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H

        # print('kernel view',kernel.shape)
        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H

        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H
        # print(output.shape)
        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')
        mask = mask.unsqueeze(1) 

        refmask = F.interpolate(refmask.detach(), size=input.size()[2:], mode='nearest')
        refmask = refmask.unsqueeze(1) 
        inv_msak = refmask
        guide_mask = torch.cat((mask, inv_msak), 1)
        # print(guide_mask.shape)
        output = torch.sum(output * guide_mask, dim=1)
        return output

class SepDRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=1, **kwargs):
        super(SepDRConv2d, self).__init__()
        self.region_num = 2

        self.conv_kernel = nn.Sequential(
            # nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels*self.region_num, self.region_num * self.region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(self.region_num * self.region_num, self.region_num * in_channels * out_channels, kernel_size=1, groups=self.region_num)
        )
        # self.refregion_num=3
        # self.refconv_kernel = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
        #     nn.Conv2d(in_channels*self.refregion_num, self.refregion_num * self.refregion_num, kernel_size=1),
        #     nn.Sigmoid(),
        #     nn.Conv2d(self.refregion_num * self.refregion_num, self.refregion_num * in_channels * out_channels, kernel_size=1, groups=self.refregion_num)
        # )

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.reffused = nn.Conv2d(in_channels*3, in_channels*1, kernel_size=1,stride=1)

        self.act = nn.Sigmoid()
    def forward(self, input, mask):
        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')
        if input.shape[-1]>input.shape[-2]:

            back_mask = 1 - mask
            inv_mask = 1 - mask

            back_mask[:,:,:,mask.shape[-2]:]=0
            ref_mask = 1 - mask
            ref_mask[:,:,:,0:mask.shape[-2]]=0
            fore_input = torch.sum(input*mask,[2,3],keepdim=True) / torch.sum(mask,[2,3],keepdim=True)
            back_input = torch.sum(input*back_mask,[2,3],keepdim=True) / torch.sum(back_mask,[2,3],keepdim=True)
            ref_input = torch.sum(input*ref_mask,[2,3],keepdim=True) / torch.sum(ref_mask,[2,3],keepdim=True)
            # print(torch.cat([fore_input,back_input,ref_input],1).shape)
            fusedfeats = self.reffused(torch.cat([fore_input,back_input,ref_input],1))
            kernel = self.conv_kernel(torch.cat([fore_input,fusedfeats],1))
            kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H
            # print('after',kernel.shape)

            # print(kernel.shape)
            output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
            # print('output',input.shape,output.shape,kernel.shape)

            output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H
            # print(output.shape)

            mask = mask.unsqueeze(1) 
            back_mask = back_mask.unsqueeze(1) 
            ref_mask = ref_mask.unsqueeze(1) 
            inv_mask = inv_mask.unsqueeze(1) 
            guide_mask = torch.cat((mask, inv_mask), 1)
            # print(guide_mask.shape)
            output = torch.sum(output * guide_mask, dim=1)
            # print(input.shape,output.shape,output_f.shape,mask.shape,output_b.shape)
            return output
        else:
            inv_mask = 1 - mask

            fore_input = torch.sum(input*mask,[2,3],keepdim=True) / torch.sum(mask,[2,3],keepdim=True)
            back_input = torch.sum(input*inv_mask,[2,3],keepdim=True) / torch.sum(inv_mask,[2,3],keepdim=True)
                
            kernel = self.conv_kernel(torch.cat([fore_input,back_input],1))
            # kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H

            output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
            output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H
            # print(torch.sum(outputview[:,0,:,:,:]!=output[:,:output.shape[1]//2,:,:]))
            mask = mask.unsqueeze(1) 
            inv_mask = inv_mask.unsqueeze(1) 

            guide_mask = torch.cat((mask, inv_mask), 1)
            # print(guide_mask.shape)
            output = torch.sum(output * guide_mask, dim=1)

            return output


class SepNoRefDRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=1, **kwargs):
        super(SepNoRefDRConv2d, self).__init__()
        self.region_num = 2

        self.conv_kernel = nn.Sequential(
            # nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels*self.region_num, self.region_num * self.region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(self.region_num * self.region_num, self.region_num * in_channels * out_channels, kernel_size=1, groups=self.region_num)
        )
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.act = nn.Sigmoid()
    def forward(self, input, mask):
        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')

        inv_mask = 1 - mask

        fore_input = torch.sum(input*mask,[2,3],keepdim=True) / torch.sum(mask,[2,3],keepdim=True)
        back_input = torch.sum(input*inv_mask,[2,3],keepdim=True) / torch.sum(inv_mask,[2,3],keepdim=True)
            
        kernel = self.conv_kernel(torch.cat([fore_input,back_input],1))

        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
        channel = output.shape[1]//2

        output_f = output[:,0*channel:1*channel,:,:]
        output_b = output[:,1*channel:2*channel,:,:]

        return output_f*mask + output_b*inv_mask

class SepNoRefLargeDRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=1, **kwargs):
        super(SepNoRefLargeDRConv2d, self).__init__()
        self.region_num = 2

        self.conv_kernel = nn.Sequential(
            # nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels*self.region_num, in_channels * self.region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels * self.region_num, self.region_num * in_channels * out_channels, kernel_size=1, groups=self.region_num)
        )
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.act = nn.Sigmoid()
    def forward(self, input, mask):
        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')

        inv_mask = 1 - mask

        fore_input = torch.sum(input*mask,[2,3],keepdim=True) / torch.sum(mask,[2,3],keepdim=True)
        back_input = torch.sum(input*inv_mask,[2,3],keepdim=True) / torch.sum(inv_mask,[2,3],keepdim=True)
            
        kernel = self.conv_kernel(torch.cat([fore_input,back_input],1))

        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
        channel = output.shape[1]//2

        output_f = output[:,0*channel:1*channel,:,:]
        output_b = output[:,1*channel:2*channel,:,:]

        return output_f*mask + output_b*inv_mask
