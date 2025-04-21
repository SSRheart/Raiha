import torch
import torch.nn as nn
import torch.nn.functional as F

class RAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        
        mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
        normalized = (x - mean_back) / std_back

        normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +
                                 self.background_beta[None, :, None, None]) * (1 - mask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore * std_back + mean_back
        normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) +
                                self.foreground_beta[None, :, None, None]) * mask

        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)


class MAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(MAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        # self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        # self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps
    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        
        # mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
        # normalized = (x - mean_back) / std_back

        # normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +
        #                          self.background_beta[None, :, None, None]) * (1 - mask)
        normalized_background = x * (1-mask)
        refmask = torch.ones(mask.size()).to(x.device)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore
        normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) +
                                self.foreground_beta[None, :, None, None]) * mask
        # print(normalized_foreground.min(),normalized_foreground.max())
        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)

class RAINV3(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAINV3, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask,refmask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        refmask = F.interpolate(refmask.detach(), size=x.size()[2:], mode='nearest')
        mean_back, std_back = self.get_foreground_mean_std(x * refmask, refmask) # the background features
        normalized = (x - mean_back) / std_back

        normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +
                                 self.background_beta[None, :, None, None]) * (refmask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore * std_back + mean_back
        normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) +
                                self.foreground_beta[None, :, None, None]) * mask

        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)


class RAINV2(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAINV2, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.reference_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.reference_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)

        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        # print('aaaaaaaaaaaaa')
        # print(mask.shape)
        if x.shape[-1]>x.shape[-2]:
            x_inter = x[:,:,:,:x.shape[-2]]
            mask_inter = mask[:,:,:,:x.shape[-2]]
            x_external = x[:,:,:,x.shape[-2]:]
            # mask_inter = mask[:,:,:,:x.shape[-1]]

            mean_back, std_back = self.get_foreground_mean_std(x_inter * (1-mask_inter), 1 - mask_inter) # the background features
            normalized = (x_inter - mean_back) / std_back

            normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +
                                    self.background_beta[None, :, None, None]) * (1 - mask_inter)

            mean_fore, std_fore = self.get_foreground_mean_std(x_inter * mask_inter, mask_inter) # the background features
            normalized = (x_inter - mean_fore) / std_fore * std_back + mean_back
            normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) +
                                    self.foreground_beta[None, :, None, None]) * mask_inter


            mask_exter = torch.ones_like(mask_inter).to(x.device)
            mean_refer, std_refer = self.get_foreground_mean_std(x_external ,mask_exter ) # the background features
            normalized = (x_external - mean_refer) / std_refer
            normalized_external = (normalized * (1 + self.reference_gamma[None, :, None, None]) +
                                    self.reference_beta[None, :, None, None]) * mask_exter
            return torch.cat([normalized_foreground + normalized_background,normalized_external],-1)

        else:
                
            mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
            normalized = (x - mean_back) / std_back

            normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +
                                    self.background_beta[None, :, None, None]) * (1 - mask)

            mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
            normalized = (x - mean_fore) / std_fore * std_back + mean_back
            normalized_foreground = (normalized * (1 + self.foreground_gamma[None, :, None, None]) +
                                    self.foreground_beta[None, :, None, None]) * mask

            return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)