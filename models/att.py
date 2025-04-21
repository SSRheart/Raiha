import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDynamics(nn.Module):
    def __init__(self, dims_in):

        super(LocalDynamics, self).__init__()
        self.linear = nn.Conv1d(dims_in*2, dims_in, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim = -1)
        self.k = 1
    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        B, C, H, W = x.shape
        # print(x.shape)
        flattened_features_query = x * mask
        flattened_features_support = x * (1-mask)
        
        flattened_features_query = flattened_features_query.view(B, C, -1)
        flattened_features_support = flattened_features_support.view(B, C, -1)
        attmask = []
        masked_features = []
        for i in range(B):

            query_features = flattened_features_query[i].view(C, -1)
            support_features = flattened_features_support[i].view(C, -1)
            
            query = query_features.unsqueeze(0)
            support = support_features.unsqueeze(0)

            simi_matrix = torch.matmul(query.permute(0, 2, 1), support)
            
            weights_value, index = torch.topk(simi_matrix, dim=2, k=self.k)
            attmask = index
            views = [support.shape[0]] + [1 if i != 2 else -1 for i in range(1, len(support.shape))]
            expanse = list(support.shape)
            expanse[0] = -1
            expanse[2] = -1
            index = index.view(views)
            index = index.expand(expanse)   
            weights_value = weights_value.view(views)
            weights_value = weights_value.expand(expanse)
            select_value = torch.gather(support, 2, index)
 
            select_value = select_value.view(1, C, self.k, -1)
            weights_value = weights_value.view(1, C, self.k, -1)
            weights_value = self.softmax(weights_value)
            # print(weights_value.shape,weights_value[0,1,0,:]-weights_value[0,2,0,:])
            fuse_tensor = weights_value * select_value
            fuse_tensor = torch.sum(fuse_tensor, -2)
            # print(fuse_tensor.shape)
            
            hybrid_feat = torch.cat((fuse_tensor, query), 1)
            hybrid_feat = self.linear(hybrid_feat)
            masked_features.append(hybrid_feat)
        # print(attmask.shape)
        attmap = torch.zeros([B,H*W]).to(x.device)
        attmap[:,attmask[:,:,0]] = 1
        # print(attmap.shape)

        attmap = attmap.view([B,1,H,W])
        # print(attmask.shape,attmask.max(),attmask.min())
        # attmap = torch.mean(weights_value,1).view(1,1,H,W)
        attmap = F.interpolate(attmap, size=[H*8,W*8], mode='nearest')
        refined_feat = torch.cat(masked_features, 0)
        refined_feat = refined_feat.view(B, C, H, W)
        return refined_feat * mask + x * (1-mask),attmap

class CosLocalDynamics(nn.Module):
    def __init__(self, dims_in):

        super(CosLocalDynamics, self).__init__()
        self.linear = nn.Conv1d(dims_in*2, dims_in, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim = -1)
        self.k = 1
    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        B, C, H, W = x.shape
        # print(x.shape)
        flattened_features_query = x * mask
        flattened_features_support = x * (1-mask)
        
        flattened_features_query = flattened_features_query.view(B, C, -1)
        flattened_features_support = flattened_features_support.view(B, C, -1)
        flattened_features_query = flattened_features_query / (flattened_features_query.norm(dim=1, keepdim=True)+1e-8)
        flattened_features_support = flattened_features_support / (flattened_features_support.norm(dim=1, keepdim=True)+1e-8)

        attmask = []
        masked_features = []
        for i in range(B):

            query_features = flattened_features_query[i].view(C, -1)
            support_features = flattened_features_support[i].view(C, -1)
            
            query = query_features.unsqueeze(0)
            support = support_features.unsqueeze(0)

            simi_matrix = torch.matmul(query.permute(0, 2, 1), support)
            
            weights_value, index = torch.topk(simi_matrix, dim=2, k=self.k)
            attmask = index
            views = [support.shape[0]] + [1 if i != 2 else -1 for i in range(1, len(support.shape))]
            expanse = list(support.shape)
            expanse[0] = -1
            expanse[2] = -1
            index = index.view(views)
            index = index.expand(expanse)   
            weights_value = weights_value.view(views)
            weights_value = weights_value.expand(expanse)
            select_value = torch.gather(support, 2, index)
 
            select_value = select_value.view(1, C, self.k, -1)
            weights_value = weights_value.view(1, C, self.k, -1)
            weights_value = self.softmax(weights_value)
            # print(weights_value.shape,weights_value[0,1,0,:]-weights_value[0,2,0,:])
            fuse_tensor = weights_value * select_value
            fuse_tensor = torch.sum(fuse_tensor, -2)
            # print(fuse_tensor.shape)
            
            hybrid_feat = torch.cat((fuse_tensor, query), 1)
            hybrid_feat = self.linear(hybrid_feat)
            masked_features.append(hybrid_feat)
        # print(attmask.shape)
        attmap = torch.zeros([B,H*W]).to(x.device)
        attmap[:,attmask[:,:,0]] = 1
        # print(attmap.shape)

        attmap = attmap.view([B,1,H,W])
        # print(attmask.shape,attmask.max(),attmask.min())
        # attmap = torch.mean(weights_value,1).view(1,1,H,W)
        attmap = F.interpolate(attmap, size=[H*8,W*8], mode='nearest')
        refined_feat = torch.cat(masked_features, 0)
        refined_feat = refined_feat.view(B, C, H, W)
        return refined_feat * mask + x * (1-mask),attmap

class CosLocalDynamicsV2(nn.Module):
    def __init__(self, dims_in):

        super(CosLocalDynamicsV2, self).__init__()
        self.linear = nn.Conv1d(dims_in*2, dims_in, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim = -1)
        self.k = 1
    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        B, C, H, W = x.shape
        # print(x.shape)
        flattened_features_query = x * mask
        flattened_features_support = x * (1-mask)
        
        flattened_features_query = flattened_features_query.view(B, C, -1)
        flattened_features_support = flattened_features_support.view(B, C, -1)
        flattened_features_query = flattened_features_query / (flattened_features_query.norm(dim=1, keepdim=True)+1e-8)
        flattened_features_support = flattened_features_support / (flattened_features_support.norm(dim=1, keepdim=True)+1e-8)
        attmask = []
        masked_features = []
        for i in range(B):

            query_features = flattened_features_query[i].view(C, -1)
            support_features = flattened_features_support[i].view(C, -1)
            # print(mask.shape)
            fore_mask = mask[i].view([1,-1]) # [1,1024,1]

            query = query_features.unsqueeze(0)
            support = support_features.unsqueeze(0)

            simi_matrix = torch.matmul(query.permute(0, 2, 1), support)            
            weights_value, index = torch.topk(simi_matrix, dim=2, k=self.k)  # [1,1024,1]


            fore_weights_value, _ = torch.topk(simi_matrix, dim=1, k=self.k)  # [1,1024,1]

            # print(weights_value.max().item())
            attmask = index
            views = [support.shape[0]] + [1 if i != 2 else -1 for i in range(1, len(support.shape))]
            expanse = list(support.shape)
            expanse[0] = -1
            expanse[2] = -1
            index = index.view(views)
            index = index.expand(expanse)   
            validmask = (fore_weights_value > 0.5)
            validmask = validmask * (fore_mask[None,:,:])
            validmask = validmask[0,:,0] # 1024  h*w
            # validmask = validmask.view([H,W])
            validmask = validmask[None,None,:]
            # print(validmask.shape,weights_value.shape,fore_mask.shape)

            weights_value = weights_value.view(views)
            weights_value = weights_value.expand(expanse)
            select_value = torch.gather(support, 2, index)
 
            select_value = select_value.view(1, C, self.k, -1)
            weights_value = weights_value.view(1, C, self.k, -1)
            weights_value = self.softmax(weights_value)
            # print(weights_value.shape,weights_value[0,1,0,:]-weights_value[0,2,0,:])
            fuse_tensor = weights_value * select_value
            fuse_tensor = torch.sum(fuse_tensor, -2)
            # print(fuse_tensor.shape)
            # fuse_tensor = fuse_tensor * validmask[None,None,:] + query * (1-validmask[None,None,:])
            # validmask
            hybrid_feat = torch.cat((fuse_tensor, query), 1)
            hybrid_feat = self.linear(hybrid_feat)
            # print()
            # if validmask.sum().item()>0:
            #     print(validmask.sum().item(),fore_mask.sum().item())
            hybrid_feat = (hybrid_feat * validmask ) + (query * (1-validmask))
            masked_features.append(hybrid_feat)
        # print(attmask.shape)
        attmap = torch.zeros([B,H*W]).to(x.device)
        attmap[:,attmask[:,:,0]] = 1
        # print(attmap.shape)

        attmap = attmap.view([B,1,H,W])
        # print(attmask.shape,attmask.max(),attmask.min())
        # attmap = torch.mean(weights_value,1).view(1,1,H,W)
        attmap = F.interpolate(attmap, size=[H*8,W*8], mode='nearest')
        refined_feat = torch.cat(masked_features, 0)
        refined_feat = refined_feat.view(B, C, H, W)
        return refined_feat * mask + x * (1-mask),attmap




    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        shortcut = self.shortcut(residual)
        x += shortcut
        x = self.relu(x)
        
        return x

class RCTV2(nn.Module):
    def __init__(self,in_channels):
        super(RCTV2, self).__init__()
        self.K_conv = ResidualBlock(in_channels,in_channels//1)
        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)

        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(8,8)
    def forward(self, x, mask):
        b,c,h,w = x.shape

        # mask = torch.nn.functional.interpolate(mask[:,0:1,:,:],[h,w],mode='nearest')
        mask = self.maxpool(mask)
        # print(mask.max())
        mask = (mask>0).float()
        backmask = 1 - mask
        # backmask[]
        K_fea = self.K_conv(x)*backmask
        Q_fea = self.K_conv(x)*(mask)
        V_fea = x * backmask # mask

        K_fea = (K_fea).view(b,c//1,w*h) # b Cb c
        K_fea = K_fea / (K_fea.norm(dim=1, keepdim=True)+1e-8)
        Q_fea = (Q_fea).view(b,c//1,w*h).permute(0,2,1)  # b c Cb
        Q_fea = Q_fea / (Q_fea.norm(dim=2, keepdim=True)+1e-8)
        V_fea = (V_fea).view(b,c,w*h) # b Cb c
        scoremap = torch.bmm(Q_fea,K_fea) * 20

        Attmap = self.softmaxmask(scoremap,mask,backmask) # b Cf Cb

        att_fore = torch.bmm(V_fea.view(b,c,w*h) , Attmap.permute(0,2,1))
        att_fore = att_fore.view([b,c,h,w])
        fused_feats = self.fuse(torch.cat([att_fore,x],1))

        M_fea = mask.view(b,1,w*h)
        # attmask = attmask / attmask.max()
        visatt = torch.sum(Attmap,1,keepdim=True) 
        visatt = visatt.view([b,1,h,w])
        attmask = F.interpolate(visatt, size=[h*8,w*8], mode='nearest')
        attmask = attmask / attmask.max()
        # print(attmask.shape,attmask.max())
        return fused_feats*mask+x*(1-mask),attmask

    def softmaxmask(self,x,querymask,keymask):  #querymask 前景mask keymask  reference对应的mask
        b,c,n= x.shape
        querymask = querymask.view([b,1,-1])
        querymask = (querymask>0).float()
        keymask = keymask.view([b,1,-1])
        keymask = (keymask>0).float()
        onesmask = torch.ones([b,querymask.shape[-1],keymask.shape[-1]]).to(x.device)
        # zerosmask = torch.ones([b,n,n]).to(x.device)
        validmask = onesmask * (querymask.permute([0,2,1])) * (keymask)
        validx = x+(1-keymask)*(-1e6)+(1-querymask.permute([0,2,1]))*(-1e6)
        attmap  = self.softmax(validx)
        attmap = attmap * validmask
        return attmap
class DVTATT(nn.Module):
    def __init__(self,in_channels):
        super(DVTATT, self).__init__()
        self.K_conv = nn.Conv2d(in_channels+768,in_channels//1,1,1)
        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)

        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(8,8)
    def forward(self, x, mask, dvtfeats):
        b,c,h,w = x.shape
        # print(dvtfeats.shape,x.shape)
        # mask = torch.nn.functional.interpolate(mask[:,0:1,:,:],[h,w],mode='nearest')
        mask = self.maxpool(mask)
        # print(mask.max())
        mask = (mask>0).float()
        backmask = 1 - mask
        # backmask[]
        # print(x.shape,dvtfeats.shape,mask.shape)
        K_fea = self.K_conv(torch.cat([x,dvtfeats],1))*backmask
        Q_fea = self.K_conv(torch.cat([x,dvtfeats],1))*(mask)
        V_fea = x * backmask # mask

        K_fea = (K_fea).view(b,c//1,w*h) # b Cb c
        K_fea = K_fea / (K_fea.norm(dim=1, keepdim=True)+1e-8)
        Q_fea = (Q_fea).view(b,c//1,w*h).permute(0,2,1)  # b c Cb
        Q_fea = Q_fea / (Q_fea.norm(dim=2, keepdim=True)+1e-8)
        V_fea = (V_fea).view(b,c,w*h) # b Cb c
        scoremap = torch.bmm(Q_fea,K_fea) * 20

        Attmap = self.softmaxmask(scoremap,mask,backmask) # b Cf Cb

        att_fore = torch.bmm(V_fea.view(b,c,w*h) , Attmap.permute(0,2,1))
        att_fore = att_fore.view([b,c,h,w])
        fused_feats = self.fuse(torch.cat([att_fore,x],1))

        M_fea = mask.view(b,1,w*h)
        # attmask = attmask / attmask.max()
        visatt = torch.sum(Attmap,1,keepdim=True) 
        visatt = visatt.view([b,1,h,w])
        attmask = F.interpolate(visatt, size=[h*8,w*8], mode='nearest')
        attmask = attmask / attmask.max()
        # print(attmask.shape,attmask.max())
        return fused_feats*mask+x*(1-mask),attmask

    def softmaxmask(self,x,querymask,keymask):  #querymask 前景mask keymask  reference对应的mask
        b,c,n= x.shape
        querymask = querymask.view([b,1,-1])
        querymask = (querymask>0).float()
        keymask = keymask.view([b,1,-1])
        keymask = (keymask>0).float()
        onesmask = torch.ones([b,querymask.shape[-1],keymask.shape[-1]]).to(x.device)
        # zerosmask = torch.ones([b,n,n]).to(x.device)
        validmask = onesmask * (querymask.permute([0,2,1])) * (keymask)
        validx = x+(1-keymask)*(-1e6)+(1-querymask.permute([0,2,1]))*(-1e6)
        attmap  = self.softmax(validx)
        attmap = attmap * validmask
        return attmap
class SGF(nn.Module):
    def __init__(self,in_channels):
        super(SGF, self).__init__()
        self.K_conv = nn.Conv2d(in_channels+768,in_channels//1,1,1)
        self.Q_conv = nn.Conv2d(in_channels+768,in_channels//1,1,1)

        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)

        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(8,8)
    def forward(self, x, mask, dvtfeats):
        b,c,h,w = x.shape
        mask = self.maxpool(mask)
        mask = (mask>0).float()
        backmask = 1 - mask

        K_fea = self.K_conv(torch.cat([x,dvtfeats],1))
        Q_fea = self.Q_conv(torch.cat([x,dvtfeats],1))
        V_fea = x # mask

        K_fea = (K_fea).view(b,c//1,w*h) # b Cb c
        K_fea = K_fea / (K_fea.norm(dim=1, keepdim=True)+1e-8)
        Q_fea = (Q_fea).view(b,c//1,w*h).permute(0,2,1)  # b c Cb
        Q_fea = Q_fea / (Q_fea.norm(dim=2, keepdim=True)+1e-8)
        V_fea = (V_fea).view(b,c,w*h) # b Cb c
        scoremap = torch.bmm(Q_fea,K_fea) * 10

        Attmap = self.softmaxmask(scoremap,mask,backmask) # b Cf Cb

        att_fore = torch.bmm(V_fea.view(b,c,w*h) , Attmap.permute(0,2,1))
        att_fore = att_fore.view([b,c,h,w])
        fused_feats = self.fuse(torch.cat([att_fore,x],1))

        M_fea = mask.view(b,1,w*h)
        visatt = torch.sum(Attmap,1,keepdim=True) 
        visatt = visatt.view([b,1,h,w])
        attmask = F.interpolate(visatt, size=[h*8,w*8], mode='nearest')
        attmask = attmask / attmask.max()
        return fused_feats*mask+x*(1-mask),attmask

    def softmaxmask(self,x,querymask,keymask):  #querymask 前景mask keymask  reference对应的mask
        b,c,n= x.shape
        querymask = querymask.view([b,1,-1])
        querymask = (querymask>0).float()
        keymask = keymask.view([b,1,-1])
        keymask = (keymask>0).float()
        onesmask = torch.ones([b,querymask.shape[-1],keymask.shape[-1]]).to(x.device)
        validmask = onesmask * (querymask.permute([0,2,1])) * (keymask)
        validx = x+(1-keymask)*(-1e6)+(1-querymask.permute([0,2,1]))*(-1e6)
        attmap  = self.softmax(validx)
        attmap = attmap * validmask
        return attmap

class NODVTATT(nn.Module):
    def __init__(self,in_channels):
        super(NODVTATT, self).__init__()
        self.K_conv = nn.Conv2d(in_channels,in_channels//1,1,1)
        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)

        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(8,8)
    def forward(self, x, mask):
        b,c,h,w = x.shape
        # print(dvtfeats.shape,x.shape)
        # mask = torch.nn.functional.interpolate(mask[:,0:1,:,:],[h,w],mode='nearest')
        mask = self.maxpool(mask)
        # print(mask.max())
        mask = (mask>0).float()
        backmask = 1 - mask
        # backmask[]
        # K_fea = self.K_conv(torch.cat([x,dvtfeats],1))*backmask
        # Q_fea = self.K_conv(torch.cat([x,dvtfeats],1))*(mask)
        K_fea = self.K_conv(x)*backmask
        Q_fea = self.K_conv(x)*(mask)

        V_fea = x * backmask # mask

        K_fea = (K_fea).view(b,c//1,w*h) # b Cb c
        K_fea = K_fea / (K_fea.norm(dim=1, keepdim=True)+1e-8)
        Q_fea = (Q_fea).view(b,c//1,w*h).permute(0,2,1)  # b c Cb
        Q_fea = Q_fea / (Q_fea.norm(dim=2, keepdim=True)+1e-8)
        V_fea = (V_fea).view(b,c,w*h) # b Cb c
        scoremap = torch.bmm(Q_fea,K_fea) * 20

        Attmap = self.softmaxmask(scoremap,mask,backmask) # b Cf Cb

        att_fore = torch.bmm(V_fea.view(b,c,w*h) , Attmap.permute(0,2,1))
        att_fore = att_fore.view([b,c,h,w])
        fused_feats = self.fuse(torch.cat([att_fore,x],1))

        M_fea = mask.view(b,1,w*h)
        # attmask = attmask / attmask.max()
        visatt = torch.sum(Attmap,1,keepdim=True) 
        visatt = visatt.view([b,1,h,w])
        attmask = F.interpolate(visatt, size=[h*8,w*8], mode='nearest')
        attmask = attmask / attmask.max()
        # print(attmask.shape,attmask.max())
        return fused_feats*mask+x*(1-mask),attmask

    def softmaxmask(self,x,querymask,keymask):  #querymask 前景mask keymask  reference对应的mask
        b,c,n= x.shape
        querymask = querymask.view([b,1,-1])
        querymask = (querymask>0).float()
        keymask = keymask.view([b,1,-1])
        keymask = (keymask>0).float()
        onesmask = torch.ones([b,querymask.shape[-1],keymask.shape[-1]]).to(x.device)
        # zerosmask = torch.ones([b,n,n]).to(x.device)
        validmask = onesmask * (querymask.permute([0,2,1])) * (keymask)
        validx = x+(1-keymask)*(-1e6)+(1-querymask.permute([0,2,1]))*(-1e6)
        attmap  = self.softmax(validx)
        attmap = attmap * validmask
        return attmap


class DVTGuidedATT(nn.Module):
    def __init__(self,in_channels):
        super(DVTGuidedATT, self).__init__()
        self.K_conv = nn.Conv2d(in_channels,in_channels//1,1,1)
        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)

        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(8,8)
    def forward(self, x, foremask,refmask):
        b,c,h,w = x.shape
        # print(refmask.shape,foremask.shape,x.shape)
        # mask = torch.nn.functional.interpolate(mask[:,0:1,:,:],[h,w],mode='nearest')
        mask = self.maxpool(foremask)
        # print(mask.max())
        mask = (mask>0).float()
        backmask =self.maxpool(refmask)
        backmask = (backmask>0).float()
        K_fea = self.K_conv(x)*backmask
        Q_fea = self.K_conv(x)*(mask)

        V_fea = x * backmask # mask

        K_fea = (K_fea).view(b,c//1,w*h) # b Cb c
        K_fea = K_fea / (K_fea.norm(dim=1, keepdim=True)+1e-8)
        Q_fea = (Q_fea).view(b,c//1,w*h).permute(0,2,1)  # b c Cb
        Q_fea = Q_fea / (Q_fea.norm(dim=2, keepdim=True)+1e-8)
        V_fea = (V_fea).view(b,c,w*h) # b Cb c
        scoremap = torch.bmm(Q_fea,K_fea) * 20
        # print('attention',scoremap.max(),scoremap.min())
        Attmap = self.softmaxmask(scoremap,mask,backmask) # b Cf Cb

        att_fore = torch.bmm(V_fea.view(b,c,w*h) , Attmap.permute(0,2,1))
        att_fore = att_fore.view([b,c,h,w])
        fused_feats = self.fuse(torch.cat([att_fore,x],1))

        M_fea = mask.view(b,1,w*h)
        # attmask = attmask / attmask.max()
        visatt = torch.sum(Attmap,1,keepdim=True) 
        visatt = visatt.view([b,1,h,w])
        attmask = F.interpolate(visatt, size=[h*8,w*8], mode='nearest')
        attmask = attmask / attmask.max()
        # print(attmask.shape,attmask.max())
        return fused_feats*mask+x*refmask,attmask

    def softmaxmask(self,x,querymask,keymask):  #querymask 前景mask keymask  reference对应的mask
        b,c,n= x.shape
        querymask = querymask.view([b,1,-1])
        querymask = (querymask>0).float()
        keymask = keymask.view([b,1,-1])
        keymask = (keymask>0).float()
        onesmask = torch.ones([b,querymask.shape[-1],keymask.shape[-1]]).to(x.device)
        # zerosmask = torch.ones([b,n,n]).to(x.device)
        validmask = onesmask * (querymask.permute([0,2,1])) * (keymask)
        validx = x+(1-keymask)*(-1e6)+(1-querymask.permute([0,2,1]))*(-1e6)
        attmap  = self.softmax(validx)
        attmap = attmap * validmask
        return attmap



class AdaptiveFusion(nn.Module):
    def __init__(self,in_channels):
        super(AdaptiveFusion, self).__init__()
        self.K_conv = ResidualBlock(in_channels,in_channels//1)
        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)
        self.project = nn.Conv2d(in_channels*2,in_channels,1,1)
        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(8,8)
        self.fused = Fusion_score(in_channels)
    def forward(self, x, mask):
        b,c,h,w = x.shape

        # mask = torch.nn.functional.interpolate(mask[:,0:1,:,:],[h,w],mode='nearest')
        mask = self.maxpool(mask)
        # print(mask.max())
        mask = (mask>0).float()
        backmask = 1 - mask
        # backmask[]
        K_fea = self.K_conv(x)*backmask
        Q_fea = self.K_conv(x)*(mask)
        V_fea = x * backmask # mask

        K_fea = (K_fea).view(b,c//1,w*h) # b Cb c
        K_fea = K_fea / (K_fea.norm(dim=1, keepdim=True)+1e-8)
        Q_fea = (Q_fea).view(b,c//1,w*h).permute(0,2,1)  # b c Cb
        Q_fea = Q_fea / (Q_fea.norm(dim=2, keepdim=True)+1e-8)
        V_fea = (V_fea).view(b,c,w*h) # b Cb c
        scoremap = torch.bmm(Q_fea,K_fea) * 20

        Attmap = self.softmaxmask(scoremap,mask,backmask) # b Cf Cb

        att_fore = torch.bmm(V_fea.view(b,c,w*h) , Attmap.permute(0,2,1))
        att_fore = att_fore.view([b,c,h,w])

        x1 = self.project(torch.cat([x*mask,att_fore*mask],1))
        fused_feats = self.fused(x,att_fore,mask)

        M_fea = mask.view(b,1,w*h)
        # attmask = attmask / attmask.max()
        visatt = torch.sum(Attmap,1,keepdim=True) 
        visatt = visatt.view([b,1,h,w])
        attmask = F.interpolate(visatt, size=[h*8,w*8], mode='nearest')
        attmask = attmask / attmask.max()
        # print(attmask.shape,attmask.max())
        return fused_feats*mask+x*(1-mask),attmask

    def softmaxmask(self,x,querymask,keymask):  #querymask 前景mask keymask  reference对应的mask
        b,c,n= x.shape
        querymask = querymask.view([b,1,-1])
        querymask = (querymask>0).float()
        keymask = keymask.view([b,1,-1])
        keymask = (keymask>0).float()
        onesmask = torch.ones([b,querymask.shape[-1],keymask.shape[-1]]).to(x.device)
        # zerosmask = torch.ones([b,n,n]).to(x.device)
        validmask = onesmask * (querymask.permute([0,2,1])) * (keymask)
        validx = x+(1-keymask)*(-1e6)+(1-querymask.permute([0,2,1]))*(-1e6)
        attmap  = self.softmax(validx)
        attmap = attmap * validmask
        return attmap

class Fusion_score(nn.Module):
    def __init__(self,in_channels):
        super(Fusion_score, self).__init__()
        # self.K_conv = ResidualBlock(in_channels,in_channels//1)
        self.conv1 = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)
        self.conv2 = nn.Conv2d(in_channels*1,in_channels,1,1)
        self.sigmoid  = nn.Sigmoid()
        self.fuse = nn.Conv2d(in_channels*2,in_channels//1,1,1, bias=False)

        # self.maxpool = nn.MaxPool2d(8,8)
    def forward(self, x,y, mask):
        b,c,h,w = x.shape


        # backmask[]
        feat1 = self.conv1(torch.cat([x*mask,y*mask],1))
        feat2 = self.conv2(feat1)
        score = self.sigmoid(feat2)

        fused = self.fuse(torch.cat([x*mask, y * mask * (1-score)],1))
        # FEAT
        return fused*mask+x*(1-mask)
