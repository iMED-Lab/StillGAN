# -*- coding: utf-8 -*-

import functools
import numpy as np
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


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
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
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
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'resunet':
        net = ResUNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class LuminanceLoss(nn.Module):
    """Define Illumination Regularization.

    Illumination Regularization reflects the degree of inputs' uneven illumination to some extent.
    """

    def __init__(self, patch_height, patch_width, crop_size=384):
        """Initialize the LuminanceLoss class.

        Parameters:
            patch_height (int) - - height of patch
            patch_width (int) - - width of patch
        """
        super(LuminanceLoss, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.crop_size = crop_size
        self.avgpool = nn.AvgPool2d((patch_height, patch_width), stride=(patch_height, patch_width))

    def forward(self, inputs):
        height = inputs.size()[2]
        width = inputs.size()[3]
        assert height >= self.crop_size and width >= self.crop_size
        assert self.crop_size % self.patch_height == 0 and self.crop_size % self.patch_width == 0, "Patch size Error."

        crop_inputs = (inputs[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0
        # [batch_size, channels, self.crop_size, self.crop_size] --> [batch_size, channels, 1, 1]
        global_mean = torch.mean(crop_inputs, [2, 3], True)
        # [batch_size, channels, self.crop_size, self.crop_size] --> [batch_size, channels, N, M]
        D = self.avgpool(crop_inputs)
        E = D - global_mean.expand_as(D)  # [batch_size, channels, N, M]
        upsample = nn.Upsample(size=[self.crop_size, self.crop_size], mode='bicubic', align_corners=False)
        R = upsample(E)  # [batch_size, channels, self.crop_size, self.crop_size]

        return torch.abs(R).mean()


def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = nn.Parameter(data=_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=False)
    return window


def _ssim(img1, img2, window, window_size, channel, K1=0.01, K2=0.03, L=1.0, size_average=True):
    assert img1.size() == img2.size()
    noise = torch.Tensor(np.random.normal(0, 0.01, img1.size())).cuda(img1.get_device())
    new_img1 = clip_by_tensor(img1 + noise, torch.Tensor(np.zeros(img1.size())).cuda(img1.get_device()), torch.Tensor(np.ones(img1.size())).cuda(img1.get_device()))
    new_img2 = clip_by_tensor(img2 + noise, torch.Tensor(np.zeros(img2.size())).cuda(img2.get_device()), torch.Tensor(np.ones(img2.size())).cuda(img2.get_device()))
    mu1 = F.conv2d(new_img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(new_img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(new_img1 * new_img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(new_img2 * new_img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(new_img1 * new_img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2.0

    ssim_map = (sigma12 + C3) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class StructureLoss(nn.Module):
    """Define Structure Loss.

    Structure Loss reflects the structural difference between inputs and outputs to some extent.
    """

    def __init__(self, channel=1, window_size=11, crop_size=384, size_average=True):
        """Initialize the StructureLoss class.

        Parameters:
            channel (int) - - number of channels
            window_size (int) - - size of window
            size_average (bool) - - average of batch or not
        """
        super(StructureLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.crop_size = crop_size
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        (_, channel, height, width) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window.data = window
            self.channel = channel

        inputs1 = (img1[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0
        inputs2 = (img2[:, :, (height-self.crop_size)//2:(height+self.crop_size)//2, (width-self.crop_size)//2:(width+self.crop_size)//2] + 1.0) / 2.0

        return 1.0 - _ssim(inputs1, inputs2, window, self.window_size, channel, self.size_average)


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            norm_layer(ch_out),
        )
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, use_dropout=False):
        super(up_conv,self).__init__()
        if use_dropout:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.5)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.up(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = res_conv_block(ch_in=img_ch, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Conv2 = res_conv_block(ch_in=ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Conv3 = res_conv_block(ch_in=2 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Conv4 = res_conv_block(ch_in=4 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]
        self.Conv5 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]
        self.Conv6 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]
        self.Conv7 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]
        self.Conv8 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up8 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 128, W / 128]
        self.Up_conv8 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up7 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 64, W / 64]
        self.Up_conv7 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]

        self.Up6 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 32, W / 32]
        self.Up_conv6 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]

        self.Up5 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 16, W / 16]
        self.Up_conv5 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]

        self.Up4 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 8, W / 8]
        self.Up_conv4 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]

        self.Up3 = up_conv(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Up_conv3 = res_conv_block(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]

        self.Up2 = up_conv(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Up_conv2 = res_conv_block(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]

        self.Up1 = up_conv(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Up_conv1 = res_conv_block(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]

        self.Conv_1x1 = nn.Conv2d(ngf, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # [B, ngf, H, W]

        x2 = self.Maxpool(x1)  # [B, ngf, H / 2, W / 2]
        x2 = self.Conv2(x2)  # [B, 2 * ngf, H / 2, W / 2]

        x3 = self.Maxpool(x2)  # [B, 2 * ngf, H / 4, W / 4]
        x3 = self.Conv3(x3)  # [B, 4 * ngf, H / 4, W / 4]

        x4 = self.Maxpool(x3)  # [B, 4 * ngf, H / 8, W / 8]
        x4 = self.Conv4(x4)  # [B, 8 * ngf, H / 8, W / 8]

        x5 = self.Maxpool(x4)  # [B, 8 * ngf, H / 16, W / 16]
        x5 = self.Conv5(x5)  # [B, 8 * ngf, H / 16, W / 16]

        x6 = self.Maxpool(x5)  # [B, 8 * ngf, H / 32, W / 32]
        x6 = self.Conv6(x6)  # [B, 8 * ngf, H / 32, W / 32]

        x7 = self.Maxpool(x6)  # [B, 8 * ngf, H / 64, W / 64]
        x7 = self.Conv7(x7)  # [B, 8 * ngf, H / 64, W / 64]

        x8 = self.Maxpool(x7)  # [B, 8 * ngf, H / 128, W / 128]
        x8 = self.Conv8(x8)  # [B, 8 * ngf, H / 128, W / 128]

        x9 = self.Maxpool(x8)  # [B, 8 * ngf, H / 256, W / 256]

        # decoding + concat path
        d8 = self.Up8(x9)  # [B, 8 * ngf, H / 128, W / 128]
        d8 = torch.cat((x8, d8), dim=1)  # [B, 16 * ngf, H / 128, W / 128]
        d8 = self.Up_conv8(d8)  # [B, 8 * ngf, H / 128, W / 128]

        d7 = self.Up7(d8)  # [B, 8 * ngf, H / 64, W / 64]
        d7 = torch.cat((x7, d7), dim=1)  # [B, 16 * ngf, H / 64, W / 64]
        d7 = self.Up_conv7(d7)  # [B, 8 * ngf, H / 64, W / 64]

        d6 = self.Up6(d7)  # [B, 8 * ngf, H / 32, W / 32]
        d6 = torch.cat((x6, d6), dim=1)  # [B, 16 * ngf, H / 32, W / 32]
        d6 = self.Up_conv6(d6)  # [B, 8 * ngf, H / 32, W / 32]

        d5 = self.Up5(d6)  # [B, 8 * ngf, H / 16, W / 16]
        d5 = torch.cat((x5, d5), dim=1)  # [B, 16 * ngf, H / 16, W / 16]
        d5 = self.Up_conv5(d5)  # [B, 8 * ngf, H / 16, W / 16]

        d4 = self.Up4(d5)  # [B, 8 * ngf, H / 8, W / 8]
        d4 = torch.cat((x4, d4), dim=1)  # [B, 16 * ngf, H / 8, W / 8]
        d4 = self.Up_conv4(d4)  # [B, 8 * ngf, H / 8, W / 8]

        d3 = self.Up3(d4)  # [B, 4 * ngf, H / 4, W / 4]
        d3 = torch.cat((x3, d3), dim=1)  # [B, 8 * ngf, H / 4, W / 4]
        d3 = self.Up_conv3(d3)  # [B, 4 * ngf, H / 4, W / 4]

        d2 = self.Up2(d3)  # [B, 2 * ngf, H / 2, W / 2]
        d2 = torch.cat((x2, d2), dim=1)  # [B, 4 * ngf, H / 2, W / 2]
        d2 = self.Up_conv2(d2)  # [B, 2 * ngf, H / 2, W / 2]

        d1 = self.Up1(d2)  # [B, ngf, H, W]
        d1 = torch.cat((x1, d1), dim=1)  # [B, 2 * ngf, H, W]
        d1 = self.Up_conv1(d1)  # [B, ngf, H, W]

        out = nn.Tanh()(self.Conv_1x1(d1))

        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='bicubic'),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
