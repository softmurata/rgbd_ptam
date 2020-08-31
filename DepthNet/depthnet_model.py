from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


# depthnet parameters
depthnet_parameters = namedtuple('parameters',
                                 'encoder, '
                                 'height, width, '
                                 'batch_size, '
                                 'num_threads, '
                                 'num_epochs, '
                                 'do_stereo, '
                                 'wrap_mode, '
                                 'use_deconv, '
                                 'alpha_image_loss, '
                                 'disp_gradient_loss_weight, '
                                 'lr_loss_weight, '
                                 'full_summary')


class DepthNet(object):
    
    
    def __init__(self, params, mode, left, right, reuse_variables=tf.AUTO_REUSE, model_index=0):
        
        self.params = params
        self.mode = mode  # 'train' or 'test'
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables
        
        self.build_net()
        self.build_outputs()
        
        if self.mode == 'test':
            
            return
        
        # for training part
        
        self.build_losses()
        self.build_summaries()
        
        
    # generate image from image and disparity
    def generate_image_left(self, image, disp):
        
        return bilinear_sampler_1d_h(image, -disp)
    
    def generate_image_right(self, image, disp):
        
        return bilinear_sampler_1d_h(image, disp)
    
    # disprity smoothness
    def get_disparity_smoothness(self, disp, pyramid):
        # disparity gradient
        disp_gradient_x = [d[:, :, :-1, :] - d[:, :, 1:, :] for d in disp]
        disp_gradient_y = [d[:, :-1, :, :] - d[:, :, 1:, :] for d in disp]
        
        # image gradient
        img_gradient_x = [img[:, :, :-1, :] - img[:, :, 1:, :] for img in pyramid]
        img_gradient_y = [img[:, :-1, :, :] - img[:, :, 1:, :] for img in pyramid]
        
        # weights
        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in img_gradient_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in img_gradient_y]
        
        smooth_x = [disp_gradient_x[i] * weights_x[i] for i in range(4)]
        smooth_y = [disp_gradient_y[i] * weights_y[i] for i in range(4)]
        
        return smooth_x + smooth_y
        
    
    def scaled_pyramid_images(self, image, num_scales):
        scaled_images = [image]
        shape = tf.shape(image)  # image shape
        h = shape[1]
        w = shape[2]
                
        for i in range(num_scales - 1):
            nh = h // 2 ** (i + 1)
            nw = w // 2 ** (i + 1)
            scaled_image = tf.image.resize_area(image, [nh, nw])
            scaled_images.append(scaled_image)
            
        return scaled_images
    
    def get_disp(self, x):
        
        disp = 0.3 * self.conv(x=x, num_out_layers=2, kernel_size=3, stride=1, activation_fn=tf.nn.sigmoid)
        
        return disp
    
    # Method function for building neural network
    # convolution layer function
    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)
    
    # max pooling
    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        
        return slim.max_pool2d(p_x, kernel_size)
    
    # for vgg 16
    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x=x, num_out_layers=num_out_layers, kernel_size=kernel_size, stride=1)
        conv2 = self.conv(x=conv1, num_out_layers=num_out_layers, kernel_size=kernel_size, stride=2)
        
        return conv2
    
    # unit resnet convolution for encoder
    def resconv(self, x, num_layers, stride):
        """
        resnet block example
        ([1 * 1, 64
         3 * 3, 64
         1 * 1, 256]) * 3(num_blocks)
        """
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x=x, num_out_layers=num_layers, kernel_size=1, stride=1)
        conv2 = self.conv(x=conv1, num_out_layers=num_layers, kernel_size=3, stride=stride)
        conv3 = self.conv(x=conv2, num_out_layers=4 * num_layers, kernel_size=1, stride=1, activation_fn=None)
        
        if do_proj:
            shortcut = self.conv(x=x, num_out_layers=4 * num_layers, kernel_size=1, stride=stride, activation_fn=None)
        else:
            shortcut = x
            
        out = conv3 + shortcut  # resconv(x) = conv(x) + x
        
        return tf.nn.elu(out)
            
    # resnet block for encoder  
    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        
        return out
    
    # for decoder
    def upsample_nn(self, x, scale):
        shape = tf.shape(x)
        h = shape[1]
        w = shape[2]
        
        return tf.image.resize_nearest_neighbor(x, [h * scale, w * scale])
    
    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv
    
    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, 1)
        return conv[:, 3:-1, 3:-1, :]  # ?
    
    # build vgg
    def build_vgg(self):
        
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv
        
        # encoder 
        with tf.variable_scope('encoder'):
            # conv block
            # conv2d() + conv2d(), out_layers is same
            conv1 = self.conv_block(x=self.model_input, num_out_layers=32, kernel_size=7)  # H / 2
            conv2 = self.conv_block(x=conv1, num_out_layers=64, kernel_size=5)  # H / 4
            conv3 = self.conv_block(x=conv2, num_out_layers=128, kernel_size=3)  # H / 8
            conv4 = self.conv_block(x=conv3, num_out_layers=256, kernel_size=3)  # H / 16
            conv5 = self.conv_block(x=conv4, num_out_layers=512, kernel_size=3)  # H / 32
            conv6 = self.conv_block(x=conv5, num_out_layers=512, kernel_size=3)  # H / 64
            conv7 = self.conv_block(x=conv6, num_out_layers=512, kernel_size=3)  # H / 128
            
            
        
        # skip connection
        with tf.variable_scope('skips'):
            
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        # decoder(upsampling or deconvolution)
        with tf.variable_scope('decoder'):
            # block1
            upconv7 = upconv(x=conv7, num_out_layers=512, kernel_size=3, scale=2)  # H / 64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = self.conv(x=concat7, num_out_layers=512, kernel_size=3, stride=1)
            
            # block2
            upconv6 = upconv(x=iconv7, num_out_layers=512, kernel_size=3, scale=2)  # H / 32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = self.conv(x=concat6, num_out_layers=512, kernel_size=3, stride=1)
            
            # block3
            upconv5 = upconv(x=iconv6, num_out_layers=256, kernel_size=3, scale=2)  # H / 16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = self.conv(x=concat5, num_out_layers=256, kernel_size=3, stride=1)
            
            # block4
            upconv4 = upconv(x=iconv5, num_out_layers=128, kernel_size=3, scale=2)  # H / 8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = self.conv(x=concat4, num_out_layers=128, kernel_size=3, stride=1)
            # get disparity feature map
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)
            
            # block5
            upconv3 = upconv(x=iconv4, num_out_layers=64, kernel_size=3, scale=2)  # H / 4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = self.conv(x=concat3, num_out_layers=64, kernel_size=3, stride=1)
            # get disparity feature map
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)
            
            # block6
            upconv2 = upconv(x=iconv3, num_out_layers=32, kernel_size=3, scale=2)  # H / 2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = self.conv(x=concat2, num_out_layers=32, kernel_size=3, stride=1)
            # get disparity feature map
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)
            
            # block7
            upconv1 = upconv(x=iconv2, num_out_layers=16, kernel_size=3, scale=2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = self.conv(x=concat1, num_out_layers=16, kernel_size=3, stride=1)
            # get disparity map
            self.disp1 = self.get_disp(iconv1)
            
            
    
    # build renet50
    def build_resnet50(self):
        
        # determine deconvolution or upsampling
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv
        
        # Encoder
        with tf.variable_scope('encoder'):
            # 64 channel outputs and 7 * 7 kernel
            conv1 = self.conv(x=self.model_input, num_out_layers=64, kernel_size=7, stride=2)  # H/2 ====> 64D
            # 3 * 3 maxpool
            pool1 = self.maxpool(x=conv1, kernel_size=3)  # H / 4 ======> 64D
            # Block1
            conv2 = self.resblock(x=pool1, num_layers=64, num_blocks=3)  # H / 8 =====> 256D
            # Block2
            conv3 = self.resblock(x=conv2, num_layers=128, num_blocks=4)  # H / 16 ====> 512D
            # Block3
            conv4 = self.resblock(x=conv3, num_layers=256, num_blocks=6)  # H / 32 ========> 1024D
            # Block4
            conv5 = self.resblock(x=conv4, num_layers=512, num_blocks=3)  # H / 64 =========> 2048D
            
        # skip conncetion
        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # Decoder
        with tf.variable_scope('decoder'):
            # H / 32
            upconv6 = upconv(x=conv5, num_out_layers=512, kernel_size=3, scale=2)
            concat6 = tf.concat([upconv6, skip5], 3)  # concatenate channels axis
            iconv6 = self.conv(x=concat6, num_out_layers=512, kernel_size=3, stride=1)
            
            # H / 16
            upconv5 = upconv(x=iconv6, num_out_layers=256, kernel_size=3, scale=2)
            concat5 = tf.concat([upconv5, skip4], 3)  # concatenate channel axis
            iconv5 = self.conv(x=concat5, num_out_layers=256, kernel_size=3, stride=1)
            
            # H / 8
            upconv4 = upconv(x=iconv5, num_out_layers=128, kernel_size=3, scale=2)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = self.conv(x=concat4, num_out_layers=128, kernel_size=3, stride=1)
            # why get disp feature map(?)
            self.disp4 = self.get_disp(iconv4)
            # upsampling
            udisp4 = self.upsample_nn(self.disp4, 2)
            
            
            # H / 4
            upconv3 = upconv(x=iconv4, num_out_layers=64, kernel_size=3, scale=2)
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = self.conv(x=concat3, num_out_layers=64, kernel_size=3, stride=1)
            # get disparity feature map
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)
            
            # H / 2
            upconv2 = upconv(x=iconv3, num_out_layers=32, kernel_size=3, scale=2)
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = self.conv(x=concat2, num_out_layers=32, kernel_size=3, stride=1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)
            
            # H
            upconv1 = upconv(x=iconv2, num_out_layers=16, kernel_size=3, scale=2)
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = self.conv(x=concat1, num_out_layers=16, kernel_size=3, stride=1)
            self.disp1 = self.get_disp(iconv1)
            
            
            
    
    
    def build_net(self):
        
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                # shurink 2 ** 4
                self.left_pyramid = self.scaled_pyramid_images(image=self.left, num_scales=4)
                
                if self.mode == 'train':
                    self.right_pyramid = self.scaled_pyramid_images(image=self.right, num_scales=4)
                    
                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)  # concatenate by channels axis
                else:
                    self.model_input = self.left
                    
                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                    
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                    
                else:
                    
                    return None
    
    
    # disp_left_est, disp_right_est            
    def build_outputs(self):
        # disprity
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [tf.expand_dims(depth[:, :, :, 0], 3) for depth in self.disp_est]  # why increase dimension
            self.disp_right_est = [tf.expand_dims(depth[:, :, :, 1], 3) for depth in self.disp_est]
            
        if self.mode == 'test':
            
            return
        
        
        # generate estimated left and right image
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]
        
        # needed the following code(?)   
        # converse
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]
        
        # smoothness   
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)  
    
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        
        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)         
    
    
    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # Image reconstruction
            # L1 loss
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(r) for r in self.l1_right]
            
            # ssim
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(l) for l in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(r) for r in self.ssim_right]
            
            # weighted sum
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1.0 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [self.params.alpha_image_loss * self.ssim_right[i] + (1.0 - self.alpha_image_loss) * self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)
            
            # disparity smoothness
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)
            
            # LR consistency
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)
            
            # total loss
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss
            
    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)
   
            
    



# bilinear sampler
def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[1]
        _width        = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output





