""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
import datetime
import os
import time
import timeit
import copy
import numpy as np
import torch as th


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Conv2d
        from MSG_GAN.CustomLayers import GenGeneralConvBlock, \
            GenInitialBlock, _equalized_conv2d

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        #   这里的&可能是按位与的意思。如果是2的整数次方，二进制一定是10000这样的，减去1恰好为01111

        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs:
        if self.use_eql:
            def to_rgb(in_channels):
                return _equalized_conv2d(in_channels, 3, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                return Conv2d(in_channels, 3, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        #   注：这里的layers存的是卷积块，rgb_converters则用于将中间的表征转换为rgb图

        # create the remaining layers
        for i in range(self.depth - 1):
            #   从这里的实现来看，i=0,1,2时候，latent size维持不变
            #   后面的层级，latent以2为倍率递减
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))
        #   这个循环中，依次求得更高层次的特征，同时根据特征得到rgb输出
        #   也就是说，前向传播得到的结果是一系列（从低分辨率到高分辨率）的rgb3通道图像

        return outputs

    @staticmethod
    def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust the dynamic colour range of the given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return th.clamp(data, min=0, max=1)


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512,
                 use_eql=True, gpu_parallelize=False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """
        from torch.nn import ModuleList
        from MSG_GAN.CustomLayers import DisGeneralConvBlock, \
            DisFinalBlock, _equalized_conv2d
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                return _equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                return Conv2d(3, out_channels, (1, 1), bias=True)
        #   from_rgb用于将一个3通道的rgb图（来自于生成器，或真实图片的下采样）经过1x1的卷积转换为特定通道的特征图

        self.rgb_to_features = ModuleList()
        self.final_converter = from_rgb(self.feature_size // 2)
        #   3 -> 256

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList()
        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)
        #   512 -> 1 （flatten output raw discriminator scores，尺寸可能是4x4）

        '''
            判别器里需要做两个操作，第一是把rgb转换为隐层特征，和已经有的特征并起来
            第二是根据当前的隐层特征，给出一个判别结果
        '''

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                '''
                    在高分辨率分支，通道是依次增加的
                    如 i = 5 对应最高分辨率
                        from_rgb 3 -> 32
                        conv     64 -> 64
                '''
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                '''
                    i = 0,1,2 的时候(对应于判别器的末端，最低分辨率部分)，走的是这个分支，
                    layer = 512 -> 256
                    rgb = 3 -> 256
                '''
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)
        '''
            i   layer_cin   layer_cout  rgb_to_features_cin rgb_to_features_cout   shape_in  shape_out(经过layers)
            
            F       512           1             3                   256              4
            
            0       512         256             3                   256              8
            1       512         256             3                   256              16
            2       512         256             3                   256              32
            
            3       256         256             3                   128              64
            4       128         128             3                    64              128        64
            5        64          64             3                    32
                                                                    (64)             256        128 
            大概的逻辑是最高分辨率图像（256）通过最后一个converter和layer，变成了(b,64,128,128)的东西，随后和经过convert的input并联，经过layer
            (i=5)全分辨率(3,256,256)，经过converter = (64,256,256)，经过layer = (64,128,128)
            (i=4)/2分辨率(3,128,128)，经过converter = (64,128,128)，并联 = (128,128,128)，经过layer = (128,64,64)
            (i=3)/4分辨率(3,64,64)，经过converter = (128,64,64)，并联 = (256,64,64)，经过layer = (256,32,32)
            (i=2)/8分辨率(3,32,32)，经过converter = (256,32,32)，并联 = (512,32,32)，经过layer = (256,16,16)
            (i=1)/16分辨率(3,16,16)，经过converter = (256,16,16)，并联 = (512,16,16)，经过layer = (256,8,8)
            (i=0)/32分辨率(3,8,8)，经过converter = (256,8,8)，并联 = (512,8,8)，经过layer = (256,4,4)
            
            final/64分辨率(3,4,4)，经过converter = (256,4,4)，并联 = (512,4,4)，经过layer = (b,)
            
            inputs_idx  0   1   2   3   4   5   6
            shape       4   8  16  32  64 128 256 
        '''

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        if self.gpu_parallelize:
            for i in range(len(self.layers)):
                self.layers[i] = th.nn.DataParallel(self.layers[i])
                self.rgb_to_features[i] = th.nn.DataParallel(
                    self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"
        #
        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])    #   输入图像尺寸256，变换为64维特征
        y = self.layers[self.depth - 2](y)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = th.cat((input_part, y), dim=1)
        y = self.final_block(y)

        # return calculated y
        return y


class MSG_GAN:
    """ Unconditional TeacherGAN

        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            use_eql: whether to use the equalized learning rate
            use_ema: whether to use exponential moving averages.
            ema_decay: value of ema decay. Used only if use_ema is True
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512,
                 use_eql=True, use_ema=True, ema_decay=0.999,
                 device=th.device("cpu")):
        """ constructor for the class """
        from torch.nn import DataParallel

        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)

        # Parallelize them if required:
        if device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = Discriminator(depth, latent_size,
                                     use_eql=use_eql, gpu_parallelize=True).to(device)
        else:
            self.dis = Discriminator(depth, latent_size, use_eql=True).to(device)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.device = device

        if self.use_ema:
            from MSG_GAN.CustomLayers import update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = th.randn(num_samples, self.latent_size).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply the moving average here:
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item()

    def create_grid(self, samples, img_files):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate
        from numpy import sqrt, power

        # dynamically adjust the colour of the images
        samples = [Generator.adjust_dynamic_range(sample) for sample in samples]

        # resize the samples to have same resolution:
        for i in range(len(samples)):
            samples[i] = interpolate(samples[i],
                                     scale_factor=power(2,
                                                        self.depth - 1 - i))
        # save the images:
        for sample, img_file in zip(samples, img_files):
            save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])),
                       normalize=True, scale_each=True, padding=0)


    def extract(self,image_paths):
        from torch.nn.functional import avg_pool2d
        # extract current batch of data for training
        images = images.to(self.device)
        extracted_batch_size = images.shape[0]

        # create a list of downsampled images from the real images:
        images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                             for i in range(1, self.depth)]
        images = list(reversed(images))



    def train(self, data, gen_optim, dis_optim, loss_fn, normalize_latents=True,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=36,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param normalize_latents: whether to normalize the latent vectors during training
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        fixed_input = th.randn(num_samples, self.latent_size).to(self.device)
        if normalize_latents:
            fixed_input = (fixed_input
                           / fixed_input.norm(dim=-1, keepdim=True)
                           * (self.latent_size ** 0.5))

        # create a global time counter
        global_time = time.time()
        global_step = 0

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                images = batch.to(self.device)
                extracted_batch_size = images.shape[0]

                # create a list of downsampled images from the real images:
                images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, self.depth)]
                images = list(reversed(images))

                # sample some random latent points
                gan_input = th.randn(
                    extracted_batch_size, self.latent_size).to(self.device)

                # normalize them if asked
                if normalize_latents:
                    gan_input = (gan_input
                                 / gan_input.norm(dim=-1, keepdim=True)
                                 * (self.latent_size ** 0.5))

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)

                # optimize the generator:
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % (int(limit / feedback_factor) + 1) == 0 or i == 1:     # Avoid div by 0 error on small training sets
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(global_step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\n")

                    # create a grid of samples and save it
                    reses = [str(int(np.power(2, dep))) + "_x_"
                             + str(int(np.power(2, dep)))
                             for dep in range(2, self.depth + 2)]
                    gen_img_files = [os.path.join(sample_dir, res, "gen_" +
                                                  str(epoch) + "_" +
                                                  str(i) + ".png")
                                     for res in reses]

                    # Make sure all the required directories exist
                    # otherwise make them
                    os.makedirs(sample_dir, exist_ok=True)
                    for gen_img_file in gen_img_files:
                        os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

                    dis_optim.zero_grad()
                    gen_optim.zero_grad()
                    with th.no_grad():
                        self.create_grid(
                            self.gen(fixed_input) if not self.use_ema
                            else self.gen_shadow(fixed_input),
                            gen_img_files)

                # increment the global_step:
                global_step += 1

                if i > limit:
                    break

            # calculate the time required for the epoch
            stop_time = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop_time - start_time))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                th.save(self.gen.state_dict(), gen_save_file)
                th.save(self.dis.state_dict(), dis_save_file)
                th.save(gen_optim.state_dict(), gen_optim_save_file)
                th.save(dis_optim.state_dict(), dis_optim_save_file)

                if self.use_ema:
                    gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_"
                                                        + str(epoch) + ".pth")
                    th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()
