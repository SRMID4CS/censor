"""Mechanisms for image reconstruction from parameter gradients."""
import os

from inversefed.indices_fedcola import select_indices_fedcola
import torch
import torchvision
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from utils.data_holder import DataHolder
torch.nn.ContrastiveLoss = ContrastiveLossWithTemperature

from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from copy import deepcopy
from tqdm import tqdm
import random

import inversefed.porting as porting

import math
import time
from inversefed.utils import project_onto_l1_ball
import defense
from inversefed.consts import STYLE_LEN
import nevergrad as ng
import numpy as np

from utils.text_utils import de_embed_text, get_text_from_tokens, get_true_length, infer_label_length_convergence, infer_label_perfect_match

import logging

from transformers import AutoProcessor, CLIPModel

logger = logging.getLogger(__name__)

imsize_dict = {
    'ImageNet': 224, 'I128':128, 'I64': 64, 'I32':32,
    'CIFAR10':32, 'CIFAR100':32, 'CIFAR100_MM':224, 'FFHQ':512, 'FFHQ64':64,
    'CA256': 256, 'CA128': 128, 'CA64': 64, 'CA32': 32, 
    'PERM64': 64, 'PERM32': 32, 'IMAGENET_IO' : 64, 'OOD_IMAGENET' : 64,
    'OOD_FFHQ' : 64,
    'Coco' : 224, 'Coco_img' : 224,
}

save_interval=100
construct_group_mean_at = 500
construct_gm_every = 100
DEFAULT_CONFIG = dict(signed=False,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      bn_stat=1e-1,
                      image_norm=1e-1,
                      z_norm=0,
                      group_lazy=1e-1,
                      init='randn',
                      init_text='randn',
                      lr_decay=True,

                      dataset='CIFAR10',

                      generative_model='',
                      gen_dataset='',
                      giml=False, 
                      gias_lr=0.1,
                      gias_iterations=0,
                      gifd=False,
                      steps=[],
                      lr_io=[],
                      start_layer=0,
                      end_layer=8,
                      #projection
                      do_project_gen_out=False,
                      do_project_noises=False,
                      do_project_latent=False,
                      max_radius_gen_out=[],
                      max_radius_noises=[],
                      max_radius_latent=[],
                      # The pre-trained StyleGAN checkpoint
                      ckpt=[],
                      #For algorithm choose:
                      gias=False,
                      ggl=False,
                      yin=False,
                      geiping=False,
                      cma_budget=0,
                      KLD=0,
                      patch_prior=0,
                      CLIP_loss=0,
                      patch_size=16,
                      #LR pace for training
                      lr_same_pace=False,
                      project=False,
                      defense_method=[],
                      defense_setting=[],
                      num_sample=10,
                      model = "N/A",  # Model name, used for loading the model
                      save_intermediate_at_img=1000,  # the interval at which to save intermediate results, -1 for no intermediate saving
                      save_intermediate_at_txt=100,  # the interval at which to save intermediate results, -1 for no intermediate saving
                      stop_at_text_perf_match=False,
                      img_lr=0.1,
                      txt_lr=0.1,
                      img_recon_method='GAN_free',
                      txt_recon_method='GAN_free',
                      img_max_iterations=10,
                      txt_max_iterations=10,
                      img_indices='fedcola_img_block_img_emb',
                      txt_indices='fedcola_txt_block_txt_emb',
                      img_convergence_threshold=0.5,
                      txt_convergence_threshold=0.5,
                      )

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config

#+++++++++++++++++++++++++++++++++
#     Definition of class
#+++++++++++++++++++++++++++++++++
class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}      #sum输入的维度可以是tuple，代表依次对当前tensor的这些维度进行求和。
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])


class MappingProxy(nn.Module):
    def __init__(self,gaussian_ft):
        super(MappingProxy,self).__init__()
        self.mean = gaussian_ft["mean"]
        self.std = gaussian_ft["std"]
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.lrelu(self.std * x + self.mean)
        return x

class BNStatisticsHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        mean_var = [mean, var]

        self.mean_var = mean_var
        # must have no output

    def close(self):
        self.hook.remove()


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, device, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, G=None, bn_prior=((0.0, 1.0)) ):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.device = device
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.setup = dict(device=self.device, dtype=next(model.parameters()).dtype)
        self.num_samples = config['num_sample']  # For CMA-ES
        self.mean_std = mean_std
        self.num_images = num_images    

        #BN Statistics
        self.bn_layers = []
        if self.config['bn_stat'] > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn_layers.append(BNStatisticsHook(module))
        self.bn_prior = bn_prior
        
        #Group Regularizer
        self.do_group_mean = False
        self.group_mean = None

        if self.config['model'] == 'FedCola_IMG_TXT':
            self.loss_fn = torch.nn.functional.cosine_embedding_loss # torch.nn.ContrastiveLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.noises = [None for i in range(self.config['restarts'])]
        self.initial_noises = [None for i in range(self.config['restarts'])]
        self.gen_outs = [[None] for i in range(self.config['restarts'])]
        self.ys = [None for i in range(self.config['restarts'])]    #For biggan's cond_vector
        self.iDLG = True
        self.images = None
        self.text_embeds = None # Dummy text embedding reconstruction

        if self.config['CLIP_loss'] > 0:
            self.CLIP_model, self.CLIP_processor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device), AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.CLIP_model = self.CLIP_model.eval()
            # self.CLIP_processor = self.CLIP_processor.eval()
            for param in self.CLIP_model.parameters():
                param.requires_grad = False

        # initialization
        if G:
            logger.info("Loading G...")
            if self.config['generative_model'] == 'stylegan2':
                self.G, self.G_mapping, self.G_synthesis = G, G.G_mapping, G.G_synthesis  
                if self.num_gpus > 1:
                    self.G, self.G_mapping, self.G_synthesis = G, nn.DataParallel(self.G_mapping), nn.DataParallel(self.G_synthesis)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
                self.G_synthesis.random_noise()
            elif self.config['generative_model'] == 'stylegan2_io':
                self.G = G
                # if self.num_gpus > 1:
                #     self.G = nn.DataParallel(self.G)
              
                # self.G.to(self.device)
                self.G.requires_grad_(False)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']
                self.mpl = MappingProxy(torch.load('gaussian_fit.pt'))
                # if self.num_gpus > 1:
                #     self.mpl = nn.DataParallel(self.mpl)

            elif self.config['generative_model'] == 'BigGAN':
                self.G = G
                # if self.num_gpus > 1:
                #     self.G = nn.DataParallel(self.G)
                # self.G.to(self.device)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']
            elif self.config['generative_model'].startswith('stylegan2-ada'):
                if self.num_gpus > 1:
                    self.G, self.G_mapping, self.G_synthesis = G, nn.DataParallel(self.G_mapping), nn.DataParallel(self.G_synthesis)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
            else:
                self.G = G
                if self.num_gpus > 1:
                    self.G = nn.DataParallel(self.G)
                self.G.to(self.device)
                self.G.requires_grad_(True)
            self.G.eval() # Disable stochastic dropout and using batch stat.
        elif self.config['generative_model']:
            if self.config['generative_model'] == 'stylegan2':
                self.G, self.G_mapping, self.G_synthesis = porting.load_decoder_stylegan2(self.config, self.device, dataset=self.config['gen_dataset'])
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
                self.G_mapping.eval()
                self.G_synthesis.eval()
            elif self.config['generative_model'] in ['stylegan2_io']:

                self.G = porting.load_decoder_stylegan2_io(self.config)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']
                self.G.requires_grad_(False)
                self.mpl = MappingProxy(torch.load('gaussian_fit.pt')) 
                # if self.num_gpus > 1:
                #     self.mpl = nn.DataParallel(self.mpl)
            elif self.config['generative_model'] in ['DCGAN']:
                G = porting.load_decoder_dcgan(self.config, self.device)
                G = G.requires_grad_(True)
                self.G = G
            elif self.config['generative_model'] in ['DCGAN-untrained']:
                G = porting.load_decoder_dcgan_untrained(self.config, self.device, dataset=self.config['gen_dataset'])
                G = G.requires_grad_(True)
                self.G = G
            elif self.config['generative_model'] in ['BigGAN']:
                self.G = porting.load_decoder_biggan_io(config)
                self.G = self.G.requires_grad_(False)
                self.G.start_layer = self.config['start_layer']
                self.G.end_layer = self.config['end_layer']

            # logger.info(self.G)
            self.G.eval()
        else:
            self.G = None
        
        # if torch.cuda.device_count() > 1 and self.G:
        # # G = nn.DataParallel(G)
        #     self.G = nn.DataParallel(self.G)

        if self.config['gifd']:
            self.G_io = deepcopy(self.G)
            # if torch.cuda.device_count() > 1:
            #     self.G_io = nn.DataParallel(self.G_io)
            self.project = self.config["project"]
            self.steps = self.config["steps"]
            self.gifd_loss = self.config['cost_fn']
            
        if self.config['gias']:
            self.G_list2d = [None for _ in range(self.config['restarts'])]
            for trial in range(self.config['restarts']):
                self.G_list2d[trial] = [deepcopy(self.G) for _ in range(self.num_images)]

        self.max_iterations = self.config['max_iterations']
        self.cma_iterations = self.config['cma_budget']
        
        self.gias_iterations = self.max_iterations if self.config['gias_iterations'] == 0 else self.config['gias_iterations']

        self.generative_model_name = self.config['generative_model']
        self.initial_z = None

    def set_initial_z(self, z):
        self.initial_z = z

    def init_dummy_z(self, G, generative_model_name, num_images):
        if self.initial_z is not None:
            dummy_z = self.initial_z.clone().unsqueeze(0) \
                .expand(num_images, self.initial_z.shape[0], self.initial_z.shape[1]) \
                .to(self.device).requires_grad_(True)
        elif generative_model_name.startswith('stylegan2-ada'):
            dummy_z = torch.randn(num_images, 512).to(self.device)
            dummy_z = G.mapping(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
            dummy_z = dummy_z.detach().requires_grad_(True)
        elif generative_model_name == 'stylegan2':
            dummy_z = torch.randn(num_images, 512).to(self.device)
            if self.config['gen_dataset'].startswith('I'):
                num_latent_layers = 16
            else:
                num_latent_layers = 18
            dummy_z = self.G_mapping(dummy_z).unsqueeze(1).expand(num_images, num_latent_layers, 512).detach().clone().to(self.device).requires_grad_(True)
            #The author took traing noise into consideration.
            # dummy_noise = G.static_noise(trainable=True)
        #Here we don't map z into w.
        elif generative_model_name == "stylegan2_io":
            dummy_z = torch.randn(
                        (num_images, 18, 512),   # 图片张数 x 层数 x 维度    这里之所以有18个隐向量，是因为它允许18层的w出现偏离。
                        dtype=torch.float,
                        requires_grad=True, device='cuda')
            dummy_z = self.mpl(dummy_z).detach().clone().requires_grad_(True)
            # logger.info(dummy_z.requires_grad)
        elif generative_model_name == 'BigGAN':
            dummy_z = torch.randn(num_images, 128).to(self.device).requires_grad_(True)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_z = torch.randn(num_images, 100, 1, 1).to(self.device).requires_grad_(True)
        return dummy_z

    def gen_dummy_data(self, G, generative_model_name, dummy_z, gen_outs=[None], noise=None, ys=None, img_size=-1, start_layer = 0):
        if not torch.is_tensor(dummy_z):  # CMA-ES
            # dummy_z = [torch.Tensor(z).to(self.device) for z in dummy_z]
            dummy_z = torch.Tensor(dummy_z).to(self.device)
        running_device = dummy_z.device
        if generative_model_name.startswith('stylegan2-ada'):
            # @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
            dummy_data = G(dummy_z, noise_mode='random')
        elif generative_model_name.startswith('stylegan2'):
            if generative_model_name.endswith('io'):  #gen_outs=None when searching the latent space.
                if self.config['optim'] == 'CMA-ES':
                    with torch.no_grad():
                        dummy_data, _ = G([dummy_z.float()], input_is_latent=True, noise=noise, layer_in=gen_outs[-1])
                else:
                    dummy_data, _ = G([dummy_z.float()], input_is_latent=True, noise=noise, layer_in=gen_outs[-1])
            else:
                dummy_data = G(dummy_z)
            if self.config['gen_dataset'].startswith('I'):
                kernel_size = 512 // self.image_size
            else:
                kernel_size = 1024 // self.image_size
            dummy_data = torch.nn.functional.avg_pool2d(dummy_data, kernel_size)
        elif generative_model_name in ['BigGAN']:
            if self.config['optim'] == 'CMA-ES':
                with torch.no_grad(): 
                    dummy_z = dummy_z.tanh()
                    dummy_data, _ = G(dummy_z, ys.float(), 1) if G.start_layer == 0 else G(gen_outs[-1], ys.float(), 1)
            else:
                # logger.info("start layer:{} end_layer:{} ys:{} dummy_z's size:{}".format(start_layer, ys.shape, dummy_z.shape))
                dummy_data, _ = G(dummy_z, ys.float(), 1) if start_layer == 0 else G(gen_outs[-1], ys.float(), 1)
                # logger.info("dummy_data:{}".format(dummy_data.shape))
                # exit()
            dummy_data = torch.nn.functional.interpolate(dummy_data, size=(self.image_size, self.image_size), mode='area')

        elif generative_model_name in ['stylegan2-ada-z']:
            dummy_data = G(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_data = G(dummy_z)
        
        dm, ds = self.mean_std
        dm, ds = dm.to(running_device), ds.to(running_device)

        dummy_data = (dummy_data + 1) / 2
        dummy_data = (dummy_data - dm) / ds
        
        return dummy_data

    def count_trainable_params(self, G=None, z=None , x=None, noise=None):
        n_z, n_G, n_x, n_noise = 0,0,0,0
        if G:
            n_z = torch.numel(z) / self.num_images if z.requires_grad else 0    # single image parameter
            logger.info(f"z: {n_z}")
            if noise:
                for item in noise:
                    n_noise += torch.numel(item) / self.num_images if item.requires_grad else 0 
                logger.info(f"noise:{n_noise}")
            n_G += sum(layer.numel() for layer in G.parameters() if layer.requires_grad)
            logger.info(f"G: {n_G}")
        else:
            n_x = torch.numel(x) /self.num_images if x.requires_grad else 0
            logger.info(f"x: {n_x}")
        self.n_trainable = n_z + n_G + n_x + n_noise

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), txt_shape=(20,384), dryrun=False, tol=None):
        """Reconstruct image from gradient."""
        if torch.is_tensor(input_data[0]):  
            self.input_data = [input_data]
        else:   # mutiple gradients
            self.input_data = input_data

        if self.config['model'] != 'FedCola_IMG_TXT':
            if labels is None:
                if torch.is_tensor(input_data[0]):  
                    labels_tmp = self.infer_label(input_data, num_inputs=self.num_images)
                else:   # mutiple gradients
                    labels_tmp = [self.infer_label(grad, num_inputs=self.num_images // len(input_data)) for grad in input_data]  
                    labels_tmp = torch.stack(labels_tmp).squeeze()
                self.reconstruct_label = False
                logger.info("Infer labels:{}".format(labels_tmp))
                infer_labels = [-1 for i in range(len(labels_tmp))]
                # adjust the order of labels
                for idx, label in enumerate(labels_tmp):
                    if label in labels:
                        infer_labels[torch.nonzero(labels == label).squeeze()] = labels_tmp[idx].clone()
                for idx, label in enumerate(labels_tmp):
                    if label not in labels:
                        infer_labels[infer_labels.index(-1)] = labels_tmp[idx].clone()
                infer_labels = torch.stack(infer_labels)            
                logger.info("Infer labels in correct order:{}".format(infer_labels))
            else:
                infer_labels = labels
        else:
            infer_labels = None
            logger.info("Label inference skipped since contrastive loss is used.")
        
        
        self.image_size = img_shape[1]
        start_time = time.time()
        ans = []
        if self.generative_model_name:  # GAN applying
            self.init_var(infer_labels)
            old_TV = self.config['total_variation']
            dummy_z = [None for _ in range(self.config['restarts'])]
            for trial in range(self.config['restarts']):
                dummy_z[trial] = self.init_dummy_z(self.G, self.generative_model_name, self.num_images)
            
            if self.config['model'] == 'FedCola_IMG':
                self.images = self._init_images(img_shape)
            elif self.config['model'] == 'FedCola_TXT':
                self.text_embeds = self._init_text_embeds(txt_shape)
            elif self.config['model'] == 'FedCola_IMG_TXT':
                self.images = self._init_images(img_shape)
                self.text_embeds = self._init_text_embeds(txt_shape)
                infer_labels = self._init_text_embeds(txt_shape)
            else:
                self.images = self._init_images(img_shape)
            
            if dryrun:
                return None
            #GGL
            if self.config['ggl']:
                self.config['cost_fn'] = 'l2'
                self.config['optim'] = 'CMA-ES' 
                self.config['total_variation'] = -1
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1
                # self.image_project = False
                dummy_z_ggl = [z.detach().clone().to(self.device).requires_grad_(True) for z in dummy_z]
                _x = self.reconstruct_by_latentCode(dummy_z_ggl, infer_labels, img_shape, dryrun, self.cma_iterations)
                _, best_score, x_best, _, label_best = self.choose_optimal(_x, infer_labels, dummy_z=dummy_z_ggl, dryrun=dryrun)
                stats_ggl = {}
                stats_ggl['opt'] = best_score
                ans.append(['ggl'] + [x_best, stats_ggl, label_best])

                self.config['total_variation'] = old_TV

            #GIAS
            if self.config['gias']:
                self.config['cost_fn'] = 'sim_cmpr0'
                self.config['optim'] = 'adam'
                self.config['KLD'] = -1
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1
                #latent space search
                dummy_z_gias = [z.detach().clone().to(self.device).requires_grad_(True) for z in dummy_z]
                _x = self.reconstruct_by_latentCode(dummy_z_gias, infer_labels, img_shape, dryrun, self.max_iterations)
                optimal_z, _, _, optimal_val, label_best = self.choose_optimal(_x, infer_labels, dummy_z=dummy_z_gias, dryrun=dryrun)
                # logger.info("optimal z's shape:{} _x shape:{}".format(optimal_z.shape, _x[0].shape))
                #parameter space search
                if self.generative_model_name in ['stylegan2_io']:
                    ans.append(['gias'] + list(self.gias_param_search(optimal_z, _x, infer_labels, optimal_noise=optimal_val)) + [label_best])
                else:
                    ans.append(['gias'] + list(self.gias_param_search(optimal_z, _x, infer_labels, optimal_ys=optimal_val)) + [label_best])

            #GIFD
            if self.config['gifd']:
                self.config['cost_fn'] = self.gifd_loss 
                self.config['optim'] = 'adam'
                self.config['KLD'] = -1
                dummy_z_io = [z.detach().clone().to(self.device).requires_grad_(True) for z in dummy_z]
                ans += self.inter_optimizer(dummy_z_io, infer_labels, -1)

 
        else:  #GAN-free method
            if self.config['model'] == 'FedCola_IMG':
                self.images = self._init_images(img_shape)
            elif self.config['model'] == 'FedCola_TXT':
                self.text_embeds = self._init_text_embeds(txt_shape)
            elif self.config['model'] == 'FedCola_IMG_TXT':
                self.images = self._init_images(img_shape)
                self.text_embeds = self._init_text_embeds(txt_shape)
                infer_labels = self._init_text_embeds(txt_shape)
            else:
                self.images = self._init_images(img_shape)
            if self.config['yin']:
                self.config['cost_fn'] = 'l2'
                self.config['optim'] = 'adam'
                # self.max_iterations = 1
                _x = self.reconstruct_by_latentCode(None, infer_labels, img_shape, dryrun, self.max_iterations, txt_shape=txt_shape)
                _, best_score, x_best, _, label_best = self.choose_optimal(_x, infer_labels, dryrun=dryrun)
                stats_yin = {}
                stats_yin['opt'] = best_score
                ans.append(['Yin'] + [x_best, stats_yin, label_best])
            
            if self.config['geiping']:
                self.config['cost_fn'] = 'sim_cmpr0'
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1
                _x, optimized_labels = self.reconstruct_by_latentCode(None, infer_labels, img_shape, dryrun, self.max_iterations, txt_shape=txt_shape)
                if self.config['model'] == 'FedCola_IMG_TXT':
                    infer_labels = optimized_labels
                _, best_score, x_best, _, label_best = self.choose_optimal(_x, infer_labels, dryrun=dryrun)
                stats_gp = {}
                stats_gp['opt'] = best_score
                ans.append(['geiping'] + [x_best, stats_gp, label_best])

        logger.info(f'Total time: {time.time()-start_time}.')
        return ans

    def init_var(self, labels):
        if self.generative_model_name in ['stylegan2_io']:
            if not self.initial_noises[0]:  #Need noises
                noises_single = self.G.make_noise(self.num_images)
                logger.info(f"Length of noises:{len(noises_single)}")
                noises = []
                for noise in noises_single:    
                    noises.append(noise.normal_().to(self.device))   
                self.initial_noises = [list(noises) for i in range(self.config['restarts'])]   #For stylegan2_io 
            
            self.noises = deepcopy(self.initial_noises)
        elif self.generative_model_name in ['BigGAN']:
            if labels is None:
                self.ys = [torch.nn.functional.one_hot(torch.randint(0, 1000, (1,)), num_classes=1000).to(self.device) for i in range(self.config['restarts'])]
            else:
                self.ys = [torch.nn.functional.one_hot(labels, num_classes=1000).to(self.device) for i in range(self.config['restarts'])]

        self.gen_outs = [[None] for i in range(self.config['restarts'])]

    def invert_stylegan2(self, dummy_z, labels, start_layer, noise_list, steps, index):
        learning_rate = self.config['lr_io'][index]
        logger.info(f"Running round {index + 1} / {len(self.config['steps'])} of GIFD.")


        _x = [None for _ in range(self.config['restarts'])]        
        for trial in range(self.config['restarts']):
        # noise_list contains the indices of nodes that we will be optimizing over
            for i in range(len(self.noises[trial])):   
                if i in noise_list:
                    self.noises[trial][i].requires_grad = True
                else:
                    self.noises[trial][i].requires_grad = False

            with torch.no_grad():
                if start_layer == 0:
                    var_list = [dummy_z[trial]] + self.noises[trial]
                    self.count_trainable_params(G=self.G_io, z=dummy_z[0])
                else:
                    self.gen_outs[trial][-1].requires_grad = True     
                    self.count_trainable_params(G=self.G_io, z=self.gen_outs[trial][-1], noise=self.noises[trial])

                    var_list = [dummy_z[trial]] + self.noises[trial] + [self.gen_outs[trial][-1]]
                    prev_gen_out = torch.ones(self.gen_outs[trial][-1].shape, device=self.gen_outs[trial][-1].device) * self.gen_outs[trial][-1]
                prev_latent = torch.ones(dummy_z[trial].shape, device=dummy_z[trial].device) * dummy_z[trial]
                prev_noises = [torch.ones(noise.shape, device=noise.device) * noise for noise in
                                self.noises[trial]]

                # set network that we will be optimizing over
                self.G_io.start_layer = start_layer          #start_layer is: 0 1 2 3...
                self.G_io.end_layer = self.config['end_layer']
            
            logger.info(f"Total number of trainable parameters: {self.n_trainable}")

            if self.config['model'] == 'FedCola_IMG_TXT' and self.config['init_text'] != 'ground_truth':
                labels[trial].requires_grad = True
                labels_opt = labels[trial]
                labels_opt.requires_grad = True
                to_optimize = var_list.copy().append(labels_opt)
            else:
                labels_opt = labels
                to_optimize = var_list.copy()

            for param in to_optimize:
                param.requires_grad = True

            optimizer = torch.optim.Adam(to_optimize, lr=learning_rate)

            ps = SphericalOptimizer([dummy_z[trial]] + self.noises[trial])  #pgd
            pbar = tqdm(range(steps))

            for i in pbar:
                if self.config['lr_same_pace']:
                    total_steps = sum(self.steps)
                    t = i / total_steps
                else:
                    t = i / steps
                lr = self.get_lr(t, learning_rate)
                optimizer.param_groups[0]["lr"] = lr
                _x[trial] = self.gen_dummy_data(self.G_io, self.config['generative_model'], dummy_z[trial], gen_outs=self.gen_outs[trial], noise=self.noises[trial]) 


                #-                      Calculate loss                           -#
                losses = [0, 0, 0, 0, 0, 0, 0] # tv, bn, img_norm, group_lazy, KLD, patch, CLIP
                optimizer.zero_grad()

                closure = self._gradient_closure(optimizer, _x[trial], self.input_data, labels_opt, losses, indices=self.config['indices'])
                rec_loss = closure()

                optimizer.step()

                if self.project:
                    ps.step()      

                #project back
                if start_layer != 0 and self.config['do_project_gen_out']:
                    if self.config['max_radius_gen_out'][index] > 0:
                        deviation = project_onto_l1_ball(self.gen_outs[trial][-1] - prev_gen_out,
                                                            self.config['max_radius_gen_out'][index])
                        var_list[-1].data = (prev_gen_out + deviation).data
                if self.config['do_project_latent']:
                    if self.config['max_radius_latent'][index] > 0:
                        deviation = project_onto_l1_ball(dummy_z[trial] - prev_latent,
                                                            self.config['max_radius_latent'][index])
                        var_list[0].data = (prev_latent + deviation).data
                if self.config['do_project_noises']:
                    if self.config['max_radius_noises'][index] > 0:
                        deviations = [project_onto_l1_ball(noise - prev_noise,
                                                            self.config['max_radius_noises'][index]) for noise,
                                        prev_noise in zip(self.noises[trial], prev_noises)]
                        for i, deviation in enumerate(deviations):
                            var_list[i+1].data = (prev_noises[i] + deviation).data

                pbar.set_description(
                    (
                        f" Rec. loss: {rec_loss.item():7.4f} | tv: {losses[0]:7.4f} | KLD: {losses[4]:7.4f} | ImageNorm: {losses[2]:7.4f} | CLIP: {losses[6]:7.4f}"
                    )
                )

            # TODO: check what happens when we are in the last layer
            with torch.no_grad():
                # latent_w = self.mpl(dummy_z[trial])
                self.G_io.end_layer = self.G_io.start_layer
                intermediate_out, _  = self.G_io([dummy_z[trial]],
                                                    input_is_latent=True,
                                                    noise=self.noises[trial],
                                                    layer_in=self.gen_outs[trial][-1],
                                                    skip=None)
                self.gen_outs[trial].append(intermediate_out)   

                self.G_io.end_layer = self.config['end_layer']
            
            #project back to image
            # if self.image_project:
            dm, ds = self.mean_std  
            with torch.no_grad():
                # Project into image space
                _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

        return _x, labels
        
    def invert_biggan(self, dummy_z, labels, start_layer, steps, index, img_size=-1):
        
        logger.info("The start_layer:{}".format(start_layer))
        logger.info(f"Running round {index + 1} / {len(self.config['steps'])} of GIFD.")


        learning_rate = self.config['lr_io'][index]

        _x = [None for _ in range(self.config['restarts'])] 
        # if torch.cuda.device_count() > 1:
        #     self.G_io = deepcopy(self.G)
        #     self.G_io.start_layer = start_layer
        #     self.G_io = nn.DataParallel(self.G_io)
        #     self.G_io.to(self.device)
        #         #start_layer is: 0 1 2 3...
        # else:
        self.G_io.start_layer = start_layer

        for trial in range(self.config['restarts']):
            # if torch.cuda.device_count() > 1:
            #     self.G_io = deepcopy(self.G)
            #     self.G_io.start_layer = start_layer
            #     self.G_io = nn.DataParallel(self.G_io)
            #     self.G_io.to(self.device)
            #     #start_layer is: 0 1 2 3...
            # else:
            self.G_io.start_layer = start_layer

            if start_layer == 0:
                optim_param = [dummy_z[trial]]
                ps = SphericalOptimizer([dummy_z[trial]])  #pgd
                self.count_trainable_params(G=self.G_io, z=dummy_z[0])
            else:
                self.gen_outs[trial][-1].requires_grad = True     
                self.count_trainable_params(G=self.G_io, z=self.gen_outs[trial][-1])
                optim_param =  [self.gen_outs[trial][-1]]
                prev_gen_out = torch.ones(self.gen_outs[trial][-1].shape, device=self.gen_outs[trial][-1].device) * self.gen_outs[trial][-1]
            
            if self.config['model'] == 'FedCola_IMG_TXT' and self.config['init_text'] != 'ground_truth':
                labels[trial].requires_grad = True
                labels_opt = labels[trial]
                labels_opt.requires_grad = True
                optim_param.append(labels_opt)
            else:
                labels_opt = labels

            for param in optim_param:
                param.requires_grad = True

            logger.info(f"Total number of trainable parameters: {self.n_trainable}")

            optimizer = torch.optim.Adam(optim_param, lr=learning_rate)

            # logger.info("_invert z:{}".format(z.shape))
            pbar = tqdm(range(steps))
            # self.match_min = np.inf
            
            # c = torch.nn.functional.one_hot(self.labels, num_classes = self.fl_setting['num_classes']).to(self.input_gradient[0].device)


            for current_step in pbar:
                # img_gen = self.generator(z, c.float(), 1)
                lr = self.get_lr(current_step / steps, learning_rate)
                # optimizer = torch.optim.Adam([optim_param[0][select_idx]], lr=learning_rate)
                optimizer.param_groups[0]['lr'] = lr

                _x[trial] = self.gen_dummy_data(self.G_io, self.config['generative_model'], dummy_z[trial], gen_outs=self.gen_outs[trial], ys=self.ys[trial], img_size=img_size, start_layer=start_layer) 
                losses = [0, 0, 0, 0, 0, 0, 0] # tv, bn, img_norm, group_lazy, KLD, patch, CLIP
                optimizer.zero_grad()
                self.dummy_z = dummy_z[trial]
                

                closure = self._gradient_closure(optimizer, _x[trial], self.input_data, labels_opt, losses, indices=self.config['indices'])
                rec_loss = closure()

                optimizer.step()

                if self.project and start_layer == 0:
                    ps.step()   

                if start_layer != 0 and self.config['do_project_gen_out']:
                    if self.config['max_radius_gen_out'][index] > 0:
                        deviation = project_onto_l1_ball(self.gen_outs[trial][-1] - prev_gen_out,
                                                        self.config['max_radius_gen_out'][index])
                        self.gen_outs[trial][-1].data = (prev_gen_out + deviation).data

                pbar.set_description(
                    (
                        f" Rec. loss: {rec_loss.item():7.4f} | tv: {losses[0]:7.4f} | KLD: {losses[4]:7.4f} | ImageNorm: {losses[2]:7.4f} | CLIP: {losses[6]:7.4f}"
                    )
                )

            with torch.no_grad():
                # if torch.cuda.device_count() > 1:
                #     self.G_io = deepcopy(self.G)
                    # self.G_io.start_layer = start_layer
                self.G_io.end_layer = start_layer + 1
                    # self.G_io = nn.DataParallel(self.G_io)
                    # self.G_io.to(self.device)
                intermediate_out, new_ys = self.G_io(self.gen_outs[trial][-1], self.ys[trial].float(), 1)   if start_layer > 0 else self.G_io(dummy_z[trial], self.ys[trial].float(), 1)
                self.gen_outs[trial].append(intermediate_out)   
                self.ys[trial] = new_ys
                self.G_io.end_layer = self.config['end_layer']
                # self.G_io = nn.DataParallel(self.G_io)
            # if self.image_project:
            dm, ds = self.mean_std  
            with torch.no_grad():
                # Project into image space
                _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)
                _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

        return _x, labels

    def inter_optimizer(self, dummy_z, labels, img_size, dryrun=False, prefix=''):
        self.model.eval()
        self.G_io.to(self.device)
        # if torch.is_tensor(input_data[0]):
        #     input_data = [input_data]

        res = []

        if self.generative_model_name == 'stylegan2_io' or self.config['start_layer'] > 0:
            self.config['KLD'] = 0

        logger.info("-------------Start intermidiate space search---------------")
        best_layer_img = None
        best_layer_label = None
        best_layer_score = {'opt':np.inf}
        res = [[prefix + f'layer{i}', None, {'opt':-1}, None] for i in range(len(self.config["steps"]))]

        for i, steps in enumerate(self.config["steps"]):
            begin_layer = i + self.config['start_layer']

            if begin_layer > self.config['end_layer']:
                raise Exception('Attemping to go after end layer')
            if self.generative_model_name == 'stylegan2_io':
                _x, labels = self.invert_stylegan2(dummy_z, labels, begin_layer, range(5 + 2 *begin_layer), int(steps), i)
            elif self.generative_model_name == 'BigGAN':
                _x, labels = self.invert_biggan(dummy_z, labels, begin_layer, int(steps), i, img_size)
                self.config['KLD'] = 0
            #_x is not in the real image space.
            #TO DO: compute score
            stats = {}
            optimal_z, stats['opt'], opt_img, _, opt_label = self.choose_optimal(_x, labels, dummy_z, dryrun=dryrun)
            if stats['opt'] < best_layer_score['opt']:  #save the best layer output
                # best_layer_name = 'Best_' + prefix + 'output' 
                # best_layer_num = i
                best_layer_img = opt_img.detach().clone()
                best_layer_score = dict(stats)
                best_layer_label = opt_label.detach().clone() if opt_label is not None else None

            res[i] = [prefix + f'layer{i}', opt_img.detach().clone(), stats, opt_label.detach().clone() if opt_label is not None else None]
            res.append(['Best_' + prefix + 'first_' + str(i) + '_layer' , best_layer_img, best_layer_score, best_layer_label])

        return res

    """
    @brief Return optimal latent code and image according to the gradient match loss
    """
    def choose_optimal(self, _x, labels, dummy_z=None, tol=None, dryrun=False, G=None):

        restarts = self.config['restarts']
        scores = torch.zeros(restarts)
        x = [None for i in range(restarts)]
        _labels = None
        if self.config['model'] == 'FedCola_IMG_TXT':
            _labels = [None for i in range(restarts)]
        # logger.info(f"choose_optimal label type:{type(labels)}")
        for trial in range(restarts):
            x[trial] = _x[trial].detach()
            if self.config['model'] == 'FedCola_IMG_TXT':
                _labels[trial] = labels[trial].detach()
                scores[trial] = self._score_trial(x[trial], self.input_data, _labels[trial])
            else:
                scores[trial] = self._score_trial(x[trial], self.input_data, labels)
            if tol is not None and scores[trial] <= tol:
                break
            if dryrun:
                break

        valid_mask = torch.isfinite(scores)
        if not torch.any(valid_mask):
            raise ValueError("All scores are invalid (NaN or Inf)")

        valid_scores = scores[valid_mask] # guard against NaN/-Inf scores?
        valid_indices = torch.arange(len(scores))[valid_mask]
        best_index_in_valid = torch.argmin(valid_scores)
        optimal_index = valid_indices[best_index_in_valid]

        logger.info(f'Score: {scores}')
        logger.info(f'Optimal result score: {scores[optimal_index]:2.4f}')


        if G:   #For GIAS
            logger.info(f'Choosing optimal G... : {optimal_index}')
            return  G[optimal_index], scores[optimal_index].item(), x[optimal_index].clone(), None, _labels[optimal_index] if _labels is not None else None
        
        
        if self.generative_model_name in ['stylegan2_io']:
            logger.info(f'Choosing optimal z and noise... : {optimal_index}')
            return dummy_z[optimal_index].detach().clone(), scores[optimal_index].item(), x[optimal_index].clone(), self.noises[optimal_index], _labels[optimal_index] if _labels is not None else None
        elif self.generative_model_name in ['BigGAN']:
            logger.info(f'Choosing optimal z and ys... : {optimal_index}')
            return dummy_z[optimal_index].detach().clone(),  scores[optimal_index].item(), x[optimal_index].clone(), self.ys[optimal_index], _labels[optimal_index] if _labels is not None else None
        elif self.generative_model_name:
            logger.info(f'Choosing optimal z... : {optimal_index}')
            return dummy_z[optimal_index].detach().clone(),  scores[optimal_index].item(), x[optimal_index].clone(), None, _labels[optimal_index] if _labels is not None else None
        else:
            logger.info(f'Choosing optimal x... : {optimal_index}')
            return None, scores[optimal_index].item(), x[optimal_index].clone(), None, _labels[optimal_index].clone() if _labels is not None else None

    def reconstruct_by_latentCode(self, dummy_z, labels, img_shape, dryrun, max_iterations=500, txt_shape=(20,384)):
        self.model.eval()

        data_holder = DataHolder()

        max_iterations = max_iterations
        if self.config['model'] == 'FedCola_IMG':
            x = self._init_images(img_shape)
        elif self.config['model'] == 'FedCola_TXT':
            x = self._init_text_embeds(txt_shape)
        elif self.config['model'] == 'FedCola_IMG_TXT':
            x = self._init_images(img_shape)
            # TODO Check
            labels = self._init_text_embeds(txt_shape)
        else:
            x = self._init_images(img_shape)
        # scores = torch.zeros(self.config['restarts'])
        
        try:
            # labels = [None for _ in range(self.config['restarts'])]
            optimizer = [None for _ in range(self.config['restarts'])]
            scheduler = [None for _ in range(self.config['restarts'])]
            _x = [None for _ in range(self.config['restarts'])]
            if self.config['model'] == 'FedCola_IMG_TXT':
                _labels = [None for _ in range(self.config['restarts'])]

            if (self.config['model'] == 'FedCola_IMG_TXT' or self.config['model'] == 'FedCola_TXT') and init_txt != 'ground_truth':
                # TODO : check of num images / batches later
                label_convergence_metrics = [{
                    'length_infered': False,
                    'perfect_match': False,
                    'length_iter': -1,
                    'perf_iter': -1,
                    'inferred_length': -1,
                    'true_length': get_true_length(data_holder.get('ground_truth_text')[nn].unsqueeze(0))
                } for nn in range(self.num_images)]

            for trial in range(self.config['restarts']):
                _x[trial] = x[trial]
                if self.config['model'] == 'FedCola_IMG_TXT':
                    _labels[trial] = labels[trial]

                if self.G:
                    self.G.to(self.device)
                    if self.config['model'] == 'FedCola_IMG_TXT' and init_txt != 'ground_truth':
                        _labels[trial].requires_grad = True
                        to_optimize = [dummy_z[trial], _labels[trial]]
                    else:
                        to_optimize = [dummy_z[trial]]

                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam(to_optimize, lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer[trial] = torch.optim.SGD(to_optimize, lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer[trial] = torch.optim.LBFGS(to_optimize)
                    elif self.config['optim'] == 'CMA-ES':
                        parametrization = ng.p.Array(init=dummy_z[trial].cpu().detach().numpy())
                        optimizer[trial] = ng.optimizers.registry['CMA'](parametrization=parametrization, budget=self.config['cma_budget'])
                    else:
                        raise ValueError()
                else:
                    #Make labels and x trainable conditionally
                    _x[trial].requires_grad = True
                    init_txt = self.config.get('init_text', self.config['init'])
                    if self.config['model'] == 'FedCola_IMG_TXT' and init_txt != 'ground_truth':
                        _labels[trial].requires_grad = True
                        to_optimize = [_x[trial], _labels[trial]]
                    else:
                        to_optimize = [_x[trial]]

                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam(to_optimize, lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer[trial] = torch.optim.SGD(to_optimize, lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer[trial] = torch.optim.LBFGS(to_optimize)
                    else:
                        raise ValueError()

                if self.config['lr_decay'] and not self.config['optim'] == 'CMA-ES':
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                                                        milestones=[self.max_iterations // 2.667, self.max_iterations // 1.6,

                                                                                    self.max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            dm, ds = self.mean_std
            
            if self.G:
                logger.info("Start latent space search")
                self.count_trainable_params(G=self.G, z=dummy_z[0])
            else:
                logger.info("Start original space search")
                self.count_trainable_params(x=_x[0])
            logger.info(f"Total number of trainable parameters: {self.n_trainable}")
            

            for iteration in range(max_iterations):
                for trial in range(self.config['restarts']):
                    if self.config['model'] == 'FedCola_IMG_TXT' and self.config['init_text'] != 'ground_truth':
                        _labels[trial].requires_grad = True
                        labels_opt = _labels[trial]
                        labels_opt.requires_grad = True
                    elif self.config['model'] == 'FedCola_IMG_TXT' and self.config['init_text'] == 'ground_truth':
                        labels_opt = _labels[trial]
                    else:
                        labels_opt = labels
                    losses = [0, 0, 0, 0, 0, 0, 0] # tv, bn, img_norm, group_lazy, KLD, patch, CLIP
                    #Group Regularizer
                    if trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                        self.do_group_mean = True
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                    if self.do_group_mean and trial == 0 and (iteration + 1) % construct_gm_every == 0:
                        logger.info("construct group mean")
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                    if self.G:
                        dummy_zs = dummy_z[trial]

                        if self.config['optim'] == 'CMA-ES':
                            ng_data = [optimizer[trial].ask() for _ in range(self.num_samples)]
                            _x_gen = [self.gen_dummy_data(self.G, self.generative_model_name, ng_data[i].value, noise=self.noises[trial], ys=self.ys[trial]) for i in range(self.num_samples)]

                        elif self.generative_model_name in ['stylegan2','stylegan2-ada','stylegan2-ada-untrained']:
                            _x[trial] = self.gen_dummy_data(self.G_synthesis, self.generative_model_name, dummy_zs)

                        elif self.generative_model_name in ['stylegan2_io']:
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_zs, noise=self.noises[trial])

                        elif self.generative_model_name in ['BigGAN']:  #For gias over BigGAN
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_zs, ys=self.ys[trial])
                        else:
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_zs)
                        self.dummy_z = dummy_z[trial]
                        
                    else:
                        self.dummy_z = None
                    
                    if self.config['optim'] == 'CMA-ES':
                        # ng_data = [optimizer[trial].ask() for _ in range(self.num_samples)]
                        loss = []
                        loss_detail = []
                        for i in range(self.num_samples):
                            self.dummy_z = torch.Tensor(ng_data[i].value).to(self.device)
                            closure = self._gradient_closure(optimizer[trial], _x_gen[i], self.input_data, labels_opt, losses, indices=self.config['indices'])
                            rec_loss = closure()
                            loss.append(rec_loss.item())
                            loss_detail.append(losses)   #record every ask's losses
                        losses = list(np.array(loss_detail).mean(axis=0))
                        rec_loss = sum(loss) / len(loss)
                        for z, l in zip(ng_data, loss):
                            optimizer[trial].tell(z, l)
                    elif self.G:
                        closure = self._gradient_closure(optimizer[trial], _x[trial], self.input_data, labels_opt, losses, indices=self.config['indices'])
                        rec_loss = optimizer[trial].step(closure)
                        rec_loss = rec_loss.item()
                    else:
                        imgs = _x[trial]

                        closure = self._gradient_closure(optimizer[trial], imgs, self.input_data, labels_opt, losses, indices=self.config['indices'])
                        rec_loss = optimizer[trial].step(closure)
                        rec_loss = rec_loss.item()

                    if self.config['lr_decay'] and not self.config['optim'] == 'CMA-ES':
                        scheduler[trial].step()

                    with torch.no_grad():
                        # Project into image space
                        if self.config['save_intermediate_at_img'] > 0 and (iteration % self.config['save_intermediate_at_img'] == 0):
                            logger.info(f'Saving intermediate IMG at iteration {iteration}...')
                            if self.config['model'] == 'FedCola_IMG_TXT' or self.config['model'] == 'FedCola_IMG':
                                for num_img in range(self.num_images):
                                    dir_path = os.path.join(data_holder.get('save_dir'), f'{num_img}/')
                                    os.makedirs(dir_path, exist_ok=True)
                                    torchvision.utils.save_image(torch.clamp(imgs.detach().clone() * ds + dm, 0, 1)[num_img:num_img + 1, ...], os.path.join(dir_path, f'{num_img}_trial_{trial}_it_{iteration}.png'))
                        if self.config['save_intermediate_at_txt'] > 0 and (iteration % self.config['save_intermediate_at_txt'] == 0):
                            logger.info(f'Saving intermediate TXT at iteration {iteration}...')
                            if self.config['model'] == 'FedCola_IMG_TXT' and self.config['init_text'] != 'ground_truth':
                                for num_txt in range(self.num_images):
                                    recon_sentence, tokens = de_embed_text(labels_opt[num_txt], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'))
                                    logger.info(f'Recon Sentence : {recon_sentence}')
                                    dir_path = os.path.join(data_holder.get('save_dir'), f'{num_txt}/')
                                    os.makedirs(dir_path, exist_ok=True)
                                    with open(os.path.join(dir_path, f'{num_txt}_trial_{trial}_it_{iteration}.txt'), 'w') as f:
                                        f.write(recon_sentence)
                        if self.config['save_intermediate_at_txt'] > 0 and (iteration % self.config['save_intermediate_at_txt'] == 0):
                            logger.info(f'Saving intermediate TXT at iteration {iteration}...')
                            if self.config['model'] == 'FedCola_TXT':
                                for num_txt in range(self.num_images):
                                    recon_sentence, tokens = de_embed_text(imgs[num_txt], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'))
                                    logger.info(f'Recon Sentence : {recon_sentence}')
                                    dir_path = os.path.join(data_holder.get('save_dir'), f'{num_txt}/')
                                    os.makedirs(dir_path, exist_ok=True)
                                    with open(os.path.join(dir_path, f'{num_txt}_trial_{trial}_it_{iteration}.txt'), 'w') as f:
                                        f.write(recon_sentence)

                        if (iteration + 1 == self.max_iterations) or iteration % save_interval == 0:
                            logger.info(f'It: {iteration}. Rec. loss: {rec_loss:2.4f} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | ImageNorm: {losses[2]:7.4f} | gr: {losses[3]:7.4f} | kld: {losses[4]:7.4f} | patch: {losses[5]:7.4f} | CLIP: {losses[6]:7.4f} ')
                            if self.config['z_norm'] > 0:
                                logger.info(torch.norm(dummy_z[trial], 2).item())
                        if iteration + 1 == max_iterations and self.config['optim'] == 'CMA-ES':
                            recommendation = optimizer[trial].provide_recommendation()
                            dummy_z[trial] = torch.from_numpy(recommendation.value).float().to(self.device) 
                            with torch.no_grad():
                                _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_z[trial], noise=self.noises[trial], ys=self.ys[trial])  


                        _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                        # check length convergence and perfect match convergence for text
                        if (self.config['model'] == 'FedCola_IMG_TXT' or self.config['model'] == 'FedCola_TXT') and self.config['init_text'] != 'ground_truth' and iteration % 50 == 0:
                            if self.config['model'] == 'FedCola_TXT':
                                cap_opt = _x[trial]
                            else:
                                cap_opt = _labels[trial]
                            for num_img in range(self.num_images):
                                if not label_convergence_metrics[num_img]['length_infered']:
                                    inferred_length, is_length_correct = infer_label_length_convergence(cap_opt[num_img], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'), true_length=label_convergence_metrics[num_img]['true_length'])
                                    if inferred_length > 0:
                                        label_convergence_metrics[num_img]['length_infered'] = is_length_correct
                                        label_convergence_metrics[num_img]['length_iter'] = iteration
                                        label_convergence_metrics[num_img]['inferred_length'] = inferred_length
                                        logger.info(f"Trial {trial}: Length inferred at iteration {iteration}, correct: {is_length_correct}")
                                if not label_convergence_metrics[num_img]['perfect_match']:
                                    is_perfect = infer_label_perfect_match(cap_opt[num_img], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'), ground_truth_text_token_ids=data_holder.get('ground_truth_text')[num_img])
                                    if is_perfect:
                                        label_convergence_metrics[num_img]['perfect_match'] = True
                                        label_convergence_metrics[num_img]['perf_iter'] = iteration
                                        logger.info(f"Trial {trial}: Perfect match achieved at iteration {iteration}")
                            data_holder.set('label_convergence_metrics', label_convergence_metrics)
                            # stop condition if any perfect matched trial achieved for all num images
                            if self.config['stop_at_text_perf_match']:
                                all_perf_matched = all([label_convergence_metrics[num_img]['perfect_match'] for num_img in range(self.num_images)])
                                if all_perf_matched:
                                    logger.info("Perfect text match achieved, stopping optimization.")
                                    dryrun = True

                    if dryrun:
                        break

                if dryrun:
                    break

        except KeyboardInterrupt:
            logger.info(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        if self.G:
            self.G.to("cpu")

        if self.config['model'] == 'FedCola_IMG_TXT':
            return _x, labels
        else: 
            return _x, None

    def gias_param_search(self, optimal_z, _x, labels, optimal_noise=None, optimal_ys=None, tol=None, dryrun=False):
        
        stats = defaultdict(list)
        optimizer = [None for _ in range(self.config['restarts'])]
        scheduler = [None for _ in range(self.config['restarts'])]
        dm, ds = self.mean_std
        try:
            
            self.dummy_z = optimal_z.detach().clone().cpu()

            self.dummy_zs = [None for k in range(self.num_images)]
            
            # When optimal_noise is empty list(Biggan as generative model), self.noise_zs won't be used. 
            self.noise_zs = [None for k in range(self.num_images)]  
            self.ys_zs = [None for k in range(self.num_images)]

            if optimal_noise is not None:
                for k in range(self.num_images):
                    self.noise_zs[k] = [torch.unsqueeze(noise[k], 0)  for noise in optimal_noise]
                    if self.num_gpus > 1:
                        for i in range(len(self.noise_zs[k])):
                            self.noise_zs[k][i] = self.noise_zs[k][i].to(f'cuda:{k%self.num_gpus}')  
                            self.noise_zs[k][i].requires_grad_(False)
                    else:
                        for i in range(len(self.noise_zs[k])):
                            self.noise_zs[k][i] = self.noise_zs[k][i].to(self.device)
                            self.noise_zs[k][i].requires_grad_(False)
            # logger.info("optimal_ys:{}".format(optimal_ys))
            if optimal_ys is not None:
                for k in range(self.num_images):
                    self.ys_zs[k] = torch.unsqueeze(optimal_ys[k], 0) 
                    if self.num_gpus > 1:                     
                        self.ys_zs[k] = self.ys_zs[k].to(f'cuda:{k%self.num_gpus}')
                    else:
                        self.ys_zs[k] = self.ys_zs[k].to(self.device)        
                    self.ys_zs[k].requires_grad_(False)
                    # logger.info("ys_zs[k]:{}".format(self.ys_zs[k]))           
            # if optimal_ys:
            #     for k in range(self.num_images):
            #         self.ys_zs[k] = torch.unsqueeze(self.dummy_z[k], 0)
            # WIP: multiple GPUs                   
            for k in range(self.num_images):
                self.dummy_zs[k] = torch.unsqueeze(self.dummy_z[k], 0)


            
            # split generator into GPUS manually
            if self.num_gpus > 1:
                logger.info(f"Spliting generators into {self.num_gpus} GPUs...")
                for trial in range(self.config['restarts']):
                    for k in range(self.num_images):
                        self.G_list2d[trial][k] = self.G_list2d[trial][k].to(f'cuda:{k%self.num_gpus}')
                        self.G_list2d[trial][k].requires_grad_(True)
                        self.dummy_zs[k] = self.dummy_zs[k].to(f'cuda:{k%self.num_gpus}')
                        self.dummy_zs[k].requires_grad_(False)

            else:
                for trial in range(self.config['restarts']):
                    for k in range(self.num_images):
                        self.G_list2d[trial][k] = self.G_list2d[trial][k].to(self.device)
                        self.G_list2d[trial][k].requires_grad_(True)
                        self.dummy_zs[k] = self.dummy_zs[k].to(self.device)
                        self.dummy_zs[k].requires_grad_(False)


            for trial in range(self.config['restarts']):
                if self.config['optim'] == 'adam':
                    optimizer[trial] = torch.optim.Adam([{'params': self.G_list2d[trial][k].parameters()} for k in range(self.num_images)], lr=self.config['gias_lr'])
                else:
                    raise ValueError()
    
                if self.config['lr_decay']:
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                        milestones=[self.gias_iterations // 2.667, self.gias_iterations // 1.6,
                                        self.gias_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

            #Unload G to CPU
            for trial in range(self.config['restarts']):
                for k in range(self.num_images):
                    self.G_list2d[trial][k].cpu()
            

            self.count_trainable_params(G=self.G_list2d[0][0], z=self.dummy_zs[0])
            logger.info(f"Total number of trainable parameters: {self.n_trainable}")

            logger.info("Start Parameter search")
            # count = 0
            for trial in range(self.config['restarts']):  #Trial is model-wise

                for k in range(self.num_images):
                    self.G_list2d[trial][k].to(f'cuda:{k%self.num_gpus}')
                for iteration in range(self.gias_iterations):
                    losses = [0, 0, 0, 0, 0, 0, 0] # tv, bn, img_norm, group_lazy, KLD, patch, CLIP

                    _x_trial = [self.gen_dummy_data(self.G_list2d[trial][k], self.generative_model_name, self.dummy_zs[k], noise=self.noise_zs[k], ys=self.ys_zs[k]).to('cpu') for k in range(self.num_images)]
                    _x[trial] = torch.stack(_x_trial).squeeze(1).to(self.device)
                    closure = self._gradient_closure(optimizer[trial], _x[trial], self.input_data, labels, losses, indices=self.config['indices'])
                    rec_loss = optimizer[trial].step(closure)
                    if self.config['lr_decay']:
                        scheduler[trial].step()
                    
                    with torch.no_grad():
                        # Project into image space
                        
                        _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                        if (iteration + 1 == self.gias_iterations) or iteration % save_interval == 0:
                            logger.info(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | ImageNorm: {losses[2]:7.4f} | gr: {losses[3]:7.4f} | patch: {losses[5]:7.4f} | CLIP: {losses[6]:7.4f}')

                    # Unload G to CPU
                    # for k in range(self.num_images):
                    #     self.G_list2d[trial][k].cpu()
                if dryrun:
                    break
                
                for k in range(self.num_images):
                    self.G_list2d[trial][k].cpu()
                
                if dryrun:
                    break

        except KeyboardInterrupt:
            logger.info(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        #Unload G to CPU
        for trial in range(self.config['restarts']):
            for k in range(self.num_images):
                self.G_list2d[trial][k].cpu()

        self.G, stats['opt'], x_optimal, _, label_optimal = self.choose_optimal(_x, labels, G=self.G_list2d)
        #the returned self.G is a list for a batch of imgs


        return x_optimal.detach(), stats

    def _init_text_embeds(self, shape):
        """
        Initialize dummy text embeddings for reconstruction attack.
        Mirrors _init_images logic but for text data.

        Returns:
            List[Tensor]: List of shape (restarts, num_images, seq_len, embed_dim)
        """
        logger.info(f"Initializing text embeddings with shape: {shape}")
        shape = (self.config['restarts'], self.num_images, shape[-2], shape[-1])  # shape is (seq_len, embed_dim)
        init_txt = self.config.get('init_text', self.config['init'])

        data_holder = DataHolder()

        # if self.text_embeds is not None:
        #     # Reuse old data if present, resized if needed
        #     return [text.detach().clone().to(self.device) for text in self.text_embeds]
        if init_txt == 'randn':
            return torch.randn(shape, **self.setup)
        elif init_txt == 'rand':
            return (torch.rand(shape, **self.setup) - 0.5) * 2
        elif init_txt == 'zeros':
            return torch.zeros(shape, **self.setup)
        elif init_txt == 'smart':
            embed = torch.randn(shape, **self.setup)
            # first token is always [CLS]
            cls_token_embedding = data_holder.get('cls_token_embedding').detach().clone()
            cls_token_embedding = cls_token_embedding.unsqueeze(0).unsqueeze(0).detach().clone().repeat(shape[0], shape[1], 1)
            embed[:, :, 0, :] = cls_token_embedding
            # last 1/2 of tokens are [PAD]
            pad_token_embedding = data_holder.get('pad_token_embedding').detach().clone()
            pad_token_embedding = pad_token_embedding.unsqueeze(0).unsqueeze(0).detach().clone().repeat(shape[0], shape[1], 1)
            half = shape[2] // 2
            embed[:, :, half:, :] = pad_token_embedding.unsqueeze(2).repeat(1, 1, shape[2] - half, 1)
            embed = embed.detach().clone().to(self.device)
            return embed
        elif init_txt == 'ground_truth':
            # provide ground truth text for mm reconstruction 
            # to see impact of perfect text reconstruction
            gt = data_holder.get('ground_truth_text')
            # correct gt list of len (num_images) [(seq_len, embed_dim)] to (restarts, num_images, seq_len, embed_dim)
            gt_corrected = [torch.stack([gt[i].detach().clone().to(self.device) for i in range(len(gt))]) for j in range(self.config['restarts'])]
            return gt_corrected
        else:
            raise ValueError(f"Unknown init type: {self.config['init']}")


    def _init_images(self, img_shape):
        # if self.images is not None:
        #     return [img.detach().clone().to(self.device) for img in self.images]
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses, indices='def'):

        data_holder = DataHolder()
        def closure():
            # logger.info(f"label:{label}")
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            # logger.info("num_images:{} num_gradients:{}".format(num_images, num_gradients))
            batch_size = num_images // num_gradients
            # logger.info("batch_size:{}".format(batch_size))
            num_batch = num_images // batch_size

            total_loss = 0
            if self.config['optim'] != "CMA-ES":
                optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]
                if self.config['model'] == "FedCola_IMG":
                    loss = self.loss_fn(self.model([batch_input, None])[0], batch_label)
                elif self.config['model'] == "FedCola_TXT":
                    loss = self.loss_fn(self.model([None, batch_input])[1], batch_label)
                elif self.config['model'] == "FedCola_IMG_TXT":
                    loss = self.loss_fn(*self.model([batch_input, batch_label], feat_out=True), torch.tensor([1.0]).to(self.device))
                else:
                    loss = self.loss_fn(self.model(batch_input), batch_label)
                # Fix to allow unused since the attack bypasses the BERT model, 
                # only the positional and type encodings are used and doesn't participate in the loss
                gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, allow_unused=True)
                # Uncomment the following line to debug unused parameters
                # for g, p in zip(gradient, self.model.parameters()):
                #     if g is None:
                #         print(f"Parameter {p.shape} is unused in this forward pass.")
                # Output would be like: "Parameter torch.Size([30522, 384]) is unused in this forward pass."
                gradient = [g if g is not None else torch.zeros_like(p) for g, p in zip(gradient, self.model.parameters())]
                #apply defense
                if self.config['defense_method'] is not None:
                    if 'noise' in self.config['defense_method']:
                        # gradient = defense.additive_noise(gradient, std=self.config['defense_setting']['noise'])
                        pass
                    if 'clipping' in self.config['defense_method']:
                        gradient = defense.gradient_clipping(gradient, bound=self.config['defense_setting']['clipping'])
                    if 'compression' in self.config['defense_method']:
                        gradient = defense.gradient_compression(gradient, percentage=self.config['defense_setting']['compression'])
                    if 'representation' in self.config['defense_method']: # for ResNet
                        mask = input_gradient[0][-2][0]!=0
                        gradient[-2] = gradient[-2] * mask
                torch.cuda.empty_cache()
                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=indices,
                                                weights=self.config['weights'], model = self.model)

                if self.config['total_variation'] > 0 and (self.config['model'] == 'FedCola_IMG' or self.config['model'] == 'FedCola_IMG_TXT'):
                    tv_loss = TV(x_trial)
                    rec_loss += self.config['total_variation'] * tv_loss
                    losses[0] = tv_loss.item()
                if self.config['bn_stat'] > 0:
                    bn_loss = 0
                    first_bn_multiplier = 10.
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                    for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                        bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                    rec_loss += self.config['bn_stat'] * bn_loss
                    losses[1] = bn_loss.item()
                if self.config['image_norm'] > 0:
                    norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['image_norm'] * norm_loss
                    losses[2] = norm_loss.item()
                if self.do_group_mean and self.config['group_lazy'] > 0:
                    group_loss =  torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['group_lazy'] * group_loss
                    losses[3] = group_loss.item()
                if self.config['z_norm'] > 0:
                    if self.dummy_z != None:
                        z_loss = torch.norm(self.dummy_z, 2)
                        rec_loss += self.config['z_norm'] * z_loss
                if self.config['KLD'] > 0:   
                    if self.generative_model_name == 'BigGAN': 
                        KLD = -0.5 * torch.sum(1 + torch.log(torch.std(self.dummy_z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(self.dummy_z.squeeze(), axis=-1).pow(2) - torch.std(self.dummy_z.squeeze(), unbiased=False, axis=-1).pow(2))
                        rec_loss += self.config['KLD'] * KLD
                        losses[4] = KLD.item()

                if self.config['patch_prior'] > 0 and (self.config['model'] == 'FedCola_IMG' or self.config['model'] == 'FedCola_IMG_TXT'):
                    patch_prior_loss_value = patch_prior_loss(x_trial, patch_size=self.config['patch_size'])
                    rec_loss += patch_prior_loss_value * self.config['patch_prior']
                    losses[5] = patch_prior_loss_value.item()
                
                if self.config['CLIP_loss'] > 0 and self.config['model'] == 'FedCola_IMG_TXT':
                    dm, ds = self.mean_std
                    x_trial_clamp = torch.clamp(x_trial * ds + dm, 0, 1)
                    if self.config['init_text'] != 'ground_truth':
                        recon_sentence, tokens = de_embed_text(batch_label[0], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'))
                    else:
                        recon_sentence = get_text_from_tokens(batch_label[0], tokenizer=data_holder.get('bert_tokenizer'))
                    clip_loss = 1 - self.clip_similarity(x_trial_clamp.detach(), recon_sentence, device=self.device)
                    rec_loss += self.config['CLIP_loss'] * clip_loss
                    losses[6] = clip_loss.item()

                total_loss += rec_loss

            if self.config['optim'] != "CMA-ES":
                total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        # logger.info(f"score_trial label type:{type(label)}")
        num_images = label.shape[0]

        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            if self.config['model'] == "FedCola_IMG":
                loss = self.loss_fn(self.model([batch_input, None])[0], batch_label)
            elif self.config['model'] == "FedCola_TXT":
                loss = self.loss_fn(self.model([None, batch_input])[1], batch_label)
            elif self.config['model'] == "FedCola_IMG_TXT":
                loss = self.loss_fn(*self.model([batch_input, batch_label], feat_out=True), torch.tensor([1.0]).to(self.device))
            else:
                loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False, allow_unused=True)
            # Fix to allow unused since the attack bypasses the BERT model,
            # only the positional and type encodings are used and doesn't participate in the loss
            gradient = [g if g is not None else torch.zeros_like(p) for g, p in zip(gradient, self.model.parameters())]
            gradient = [grad for grad in gradient]
            #apply defense
            if self.config['defense_method'] is not None:
                if 'noise' in self.config['defense_method']:
                    # gradient = defense.additive_noise(gradient, std=self.config['defense_setting']['noise'])
                    pass
                if 'clipping' in self.config['defense_method']:
                    gradient = defense.gradient_clipping(gradient, bound=self.config['defense_setting']['clipping'])
                if 'compression' in self.config['defense_method']:
                    gradient = defense.gradient_compression(gradient, percentage=self.config['defense_setting']['compression'])
                if 'representation' in self.config['defense_method']: # for ResNet
                    mask = input_gradient[0][-2][0]!=0
                    gradient[-2] = gradient[-2] * mask

            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'], model = self.model)
            total_loss += rec_loss
        return total_loss

    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    @staticmethod
    def infer_label(input_gradient, num_inputs=1):  
        last_weight_min = torch.argsort(torch.sum(input_gradient[-2], dim=-1), dim=-1)[:num_inputs]
        labels = torch.sort(last_weight_min.detach().reshape((-1,)).requires_grad_(False))[0]     # Use sort to adjust the order of labels as the same to grouth truth 
        return labels

    def clip_similarity(self, image, text, device='cuda'):
        # Convert tensor to proper format for CLIP processor  
        if isinstance(image, torch.Tensor):
            # Convert from tensor to numpy array and ensure proper format
            image_np = image.detach().cpu().numpy()
            # Handle batch dimension and channel order (C,H,W) -> (H,W,C)
            if len(image_np.shape) == 4:  # Batch of images
                image_np = image_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
                image_np = image_np.squeeze(0)  # Remove batch dimension  
            elif len(image_np.shape) == 3:  # Single image
                image_np = image_np.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
            
            # Ensure values are in [0, 255] range for PIL
            image_np = (image_np * 255).astype(np.uint8)
            image = image_np

        inputs = self.CLIP_processor(text=text, images=image, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = self.CLIP_model(**inputs)

        cosine_similarity = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=-1).squeeze()
        
        return cosine_similarity


class MultimodalJointGradientReconstructor(GradientReconstructor):
    """Reconstruct image and text seperately using the same forward pass, with two optimizers, different methods and gradient indices per each modality."""
    

    def __init__(self, model, device, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, G=None, bn_prior=((0.0, 1.0)) ):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, device, mean_std, config, num_images, G=G, bn_prior=bn_prior)

        self.image_recon_done = False
        self.text_recon_done = False

        self.img_max_iterations = self.config.get('img_max_iterations', self.max_iterations)
        self.txt_max_iterations = self.config.get('txt_max_iterations', self.max_iterations)
        self.max_iterations = max(self.img_max_iterations, self.txt_max_iterations)


    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), txt_shape=(20,384), dryrun=False, tol=None):
        """Reconstruct image from gradient."""
        if torch.is_tensor(input_data[0]):  
            self.input_data = [input_data]
        else:   # mutiple gradients
            self.input_data = input_data

        self.image_size = img_shape[1]
        start_time = time.time()
        ans = []

        self.img_shape = img_shape
        self.txt_shape = txt_shape

        self.images = self._init_images(img_shape)
        self.text_embeds = self._init_text_embeds(txt_shape)

        # Initialize dummy_z for GAN-based reconstruction
        self.dummy_z_global = None
        self.dummy_z_io = None

        if self.config['img_recon_method'] == 'GAN_based':  # GAN applying
            self.init_var(self.text_embeds)
            self.dummy_z_global = [None for _ in range(self.config['restarts'])]
            for trial in range(self.config['restarts']):
                self.dummy_z_global[trial] = self.init_dummy_z(self.G, self.generative_model_name, self.num_images)

            #GIFD
            if self.config['gifd']:
                self.config['cost_fn'] = self.gifd_loss 
                self.config['optim'] = 'adam'
                self.config['KLD'] = -1
                self.dummy_z_io = [z.detach().clone().to(self.device).requires_grad_(True) for z in self.dummy_z_global]
                # ans += self.inter_optimizer(self.dummy_z_io, infer_labels, -1)


        if self.config['img_recon_method'] == 'GAN_free':  #GAN-free method

            if self.config['geiping']:
                self.config['cost_fn'] = 'sim_cmpr0'
                self.config['image_norm'] = -1
                self.config['group_lazy'] = -1

        x_i_hat, x_t_hat = self.joint_reconstructor()
        _, best_score, x_i_hat_best, _, x_t_hat_best = self.choose_optimal(x_i_hat, x_t_hat, dryrun=dryrun)
        stats_gp = {}
        stats_gp['opt'] = best_score
        ans.append(['joint'] + [x_i_hat_best, stats_gp, x_t_hat_best])

        logger.info(f'Total time: {time.time()-start_time}.')
        return ans

    def joint_reconstructor(self):
        self.model.eval()

        #initialize data holder
        data_holder = DataHolder()
        
        self.max_iterations

        x_i_hat = self.images
        x_t_hat = self.text_embeds

        try:

            # TODO : add scheduler if needed
            image_optimizer = [None for _ in range(self.config['restarts'])]
            image_scheduler = [None for _ in range(self.config['restarts'])]

            text_optimizer = [None for _ in range(self.config['restarts'])]
            text_scheduler = [None for _ in range(self.config['restarts'])]

            # initialize label convergence metrics for text
            label_convergence_metrics = [{
                'length_infered': False,
                'perfect_match': False,
                'length_iter': -1,
                'perf_iter': -1,
                'inferred_length': -1,
                'true_length': get_true_length(data_holder.get('ground_truth_text')[nn].unsqueeze(0))
            } for nn in range(self.num_images)]
            
            # send generator into GPU        
            if self.G:
                    self.G.to(self.device)


            dm, ds = self.mean_std
            early_stopping = False

            x_i_hat_to_opt = [None for _ in range(self.config['restarts'])]
            x_t_hat_to_opt = [None for _ in range(self.config['restarts'])]


            for trial in range(self.config['restarts']):
                x_i_hat_to_opt[trial] = x_i_hat[trial]
                x_t_hat_to_opt[trial] = x_t_hat[trial]
                x_i_hat_to_opt[trial].requires_grad = True
                x_t_hat_to_opt[trial].requires_grad = True

                # initialize optimizers
                image_optimizer[trial] = torch.optim.Adam([x_i_hat_to_opt[trial]], lr=self.config['img_lr'])
                text_optimizer[trial] = torch.optim.Adam([x_t_hat_to_opt[trial]], lr=self.config['txt_lr'])

                if self.config['lr_decay'] and not self.config['optim'] == 'CMA-ES':
                    image_scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(image_optimizer[trial],
                        milestones=[self.max_iterations // 2.667, self.max_iterations // 1.6, self.max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
                    text_scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(text_optimizer[trial],
                        milestones=[self.max_iterations // 2.667, self.max_iterations // 1.6, self.max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

            for iteration in range(self.max_iterations):
                for trial in range(self.config['restarts']):

                    # tv, bn, img_norm, group_lazy, KLD, patch, CLIP
                    image_losses = [0, 0, 0, 0, 0, 0, 0]
                    text_losses = [0, 0, 0, 0, 0, 0, 0]

                    if self.G:
                        dummy_z_trial = self.dummy_z_global[trial]

                        if self.generative_model_name in ['stylegan2','stylegan2-ada','stylegan2-ada-untrained']:
                            x_t_hat_to_opt[trial] = self.gen_dummy_data(self.G_synthesis, self.generative_model_name, dummy_z_trial)

                        elif self.generative_model_name in ['stylegan2_io']:
                            x_i_hat_to_opt[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_z_trial, noise=self.noises[trial])

                        elif self.generative_model_name in ['BigGAN']:  #For gias over BigGAN
                            x_i_hat_to_opt[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_z_trial, ys=self.ys[trial])
                        else:
                            x_i_hat_to_opt[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_z_trial)
                        self.dummy_z = dummy_z_trial
                        
                    else:
                        self.dummy_z = None
    
                    if self.config['img_recon_method'] == 'GAN_free' and iteration < self.img_max_iterations:
                        image_closure = self._gradient_closure(image_optimizer[trial], x_i_hat_to_opt[trial], self.input_data, x_t_hat_to_opt[trial].detach().clone(), image_losses, indices=self.config.get('img_indices'))
                        image_rec_loss = image_optimizer[trial].step(image_closure)
                        image_rec_loss = image_rec_loss.item()

                    if self.config['txt_recon_method'] == 'GAN_free' and iteration < self.txt_max_iterations:
                        text_closure = self._gradient_closure(text_optimizer[trial], x_i_hat_to_opt[trial].detach().clone(), self.input_data, x_t_hat_to_opt[trial], text_losses, indices=self.config.get('txt_indices'))
                        text_rec_loss = text_optimizer[trial].step(text_closure)
                        text_rec_loss = text_rec_loss.item()

                    if self.config['lr_decay'] and not self.config['optim'] == 'CMA-ES':
                        image_scheduler[trial].step()
                        text_scheduler[trial].step()

                    with torch.no_grad():
                        if self.config['save_intermediate_at_img'] > 0 and (iteration % self.config['save_intermediate_at_img'] == 0):
                            logger.info(f'Saving intermediate IMG at iteration {iteration}...')
                            for num_img in range(self.num_images):
                                dir_path = os.path.join(data_holder.get('save_dir'), f'{num_img}/')
                                os.makedirs(dir_path, exist_ok=True)
                                torchvision.utils.save_image(torch.clamp(x_i_hat_to_opt[trial].detach().clone() * ds + dm, 0, 1)[num_img:num_img + 1, ...], os.path.join(dir_path, f'{num_img}_trial_{trial}_it_{iteration}.png'))
                        if self.config['save_intermediate_at_txt'] > 0 and (iteration % self.config['save_intermediate_at_txt'] == 0):
                            logger.info(f'Saving intermediate TXT at iteration {iteration}...')
                            for num_txt in range(self.num_images):
                                recon_sentence, tokens = de_embed_text(x_t_hat_to_opt[trial][num_txt], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'))
                                logger.info(f'Recon Sentence : {recon_sentence}')
                                dir_path = os.path.join(data_holder.get('save_dir'), f'{num_txt}/')
                                os.makedirs(dir_path, exist_ok=True)
                                with open(os.path.join(dir_path, f'{num_txt}_trial_{trial}_it_{iteration}.txt'), 'w') as f:
                                    f.write(recon_sentence)

                        if (iteration + 1 == self.max_iterations) or iteration % save_interval == 0:
                            logger.info(f'It: {iteration}. Image Rec. loss: {image_rec_loss:2.4f} | tv: {image_losses[0]:7.4f} | bn: {image_losses[1]:7.4f} | ImageNorm: {image_losses[2]:7.4f} | gr: {image_losses[3]:7.4f} | kld: {image_losses[4]:7.4f} | patch: {image_losses[5]:7.4f} | CLIP: {image_losses[6]:7.4f} ')
                            logger.info(f'It: {iteration}. Text Rec. loss: {text_rec_loss:2.4f} ')
                            if self.config['z_norm'] > 0:
                                logger.info(torch.norm(dummy_z[trial], 2).item())

                        x_i_hat_to_opt[trial].data = torch.max(torch.min(x_i_hat_to_opt[trial], (1 - dm) / ds), -dm / ds)
                        x_t_hat_to_opt[trial].data = x_t_hat_to_opt[trial]

                        # check length convergence and perfect match convergence for text
                        if (self.config['model'] == 'FedCola_IMG_TXT' or self.config['model'] == 'FedCola_TXT') and self.config['init_text'] != 'ground_truth' and iteration % 50 == 0:
                            cap_opt = x_t_hat_to_opt[trial].detach().clone()
                            for num_img in range(self.num_images):
                                if not label_convergence_metrics[num_img]['length_infered']:
                                    inferred_length, is_length_correct = infer_label_length_convergence(cap_opt[num_img], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'), true_length=label_convergence_metrics[num_img]['true_length'])
                                    if inferred_length > 0:
                                        label_convergence_metrics[num_img]['length_infered'] = is_length_correct
                                        label_convergence_metrics[num_img]['length_iter'] = iteration
                                        label_convergence_metrics[num_img]['inferred_length'] = inferred_length
                                        logger.info(f"Trial {trial}: Length inferred at iteration {iteration}, correct: {is_length_correct}")
                                if not label_convergence_metrics[num_img]['perfect_match']:
                                    is_perfect = infer_label_perfect_match(cap_opt[num_img], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'), ground_truth_text_token_ids=data_holder.get('ground_truth_text')[num_img])
                                    if is_perfect:
                                        label_convergence_metrics[num_img]['perfect_match'] = True
                                        label_convergence_metrics[num_img]['perf_iter'] = iteration
                                        logger.info(f"Trial {trial}: Perfect match achieved at iteration {iteration}")
                            data_holder.set('label_convergence_metrics', label_convergence_metrics)
                            # stop condition if any perfect matched trial achieved for all num images
                            if self.config['stop_at_text_perf_match']:
                                all_perf_matched = all([label_convergence_metrics[num_img]['perfect_match'] for num_img in range(self.num_images)])
                                if all_perf_matched:
                                    logger.info("Perfect text match achieved, stopping optimization.")
                                    early_stopping = True

                    if early_stopping:
                        break

                if early_stopping:
                    break

                if iteration == self.img_max_iterations -1:
                    self.image_recon_done = True
                    logger.info("=== Image reconstruction iterations maxed out ===")
                if iteration == self.txt_max_iterations -1:
                    self.text_recon_done = True
                    logger.info("=== Text reconstruction iterations maxed out ===")

            # Update the original variables with the optimized results
            for trial in range(self.config['restarts']):
                x_i_hat[trial] = x_i_hat_to_opt[trial].detach().clone()
                x_t_hat[trial] = x_t_hat_to_opt[trial].detach().clone()

        except KeyboardInterrupt:
            logger.info(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        if self.G:
            self.G.to("cpu")

        return x_i_hat, x_t_hat

class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0, 
                 G=None):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images, G=G)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses, indices='def'):

        data_holder = DataHolder()
        def closure():
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size

            total_loss = 0
            optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]

                # loss = self.loss_fn(self.model(batch_input), batch_label)
                # gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                
                gradient = loss_steps(self.model, batch_input, batch_label, loss_fn=self.loss_fn,
                                        local_steps=self.local_steps, lr=self.local_lr,
                                        use_updates=self.use_updates,
                                        batch_size=self.batch_size,
                                        config=self.config)

                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=indices,
                                                weights=self.config['weights'], model = self.model)

                if self.config['total_variation'] > 0 and (self.config['model'] == 'FedCola_IMG' or self.config['model'] == 'FedCola_IMG_TXT'):
                    tv_loss = TV(x_trial)
                    rec_loss += self.config['total_variation'] * tv_loss
                    losses[0] = tv_loss
                if self.config['bn_stat'] > 0:
                    bn_loss = 0
                    first_bn_multiplier = 10.
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                    for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                        bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                    rec_loss += self.config['bn_stat'] * bn_loss
                    losses[1] = bn_loss
                if self.config['image_norm'] > 0:
                    norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['image_norm'] * norm_loss
                    losses[2] = norm_loss
                if self.do_group_mean and self.config['group_lazy'] > 0:
                    group_loss =  torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['group_lazy'] * group_loss
                    losses[3] = group_loss

                if self.config['KLD'] > 0:   
                    if self.generative_model_name == 'BigGAN': 
                        KLD = -0.5 * torch.sum(1 + torch.log(torch.std(self.dummy_z.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(self.dummy_z.squeeze(), axis=-1).pow(2) - torch.std(self.dummy_z.squeeze(), unbiased=False, axis=-1).pow(2))
                        rec_loss += self.config['KLD'] * KLD
                        losses[4] = KLD.item()

                if self.config['patch_prior'] > 0 and (self.config['model'] == 'FedCola_IMG' or self.config['model'] == 'FedCola_IMG_TXT'):
                    patch_prior_loss_value = patch_prior_loss(x_trial, patch_size=self.config['patch_size'])
                    rec_loss += patch_prior_loss_value * self.config['patch_prior']
                    losses[5] = patch_prior_loss_value.item()

                if self.config['CLIP_loss'] > 0 and self.config['model'] == 'FedCola_IMG_TXT':
                    dm, ds = self.mean_std
                    x_trial_clamp = torch.clamp(x_trial * ds + dm, 0, 1)
                    recon_sentence, tokens = de_embed_text(batch_label[0], bert_embedding=data_holder.get('bert_embedding'), tokenizer=data_holder.get('bert_tokenizer'))
                    clip_loss = 1 - self.clip_similarity(x_trial_clamp.detach(), recon_sentence, device=self.device)
                    rec_loss += self.config['CLIP_loss'] * clip_loss
                    losses[6] = clip_loss.item()

                if self.config['z_norm'] > 0:
                    if self.dummy_z != None:
                        z_loss = torch.norm(self.dummy_z, 2)
                        rec_loss += 1e-3 * z_loss

                total_loss += rec_loss


            total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        self.model.zero_grad()
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            # loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = loss_steps(self.model, batch_input, batch_label, loss_fn=self.loss_fn,
                                local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates, config=self.config)
            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'], model = self.model)
            total_loss += rec_loss
        return total_loss

def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0, config=DEFAULT_CONFIG):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            # TODO
            if config['model'] == 'FedCola_IMG_TXT':
                outputs = patched_model([inputs, labels], patched_model.parameters, feat_out=True)
            elif config['model'] == 'FedCola_IMG':
                outputs = patched_model([inputs,None], patched_model.parameters)[0]
            elif config['model'] == 'FedCola_TXT':
                outputs = patched_model([None, inputs], patched_model.parameters)[1]
            else:
                outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            # TODO
            if config['model'] == 'FedCola_IMG_TXT':
                outputs = patched_model([inputs, labels], patched_model.parameters, feat_out=True)
            elif config['model'] == 'FedCola_IMG':
                outputs = patched_model([inputs,None], patched_model.parameters)[0]
            elif config['model'] == 'FedCola_TXT':
                outputs = patched_model([None, inputs], patched_model.parameters)[1]
            else:
                outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        if config['model'] == 'FedCola_IMG_TXT':
            loss = loss_fn(*outputs, torch.tensor([1.0]).to(outputs[0].device)).sum()
        else:
            loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())

def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal', model=None):
    """Input gradient is given data."""
    # logger.info("The length of gradients:{}".format(len(gradients)))
    # logger.info("The length of gradients:{}".format(len(input_gradient)))

    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices.startswith('fedcola') and model is not None:
        to_select = []
        if 'img_emb' in indices:
            to_select.append("img_embedding")
        if 'txt_emb' in indices:
            to_select.append("txt_embedding")
        if 'img_block' in indices:
            to_select.append("img_blocks")
        if 'txt_block' in indices:
            to_select.append("txt_blocks")
        if 'head' in indices:
            to_select.append("heads")

        indices = select_indices_fedcola(model, select=to_select)
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        # logger.info("Indices:{}".format(indices))
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('compressed'):
                ratio = float(cost_fn[10:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)

                trial_flatten = trial_gradient[i].flatten()
                trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * trial_mask

                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask
                costs += ((trial_compressed - input_compressed).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('sim_cmpr'):
                ratio = float(cost_fn[8:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)
                
                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask

                trial_flatten = trial_gradient[i].flatten()
                # trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                # trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * input_mask

                
                costs -= (trial_compressed * input_compressed).sum() * weights[i]
                pnorm[0] += trial_compressed.pow(2).sum() * weights[i]
                pnorm[1] += input_compressed.pow(2).sum() * weights[i]

            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                input_gradient[i].flatten(),
                                                                0, 1e-10) * weights[i]
        if cost_fn.startswith('sim'):
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        if cost_fn == 'l2':
            costs /= len(indices)
        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def patch_prior_loss(x_hat, patch_size=16):
    """
    Patch prior (GradViT): penalize discontinuities at patch boundaries.
    Uses squared L2 across channels at each boundary pixel.
    x_hat: [N, C, H, W]
    """
    N, C, H, W = x_hat.shape
    device = x_hat.device

    loss = x_hat.new_zeros(())
    # Horizontal boundaries (rows at k*patch_size)
    if H >= patch_size:
        h_idx = torch.arange(patch_size, H, patch_size, device=device)
        dh = x_hat[:, :, h_idx, :] - x_hat[:, :, h_idx - 1, :]
        loss = loss + dh.pow(2).sum()

    # Vertical boundaries (cols at k*patch_size)
    if W >= patch_size:
        w_idx = torch.arange(patch_size, W, patch_size, device=device)
        dv = x_hat[:, :, :, w_idx] - x_hat[:, :, :, w_idx - 1]
        loss = loss + dv.pow(2).sum()

    return loss
