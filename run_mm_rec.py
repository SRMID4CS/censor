"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""
import os

from utils.data_holder import DataHolder
#limit the visual gpus
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision
import torch.nn as nn
import yaml
import numpy as np

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time

import json
import hashlib
import csv
import copy
import pickle
import defense
import lpips

import datetime
import logging

from FedCola.src.fedcola_options import add_fedcola_args
import FedCola.src.models.mome

from utils.text_utils import de_embed_text
from transformers import BertTokenizer

import evaluate


def init_logger(output_dir, log_level=logging.INFO):
    """Initialize and configure the root logger."""
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # File Handler
    fh = logging.FileHandler(os.path.join(output_dir, "main.log"))
    fh.setLevel(log_level)
    fh_formatter = logging.Formatter('%(message)s')  # Only message content
    fh.setFormatter(fh_formatter)
    root_logger.addHandler(fh)

    # Stream Handler (Console)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh_formatter = logging.Formatter('%(message)s')  # Only message content
    sh.setFormatter(sh_formatter)
    root_logger.addHandler(sh)

    root_logger.info("-" * 80)

    return root_logger

nclass_dict = {'I32': 1000, 'I64': 1000, 'I128': 1000, 
               'CIFAR10': 10, 'CIFAR100': 100, 'CA': 8, 'ImageNet':1000, 'IMAGENET_IO' : 1000,
               'FFHQ': 10, 'FFHQ64': 10, 'FFHQ128': 10, 'OOD_FFHQ':10, 'OOD_IMAGENET':1000,
               'Coco': 25499, 'Coco_img': 25499, 'Coco_txt': 25499,
               }
# Parse input arguments

parser = inversefed.options()

# parser.add_argument('--seed', default=1234, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--batch_size', default=4, type=int, help='Number of mini batch for federated averaging')
parser.add_argument('--local_lr', default=1e-4, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--checkpoint_path', default='', type=str, help='Local learning rate for federated averaging')
parser.add_argument('--gan', default='stylegan2', type=str, help='GAN model option:[stylegan2, biggan]')
parser.add_argument('--config', default='./config_stylegan2', type=str, help='Path of selected config file.')


# Add Args for FedCola

parser = add_fedcola_args(parser)


args = parser.parse_args()
if args.target_id is None:
    args.target_id = 0
args.save_image = True

# Parse training strategy
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config

def save_experiment_config(config_path, save_dir):
    # Generate a unique identifier for the experiment
    experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    saved_config_file = os.path.join(save_dir, f"experiment_config_{experiment_id}.yml")

    # Read and save a copy of the configuration
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(saved_config_file, 'w') as file:
        yaml.dump(config, file)

    return saved_config_file



def log_model_dropout_rates(model, logger):
    """Logs the dropout rates of all dropout and DropPath layers in the model."""
    dropout_layers = []
    droppath_layers = []
    # DropPath may be from timm or custom, so check by class name
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append((name, module.p))
        elif module.__class__.__name__ == 'DropPath':
            # DropPath usually has attribute drop_prob or drop_path_rate
            drop_prob = getattr(module, 'drop_prob', None)
            if drop_prob is None:
                drop_prob = getattr(module, 'drop_path_rate', None)
            droppath_layers.append((name, drop_prob))
    if dropout_layers:
        logger.info("Dropout rates in the model:")
        for name, rate in dropout_layers:
            logger.info(f"Dropout layer: {name}, rate: {rate}")
    else:
        logger.info("No nn.Dropout layers found in the model.")
    if droppath_layers:
        logger.info("DropPath rates in the model:")
        for name, rate in droppath_layers:
            logger.info(f"DropPath layer: {name}, drop_prob: {rate}")
    else:
        logger.info("No DropPath layers found in the model.")


if __name__ == "__main__":
    # Choose GPU device and print status information:
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    
    start_time = time.time()
    #read config
    config_path = args.config
    config = load_config(config_path=config_path)

    # change the exp path
    config['exp_name'] = config['exp_name'] + '_num_images_' + str(config['num_images']) + '_defense_'+ str(config['defense_method']) + '_' + current_time
    save_dir = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    logger = init_logger(save_dir)
    # Log the process ID
    pid = os.getpid()
    logging.info(f"Process ID: {pid}")
    logger.info("output_dir: {}".format(save_dir))


    setup = inversefed.utils.system_startup(args)

    save_experiment_config(config_path, save_dir)
    logger.info(f"Config for this experiment is saved")
    # Prepare for training
    # Get data:

    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(config['dataset'], defs, data_path=config['data_path'], args=args)

    set_seed = config['set_seed']
    if isinstance(set_seed, int):
        logger.info("Set seed:{}".format(set_seed))
        torch.manual_seed(set_seed)
    
    model, model_seed = inversefed.construct_model(config['model'], num_classes=nclass_dict[config['dataset']], num_channels=3, seed=set_seed, args=args)
    model.to(**setup)

    logger.info("Model loaded: {}".format(model))

    try:
        log_model_dropout_rates(model, logger)
    except Exception as e:
        logger.error(f"Error logging dropout rates: {e}")

    bert_embedding = None
    if hasattr(model, 'embeddings'):
        for emb in model.embeddings:
            # Find the text embedding module
            if hasattr(emb, 'text_embeddings'):
                bert_embedding = copy.deepcopy(emb.text_embeddings)
                break

    if bert_embedding is not None:
        logger.info("BERT model loaded for text embedding: {}".format(bert_embedding))
        
        bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case="uncased" in 'bert_base_uncased'
        )

        data_holder = DataHolder()

        cls_token_embedding = bert_embedding(torch.tensor([[bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.cls_token)]], device=setup['device']))[0]
        pad_token_embedding = bert_embedding(torch.tensor([[bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.pad_token)]], device=setup['device']))[0]

        data_holder.set('cls_token_embedding', cls_token_embedding)
        data_holder.set('pad_token_embedding', pad_token_embedding)
    else:
        logger.info("No BERT model found, using default model.")



    if config['dataset'].startswith('FFHQ') or config['dataset'].endswith('FFHQ'):
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    else:
        dataset = config['dataset']
        dm = torch.as_tensor(getattr(inversefed.consts, f'{dataset.lower()}_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'{dataset.lower()}_std'), **setup)[:, None, None]

    # optimizer, scheduler = inversefed.training.set_optimizer(model, defs)
    optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay)
    logger.info(f"Optimizer: {optimizer}")

    
    
    # train the model and save the model checkpoint at each epoch
    for epoch in range(config['train_epochs']):
        # add the epoch folder
        save_dir = os.path.join(config['output_dir'], config['exp_name'], f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Epoch {epoch} started,saved at {save_dir}")

        # model = nn.DataParallel(model)
        model.eval()

        if config['optim'] == 'GAN_based':
            config_m = dict(cost_fn=config['cost_fn'],
                        indices=config['indices'],
                        weights=config['weights'],
                        lr=config['lr'] if config['lr'] is not None else 0.1,
                        optim='adam',
                        restarts=config['restarts'],
                        max_iterations=config['max_iterations'],
                        total_variation=config['total_variation'],
                        bn_stat=config['bn_stat'],
                        image_norm=config['image_norm'],
                        z_norm= args.z_norm,
                        group_lazy=config['group_lazy'],
                        init=config['init'],
                        lr_decay=True,
                        dataset=config['dataset'],
                        #params for inter optim
                        ckpt= config['ckpt'],
                        gifd = config['gifd'],
                        steps =  config['steps'],
                        lr_io =  config['lr_io'],
                        start_layer = config['start_layer'],
                        end_layer = config['end_layer'],  
                        do_project_gen_out = config['do_project_gen_out'],
                        do_project_noises = config['do_project_noises'],
                        do_project_latent = config['do_project_latent'],
                        max_radius_gen_out = config['max_radius_gen_out'],
                        max_radius_noises = config['max_radius_noises'],
                        max_radius_latent = config['max_radius_latent'],
                        #defense
                        defense_method = config['defense_method'],
                        defense_setting = config['defense_setting'],

                                                    
                        generative_model=config['generative_model'],
                        gen_dataset=config['gen_dataset'],
                        giml='',
                        gias= config['gias'],
                        ggl = config['ggl'],
                        cma_budget = config['cma_budget'],
                        num_sample = config['num_sample'],
                        KLD = config['KLD'],
                        gias_lr=config['gias_lr'],
                        gias_iterations=config['gias_iterations'],
                        model=config['model'],
                        )
        elif config['optim'] == 'GAN_free':
            config_m = dict(cost_fn=config['cost_fn'],
                        indices=config['indices'],
                        weights=config['weights'],
                        lr=config['lr'] if config['lr'] is not None else 0.1,
                        optim='adam',
                        restarts=config['restarts'],
                        max_iterations=config['max_iterations'],
                        total_variation=config['total_variation'],
                        bn_stat=config['bn_stat'],
                        image_norm=config['image_norm'],
                        z_norm=args.z_norm,
                        group_lazy=config['group_lazy'],
                        init=config['init'],
                        init_text=config['init_text'],
                        lr_decay=True,
                        dataset=config['dataset'],
                        geiping=config['geiping'],
                        yin=config['yin'],
                        generative_model='',
                        gen_dataset='',
                        giml=False,
                        gias=False,
                        gias_lr=0.0,
                        gias_iterations=0,
                        model=config['model'],
                        )

        G = None
        if args.checkpoint_path:
            with open(args.checkpoint_path, 'rb') as f:
                G, _ = pickle.load(f)
                G = G.requires_grad_(True).to(setup['device'])

        config_to_save = dict(config)  # Make a shallow copy
        config_to_save.pop('cls_token_embedding', None)
        config_to_save.pop('pad_token_embedding', None)
        #Save the config file first
        inversefed.utils.save_to_table(os.path.join(config['output_dir'], config['exp_name']), name='configs', dryrun=args.dryrun, **config_to_save)
        target_id = config['target_id']
        target_id_increment = config['target_id_increment']
        iter_dryrun = False

        for i in range(config['num_exp']): 

            # indicator dictionary
            psnrs = {}
            lpips_sc ={}
            lpips_sc_a = {}
            ssim = {}
            mse_i = {}

            target_id = config['target_id'] + i * target_id_increment

            tid_list = []

            print(f"Dataset size: {len(validloader.dataset)}")
            print(f"Attempting to access index: {target_id}")

            if config['num_images'] == 1:
                ground_truth, labels = validloader.dataset[target_id]
                ground_truth = ground_truth.unsqueeze(0).to(**setup)
                if config['model'] == 'FedCola_TXT':
                    ground_truth = ground_truth.long()
                if config['model'] == 'FedCola_IMG_TXT':
                    labels = labels.unsqueeze(0).to(**setup)
                    labels = labels.long()
                else:
                    labels =  torch.as_tensor((labels,), device=setup['device'])
                target_id_ = target_id + 1
                logger.info(f"loaded img {target_id_ - 1}")
                tid_list.append(target_id_ - 1)
            else:
                ground_truth, labels = [], []
                target_id_ = target_id
                while len(labels) < config['num_images']:
                    img, label = validloader.dataset[target_id_]
                    target_id_ += config['target_id_increment']
                    label_in_device = label.to(setup['device']) if torch.is_tensor(label) else torch.tensor(label, device=setup['device'])
                    if config['model'] == 'FedCola_IMG_TXT' or (label_in_device not in labels):
                        logger.info("loaded img %d" % (target_id_ - 1))
                        labels.append(label_in_device)
                        ground_truth.append(img.to(**setup))
                        tid_list.append(target_id_ - 1)

                ground_truth = torch.stack(ground_truth)

                if config['model'] == 'FedCola_TXT':
                    ground_truth = ground_truth.long()
                if config['model'] == 'FedCola_IMG_TXT':
                    labels = torch.stack(labels)
                    labels = labels.long()
                else:
                    labels = torch.cat(labels)

            if config['model'] == 'FedCola_IMG_TXT':
                input_shape = (3, ground_truth.shape[-2], ground_truth.shape[-1])
                logger.info("Input shape for FedCola_IMG_TXT: {}".format(input_shape))
            elif config['model'] == 'FedCola_IMG':
                input_shape = (3, ground_truth.shape[-2], ground_truth.shape[-1])
                logger.info("Input shape for FedCola_IMG: {}".format(input_shape))
            elif config['model'] == 'FedCola_TXT':
                input_shape = (ground_truth.shape[-2], ground_truth.shape[-1])
                logger.info("Input shape for FedCola_TXT: {}".format(input_shape))
            else:
                input_shape = (3, ground_truth.shape[2], ground_truth.shape[3])
                logger.info("Input shape: {}".format(input_shape))

            # Run reconstruction
            if config['bn_stat'] > 0:
                bn_layers = []
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn_layers.append(inversefed.BNStatisticsHook(module))

            # store ground truth text to be used in mm reconstruction
            # to see impact of perfect text reconstruction
            if config['model'] == 'FedCola_IMG_TXT' and config['init_text'] == 'ground_truth':
                data_holder.set('ground_truth_text', labels)

            if args.accumulation == 0:
                logger.info("Ground truth's size:{}".format(ground_truth[0].shape))
                logger.info("Ground truth's size RAW:{}".format(ground_truth.shape))
                if 'FedCola' in config['model']:
                    if config['model'] == 'FedCola_IMG_TXT':
                        outputs_for_rec = model([ground_truth, labels], feat_out=True)
                    elif config['model'] == 'FedCola_IMG':
                        outputs_for_rec = model([ground_truth,None])[0]
                    elif config['model'] == 'FedCola_TXT':
                        outputs_for_rec = model([None, ground_truth])[1]
                else:
                    outputs_for_rec = model(ground_truth)
                if config['model'] == 'FedCola_IMG_TXT':
                    target_loss = loss_fn(*outputs_for_rec, torch.tensor([1.0]).to(**setup))
                else:
                    target_loss, _, _ = loss_fn(outputs_for_rec, labels)
                input_gradient = torch.autograd.grad(target_loss, model.parameters())

                # compute the input_gradient norm
                input_gradient_norm = torch.norm(torch.cat([g.view(-1) for g in input_gradient]), p=2)
                # move the input_gradient norm to the cpu
                input_gradient_tmp = input_gradient_norm.cpu().detach().numpy()

                logger.info(f"Input gradient norm: {input_gradient_norm}")
                logger.info(f"Target loss: {target_loss.item()}")

                # save the input_gradient norm and the target_loss to a csv file, use the save to table function, only save the input_gradient norm and the target_loss
                inversefed.utils.save_to_table(os.path.join(save_dir), name=f'{epoch}_epoch_input_gradient_norm', dryrun=args.dryrun, input_gradient_norm=input_gradient_tmp, target_loss=target_loss.item(), target_id=target_id, seed=model_seed)

                best_noise_loss = target_loss

                bn_prior = []
                if config['bn_stat'] > 0:
                    for idx, mod in enumerate(bn_layers):
                        mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
                        bn_prior.append(mean_var)

                #apply defense strategy
                if config['defense_method'] is None:
                    logger.info('No defense applied.')
                    d_param = config['defense_setting']
                else:
                    if config['defense_method'] == 'noise':
                        d_param = 0.01 if config['defense_setting']['noise'] is None else config['defense_setting']['noise']
                        input_gradient = defense.additive_noise(model, input_gradient, save_dir, std=d_param)
                        overhead_end_time = time.time()
                    if config['defense_method'] == 'clipping':
                        d_param = 4 if  config['defense_setting']['clipping'] is None else config['defense_setting']['clipping']
                        input_gradient = defense.gradient_clipping(input_gradient, bound=d_param)
                    if config['defense_method'] == 'compression':
                        d_param = 20 if  config['defense_setting']['compression'] is None else config['defense_setting']['compression']
                        input_gradient = defense.gradient_compression(input_gradient, percentage=d_param)
                    if config['defense_method'] == 'representation':
                        d_param = 10 if config['defense_setting']['representation'] is None else config['defense_setting']['representation']
                        input_gradient = defense.perturb_representation(input_gradient, model, ground_truth, pruning_rate=d_param)
                    if config['defense_method'] == 'orthogonal' and epoch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        logger.info('Orthogonal applied in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].')
                        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9,
                                    weight_decay=defs.weight_decay)
                        d_param = 1e-4 if config['defense_setting']['orthogonal'] is None else config['defense_setting']['orthogonal']
                        input_gradient, best_noise_loss = defense.orthogonal_gradient(input_gradient, model, ground_truth, labels, trials=config['our_num_tries'], epsilon=d_param, best_loss = target_loss)
                    elif config['defense_method'] == 'orthogonal' and epoch not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay)
                        logger.info('Orthogonal applied in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], no defense applied later.')

                rec_machine = inversefed.GradientReconstructor(model, setup['device'], (dm, ds), config_m, num_images=config['num_images'], bn_prior=bn_prior, G=G)

                if G is None:
                    G = rec_machine.G

                if config['model'] == 'FedCola_IMG_TXT':
                    labels_to_support = None
                    logger.info("Reconstructing FedCola_IMG_TXT : Labels set to None")
                else:
                    labels_to_support = labels
                    logger.info("Real labels:{}".format(labels))

                result = rec_machine.reconstruct(input_gradient, labels_to_support, img_shape=input_shape, dryrun=iter_dryrun)
                if iter_dryrun:
                    continue
            else:
                local_gradient_steps = args.accumulation
                local_lr = args.local_lr
                batch_size = args.batch_size
                input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth,
                                                                                labels,
                                                                                lr=local_lr,
                                                                                local_steps=local_gradient_steps, use_updates=True, batch_size=batch_size,
                                                                                config=config_m
                                                                                )
                input_parameters = [p.detach() for p in input_parameters]

                rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_gradient_steps,
                                                            local_lr, config_m,
                                                            num_images=config['num_images'], use_updates=True,
                                                            batch_size=batch_size)
                if G is None:
                    if rec_machine.generative_model_name in ['stylegan2']:
                        G = rec_machine.G_synthesis
                    else:
                        G = rec_machine.G
                
                if config['model'] == 'FedCola_IMG_TXT':
                    labels_to_support = None
                    logger.info("Reconstructing FedCola_IMG_TXT : Labels set to None")
                else:
                    labels_to_support = labels
                    logger.info("Real labels:{}".format(labels))

                result = rec_machine.reconstruct(input_parameters, labels_to_support, img_shape=input_shape, dryrun=args.dryrun)

            #lpips
            lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(**setup)
            lpips_loss_a = lpips.LPIPS(net='alex', spatial=False).to(**setup)

            #Record the best layer if GIFD is applied
            Best_layer_num = -1
            for idx, item in enumerate(result):
                # Compute stats and save to a table:
                file_name = item[0]
                output = item[1]
                stats = item[2]
                label_best = item[3]

                if file_name == "Best_layer_num":
                    Best_layer_num = int(output) 
                    continue
                    
                if output is None :  #some layers were skiped 
                    test_psnr = -1
                    lpips_score = -1
                    lpips_score_a = -1
                    ssim_score = -1
                    test_mse = -1
                    feat_mse = -1
                elif output.shape[-1] != ground_truth.shape[-1]:
                    test_psnr = -1
                    lpips_score = -1
                    lpips_score_a = -1
                    ssim_score = -1
                    test_mse = -1
                    feat_mse = -1
                    output_den = torch.clamp(output * ds + dm, 0, 1)
                else:
                    output_den = torch.clamp(output * ds + dm, 0, 1)
                    ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
                    # logger.info("output's dimension:{} ground_truth's dimension:{}".format(output.shape, ground_truth.shape))
                    if 'FedCola' in config['model']:
                        if config['model'] == 'FedCola_IMG_TXT':
                            reconstructed_output_feat = model([output, labels], feat_out=True)
                            reconstructed_output_feat = reconstructed_output_feat[0] # TEMP TODO fix for text
                            reconstructed_ground_truth_feat = model([ground_truth, labels], feat_out=True)
                            reconstructed_ground_truth_feat = reconstructed_ground_truth_feat[0] # TEMP TODO fix for text
                        elif config['model'] == 'FedCola_IMG':
                            reconstructed_output_feat = model([output,None])[0]
                            reconstructed_ground_truth_feat = model([ground_truth,None])[0]
                        elif config['model'] == 'FedCola_TXT':
                            reconstructed_output_feat = model([None, output])[1]
                            reconstructed_ground_truth_feat = model([None, ground_truth])[1]
                    else:
                        reconstructed_output_feat = model(output)
                        reconstructed_ground_truth_feat = model(ground_truth)
                    feat_mse = (reconstructed_output_feat - reconstructed_ground_truth_feat).pow(2).mean().item()
                    test_mse = (output_den - ground_truth_den).pow(2).mean().item()
                    ssim_score, _ = inversefed.metrics.ssim_batch(output, ground_truth)
                    with torch.no_grad():
                        lpips_score = lpips_loss(output, ground_truth).squeeze().mean().item()
                        lpips_score_a = lpips_loss_a(output, ground_truth).squeeze().mean().item()
                    logger.info("output_den's dimension:{}".format(output_den.shape))
                    test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
                    logger.info(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | LPIPS(VGG): {lpips_score:2.4f} | LPIPS(ALEX): {lpips_score_a:2.4f} | SSIM: {ssim_score:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | ")

                ouput_dir = os.path.join(save_dir, file_name)
                
                psnrs[file_name +'_psnr'] = test_psnr
                lpips_sc[file_name +'_lpips(vgg)'] = lpips_score
                lpips_sc_a[file_name +'_lpips(alex)'] = lpips_score_a
                ssim[file_name +'_ssim'] = ssim_score
                mse_i[file_name + '_mse_i'] = test_mse

                os.makedirs(os.path.join(ouput_dir), exist_ok=True)

                exp_name = config['exp_name']
                inversefed.utils.save_to_table(os.path.join(ouput_dir), name=f'{exp_name}', dryrun=args.dryrun,
                                            rec_loss=stats["opt"],
                                            psnr=test_psnr,
                                            LPIPS_VGG=lpips_score,
                                            LPIPS_ALEX=lpips_score_a,
                                            ssim=ssim_score,
                                            test_mse=test_mse,
                                            feat_mse=feat_mse,

                                            target_id=target_id,
                                            seed=model_seed
                                            )


                # Save the resulting image
                if args.save_image and output is not None:
                    if config['model'] == 'FedCola_IMG' or config['model'] == 'FedCola_IMG_TXT':
                        # Same as Default - Save the image
                        for j in range(config['num_images']):
                            torchvision.utils.save_image(output_den[j:j + 1, ...], os.path.join(ouput_dir, f'{tid_list[j]}_gen.png'))

                    if config['model'] == 'FedCola_TXT' or (config['model'] == 'FedCola_IMG_TXT' and config['init_text'] != 'ground_truth'):
                        # Save the text after deembedding
                        recon_sentence_list = []
                        for j in range(config['num_images']):
                            recon_sentence_embedding_seq = output_den[j:j + 1, ...] if config['model'] == 'FedCola_TXT' else label_best[j:j + 1, ...]
                            # convert to text
                            recon_sentence = de_embed_text(recon_sentence_embedding_seq[0], bert_embedding=bert_embedding, tokenizer=bert_tokenizer)
                            recon_sentence_list.append(recon_sentence)
                            with open(os.path.join(ouput_dir, f'{tid_list[j]}_gen.txt'), 'w') as f:
                                f.write(recon_sentence)

                    else:
                        # Default - Save the image
                        for j in range(config['num_images']):
                            torchvision.utils.save_image(output_den[j:j + 1, ...], os.path.join(ouput_dir, f'{tid_list[j]}_gen.png'))

                # Update target id
                target_id = target_id_
                
            if Best_layer_num >= 0:                    
                inversefed.utils.save_to_table(os.path.join(save_dir), name='Metrics', dryrun=args.dryrun, target_id=int(target_id - 1), Best_layer_num=Best_layer_num ,**psnrs, **lpips_sc, **lpips_sc_a, **ssim, **mse_i)
            else:
                inversefed.utils.save_to_table(os.path.join(save_dir), name='Metrics', dryrun=args.dryrun, target_id=int(target_id - 1), **psnrs, **lpips_sc, **lpips_sc_a, **ssim, **mse_i)
                

            for j in range(config['num_images']):
                # TODO Save the ground truth txt
                if config['model'] == 'FedCola_IMG' or config['model'] == 'FedCola_IMG_TXT':
                    torchvision.utils.save_image(ground_truth_den[j:j + 1, ...], os.path.join(save_dir, f'{tid_list[j]}_gt.png'))

                if config['model'] == 'FedCola_TXT' or config['model'] == 'FedCola_IMG_TXT':
                    # Save the text after deembedding
                    sentence_token_ids = ground_truth[j:j + 1, ...] if config['model'] == 'FedCola_TXT' else labels[j:j + 1, ...]
                    # convert to text
                    sentence = bert_tokenizer.convert_ids_to_tokens(sentence_token_ids.squeeze().tolist())
                    print("Ground truth tokens:", sentence)
                    sentence = bert_tokenizer.convert_tokens_to_string(sentence)
                    print("Ground truth sentence:", sentence)
                    with open(os.path.join(save_dir, f'{tid_list[j]}_gt.txt'), 'w') as f:
                        f.write(sentence)

                    # calculate BLEU and ROGUE scores
                    bleu_scorer = evaluate.load("bleu")
                    rouge_scorer = evaluate.load("rouge")

                    bleu_result = bleu_scorer.compute(predictions=recon_sentence_list, references=[sentence])
                    rouge_result = rouge_scorer.compute(predictions=recon_sentence_list, references=[sentence])

                    print("BLEU score:", bleu_result)
                    print("ROUGE score:", rouge_result)
                    inversefed.utils.save_to_table(os.path.join(save_dir), name=f'Metrics_Text_{tid_list[j]}', dryrun=args.dryrun, target_id=int(tid_list[j]), **bleu_result, **rouge_result)
                else:
                    torchvision.utils.save_image(ground_truth_den[j:j + 1, ...], os.path.join(save_dir, f'{tid_list[j]}_gt.png'))
            #one row represents psnrs of a batch
        
        skip_update_for_attack_sample = True
        if skip_update_for_attack_sample:
            logger.info("Skipping model update for attack sample.")
        else:
            learning_rate = 0.001
            # after inversion, we need to update the model with input_gradient
            with torch.no_grad():
                for param, best_grad in zip(model.parameters(), input_gradient):
                    param.data -= best_grad * learning_rate
                logger.info("Model updated with best gradient.")

        noisy_input_gradient_norm = torch.norm(torch.stack([g.norm() for g in input_gradient]), 2)
        logger.info('Best noise gradient L2 norm: {}'.format(noisy_input_gradient_norm))
        inversefed.utils.save_to_table(os.path.join(save_dir), name=f'{epoch}_epoch_noise_gradient_norm', dryrun=args.dryrun, noisy_input_gradient_norm=str(noisy_input_gradient_norm.item()), best_noise_loss=best_noise_loss.item(), target_id=target_id, seed=model_seed)

        # load saved model for the epoch if provided to save time, no need to train
        if config['use_saved_models'] == True:
            model_load_dir = config['model_load_dir']
            model.load_state_dict(torch.load(os.path.join(model_load_dir, f"epoch_{epoch}", f"model_epoch_{epoch}.pt")))
        else:
            # simulate FL training, train the model with more instances, then evaluate the model
            model.train()
            logger.info(f"Model name: {config['model']}")
            for i, (inputs, targets) in enumerate(trainloader):
                logger.info(f"Epoch {epoch} batch {i} started")
                optimizer.zero_grad()
                inputs = inputs.to(**setup)
                targets = targets.to(**setup)
                if config['model'] == 'FedCola_TXT':
                    inputs = inputs.long()
                if config['model'] == 'FedCola_IMG_TXT':
                    targets = targets.long()
                targets = targets.long()
                if 'FedCola' in config['model']:
                    if config['model'] == 'FedCola_IMG_TXT':
                        outputs = model([inputs, targets], feat_out=True)
                    elif config['model'] == 'FedCola_IMG':
                        outputs = model([inputs,None])[0]
                    elif config['model'] == 'FedCola_TXT':
                        outputs = model([None, inputs])[1]
                else:
                    outputs = model(inputs)
                if config['model'] == 'FedCola_IMG_TXT':
                    loss = loss_fn(*outputs, torch.tensor([1.0]).to(**setup))
                else:
                    loss, _, _ = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                logger.info(f"loss: {loss.item()} at epoch {epoch} batch {i}")
        
            logger.info(f"Epoch {epoch} training loss: {loss.item()}")

            # save the model checkpoint at each epoch
            save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model {epoch} epoch checkpoint saved at {save_path}")

    # Print final timestamp
    logger.info(datetime.datetime.now().strftime("%A, %d %B %Y %I:%M%p"))
    logger.info('---------------------------------------------------')
    logger.info(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    logger.info("output_dir: {}".format(save_dir))
    logger.info('-------------Job finished.-------------------------')

