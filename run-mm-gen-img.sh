goal=YourGoal

ic=1 # number of img clients
tc=1 # number of txt clients
mc=1 # number of img+txt clients
cncntrtn=0.5 # concentration parameter for Dirichlet distribution
c=0.25 # sampling ratio for clients
nt=8 # number of threads for parallel training
b=32 # batch size
root='/home/ubuntu/mnt-disk/repos/censor/data/' # root path of the dataset

python run_mm_rec.py --config configs_biggan_fedcola_coco_img.yml --data_path $root --exp_name FedCola --shared_param attn --share_scope modality  --colearn_param none --compensation --with_aux --aux_trained --seed 1 --multi-task --modalities img txt img+txt img+txt --Ks $ic $tc $mc --test_size -1 --split_type diri --cncntrtn $cncntrtn --model_name mome_small_patch32 --resize 224 --imnorm --algorithm fedavg --eval_type global --eval_every 1 --eval_metrics acc1 --R 5 --C $c --E 5 --B $b --beta1 0 --optimizer AdamW --lr 1e-4 --lr_decay 0.99 --lr_decay_step 1 --criterion CrossEntropyLoss --num_thread $nt --use_bert_tokenizer --pretrained --goal $goal --equal_sampled --eval_batch_size 512 --reduce_samples 1000 --dropout 0
