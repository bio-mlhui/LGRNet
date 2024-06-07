# Tutorial for Training

## Pretrained Backbone Weights

Train on SUN-SEG

CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online SLURM_NNODES=1 SLURM_NODEID=0 TORCH_NUM_WORKERS=8 python main.py --config_file output/VIS/sunseg/pvt.py --trainer_mode train_attmpt



