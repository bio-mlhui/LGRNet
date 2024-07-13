# Train & Eval on SUN-SEG

`
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=disabled TORCH_NUM_WORKERS=8 python main.py --config_file output/VIS/sunseg/pvt.py --trainer_mode train_attmpt
`

You can change the number of gpu, wandb mode, torch workers according to your hardware.




