export PYTHONPATH=$PYTHONPATH:
CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node 1 --master_port 15675  ../DiffLoss/scripts/image_train.py  \
                            --data_dir Enter your dict \
                            --gt_dir Enter your dict \
                            --testdata_dir Enter your dict \
                            --testgt_dir Enter your dict \
                            --model_path ../DiffLoss/pretrain/256x256_diffusion_uncond.pt \
                            --attention_resolutions 32,16,8 --class_cond False \
                            --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear \
                            --num_channels 256  --num_head_channels 64 \
                            --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --lr 1e-3 --batch_size 4 \
                            --rescale_learned_sigmas True  \
                            --log_dir ../DiffLoss/logs  \


