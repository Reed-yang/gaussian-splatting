sleep 15000

export CUDA_VISIBLE_DEVICES=2

python train.py --disable_viewer \
                -s CityFusionData/beijing/beijing_block10 \
                -m outputs/1107_beijing-block10_8e6init_dis-adc_diffusion \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/beijing-block10 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001

python render.py                 -s CityFusionData/beijing/beijing_block10 \

python render.py \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_not-use-conf_loss-lpips_early-higher-weight

python metrics_all.py -m outputs/1107_beijing-block10_8e6init_dis-adc_diffusion