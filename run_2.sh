sleep 3600

export CUDA_VISIBLE_DEVICES=2

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_w-conf \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15


python render.py  --skip_train\
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_w-conf \

python metrics.py -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_w-conf

python render.py  --skip_train \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1106_ningbo-block15_8e6init_dis-adc_diffusion

python metrics.py -m outputs/1106_ningbo-block15_8e6init_dis-adc_diffusion


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_not-use-conf_loss-lpips \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_not-use-conf_loss-lpips_early-high-weight \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1107_ningbo-block15_8e6init_dis-adc_diffusion_not-use-conf_loss-lpips_early-higher-weight \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15

python render.py \
                -s CityFusionData/beijing/beijing_block10 \
                -m outputs/1107_beijing-block10_8e6init_dis-adc_diffusion

python metrics.py -m outputs/1107_beijing-block10_8e6init_dis-adc_diffusion


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1108_ningbo-block15_8e6init_dis-adc_diffusion_conf_point-mask-lpips \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1108_ningbo-block15_8e6init_dis-adc_diffusion_point-render-as-l1 \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block15 \
                -m outputs/1108_ningbo-block15_8e6init_dis-adc_diffusion_rand-mask-top-0.1 \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors/ningbo-block15 \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001 \
                --conf_path conf-map/ningbo-blcck15

python train.py --disable_viewer \
                -s CityFusionData/zhangjiang/small_area \
                -m outputs/debug\
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --disable_adc \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001
