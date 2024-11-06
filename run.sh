# 通过--eval来指定是否把ground view载入训练

python train.py -s CityFusionData/ningbo/ningbo_block21 \
                -m outputs/1030_ningbo-block21_w_ground \
                --data_device cpu

python train.py -s data/ningbo_block0 \
                -m outputs/debug \
                --data_device cpu

python train.py -s data/ningbo_block0/flight \
                -m outputs/debug \
                --data_device cpu

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1030_ningbo-block20_wo_ground \
                --data_device cpu \
                --eval

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1031_ningbo-block20_w_ground_test_skybox-0 \
                --data_device cpu

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1031_ningbo-block20_w_ground_test_skybox-1_disable-adc \
                --data_device cpu \
                --use_skybox \
                --disable_adc

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1031_ningbo-block20_w_ground_test_skybox-1_disable-adc_longtrain \
                --data_device cpu \
                --use_skybox \
                --disable_adc

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1101_ningbo-block20_w_ground_test_skybox-1_open-adc_test-sky-num \
                --data_device cpu \
                --use_skybox


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1101_ningbo-block20_w_ground_test_skybox-2_zero-sky-grad_adc-100 \
                --data_device cpu \
                --use_skybox \
                --iterations 50000 \
                --densification_interval 100

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1102_ningbo-block20_w_ground_disadc_ground-extra-start-1w \
                --data_device cpu \
                --use_skybox \
                --iterations 50000 \
                --densification_interval 1000 \
                --disable_adc \
                --train_ground_extra

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1102_ningbo-block20_only-train-ground \
                --data_device cpu \
                --use_skybox \
                --iterations 50000 \
                --densification_interval 1000 \
                --disable_adc \
                --train_ground_extra \
                --ground_extra_start 50000

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1102_ningbo-block20_use6e6pc-init_adc-100_4w5-adc_train-ground \
                --data_device cpu \
                --use_skybox \
                --iterations 80000 \
                --densify_until_iter 45000 \
                --densification_interval 100 \
                --train_ground_extra \
                --opacity_reset_interval 8000


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1102_ningbo-block20_use6e6pc-init_adc-100_4w5-adc_train-diffusion_test \
                --data_device cpu \
                --use_skybox \
                --iterations 80000 \
                --densify_until_iter 45000 \
                --densification_interval 100 \
                --train_ground_extra \
                --opacity_reset_interval 8000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors \
                --eval


python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1102_ningbo-block20_use8e6pc-init_dis-adc_train-ground \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --train_ground_extra

python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1103_ningbo-block20_use8e6pc-init_dis-adc_train-diffusion_test \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors \
                --eval



python train.py --disable_viewer \
                -s CityFusionData/ningbo/ningbo_block20 \
                -m outputs/1103_ningbo-block20_use8e6pc-init_dis-adc_train-diffusion_test_change-position-scaling-lr \
                --data_device cpu \
                --use_skybox \
                --iterations 100000 \
                --use_diffusion_prior \
                --diffusion_path diffusion_priors \
                --disable_adc \
                --eval \
                --position_lr_init 0.000016 \
                --position_lr_final 0.00000016 \
                --scaling_lr 0.001