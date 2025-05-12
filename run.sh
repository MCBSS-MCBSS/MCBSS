python Train_Cifar.py  --device 0 --batch_size 128  --lr 0.01  --num_epochs 200  --seed 2025\
                       --lambd 1  --barycenter_number 1  --ulb_loss_ratio 1.0 --p_cutoff 0.8  --Nb 128\
                       --output_file  cifar10-imb01-noise02.txt \
                       --dataset cifar10  --closeset_ratio 0.2  --noise_type unif  --imb_factor 0.1