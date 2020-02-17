CUDA_VISIBLE_DEVICES=1 python main.py --data ./data/FB15k-237/ \
--epochs_gat 3000 --epochs_conv 150 --weight_decay_gat 0.00001 \
--get_2hop True --partial_2hop True --batch_size_gat 272115 --margin 1 \
--out_channels 50 --drop_conv 0.3 \
--output_folder /scratch/cse/dual/cs5150286/col868/tests/distmult/checkpoints/fb/out/ | tee fb15_out_distmult.txt