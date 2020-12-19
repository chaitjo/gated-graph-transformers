#!/bin/bash
code=main_mol.py

dataset=ogbg-molhiv
expt=submission

gnn=gated-gcn
num_layer=10
emb_dim=256
pos_enc_dim=20
pooling='mean'

batch_size=256
epochs=50
lr=1e-3

seed0=0 
seed1=1
seed2=2 
seed3=3
seed4=4 
seed5=5
seed6=6 
seed7=7
seed8=8 
seed9=9

tmux new -s $expt -d
tmux send-keys "conda activate ogb" C-m
tmux send-keys "
python $code --dataset $dataset --device 0 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed0 &
python $code --dataset $dataset --device 1 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed1 &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --device 0 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed2 &
python $code --dataset $dataset --device 1 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed3 &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --device 0 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed4 &
python $code --dataset $dataset --device 1 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed5 &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --device 0 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed6 &
python $code --dataset $dataset --device 1 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed7 &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --device 0 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed8 &
python $code --dataset $dataset --device 1 --expt_name $expt --gnn $gnn --num_layer $num_layer --emb_dim $emb_dim --pos_enc_dim $pos_enc_dim --pooling $pooling --batch_size $batch_size --epochs $epochs --lr $lr --seed $seed9 &
wait" C-m
tmux send-keys "tmux kill-session -t $expt" C-m
