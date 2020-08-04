


command_file=`basename "$0"`
gpu=0
model=model_PNv2_SAP-1
# model=model_PNv2_ASAP-1
data=/mnt/sdb/hanwen/processed_pc
num_point=16384
num_frame=3
max_epoch=150
batch_size=2
learning_rate=0.0016
# model_path=log_${model}_labelweights_1.2_new_radius_step_1/model-17.ckpt
model_path=None
log_dir=log_${model}


python train_model.py \
    --gpu $gpu \
    --data $data \
    --model $model \
    --learning_rate $learning_rate \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --max_epoch $max_epoch \
    --batch_size $batch_size \
    #> $log_dir.txt 2>&1 &
