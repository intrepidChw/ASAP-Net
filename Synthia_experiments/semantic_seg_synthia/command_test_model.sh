


command_file=`basename "$0"`
gpu=1
# model=model_PNv2_SAP-1
model=model_PNv2_ASAP-1
data=/mnt/sdb/hanwen/processed_pc
num_point=16384
num_frame=3
batch_size=32
model_path=log_tpc_atten_0.827/model-148.ckpt
# model_path=None
log_dir=log_${model}_test
save=True
save_dir=results_${model}


python test_model.py \
    --gpu $gpu \
    --data $data \
    --model $model \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --save $save \
    --save_dir $save_dir \
    #> $log_dir.txt 2>&1 &
