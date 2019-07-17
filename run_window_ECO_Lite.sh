

##################################################################### 
# Parameters!
mainFolder="net_runs"
subFolder="ECO_lite_finetune_UCF101_run1"
snap_pref="eco_lite_finetune_UCF101"

train_path="data/ucf101_rgb_train_split_1.txt"
val_path="data/ucf101_rgb_val_split_1.txt"

n2D_model="nll"
n3D_model="nll"

nECO_model="models/ECO_Lite_rgb_model_Kinetics.pth.tar"

frame_prefix="frame_"
#############################################
#--- training hyperparams ---
dataset_name="ucf101"
netType="ECO"
batch_size=1
learning_rate=0.001
num_segments=4
dropout=0.3
iter_size=4
num_workers=4

##################################################################### 
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}/training

echo "Current network folder: "
echo ${mainFolder}/${subFolder}


##################################################################### 
# Find the latest checkpoint of network 
checkpointIter="$(ls ${mainFolder}/${subFolder}/*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
##################################################################### 


echo "${checkpointIter}"

##################################################################### 
# If there is a checkpoint then continue training otherwise train from scratch
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}/${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

python3 -u predict_window.py ${dataset_name} RGB ${train_path} ${val_path}  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 5 --epochs 20 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix ${frame_prefix} --pretrained_parts finetune --no_partialbn  --nesterov "True" --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt    

else
     echo "Checkpoint is missing"
fi

##################################################################### 


