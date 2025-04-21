CUDA_VISIBLE_DEVICES=0 python evaluate_raihanet_final.py \
--dataset_root ../iHarmony4Resized/ \
--batch_size 1 \
--model raihanet \
--netG rhnet \
--name 4e4_40_20 \
--dataset_mode ragiharmony4 \
--jsonl_path cont07_final_random1 \
--continue_train \
--epoch 60 \
--saveimg 0 \
--att 'nodvt' \
--use_MGD 1 \
--addpos 1 \
--dvt 1 &


CUDA_VISIBLE_DEVICES=0 python evaluate_raihanet_final.py \
--dataset_root ../iHarmony4Resized/ \
--batch_size 1 \
--model raihanet \
--netG rhnet \
--name 4e4_40_20 \
--dataset_mode ragiharmony4 \
--jsonl_path cont07_final_random2 \
--continue_train \
--epoch 60 \
--saveimg 0 \
--att 'nodvt' \
--use_MGD 1 \
--addpos 1 \
--dvt 1 &


CUDA_VISIBLE_DEVICES=0 python evaluate_raihanet_final.py \
--dataset_root ../iHarmony4Resized/ \
--batch_size 1 \
--model raihanet \
--netG rhnet \
--name 4e4_40_20 \
--dataset_mode ragiharmony4 \
--jsonl_path cont07_final_random3 \
--continue_train \
--epoch 60 \
--saveimg 0 \
--att 'nodvt' \
--use_MGD 1 \
--addpos 1 \
--dvt 1 &


CUDA_VISIBLE_DEVICES=0 python evaluate_raihanet_final.py \
--dataset_root ../iHarmony4Resized/ \
--batch_size 1 \
--model raihanet \
--netG rhnet \
--name 4e4_40_20 \
--dataset_mode ragiharmony4 \
--jsonl_path cont07_final_random4 \
--continue_train \
--epoch 60 \
--saveimg 0 \
--att 'nodvt' \
--use_MGD 1 \
--addpos 1 \
--dvt 1 &


CUDA_VISIBLE_DEVICES=0 python evaluate_raihanet_final.py \
--dataset_root ../iHarmony4Resized/ \
--batch_size 1 \
--model raihanet \
--netG rhnet \
--name 4e4_40_20 \
--dataset_mode ragiharmony4 \
--jsonl_path cont07_final_random5 \
--continue_train \
--epoch 60 \
--saveimg 0 \
--att 'nodvt' \
--use_MGD 1 \
--addpos 1 \
--dvt 1 &
