# nlp_hw3
use transformer infer.py

cd ChineseNMT

python infer.py --interactive --data_size 100k \
  --ckpt /mnt/c/Users/sysu/Desktop/nlp_hw3/ChineseNMT/experiment/100k_rel_rms_adam_smooth/model_100k_rel_rms_adam_smooth.pth \
  --position_encoding relative \
  --norm_type rmsnorm
 
 use rnn infer.py

 cd rnn_nmt_clean

 python infer.py --dataset 100k --interactive --attention additive --ckpt /mnt/c/Users/sysu/Desktop/nlp_hw3/rnn_nmt_clean/checkpoints_100k/model_multiplicative_50.pt --attention multiplicative