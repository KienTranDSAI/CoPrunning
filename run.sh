python prun_layer.py --layer_idx 1 --sparsity_ratio 0.5 --nsamples 10


python wanda_pruner.py \
      --model openbmb/MiniCPM-2B-sft-bf16 \
      --sparsity_ratio 0.5 \
      --use_recovery \
      --inverse_wanda_update_fraction 0.2 \
      --inverse_wanda_max_relative_update 3.0