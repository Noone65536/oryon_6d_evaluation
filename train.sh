export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false
#export WANDB_MODE=disabled
#export CUDA_VISIBLE_DEVICES=0

python run_train.py exp_name=baseline model_name=ours_encoder