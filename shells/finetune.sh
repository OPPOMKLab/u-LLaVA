pip install -i https://mirrors.aliyun.com/pypi/simple/ -r ./shells/requirements.txt

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    train_ullava.py \
    --cfg_path './configs/train/ullava_stage2.yaml'