pip install -i https://mirrors.aliyun.com/pypi/simple/ -r ./shells/requirements.txt

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    train_ullava_core.py \
    --cfg_path './configs/train/ullava_core_stage1.yaml'