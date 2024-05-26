
# install huggingface api

##

```shell
conda create --name huggingface python=3.10
conda activate huggingface
conda install jupyter
```

```shell
# huggingface_hub
pip install huggingface_hub

# 可以环境登录 或者 python 代码中登录
huggingface-cli login
```

```shell
# dataset
# pip install datasets
# python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"

# audio
# pip install datasets[audio]

# vision
pip install datasets[vision]
pip install Pillow
```