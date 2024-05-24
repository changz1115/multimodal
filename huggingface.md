
# install huggingface api

##

```shell
conda create --name huggingface python=3.10
conda activate huggingface
```

```shell
# huggingface_hub
pip install huggingface_hub
huggingface-cli login
```

```shell
# dataset
pip install datasets
python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"

# audio
pip install datasets[audio]

# vision
pip install datasets[vision]
```