
##

```shell
conda create --name huggingface python=3.10
conda activate huggingface
conda install jupyter
conda install jupyterlab
```

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[torch,metrics]
```