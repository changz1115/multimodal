
## 本地搭建jupyterlab

```shell
conda create --name huggingface python=3.10
conda activate huggingface
pip install jupyter jupyterlab -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com
pip install huggingface_hub torch -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com
jupyter-lab --allow-root
```

```shell
# 允许远程访问
jupyter notebook --generate-config
vim ~/.jupyter/jupyter_notebook_config.py
```

```python
# 输入自己的密码并得到hash值
from jupyter_server.auth import passwd
passwd()
```

```python
# 允许远程访问
# 不使用本地浏览器打开
# 允许所有IP访问
# 配置密码 上面的hash值 或者 下面这个sample 也就是 212121
c.NotebookApp.allow_remote_access = True
c.NotebookApp.open_browser = False
c.NotebookApp.ip='*'
c.NotebookApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$MAWO98xW1erTnY8AcRiF3Q$FEz7AZpG/jryQGEmaR70oB9Hs2+alUrSn8+sPUEgVsw'
```

```shell
# 带IP启动
jupyter-lab --allow-root --ip=10.1.147.25
jupyter-lab --allow-root --ip=10.1.147.52
```