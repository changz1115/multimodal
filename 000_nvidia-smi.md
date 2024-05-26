
# install nvidia driver

```shell
add-apt-repository ppa:graphics-drivers/ppa
apt update
ubuntu-drivers devices
#systemctl status nvidia-drm
```

```
 ubuntu-drivers devices
ERROR:root:aplay command not found
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00002803sv00007377sd00001500bc03sc00i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-525 - third-party non-free
driver   : nvidia-driver-550-open - third-party non-free
driver   : nvidia-driver-545-open - distro non-free
driver   : nvidia-driver-535-server-open - distro non-free
driver   : nvidia-driver-535 - third-party non-free
driver   : nvidia-driver-545 - third-party non-free
driver   : nvidia-driver-555 - third-party non-free recommended
driver   : nvidia-driver-555-open - third-party non-free
driver   : nvidia-driver-550 - third-party non-free
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-535-open - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

```shell
# driver   : nvidia-driver-555 - third-party non-free recommended
sudo apt install nvidia-driver-555
reboot

# 重启以后
nvidia-smi
lsmod | grep nvidia
```

```shell
# ERROR:root:aplay command not found
sudo apt install alsa-utils
```

```shell
# 验证一下
# ollama 在nvidia 驱动正常安装的情况下，使用GPT
ollama list
ollama run llava

# 在开一个session 就会发现被ollama使用了
nvidia-smi
```

```txt
nvidia-smi
Sat May 25 12:39:42 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.02              Driver Version: 555.42.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     Off |   00000000:01:00.0 Off |                  N/A |
|  0%   43C    P0             25W /  165W |    5314MiB /  16380MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1452      C   ...unners/cuda_v11/ollama_llama_server       5304MiB |
+-----------------------------------------------------------------------------------------+
```
