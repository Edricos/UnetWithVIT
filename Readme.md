## 环境配置 
### wsl2的ubuntu22.04子系统下，搭建cuda深度学习环境

### 1. cudaToolKit == 11.8

```
sudo apt install cuda-toolkit-11-8
```
```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
```
```
#[.bashrc]
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
```
source ~/.bashrc
```
```
nvcc -V
```
wsl2安装cudaToolKit教程：https://blog.csdn.net/weixin_42062018/article/details/125777391


### 2. cuDNN == 8.9.0.131


```
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
```
```
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
```
```
sudo apt-get update
```
```
sudo apt-get install libcudnn8=8.9.0.131-1+cuda11.8
```
```
sudo apt-get install libcudnn8-dev=8.9.0.131-1+cuda11.8
```
```
sudo apt-get install libcudnn8-samples=8.9.0.131-1+cuda11.8
```
cuDNN官网教程：https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb   
cuDNN下载：https://developer.nvidia.com/rdp/cudnn-archive

### 3. python == 3.9

```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```


### others
Cython相关：
```
sudo apt-get install python3.9-dev
```







