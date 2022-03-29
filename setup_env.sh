 pip install  tensorflow-gpu==2.7.0 blobfile  mpi4py tqdm requests pandas
     conda install -y openmpi

WORKSPACE = "/workspace/guided-diffusion/pretrain_model"

cd $WORKSPACE
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_imagenet256.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt