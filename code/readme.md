
# Usage
article :[Unsupervised Domain Adaptation with Variational Approximation for Cardiac Segmentation](https://arxiv.org/abs/2106.08752v1)
or [paper](https://ieeexplore.ieee.org/document/9459711)

## Change your path

1. edit the following pathway in the 'Adversarial_DA_seg_trainer.py':

* prefix='~/experiments/loss_tSNE'
- dataset_dir = '~/Dataset/Patch192'
* TestDir=dataset_dir+'/LGE_Vali/'


## tune the hyperparameters

1. edit the following hyperparameters in the 'Adversarial_DA_seg_trainer.py':

* Alpha=  (1e0; weight of target vae; change according to your tasks)
- Beta=   (1e-3; weight of distance; change according to your tasks)
- PredLamda=    (1e3; weight of prediction loss in vae; change according to your tasks)
* KLDLamda=   (1.0;weight of kld loss in vae; change according to your tasks)


## prepare dataset

1. We first cropped all cardiac imgaes into 192\time 192, the LGE and C0 dataset can be found in the file 'Dataset'


## run
1. python Adversarial_DA_seg_trainer.py
2. note: if using the regularization term only, select the code in 'general' file, if using the slice position infomation to constrain the network, select the code in 'add_position_info' file.


## Citation

Please cite the paper if ypu found this repository useful:

'''

@ARTICLE{9459711,
  author={Wu, Fuping and Zhuang, Xiahai},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Unsupervised Domain Adaptation with Variational Approximation for Cardiac Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMI.2021.3090412}}
 
 '''

