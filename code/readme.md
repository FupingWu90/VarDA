
# Usage
article :[Unsupervised Domain Adaptation with Variational Approximation for Cardiac Segmentation](https://arxiv.org/abs/2106.08752v1)

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


## Citation

Please cite the paper if ypu found this repository useful:

```
@misc{wu2021unsupervised,
      title={Unsupervised Domain Adaptation with Variational Approximation for Cardiac Segmentation}, 
      author={Fuping Wu and Xiahai Zhuang},
      year={2021},
      eprint={2106.08752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
