
#Usage

## Change your path

1. edit the following pathway in the 'Adversarial_DA_seg_trainer.py':

* prefix='~/experiments/loss_tSNE'
- dataset_dir = '~/Dataset/Patch192'
* TestDir=dataset_dir+'/LGE_Vali/'


## tune the hyperparameters

1. edit the following hyperparameters in the 'Adversarial_DA_seg_trainer.py':

* Alpha=  (1e0; weight of target vae)
- Beta=   (1e-3; weight of distance)
- PredLamda=    (1e3; weight of prediction loss in vae)
* KLDLamda=   (1.0;weight of kld loss in vae)


## prepare dataset

1. We first cropped all cardiac imgaes into 192\time 192, the LGE and C0 dataset can be found in the file 'Dataset'


## run
1. python Adversarial_DA_seg_trainer.py