# Usage

## C0 dataset

This dataset was from the [MS-CMRSeg2019 challenge:](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg19/) ;
The 35 labeled data was all cropped into 192\time 192;
This dataset was used as the source data


## LGE dataset
This dataset was from  the [MS-CMRSeg2019 challenge:](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg19/);
The 45 labeled data was all cropped into 192\time 192;
These 45 images was used as the target data;
images of id 1-5 with labels were used as validation dataset, and the other were used as training and test dataset (transductive learning).
(Note: users can also split the 45 LGE images into training and validation dataset for cross-validation.)


## Citation

Please cite these two papers when you use the data for publications.

```
[1] 	Fuping Wu and Xiahai Zhuang: Unsupervised Domain Adaptation with Variational Approximation for Cardiac Segmentation. IEEE Transactions on Medical Imaging, 2021, doi: 10.1109/TMI.2021.3090412..
```
and
```
[2] Xiahai Zhuang: Multivariate mixture model for myocardial segmentation combining multi-source images. IEEE Transactions on Pattern Analysis and Machine Intelligence (T PAMI), vol. 41, no. 12, 2933-2946, Dec 2019. link.
```



# CT-MR dataset
The methods are all almost failed on the preprocessed 3D data, but satisfactory on 2D slices. Hence, we provided another CT-MR dataset for users to validate methods. Users can find this dataset in 'https://github.com/FupingWu90/CT_MR_2D_Dataset_DA' : [CT-MR-Dataset](https://github.com/FupingWu90/CT_MR_2D_Dataset_DA). This dataset contains 2D slices selected around the center of heart, and the target is to segment myocardium (Myo) and left ventricle (LV).

Please cite the below paper when you use the data for publications.
```
F. Wu and X. Zhuang, "CF Distance: A New Domain Discrepancy Metric and Application to Explicit Domain Adaptation for Cross-Modality Cardiac Image Segmentation," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2020.3016144.
```
and 

```
2.	Fuping Wu and Xiahai Zhuang: Unsupervised Domain Adaptation with Variational Approximation for Cardiac Segmentation. IEEE Transactions on Medical Imaging, 2021, doi: 10.1109/TMI.2021.3090412.
```
