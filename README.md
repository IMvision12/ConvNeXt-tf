# ConvNeXt and ConvNeXtV2

This repository is about an implementation of the research paper "A ConvNet of the 2020s" using `Tensorflow` published by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, Facebook AI Research (FAIR), UC Berkeley on the 10 January 2022.

<p align="center">
<img src="https://user-images.githubusercontent.com/8370623/180626875-fe958128-6102-4f01-9ca4-e3a30c3148f9.png" width=100% height=100% 
class="center">
</p>

# All model configurations

<p align="center">
<img src="https://github.com/IMvision12/ConvNeXt-tf/blob/main/img/configurations.PNG"
class="center">
</p>

## Results
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs |
|:---:|:---:|:---:|:---:| :---:|
| ConvNeXt-T | 224x224 | 82.1 | 28M | 4.5G |
| ConvNeXt-S | 224x224 | 83.1 | 50M | 8.7G |
| ConvNeXt-B | 224x224 | 83.8 | 89M | 15.4G |
| ConvNeXt-B | 384x384 | 85.1 | 89M | 45.0G |
| ConvNeXt-L | 224x224 | 84.3 | 198M | 34.4G |
| ConvNeXt-L | 384x384 | 85.5 | 198M | 101.0G |

### ImageNet-22K trained models

| name | resolution |acc@1 | #params | FLOPs |
|:---:|:---:|:---:|:---:| :---:|
| ConvNeXt-T | 224x224 | 82.9 | 29M | 4.5G |
| ConvNeXt-T | 384x384 | 84.1 | 29M | 13.1G |
| ConvNeXt-S | 224x224 | 84.6 | 50M | 8.7G |
| ConvNeXt-S | 384x384 | 85.8 | 50M | 25.5G |
| ConvNeXt-B | 224x224 | 85.8 | 89M | 15.4G |
| ConvNeXt-B | 384x384 | 86.8 | 89M | 47.0G |
| ConvNeXt-L | 224x224 | 86.6 | 198M | 34.4G |
| ConvNeXt-L | 384x384 | 87.5 | 198M | 101.0G |
| ConvNeXt-XL | 224x224 | 87.0 | 350M | 60.9G |
| ConvNeXt-XL | 384x384 | 87.8 | 350M | 179.0G |

# References

[1] ConvNeXt paper: https://arxiv.org/abs/2201.03545

[2] Official ConvNeXt code: https://github.com/facebookresearch/ConvNeXt
