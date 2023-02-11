### To-do
- [x] Add convnextv1 and v2 pytorch
- [x] convert pytorch to tensorflow
- [ ] weight conversion

# ConvNeXt and ConvNeXtV2

This repository is about an implementation of the research paper "A ConvNet of the 2020s" and "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" using `Tensorflow`.

ConvNeXtV1 : ConvNeXt, a pure ConvNet model constructed entirely from standard ConvNet modules. ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

ConvNeXtV2: The paper proposed a fully convolutional masked autoencoder framework (FCMAE) and a new Global Response Normalization (GRN) layer to original ConvNeXtV1 model to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks.

<p align="center">
<img src="https://github.com/IMvision12/ConvNeXt-tf/blob/main/img/model_scaling.png" width=50% height=50%
class="right">
</p>

### ConvNeXtV1 and ConvNeXtV2 block design:

<p align="center">
<img src="https://github.com/IMvision12/ConvNeXt-tf/blob/main/img/Capture.PNG" width=40% height=40%
class="right">
</p>

# References

[1] ConvNeXt paper: https://arxiv.org/abs/2201.03545

[2] ConvNeXtV2 paper: https://arxiv.org/abs/2301.00808

[3] Official ConvNeXt code: https://github.com/facebookresearch/ConvNeXt

[4] Official ConvNeXtV2 code: https://github.com/facebookresearch/ConvNeXt-V2
