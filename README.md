# CLPA-Clean-Label-Poisoning-Availability-Attacks
This is the implementation of AAAI-22 paper: https://www.aaai.org/AAAI22Papers/AAAI-3872.ZhaoB.pdf


Poisoning attacks are emerging threats to deep neural networks where the adversaries attempt to compromise the models by injecting malicious data points in the clean training data. Poisoning attacks target either the availability or integrity of a model. The availability attack aims to degrade the overall accuracy while the integrity attack causes misclassification only for specific instances without affecting the accuracy of clean data. Although clean-label integrity attacks are proven to be effective in recent studies, the feasibility of clean-label availability attacks remains unclear. This paper, for the first time, proposes a clean-label approach, CLPA, for the poisoning availability attack. We reveal that due to the intrinsic imperfection of classifiers, naturally misclassified inputs can be considered as a special type of poisoned data, which we refer to as "natural poisoned data". We then propose a two-phase generative adversarial net (GAN) based poisoned data generation framework along with a triplet loss function for synthesizing clean-label poisoned samples that locate in a similar distribution as natural poisoned data. The generated poisoned data are plausible to human perception and can also bypass the singular vector decomposition (SVD) based defense. We demonstrate the effectiveness of our approach on CIFAR-10 and ImageNet dataset over a variety type of models.

![image](https://user-images.githubusercontent.com/36553004/157361659-0dda060d-5b6e-4e10-a239-0f45c8f3c49f.png)


# Requirements

PyTorch, version 1.0.1

tqdm, numpy, scipy, and h5py

The ImageNet training set


# Useful links
Training BigGAN is difficult, you can download a pre-trained BigGAN from the official repository of BigGAN at:

https://github.com/ajbrock/BigGAN-PyTorch
