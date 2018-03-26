# GAN
PyTorch Implementations of Generative Adversarial Nets: GAN, DCGAN, WGAN, CGAN, InfoGAN

# DCGAN 
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)[2015]  

**Architecture guidelines for stable Deep Convolutional GANs**  

* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).  
* Use batchnorm in both the generator and the discriminator  
* Remove fully connected hidden layers for deeper architectures. Just use average pooling at the end.  
* Use ReLU activation in generator for all layers except for the output, which uses Tanh.  
* Use LeakyReLU activation in the discriminator for all layers.  

 ***************
