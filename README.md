# tf-WSI-dataset-utils
Input functions for training neural networks on WSI histology images using tf.dataset input pipelines to gather random WSI patches at runtime

**Abstract:**<br>
*We have created a custom input pipeline to efficiently extract random patches and labels from whole slide images (WSIs) for input to a neural network. We use a Tensorflow backend to prefetch these patches during network training, avoiding the need for WSI preparation such as WSI chopping prior to training.. This code is setup to randomly prefetch and augment patches from WSIs at training time efficiently on the CPU.*

## Useage

This should be wrapped in a tf.py_function for real time evaluation at training time.
```
  see "example_usage.py" for more info
  
```
An example of this code used as an input for CycleGAN be found [here](https://github.com/SarderLab/WSI-cycleGAN)

This code was created by [Brendon Lutnick](https://github.com/brendonlutnick)


## Resources

* [ProGAN Paper (NVIDIA research)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
* [Additional material (UBBox)](https://buffalo.box.com/s/8sl2k01svciu1a5qex4g4ziyox39204c)
  * [Pre-trained networks (human kidney biopsies)](https://buffalo.box.com/s/2jtuzqudgs27mvo6izqosib1h979hmtn)
  * [1000 generated images](https://buffalo.box.com/s/ra5gp06kwcadpd9cefnqq0p103utip9x)
  * [Video interpolation (latent walk)](https://buffalo.box.com/s/88cxodei9u65suwxpt30a5pczj7p2d65)


* Linux.
* 64-bit Python 3.6 installation with numpy 1.13.3 or newer.
* One or more high-end NVIDIA GPU with 8GB of DRAM. To use the full resolution network 16GB DRAM is needed.
* NVIDIA driver 391.25 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.1.2 or newer.
* Additional Python packages listed in `requirements-pip.txt`
