# tf-WSI-dataset-utils
Input functions for training neural networks on WSI histology images using tf.dataset input pipelines to gather random WSI patches at runtime.

This code was created by [Brendon Lutnick](https://github.com/brendonlutnick)

**Abstract:**<br>
*We have created a custom input pipeline to efficiently extract random patches and labels from whole slide images (WSIs) for input to a neural network. We use a Tensorflow backend to prefetch these patches during network training, avoiding the need for WSI preparation such as WSI chopping prior to training.. This code is setup to randomly prefetch and augment patches from WSIs at training time efficiently on the CPU.*

## Useage

This should be wrapped in a tf.py_function for real time evaluation at training time.

"dataset_util.py" contains the main functions:
1. save_wsi_thumbnail_mask()  - run this to create masks of the tissue regions
2. get_random_wsi_patch()     - function for getting stochastic regions from WSIs | wrapped in a tf.py_function() for use in network
3. get_slide_label()          - an example of how to read slide labels from a master excel sheet

See: "example_usage.py" for more info on the use of these functions.

## Resources

Examples of this code in use are avalable:
* [WSI-CycleGAN](https://github.com/SarderLab/WSI-cycleGAN)
* [WSI-ProGAN](https://github.com/SarderLab/WSI-ProGAN)

## Requirements

This code was developed using Ubuntu Linux, running tensorflow-gpu 1.15

* Python
* OpenSlide
* OpenSlide-tools
* Tensorflow >= 1.14
* OpenCV
* Pandas
* Skimage
* Matplotlib
* Pillow
* Scipy
* Numpy
