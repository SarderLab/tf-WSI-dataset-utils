# tf-WSI-dataset-utils
Input functions for training neural networks on WSI histology images using tf.dataset input pipelines to gather random WSI patches at runtime
This code was created by [Brendon Lutnick](https://github.com/brendonlutnick)

**Abstract:**<br>
*We have created a custom input pipeline to efficiently extract random patches and labels from whole slide images (WSIs) for input to a neural network. We use a Tensorflow backend to prefetch these patches during network training, avoiding the need for WSI preparation such as WSI chopping prior to training.. This code is setup to randomly prefetch and augment patches from WSIs at training time efficiently on the CPU.*

## Useage

This should be wrapped in a tf.py_function for real time evaluation at training time.
```
  See: "example_usage.py" for more info.

```

## Resources

Examples of this code in use are avalable:
* [WSI-CycleGAN](https://github.com/SarderLab/WSI-cycleGAN)
* [WSI-ProGAN](https://github.com/SarderLab/WSI-ProGAN)

## Requirements

* Python
* OpenSlide
* OpenSlide-tools
* Tensorflow
* OpenCV
* Pandas
* Skimage
* Matplotlib
* Pillow
* Scipy
* Numpy
