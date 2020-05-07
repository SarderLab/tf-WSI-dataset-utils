# tf-WSI-dataset-utils
Input functions for training neural networks on WSI histology images using tf.dataset input pipelines to gather random WSI patches at runtime

This code is setup to randomly prefetch and augment patches from WSIs at training time efficiently on the CPU. This does not require WSI chopping prior to training.  

This should be wrapped in a tf.py_function for real time evaluation at training time.
