import tensorflow as tf
from dataset_util import save_wsi_thumbnail_mask, get_random_wsi_patch

def input_setup(batch_size=32, dataset_dir='./path_to_folder_with_WSIs', wsi_ext='.svs'):

    '''
    This function sets up the input data pipeline
    takes input at tfrecord files
    '''

    #######################################################################

    # data A input pipeline
    wsi_paths = list(glob('{}*{}'.format(dataset_dir, wsi_ext)))
    wsi_paths = [str(path) for path in wsi_paths]

    # create WSI thumbnail masks
    for wsi_path in wsi_paths:
        save_wsi_thumbnail_mask(wsi_path)

    random.shuffle(wsi_paths)
    image_count = len(wsi_paths)
    print('found: {} images from WSI dataset'.format(image_count))

    # setup tf dataset using py_function
    path_ds = tf.data.Dataset.from_tensor_slices(wsi_paths)
    path_ds = path_ds.repeat(100)
    wsi_dataset = path_ds.map(lambda filename: tf.py_function(get_random_wsi_region, [filename], tf.float32), num_parallel_calls=40)
    wsi_dataset = wsi_dataset.shuffle(100)
    wsi_dataset = wsi_dataset.batch(batch_size=batch_size, drop_remainder=True)
    wsi_dataset = wsi_dataset.prefetch(buffer_size=1) # <-- very important for efficency
    iterator = tf.data.Iterator.from_structure(wsi_dataset.output_types, wsi_dataset.output_shapes)
    training_init_op = iterator.make_initializer(wsi_dataset)
    get_input = iterator.get_next()

    '''
    to use run:
        "sess.run([training_init_op])"

    "get_input"     will return the next batch of images when any part of the
                    TF-graph that relies on it is called

    for more info see:
            https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset

    '''
