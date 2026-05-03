import tensorflow as tf

IMAGE_SIZE = [224, 224]


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def parse_labeled_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, feature_description)

    image = decode_image(example["image"])
    label = tf.cast(example["class"], tf.int32)

    return image, label


def parse_unlabeled_example(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)

    image = decode_image(example["image"])
    image_id = example["id"]

    return image, image_id