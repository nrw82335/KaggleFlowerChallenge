from pathlib import Path
import tensorflow as tf

from src.tfrecord_parser import parse_labeled_example, parse_unlabeled_example

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 8
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "tfrecords-jpeg-224x224"


def get_file_lists(data_dir=DATA_DIR):
    """
    Return lists of TFRecord file paths for train, validation, and test.
    """
    data_path = Path(data_dir)

    train_files = sorted(str(p) for p in (data_path / "train").glob("*.tfrec"))
    val_files = sorted(str(p) for p in (data_path / "val").glob("*.tfrec"))
    test_files = sorted(str(p) for p in (data_path / "test").glob("*.tfrec"))

    return train_files, val_files, test_files


def make_labeled_dataset(file_list, batch_size=BATCH_SIZE, shuffle=False):
    """
    Build a dataset for train or validation data.
    """
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(parse_labeled_example, num_parallel_calls=AUTO)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


def make_unlabeled_dataset(file_list, batch_size=BATCH_SIZE):
    """
    Build a dataset for test data.
    """
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(parse_unlabeled_example, num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


def load_datasets(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Return train, validation, and test datasets.
    """
    train_files, val_files, test_files = get_file_lists(data_dir)

    train_ds = make_labeled_dataset(train_files, batch_size=batch_size, shuffle=True)
    val_ds = make_labeled_dataset(val_files, batch_size=batch_size, shuffle=False)
    test_ds = make_unlabeled_dataset(test_files, batch_size=batch_size)

    return train_ds, val_ds, test_ds