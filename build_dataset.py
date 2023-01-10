import os
import sys
import pickle
import shutil
import requests
import subprocess

import numpy as np


"""
Write raw numpy data of CIFAR-10 in a more readable way for further usage
Example: python build_dataset.py untared_cifar_folder new_folder
"""


CIFAR10_LINK = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_FILE = 'cifar-10-python.tar.gz'
CIFAR10_FOLDER = 'cifar-10-batches-py'


def read_cifar_chunk(filepath: str):
    """
    Return: (Images, Labels)
    """
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    return data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(data[b'labels'], dtype=np.uint8)


if __name__ == '__main__':
    output_folder = sys.argv[1]

    print("Downloading...")

    if not os.path.exists(CIFAR10_FILE):
        with open(CIFAR10_FILE, 'wb') as f:

            response = requests.get(CIFAR10_LINK, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(80 * dl / total_length)
                    sys.stdout.write(f"\r[{'#' * done}{' ' * (80 - done)}]")
                    sys.stdout.flush()

    print("Extracting...")
    subprocess.run(['tar', '-xf', CIFAR10_FILE])

    images_train = []
    labels_train = []

    print("Reading CIFAR-10 Data...")

    train_batches = [f'data_batch_{i}' for i in range(1, 5 + 1)]
    for batch_file in train_batches:
        images, labels = read_cifar_chunk(os.path.join(CIFAR10_FOLDER, batch_file))
        
        images_train.append(images)
        labels_train.append(labels)

    images_train = np.concatenate(images_train, dtype=np.uint8)
    labels_train = np.concatenate(labels_train, dtype=np.uint8)

    images_test, labels_test = read_cifar_chunk(os.path.join(CIFAR10_FOLDER, 'test_batch'))

    with open(os.path.join(CIFAR10_FOLDER, 'batches.meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    
    label_names = [name.decode() for name in meta[b'label_names']]

    shutil.rmtree(CIFAR10_FOLDER)

    print("Writing Processed Data...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with \
        open(os.path.join(output_folder, 'train.pkl'), 'wb') as f_train, \
        open(os.path.join(output_folder, 'test.pkl'), 'wb') as f_test, \
        open(os.path.join(output_folder, 'labels.pkl'), 'wb') as f_labels:
        
        pickle.dump({ 'images': images_train, 'labels': labels_train }, f_train)
        pickle.dump({ 'images': images_test, 'labels': labels_test }, f_test)
        pickle.dump({ i: label for i, label in enumerate(label_names) }, f_labels)
