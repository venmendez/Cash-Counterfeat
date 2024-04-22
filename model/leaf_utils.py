import os
import shutil
import random
import datetime
import time
import pytz
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from functools import wraps
from typing import List, Tuple, Callable
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential


def stratified_train_test_split(X, y, test_size = 0.2, seed = 42):
    # Split the data into training and validation sets with stratified sampling
    train_files, valid_files, train_labels, valid_labels = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train = (train_files, train_labels)
    validation = (valid_files, valid_labels)

    return train, validation


def get_preprocessed_dataset(train: Tuple[List[str], List[str]], 
                            validation: Tuple[List[str], List[str]], 
                            preprocessor: Callable,
                            batch_size: int):
    """
    Creates preprocessed Tensorflow Datasets for train and validation 
    
    Args:
        train (tuple): [list of train file paths, labels]
        validation (tuple): [list of validation file paths, labels]
        preprocessor (Callable): preprocessor function
        batch_size (int): data batch size

    Returns:
        tuple: A tuple containing two preprocessed TensorFlow Datasets 
        (training and validation)
    """
    train_files, train_labels = train
    valid_files, valid_labels = validation

    # Create a tf.data.Dataset for training data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_dataset = train_dataset.map(preprocessor, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create a tf.data.Dataset for validation data
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels))
    valid_dataset = valid_dataset.map(preprocessor, 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, valid_dataset


def get_4_part_seq_model(model, augment_layer, input_shape, n_classes):
    """
    Returns a sequential model with 4 parts:
        - the augmentation layer
        - the (frozen) base model with `include_top=False`
        - GlobalAveragePooling2D
        - N-neuron Dense layer with softmax activation
    """
    # Load pre-trained model with imagenet weights and without top layers
    base_model = model(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model
    base_model.trainable = False

    seq_model = Sequential([
        augment_layer,
        base_model,
        GlobalAveragePooling2D(),
        Dense(n_classes, activation='softmax')
    ])

    return seq_model


def timer(func: Callable):
    """
    A decorator that prints how long a function took to run
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_total = time.time() - t_start

        hours = int(t_total // 3600)
        minutes = int((t_total % 3600) // 60)
        seconds = t_total % 60

        print(f"{func.__name__} ran for {hours:02d}:{minutes:02d}:{seconds:02f}s")
        return result
    return wrapper


def get_model_train_with_storing(file_prefix: str, run_dir: str) -> Callable:
    """
    Return a customizable train function with ModelCheckpoint callback.
    The model is serialized to JSON after training

    Args:
        file_prefix (str): filename for the hdf5 file
        run_dir (str): path to current run folder
    Returns:
        Callable: a customized model train function
    """
    # save weights
    weights_path = f"{run_dir}/weights/{file_prefix}.hdf5"
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    callbacks_list = [checkpoint]

    @timer
    def train_model(model, train_data, valid_data, optimizer, epochs=100):
        # Compile the model
        # optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(train_data, 
                            validation_data=valid_data, 
                            callbacks=callbacks_list, 
                            epochs=epochs)
        serialize_model(model)
        return model, history

    def serialize_model(model):
        """
        Input must be the model object after training (to avoid possible errors)
        """
        # serialize model to JSON
        model_path = f'{run_dir}/{file_prefix}_model.json'
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)

    return train_model


def get_model_train_callable():
    def train_model(model, train_data, valid_data, lr=0.001, epochs=100):
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(train_data, 
                            validation_data=valid_data,  
                            epochs=epochs)
        return model, history


def create_run_folder(dest: str) -> str:
    """
    Creates a folder for storing the files for a specific run/session

    Args:
        dest (str): path to destination folder where run folder will be created
    Returns:
        str: path to the run folder
    """
    timezone = pytz.timezone('Asia/Manila')
    runs_dir = f"{dest}/ensemble_runs"

    # Get the current date and time
    current_datetime = datetime.datetime.now(timezone)

    # Format the date and time as a string (you can customize the format as needed)
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Create a folder with the formatted datetime as its name
    folder_path = f"{runs_dir}/run_{formatted_datetime}"
    os.makedirs(folder_path)
    return folder_path

@timer
def create_image_preprocessor(image_size: Tuple[int, int], unique_labels: list) -> Callable:
    """
    Creates a customizable function for decoding and preprocessing an image
    based on the desired image dimensions
    """
    
    # Create a mapping from string labels to numerical indices
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(unique_labels, tf.range(len(unique_labels), dtype=tf.int64)),
        -1)
    
    def load_and_preprocess_image(file_path, label):
        # Load and decode the image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Resize and normalize the image
        img = tf.image.resize(img, image_size)
        img = img / 255.0  # Normalize pixel values to [0, 1]

        # Convert string label to numerical index using the lookup table
        label_index = table.lookup(label)

        # Apply one-hot encoding to the label
        one_hot_label = tf.one_hot(label_index, depth=len(unique_labels))

        return img, one_hot_label

    return load_and_preprocess_image


def get_filepaths_labels(src: str) -> Tuple[List[str], List[str]]:
    """
    Returns two lists: file paths and the corresponding labels
    """
    # Get the list of class labels (subfolder names)
    class_labels = sorted(os.listdir(src))

    # Create a list of file paths and corresponding labels
    file_paths = []
    labels = []
    for label in class_labels:
        class_folder = os.path.join(src, label)
        # Get list of all files in the class folder
        class_files = [os.path.join(class_folder, file) for file in os.listdir(class_folder)]
        # Append file paths and labels
        file_paths.extend(class_files)
        labels.extend([label] * len(class_files))

    return file_paths, labels

@timer
def create_dataset(src: str,
                    dest: str, 
                    N_samples: int, 
                    selected_classes: List[str], 
                    size: Tuple[int, int]) -> None:
    """
    Creates two folders: a dataset folder with subfolders for the specified classes each
    containing N randomly selected samples; A holdout folder containing the remaining samples 
    for each class
    Args:
        src (str): the path to the image source parent directory
        dest (str): the path where the dataset and holdout folders will be created.
        N_samples (str): the number of samples for each class.
        selected_classes (list): a list of class names to include in the datasets
    """

    source = f"{src}/dataset"
    destination = f"{dest}/ensemble_dataset"
    holdout = f"{dest}/holdout"

    os.makedirs(destination, exist_ok=True)
    os.makedirs(holdout, exist_ok=True)

    # Check if the selected subdirectories exist
    try:
        # Iterate through subfolders in source
        for class_folder in selected_classes:
            class_folder_source = os.path.join(source, class_folder)
            class_folder_dest = os.path.join(destination, class_folder)
            class_folder_holdout = os.path.join(holdout, class_folder)

            # Ensure corresponding subfolders exist in destination and holdout
            os.makedirs(class_folder_dest, exist_ok=True)
            os.makedirs(class_folder_holdout, exist_ok=True)

            # List all images in the class folder
            images = os.listdir(class_folder_source)

            # Randomly select 100 images
            selected_images = random.sample(images, N_samples)

            # Copy selected images to destination
            for image in selected_images:
                src_path = os.path.join(class_folder_source, image)
                dst_path = os.path.join(class_folder_dest, image)
                # shutil.copy(src_path, dst_path)
                with Image.open(src_path) as img:
                    resized_img = img.resize(size)
                    resized_img.save(dst_path)

            # Copy remaining images to holdout
            for image in images:
                if image not in selected_images:
                    src_path = os.path.join(class_folder_source, image)
                    dst_path = os.path.join(class_folder_holdout, image)
                    # shutil.copy(src_path, dst_path)
                    with Image.open(src_path) as img:
                        resized_img = img.resize(size)
                        resized_img.save(dst_path)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    assert_equal_samples(destination, N_samples)


def assert_equal_samples(data_dir: str, N_samples: int) -> None:
    """
    Sanity check to make sure all subfolders contain N number of files each
    """
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)

    if os.path.isdir(subfolder_path):
        num_files = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
        assert num_files == N_samples, f"Subfolder '{subfolder}' does not contain exactly {N_samples} files. It contains {num_files} files."


def plot_history_metrics(history):
    """
    Plot the accuracy and loss for the training and validation data
    """
    fig = plt.figure(figsize=(15,8))

    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'b', label = 'acc')
    plt.plot(history.history['val_accuracy'], 'r', label = 'val_acc')
    plt.title("Train Accuracy vs Validation Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(222)
    plt.plot(history.history['loss'], 'b', label = 'loss')
    plt.plot(history.history['val_loss'], 'r', label = 'val_loss')
    plt.title("Train Loss vs Validation Loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()


def validation_report_callables(model, validation_data, class_dict):
    """
    Evaluates the model against the validation data and returns two functions:
        One for printing the classification report, and another for
        displaying the confusion matrix
    """

    true_labels = []
    predicted_labels = []

    for images, labels in validation_data:
        true_labels.extend(tf.argmax(labels, axis=1).numpy())  # Get true labels
        predictions = model.predict(images)
        predicted_labels.extend(tf.argmax(predictions, axis=1).numpy())  # Get predicted labels

    def print_classification_report():
        report = classification_report(true_labels, predicted_labels, target_names=class_dict.values())
        print(report)

    def plot_confusion_matrix():
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize = (9,7))
        sns.heatmap(cm, annot=True, xticklabels=class_dict.values(), yticklabels=class_dict.values())
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.title('Confusion Matrix')
        plt.tight_layout(pad=5.0)

    return print_classification_report, plot_confusion_matrix