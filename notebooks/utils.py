import h5py
import numpy as np
import tensorflow as tf
import pathlib
import pickle
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

RANDOM_SEED=42

def get_data_paths_ordered(data_path:str)->pathlib.Path:
    pattern = r'^\d{4}_\d{2}_\d{2}$'
    data_dir = pathlib.Path(data_path)
    paths = sorted([p for p in data_dir.iterdir() if p.is_dir() and bool(re.match(pattern, p.stem)) ])
    return paths
    
@tf.function(jit_compile=True)
def smooth(curr_input, max_val):
    """Smooths the input tensor by dividing by max_val if any element exceeds it."""
    def body(arr, m):
        # Divide elements exceeding m by m
        return tf.where(arr > m, arr / m, arr), m

    max_val_casted = tf.cast(max_val, tf.float64)
    # Cast the result of reduce_max to float32 before comparison
    condition = lambda x, m: tf.cast(tf.math.reduce_max(x), tf.float64) > m
    curr_input_casted=tf.cast(curr_input, tf.float64)
    arr, _ = tf.while_loop(
        condition,
        body,
        loop_vars=[curr_input_casted, max_val_casted],
        parallel_iterations=20,
        maximum_iterations=5
    )

    return arr

@tf.function(jit_compile=True)
def standartize(curr_input):
    mean = tf.math.reduce_mean(curr_input)
    std = tf.math.reduce_std(curr_input)
    return tf.math.divide_no_nan(curr_input - mean, std)

# Define the CNN-LSTM model creation function
def cnn_lstm(n_dataset_features, num_of_classes, features_per_layer, strides, pool_size, units, dropout):
    inp = tf.keras.layers.Input((n_dataset_features, 1))
    x = tf.keras.layers.Conv1D(features_per_layer, kernel_size=16, strides=strides, activation='relu')(inp)
    x = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')(x)
    x = tf.keras.layers.Conv1D(features_per_layer, kernel_size=8, strides=strides, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')(x)
    x = tf.keras.layers.LSTM(units, return_sequences=True, recurrent_activation='hard_sigmoid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    preds = tf.keras.layers.Dense(num_of_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def small_cnn_lstm(n_dataset_features, num_of_classes, features_per_layer, strides, pool_size, units, dropout,
                   kernel_size,lr=0.001):
    inp=tf.keras.layers.Input((n_dataset_features,1))
    x=tf.keras.layers.Conv1D(features_per_layer, kernel_size=kernel_size, strides=strides, activation='relu')(inp)
    x=tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')(x)
    x=tf.keras.layers.LSTM(units, return_sequences=True, recurrent_activation='hard_sigmoid')(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dropout(dropout)(x)
    preds=tf.keras.layers.Dense(num_of_classes, activation='softmax')(x)
    model=tf.keras.Model(inputs= inp,outputs=preds)
    opt=tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  # optimizer, metrics
    return model


def load_hdf5_data(path, label_encoder, n_samples=-1):
    """
    Reads data from an HDF5 file and returns features and labels.
    
    Args:
        path (str): Path to the HDF5 file.
        label_encoder: Label encoder object with a transform method.
        n_samples (int, optional): Number of samples per label to include. Defaults to -1 (all samples).
        
    Returns:
        tuple: (X_data, y_data) containing features and one-hot encoded labels.
    """
    with h5py.File(path, 'r') as hdf:
        X_data = hdf['vector'][:].astype(float)
        y_data = label_encoder.transform(hdf['site_name'][:].reshape(-1, 1)).toarray()
        
        if n_samples != -1:
            # Filter data based on n_samples per label
            X_filtered, y_filtered = [], []
            unique_labels = np.unique(np.argmax(y_data, axis=1))  # Get unique label indices

            for label in unique_labels:
                label_indices = np.where(np.argmax(y_data, axis=1) == label)[0]
                selected_indices = label_indices[:n_samples]  # Select first n_samples for this label
                X_filtered.extend(X_data[selected_indices])
                y_filtered.extend(y_data[selected_indices])

            X_data = np.array(X_filtered)
            y_data = np.array(y_filtered)
            
    return X_data, y_data


def prepare_hdf5_data(X_data, y_data, noise_function=None, batch_size=32, sample_validation=0.1, max_val=8192, verbose=0,shuffle=True):
    """
    Prepares data for training and validation with optional noise.

    Args:
        X_data (np.ndarray): The feature data.
        y_data (np.ndarray): The label data (one-hot encoded).
        noise_function (callable, optional): Function to apply noise to X_data.
            Defaults to None, in which case no noise is applied.
            The function should take X_data as input and return the noised X_data.
        batch_size (int, optional): Batch size for datasets. Defaults to 32.
        sample_validation (float, optional): Fraction of data to use for validation. Defaults to 0.1.
        max_val (float, optional): Maximum value for data smoothing. Defaults to 8192.
        verbose (int, optional): Verbos
    Returns:
        dict: A dictionary containing 'data' (training dataset) and 'val' (validation dataset).
    """
    # Apply noise if a noise function is provided
    if noise_function is not None:
        X_data = noise_function(X_data)

    # Create training and validation datasets
    if sample_validation > 0:
        X_trn, X_tst, y_trn, y_tst = train_test_split( X_data, y_data, test_size=sample_validation, random_state=42)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_trn, y_trn))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_tst, y_tst))
        validation_data = val_dataset
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
        validation_data = None

    # Process validation data
    validation_data = None if isinstance(validation_data, type(None)) else validation_data.map(
        lambda x, y: (standartize(tf.cast(x, tf.float32)), y)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE) #.map(lambda x,y:(smooth(x,max_val),y))

    # Process training data
    if shuffle:
      train_dataset = train_dataset.shuffle(len(y_data))
    train_dataset = train_dataset.map(lambda x, y: (standartize(tf.cast(x, tf.float32)), y)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE) #.map(lambda x,y:(smooth(x,max_val),y))

    return {'data': train_dataset, 'val': validation_data}


def create_subset_encoder(original_encoder, n_categories):
  """
  Creates a new OneHotEncoder with only the first n categories from the original encoder.
  
  Args:
      original_encoder: A fitted OneHotEncoder.
      n_categories: The number of categories to keep in the new encoder.
      
  Returns:
      A new OneHotEncoder with only the first n categories.
  """
  if not hasattr(original_encoder, 'categories_'):
      raise ValueError("The original encoder must be fitted before creating a subset.")
  
  # Extract the original categories
  original_categories = original_encoder.categories_
  
  # Create a new list with only the first n categories for each feature
  subset_categories = [categories[:n_categories] for categories in original_categories]
  
  # Create a new encoder with the subset of categories
  subset_encoder = OneHotEncoder(categories=subset_categories, sparse_output=True)
  
  return subset_encoder

def cache_countermeasure(X_data, noise_std, random_seed=RANDOM_SEED):
    """
    Adds absolute normal noise to the input data.

    Args:
        X_data (np.ndarray): The input data to add noise to.
        noise_std (float): The standard deviation of the normal distribution for generating noise.
        random_seed (int, optional): The random seed for reproducibility. Defaults to RANDOM_SEED.

    Returns:
        np.ndarray: The data with added noise.
    """
    rng = np.random.default_rng(random_seed)
    # abs on normal distribution, always positive
    noise = np.abs(rng.normal(0, noise_std, X_data.shape))
    X_data_noisy = X_data + noise
    return X_data_noisy


def network_countermeasure(X_data, noise_std,inserting_noise_p, random_seed=RANDOM_SEED):
    """
    Adds controlled Gaussian noise to the input data as a countermeasure, with noise inserted at random positions.
    Parameters
    ----------
    X_data : np.ndarray
        Input data array to which noise will be added.
    noise_std : float
        Standard deviation of the Gaussian noise to be added.
    inserting_noise_p : float
        Probability of inserting noise at each position in the input data.
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility. Defaults to RANDOM_SEED.
    Returns
    -------
    np.ndarray
        The input data array with noise added at randomly selected positions.
    """
    rng = np.random.default_rng(random_seed)
    # noise can be negative here
    noise = rng.normal(0, noise_std, X_data.shape)
    p_of_inserting_noise = rng.binomial(1, inserting_noise_p, size=X_data.shape)
    noise = noise * p_of_inserting_noise
    X_data_noisy = X_data + noise
    return X_data_noisy


def predict_on_hdf5(test_path, model,label_encoder,batch_size=32):
    # Open the HDF5 file for reading
    with h5py.File(test_path, 'r') as hdf:
        # Load the features and labels
        X_test = hdf[f'vector'][:]
        y_test = label_encoder.transform((hdf[f'site_name'][:]).reshape(-1,1)).toarray()

        # Create TensorFlow Dataset objects for efficient batch processing
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(lambda x, y: (standartize(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Evaluate the model on the test dataset
        res = model.predict(test_dataset)
        return res

def get_ground_truth_labels(dataset):
  def get_argmax_batch(feats,labs):
      return tf.argmax(labs, axis=1)
  argmax_dataset = dataset.map(get_argmax_batch)
  argmax_list = list(argmax_dataset.as_numpy_iterator())
  return np.concatenate(argmax_list)

def test_on_hdf5(test_path, model,label_encoder,batch_size=32):
    # Open the HDF5 file for reading
    with h5py.File(test_path, 'r') as hdf:
        # Load the features and labels
        X_test = hdf[f'vector'][:]
        y_test = label_encoder.transform((hdf[f'site_name'][:]).reshape(-1,1)).toarray()

        # Create TensorFlow Dataset objects for efficient batch processing
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(lambda x, y: (standartize(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(test_dataset)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        return accuracy


