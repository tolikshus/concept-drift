import h5py
import numpy as np
import tensorflow as tf
import pathlib
import pickle
import re
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

def get_data_paths_ordered(data_path:str)->pathlib.Path:
    pattern = r'^\d{4}_\d{2}_\d{2}$'
    data_dir = pathlib.Path(data_path)
    paths = sorted([p for p in data_dir.iterdir() if p.is_dir() and bool(re.match(pattern, p.stem)) ])
    return paths

@tf.function(jit_compile=True)
def standartize(curr_input):
    curr_input = tf.cast(curr_input, tf.float32)
    mean, variance = tf.nn.moments(curr_input, axes=None)
    std = tf.sqrt(variance) + 0.0001 # the original function was without sqrt
    return (curr_input - mean) / std

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

def prepare_hdf5_data(path, label_encoder, batch_size=32, n_samples=-1,sample_validation=0.1,verbose=0):
    # Open the HDF5 file for reading
    with h5py.File(path, 'r') as hdf:
        X_data = hdf['vector'][:]
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
        if sample_validation > 0:
            train_size = int(len(X_data) * (1 - sample_validation))
            train_dataset = tf.data.Dataset.from_tensor_slices((X_data[:train_size], y_data[:train_size]))

            val_dataset = tf.data.Dataset.from_tensor_slices((X_data[train_size:], y_data[train_size:]))
            validation_data = val_dataset
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
            validation_data = None
        validation_data = None if isinstance(validation_data,type(None)) else validation_data.map(lambda x, y: (standartize(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(len(y_data)).map(lambda x, y: (standartize(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return {'data':train_dataset,'val':validation_data}


# Function to train the model on batches from HDF5 file
def train_on_hdf5(train_path, model, label_encoder, batch_size=32, n_epochs=100, n_samples=-1,sample_validation=0.1,verbose=0):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Open the HDF5 file for reading
    with h5py.File(train_path, 'r') as hdf:
        X_data = hdf['vector'][:]
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
        if sample_validation > 0:
            train_size = int(len(X_data) * (1 - sample_validation))
            train_dataset = tf.data.Dataset.from_tensor_slices((X_data[:train_size], y_data[:train_size]))

            val_dataset = tf.data.Dataset.from_tensor_slices((X_data[train_size:], y_data[train_size:])) 
            validation_data = val_dataset
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
            validation_data = None
        validation_data = None if isinstance(validation_data,type(None)) else validation_data.map(lambda x, y: (standartize(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(len(y_data)).map(lambda x, y: (standartize(x), y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Train the model


        model.fit(
            train_dataset,
            validation_data=validation_data ,
            epochs=n_epochs,
            callbacks=[early_stopping],
            verbose=verbose
        )


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

class OpenWorldLabelEncoder:
    def __init__(self, label_encoder, max_label):
        """
        Initializes the OpenWorldLabelEncoder.

        Args:
            label_encoder: A fitted label encoder object (e.g., from sklearn.preprocessing).
            max_label: The maximum label index to keep. Labels greater than this will be mapped to max_label.
        """
        self.label_encoder = label_encoder
        self.max_label = max_label-1
        self.open_world_label = 'open_world' # Define the string representation for the open_world label

    def transform(self, labels):
          processed_labels = np.asarray(labels).reshape(-1, 1)
          clipped_transformed = self.label_encoder.transform(processed_labels)[:, :self.max_label].toarray()
          no_one_mask = np.all(clipped_transformed == 0, axis=1)
          last_column = np.zeros((clipped_transformed.shape[0], 1), dtype=clipped_transformed.dtype)
          last_column[no_one_mask, 0] = 1
          clipped_transformed =np.concatenate((clipped_transformed, last_column), axis=1)
          return csr_matrix(clipped_transformed,dtype=np.float64)
    def inverse_transform(self, numerical_labels):
        open_world_mask = (numerical_labels == self.max_label)
        original_labels = self.label_encoder.inverse_transform(numerical_labels[~open_world_mask])
        transformed_labels = np.empty(numerical_labels.shape, dtype=object)
        transformed_labels[~open_world_mask] = original_labels
        transformed_labels[open_world_mask] = self.open_world_label
        return transformed_labels

