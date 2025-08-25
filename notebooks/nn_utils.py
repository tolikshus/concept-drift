import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Input, Conv1D, MaxPool1D, LSTM, Flatten, Dropout, Dense, Lambda

class TripletSemiHardLoss(tf.keras.losses.Loss):
    """
    Computes the triplet loss with semi-hard negative mining.

    Reference:
    https://www.tensorflow.org/addons/api_docs/python/tfa/losses/TripletSemiH
    
    
    
    ardLoss
    Adapted for TensorFlow 2.x without tf_addons dependency.
    """

    def __init__(self, margin=1.0, name=None):
        super().__init__(name=name or "triplet_semihard_loss")
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        y_true: tensor of shape [batch_size] with integer class labels
        y_pred: tensor of shape [batch_size, embed_dim] with embedding vectors
        """
        # Ensure correct dtype
        labels = tf.cast(y_true, tf.int32)
        embeddings = y_pred

        # Compute pairwise squared distances
        pdist_matrix = self._pairwise_distance(embeddings)

        # Build adjacency matrix of positive (same label) and negative pairs
        adjacency = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        adjacency_not = tf.logical_not(adjacency)

        batch_size = tf.size(labels)

        # For each anchor, get the hardest positive
        # We use pdist_matrix but masked by adjacency
        mask_positives = tf.cast(adjacency, tf.float32) - tf.linalg.diag(tf.ones([batch_size]))

        # Compute the distance matrix where positives only
        positives = pdist_matrix * mask_positives

        # For each anchor, the hardest positive is the maximum distance
        hardest_positive_dist = tf.reduce_max(positives, axis=1)

        # For each anchor, find the semi-hard negative
        # We want negatives where d_an > d_ap
        diff = tf.expand_dims(hardest_positive_dist, 1) - pdist_matrix + self.margin

        # Mask out invalid negatives
        mask_negatives = tf.cast(adjacency_not, tf.float32)
        # We only keep semi-hard: ensure negative distance > positive
        cond = pdist_matrix > tf.expand_dims(hardest_positive_dist, 1)
        mask_semihard = tf.cast(cond, tf.float32)

        mask = mask_negatives * mask_semihard

        semihard_negatives = diff * mask

        # For each anchor, choose minimal semi-hard loss; if none, zero
        # To avoid taking min over all zeros when no valid, add large value
        max_val = tf.reduce_max(diff) + 1.0
        semihard_negatives = tf.where(mask > 0.0, semihard_negatives, max_val)

        hardest_negative_dist = tf.reduce_min(semihard_negatives, axis=1)

        # Compute triplet loss for each anchor
        loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)
        loss = tf.reduce_mean(loss)

        return loss

    @staticmethod
    def _pairwise_distance(embeddings):
        """
        Compute pairwise squared Euclidean distance matrix
        """
        # ||a - b||^2 = ||a||^2 - 2*a.b + ||b||^2
        square = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
        pdist = square - 2 * tf.matmul(embeddings, embeddings, transpose_b=True) + tf.transpose(square)
        # Make sure distances are non-negative due to numeric errors
        pdist = tf.maximum(pdist, 0.0)
        return pdist

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config

class L2NormalizationLayer(Layer):
  def __init__(self, axis=-1, **kwargs):
      super().__init__(**kwargs)
      self.axis = axis

  def call(self, inputs):
      return tf.math.l2_normalize(inputs, axis=self.axis)

  def get_config(self):
      config = super().get_config()
      config.update({"axis": self.axis})
      return config

  def compute_output_shape(self, input_shape):
      return input_shape

def triplet_cnn(input_vector, output_size,kernel_size, filters=32, strides=5, pool_size=4,dropout=0.8,lr=0.001):
    kernels=[]
    if type(kernel_size)is list:
        kernels=kernel_size
    else:
        kernels.extend([kernel_size,kernel_size])
    inp = Input(shape=(input_vector,1))
    x = Conv1D(filters, kernel_size=kernels[0], strides=strides, activation='relu')(inp)
    x = MaxPool1D(pool_size=pool_size, padding='same')(x)
    x=Dropout(dropout)(x)
    x = Conv1D(filters, kernel_size=kernels[1], strides=strides, activation='relu')(x)
    x = MaxPool1D(pool_size=pool_size, padding='same')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    X = Dense(output_size, activation=None)(x)
    #preds = Lambda(lambda l: tf.math.l2_normalize(l, axis=1))(X)
    preds = L2NormalizationLayer(axis=1)(X)
    model = Model(inputs=inp, outputs=preds)
    opt=tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt,loss=TripletSemiHardLoss())
    return model

def triplet_cnn_lstm(input_vector, output_size, filters, strides, pool_size, units, dropout,
                      lr=0.001) -> tf.keras.Model:
    inp = Input((input_vector, 1))
    x = Conv1D(filters, kernel_size=16, strides=strides, activation='relu')(inp)
    x = MaxPool1D(pool_size=pool_size, padding='same')(x)
    x = Conv1D(filters, kernel_size=8, strides=strides, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=pool_size, padding='same')(x)
    x = LSTM(units, return_sequences=True, recurrent_activation='hard_sigmoid')(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    X = Dense(output_size, activation=None)(x)

    # Pass the output_shape function to the Lambda layer
    preds = L2NormalizationLayer(axis=1)(X)

    model = Model(inputs=inp, outputs=preds)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # Make sure TripletSemiHardLoss is defined before compiling
    model.compile(optimizer=opt, loss=TripletSemiHardLoss())
    return model
