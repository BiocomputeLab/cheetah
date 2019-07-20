
import tensorflow as tf
from keras import layers
from keras import Input
from keras import optimizers
from keras.models import Model, load_model
from keras import backend as K
import numpy as np
from sklearn.metrics import f1_score


def custom_categorical_crossentropy(class_weights):
    '''Custom categorical crossentropy loss function'''
    def pixelwise_loss(y_true, y_pred):
        '''Computation of weighted pixelwise loss'''
        # Initialize weights tensor
        weights = np.array(class_weights)[np.newaxis, np.newaxis, :]
        w_tensor = weights * tf.ones_like(y_true)
        # Compute loss
        epsilon = tf.constant(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(y_true * tf.log(y_pred) * w_tensor,
                               len(y_pred.get_shape()) - 1)
    return pixelwise_loss
   

def custom_softmax(input_data):
    d = K.exp(input_data - K.max(input_data, axis=-1, keepdims=True))
    return d / K.sum(d, axis=-1, keepdims=True)


def u_net_model(img_height, img_width, input_chn, n_classes, act_func='elu',
                regularizer='batchnorm', dropoutrate=0.1):
    '''
    U-Net (encoder-decoder) fully convolutional network
    img_height: image height in pixels ==> height/32 must be an integer
    img_width: image width in pixels ==> width/32 must be an integer
    input_chn: number of input channels
    n_classes: number of classes on ground truth image
    act_func: activation function (layers)
              default = 'elu'
    regularizer: batch normalisation (batchnorm) or dropout (dropout)
                 default = 'batchnorm'
    dropoutrate: dropout rate (1 means 100%)
                 default = 0.1
    '''
    
    w_init = 'glorot_normal'
    kernel_size = (3, 3)
    
    # Downsampling 1
    netw_input = Input(shape=(img_height, img_width, input_chn))
    conv_d_1 = layers.BatchNormalization(axis=-1)(netw_input)
    conv_d_1 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                             kernel_initializer=w_init)(conv_d_1)
    if regularizer == 'batchnorm':
        conv_d_1 = layers.BatchNormalization(axis=-1)(conv_d_1)
    conv_d_1 = layers.Activation(act_func)(conv_d_1)
    conv_d_1 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                             kernel_initializer=w_init)(conv_d_1)
    if regularizer == 'batchnorm':
        conv_d_1 = layers.BatchNormalization(axis=-1)(conv_d_1)
    conv_d_1 = layers.Activation(act_func)(conv_d_1)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_1)
    if regularizer == 'dropout':
        pool_1 = layers.Dropout(dropoutrate)(pool_1)
    
    # Downsampling 2
    conv_d_2 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_1)
    if regularizer == 'batchnorm':
        conv_d_2 = layers.BatchNormalization(axis=-1)(conv_d_2)
    conv_d_2 = layers.Activation(act_func)(conv_d_2)
    conv_d_2 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_d_2)
    if regularizer == 'batchnorm':
        conv_d_2 = layers.BatchNormalization(axis=-1)(conv_d_2)
    conv_d_2 = layers.Activation(act_func)(conv_d_2)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_2)
    if regularizer == 'dropout':
        pool_2 = layers.Dropout(dropoutrate)(pool_2)
    
    # Downsampling 3
    conv_d_3 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_2)
    if regularizer == 'batchnorm':
        conv_d_3 = layers.BatchNormalization(axis=-1)(conv_d_3)
    conv_d_3 = layers.Activation(act_func)(conv_d_3)
    conv_d_3 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_d_3)
    if regularizer == 'batchnorm':
        conv_d_3 = layers.BatchNormalization(axis=-1)(conv_d_3)
    conv_d_3 = layers.Activation(act_func)(conv_d_3)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_3)
    if regularizer == 'dropout':
        pool_3 = layers.Dropout(dropoutrate)(pool_3)
    
    # Downsampling 4
    conv_d_4 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_3)
    if regularizer == 'batchnorm':
        conv_d_4 = layers.BatchNormalization(axis=-1)(conv_d_4)
    conv_d_4 = layers.Activation(act_func)(conv_d_4)
    conv_d_4 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_d_4)
    if regularizer == 'batchnorm':
        conv_d_4 = layers.BatchNormalization(axis=-1)(conv_d_4)
    conv_d_4 = layers.Activation(act_func)(conv_d_4)
    pool_4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_4)
    if regularizer == 'dropout':
        pool_4 = layers.Dropout(dropoutrate)(pool_4)
    
    # Bottom block
    conv_b = layers.Conv2D(1024, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_4)
    if regularizer == 'batchnorm':
        conv_b = layers.BatchNormalization(axis=-1)(conv_b)
    conv_b = layers.Activation(act_func)(conv_b)
    conv_b = layers.Conv2D(1024, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_b)
    if regularizer == 'batchnorm':
        conv_b = layers.BatchNormalization(axis=-1)(conv_b)
    conv_b = layers.Activation(act_func)(conv_b)
    
    # Upsampling 1
    up_1 = layers.UpSampling2D(size=(2, 2))(conv_b)
    if regularizer == 'dropout':
        up_1 = layers.Dropout(dropoutrate)(up_1)
    concat_u_1 = layers.concatenate([up_1, conv_d_4], axis=-1)
    conv_u_1 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(concat_u_1)
    if regularizer == 'batchnorm':
        conv_u_1 = layers.BatchNormalization(axis=-1)(conv_u_1)
    conv_u_1 = layers.Activation(act_func)(conv_u_1)
    conv_u_1 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_u_1)
    if regularizer == 'batchnorm':
        conv_u_1 = layers.BatchNormalization(axis=-1)(conv_u_1)
    conv_u_1 = layers.Activation(act_func)(conv_u_1)
    
    # Upsampling 2
    up_2 = layers.UpSampling2D(size=(2, 2))(conv_u_1)
    if regularizer == 'dropout':
        up_2 = layers.Dropout(dropoutrate)(up_2)
    concat_u_2 = layers.concatenate([up_2, conv_d_3], axis=-1)
    conv_u_2 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same', 
                            kernel_initializer=w_init)(concat_u_2)
    if regularizer == 'batchnorm':
        conv_u_2 = layers.BatchNormalization(axis=-1)(conv_u_2)
    conv_u_2 = layers.Activation(act_func)(conv_u_2)
    conv_u_2 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same', 
                            kernel_initializer=w_init)(conv_u_2)
    if regularizer == 'batchnorm':
        conv_u_2 = layers.BatchNormalization(axis=-1)(conv_u_2)
    conv_u_2 = layers.Activation(act_func)(conv_u_2)
    
    # Upsampling 3
    up_3 = layers.UpSampling2D(size=(2, 2))(conv_u_2)
    if regularizer == 'dropout':
        up_3 = layers.Dropout(dropoutrate)(up_3)
    concat_u_3 = layers.concatenate([up_3, conv_d_2], axis=-1)
    conv_u_3 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(concat_u_3)
    if regularizer == 'batchnorm':
        conv_u_3 = layers.BatchNormalization(axis=-1)(conv_u_3)
    conv_u_3 = layers.Activation(act_func)(conv_u_3)
    conv_u_3 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_u_3)
    if regularizer == 'batchnorm':
        conv_u_3 = layers.BatchNormalization(axis=-1)(conv_u_3)
    conv_u_3 = layers.Activation(act_func)(conv_u_3)
    
    # Upsampling 4
    up_4 = layers.UpSampling2D(size=(2, 2))(conv_u_3)
    if regularizer == 'dropout':
        up_4 = layers.Dropout(dropoutrate)(up_4)
    concat_u_4 = layers.concatenate([up_4, conv_d_1], axis=-1)
    conv_u_4 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(concat_u_4)
    if regularizer == 'batchnorm':
        conv_u_4 = layers.BatchNormalization(axis=-1)(conv_u_4)
    conv_u_4 = layers.Activation(act_func)(conv_u_4)
    conv_u_4 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_u_4)
    if regularizer == 'batchnorm':
        conv_u_4 = layers.BatchNormalization(axis=-1)(conv_u_4)
    conv_u_4 = layers.Activation(act_func)(conv_u_4)
    
    # Output layer
    conv_u_out = layers.Conv2D(n_classes, (1, 1), strides=(1, 1), 
                               padding='same',
                               kernel_initializer=w_init)(conv_u_4)
    netw_output = layers.Activation('softmax')(conv_u_out)
    
    # Model
    model = Model(inputs=netw_input, outputs=netw_output)
    
    return model


def load_segmenter_full (full_model_filename):
    # Full model should be a *.h5 file
    model = load_model(full_model_filename)
    return Segmenter(with_model=model)


def load_segmenter (model_filename, weights_filename):
    # Model is a *.json file and the weights are a *.h5 file
    with open(model_filename) as f:
        model_kwargs = json.load(f)
    model = netw_models.u_net_model(*model_args, **model_kwargs)
    model.load_weights(weights_filename)
    return Segmenter(with_model=model)


class Segmenter():
    '''Class for segmenting images based on a U-Net model'''
    
    def __init__(self, img_height=128, img_width=128, input_chn=1, n_classes=2, act_func='elu',
                regularizer='batchnorm', dropoutrate=0.1, with_model=None):
        '''Initialization'''
        if with_model == None:
            self.model = u_net_model(img_height, img_width, input_chn, n_classes, act_func=act_func,
                                     regularizer=regularizer, dropoutrate=dropoutrate)
        else:
            self.model = with_model
    

    def train (self, train_X, train_Y, val_X, val_Y, weighted_loss=True, class_weights=None,
               batch_size=8, n_epochs=50):
        # Extract dimensions of the data
        _, height, width, channels = train_X.shape
        n_classes = train_Y.shape[-1]

        # TODO: add checks that training data is correct shape for model - if not exit with error

        if weighted_loss == True:
            if class_weights == None:
                class_weights = [1]*n_classes
            loss_function = custom_categorical_crossentropy(class_weights)
        else:
            loss_function = 'categorical_crossentropy'
        # Compile model
        self.model.compile(loss=loss_function,
                           optimizer=optimizers.RMSprop(lr=1e-4, rho=0.9),
                           metrics=['acc'])
        # Model training
        model_fit = self.model.fit(train_X, train_Y, batch_size, n_epochs,
                                   validation_data=(val_X, val_Y), shuffle=True)
        return model_fit


    def save_full (self, filename):
        # Should be a *.h5 file
        self.model.save(filename)


    def save (self, model_filename, weights_filename):
        # Model should be a *.h5 file
        # Weights should be a *.json file
        model.save_weights(weights_filename)
        with open(model_filename, 'w') as file:
            json.dump(model_kwargs, file)
        
        
    def test(self, test_X, test_Y, verbose=1, average=None):
        '''
        Evaluate model on a test dataset
        test_X: input data (numpy array)
        test_Y: input labels (numpy array, one-hot encoded)
        verbose: 0 = no output, 1 = progress bar
                 default = 1
        average: None = F1 score per class, 'micro' = globally averaged
                 default = None
        returns: [metric_overall_acc, metric_f1_score]
        '''
        y_pred = self.model.predict(test_X, batch_size=1, verbose=verbose)
        return self._calc_metrics(test_Y, y_pred, average)
          
            
    def predict(self, pred_X, verbose=1, return_format='list'):
        '''
        Use model for prediction (unseen data)
        pred_X: input data (numpy array)
        verbose: 0 = no output, 1 = progress bar
                 default = 1
        return_format: 'list' = return list, 'array' = return numpy array
                       default = 'list'
        returns: prediction (probabilities for each class)
                 as list or numpy array (specified by return_format)
        '''
        y_pred = self.model.predict(pred_X, batch_size=1, verbose=verbose)
        if return_format == 'list': # mainly for debugging purpose
            y_pred_list = []
            for i in range(0, y_pred.shape[0]):
                y_pred_list.append(y_pred[i, ...])
            return y_pred_list
        else:
            return y_pred
        

    def _calc_metrics(self, y_true, y_pred, average):
        '''Calculate metrics (overall accuracy, F1 score)'''
        y_pred_max = np.argmax(y_pred, axis=-1)
        y_true_max = np.argmax(y_true, axis=-1)
        # Overall (pixelwise) accuracy
        # (for multiclass classification the same as Jaccard index)
        metric_overall_acc = np.sum(np.equal(
                                    y_true_max, y_pred_max)) / y_true_max.size
        # F1 score
        metric_f1_score = f1_score(y_true_max.flatten(), y_pred_max.flatten(),
                                   average=average)
        return [metric_overall_acc, metric_f1_score]
