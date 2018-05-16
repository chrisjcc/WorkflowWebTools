# ---- Import Keras deep neural network library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1, l2
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K

# Define Multilayer Perceptron architecture
def create_model(input_dim=7342, n_classes = 11, nlayers=5, nneurons=100,
                 dropout_rate=0.0, l2_norm=0.001, learning_rate=1e-3,
                 activation='relu', kernel_initializer='lecun_normal',
                 optimizer='adam', metric='accuracy', 
                 loss='sparse_categorical_crossentropy'):
    '''
    create_model: build neural network architecture
    '''
        
    # create neural network model
    model = Sequential()
    
    # Add fully connected layer with an activation function (input layer)
    model.add(Dense(units=nneurons,
                    input_dim=input_dim,
                    kernel_initializer=kernel_initializer,
                    activation=activation,
                    kernel_regularizer=l2(l2_norm)))
    
    if dropout_rate != 0.:
        model.add(Dropout(dropout_rate))
                                        
    # Indicate the number of hidden layers
    for index, layer in enumerate(range(nlayers-1)):
        model.add(Dense(units=nneurons,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        kernel_regularizer=l2(l2_norm)))
        
    # Add dropout layer
    if dropout_rate != 0.:
        model.add(Dropout(dropout_rate))
        
    # Add fully connected output layer with a sigmoid activation function
    model.add(Dense(n_classes,
                    kernel_initializer=kernel_initializer,
                    activation='softmax',
                    kernel_regularizer=l2(l2_norm)))
    
    # Compile neural network (set loss and optimize)
    model.compile(loss=loss,#'categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[metric]) #'crossentropy'
    
    # Print summary report
    if True:
        model.summary()
    
    # Return compiled network
    return model
