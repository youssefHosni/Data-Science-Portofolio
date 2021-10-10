import keras
from keras import optimizers
import sys


def choosing(optimizer):
    if optimizer=='adam':
        opt=keras.optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer=='adamax':    
        opt= keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif optimizer=='Nadama':    
        opt= keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    elif optimizer=='adadelta':    
        opt= optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer=='adagrad':    
        opt= keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    elif optimizer=='sgd':
        opt = optimizers.SGD(lr=0.01, clipnorm=1.)
    elif optimizer=='RMSprop':
        opt=keras.optimizers.RMSprop(lr=0.0006, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer==None:
        return
    else:
        print('Value Error:Optimizer took unexpected value')
        sys.exit()
    
    return opt 