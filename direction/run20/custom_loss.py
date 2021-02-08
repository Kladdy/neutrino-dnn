from radiotools import helper as hp
from generator import batch_size
import tensorflow.keras.backend as K

def angle_difference_loss(y_true, y_pred):

    for i in range(batch_size):
        angle_difference_data[i] = hp.get_angle(y_pred[i], y_true[i])
        
    return K.constant(angle_difference_data)