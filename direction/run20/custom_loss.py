from radiotools import helper as hp
from generator import batch_size
import tensorflow.keras.backend as K

def angle_difference_loss(y_true, y_pred):

    y_true_np = K.eval(y_true)
    y_pred_np = K.eval(y_pred)

    for i in range(batch_size):
        angle_difference_data[i] = hp.get_angle(y_pred_np[i], y_true_np[i])
        
    return K.constant(angle_difference_data)