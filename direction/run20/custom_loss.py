from radiotools import helper as hp
from generator import batch_size

def angle_difference_loss(y_true, y_pred):

    for i in range(batch_size):
        angle_difference_data[i] = hp.get_angle(y_pred[i], y_true[i])
        
    return angle_difference_data