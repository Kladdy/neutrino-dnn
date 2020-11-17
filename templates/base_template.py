""" base_template.py

    General info
    ------------
    Acts as a base template for all modeling files in 3 steps:
        Step 1: Does something important
        Step 2: Rests for a while
        Step 3: Starts working again

    Requirements
    ------------
    Python packages:
        gpuutils
        tensorflow
    Other:
        A sense of humor
        A great day in one of two ways
            - A sunny day
            - A day full of sunshine and rainbows
        NOTE: These requirements cannot be fulfilled anyway, so why bother?

    Authors
    -------
    Written by Sigfrid Stjärnholm (nov 2020)
"""

# This code stops tensorflow from reserving all VRAM of a GPU.
# This code snippet is mandatory for all pieces of code that
# use tensorflow. Otherwise, the code crashes if another process
# tries to access the same GPU.
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1,
                  framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
