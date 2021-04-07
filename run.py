import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.config
# enable GPU (this has to happen before additional TF stuff happens)
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)
import warnings
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tqdm import tqdm
tqdm.pandas() # allows for the use of progress_apply(...)
import validation_set
import imagenet_map
import lime
import json

# TF configuration
tensorflow.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore') 
model = tensorflow.keras.applications.inception_v3.InceptionV3() # Load pretrained model

# build imagenet class map
imagenet_class_to_label = imagenet_map.build_imagenet_map()

# build validation set
val_df, imgs = validation_set.build_validation_set(model, imagenet_class_to_label)

# perform lime on built validation set
kwargs = {
    'num_perturb': 100
}
result = lime.evaluate_set(val_df, model, imgs, **kwargs)

def stringify_dict(a):
    return ','.join('_'.join(kv) for kv in zip(a.keys(), map(str, a.values())))

result.to_csv('results@%s.csv' % stringify_dict(kwargs))