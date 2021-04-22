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
import json
import argparse
import validation_set
import imagenet_map
import lime
from consts import dataDir, dataType, annFile
from pycocotools.coco import COCO # on windows: python -m pip install pycocotools-windows


def run(kwargs: dict):
    coco = COCO(annFile)


    # TF configuration
    tensorflow.get_logger().setLevel('ERROR')
    warnings.filterwarnings('ignore') 
    model = tensorflow.keras.applications.inception_v3.InceptionV3() # Load pretrained model

    # build imagenet class map
    imagenet_class_to_label = imagenet_map.build_imagenet_map()

    # build validation set
    # val_df, imgs = validation_set.build_validation_set(model, coco, imagenet_class_to_label)
    # val_df.to_csv('val_df.csv')
    val_df = pd.read_csv('val_df.csv')

    # perform lime on built validation set
    # val_df = val_df.sample(n=2, random_state=1337)
    result, args = lime.evaluate_set(val_df, model, coco, **kwargs)

    def stringify_dict(a):
        return ','.join('_'.join(kv) for kv in zip(a.keys(), map(str, a.values())))

    result.to_csv('results/%.3f_results@%s.csv' % (result.score.mean(), stringify_dict(args)))

if __name__ == '__main__':
    # parse cmdline args
    parser = argparse.ArgumentParser(description='LIME evaluation tool')
    parser.add_argument('-n')
    parser.add_argument('--seg')
    parser.add_argument('--reg')
    parser.add_argument('--sel')
    parser.add_argument('--weights')

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}

    run(args)
