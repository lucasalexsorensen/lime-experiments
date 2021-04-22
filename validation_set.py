from skimage.io import imread
import numpy as np
from skimage.transform import resize
import pandas as pd
from tensorflow.keras.applications.imagenet_utils import decode_predictions

def build_validation_set(model, coco, imagenet_class_to_label: dict) -> (pd.DataFrame, list):
    imgIds = [coco.getImgIds(catIds=coco.getCatIds(catNms=[term])) for term in ['dog','bear','cat','boat']]

    flatten = lambda t: [item for sublist in t for item in sublist]
    imgIds = flatten(imgIds)

    imgs = coco.loadImgs(imgIds)

    blacklist = list()
    img_vector = np.zeros((len(imgs), 299, 299, 3))
    for idx, I in enumerate(imgs):
        img = imread('%s/images/%s/%s' % (dataDir,dataType,I['file_name']))
        img = resize(img, (299,299)) 

        if img.shape != (299, 299, 3):
            print('blacklisted due to wrong shape', idx)
            blacklist.append(idx)
            continue

        X = (img - 0.5)*2 #Inception pre-processing

        img_vector[idx, :, :, :] = X

    # get initial predictions
    val_predictions = model.predict(img_vector)
    val_list = list()
    for idx, raw_preds in enumerate(val_predictions):
        if idx in blacklist:
            continue
            
        decoded_preds = decode_predictions(np.array(raw_preds).reshape(1, 1000), top=5)[0]
        top_indices = np.argsort(raw_preds)[::-1]
        
        preds = [(i, imagenet_class_to_label.get(p[0], None)) for i,p in zip(top_indices, decoded_preds)]
        preds = list(filter(lambda x: x[1] is not None, preds))
        
        if len(preds) < 1:
            continue

        class_to_explain, label = preds[0]

        imgId = imgIds[idx]
        if imgId == None:
            continue
            
        val_list.append({ 'idx': idx, 'imgId': imgId, 'label': label, 'imagenet_class_to_explain': class_to_explain })
    val_df = pd.DataFrame.from_records(val_list)
    print('Built validation dataset:')
    print(val_df.label.value_counts())
    return val_df, imgs
