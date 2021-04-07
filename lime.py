import copy
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from PIL import Image
from skimage.io import imread
import sklearn.metrics
from skimage.transform import resize
from skimage.segmentation import quickshift
import random
import numpy as np
import pandas as pd
from consts import dataDir, dataType
from validation_set import coco

# applies a binary perturbation mask to given img
def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:,:,np.newaxis]
    return perturbed_image

# IoU metric
def calculate_score(img_mask, coco_mask):
    intersection = np.sum(img_mask * coco_mask)
    union = np.sum(img_mask + coco_mask)

    return intersection / union

# evaluate a set
def evaluate_set(val_df: pd.DataFrame, model, imgs: list, **kwargs):
    result = val_df.progress_apply(lambda r: evaluate_row(r, model, imgs, **kwargs), axis=1)
    return result

# evaluate a single image (i.e. row of the DF)
def evaluate_row(row, model, imgs: list, **kwargs):
    idx, label, imagenet_class_to_explain = row.idx, row.label, row.imagenet_class_to_explain
    
    kwargs.setdefault('num_perturb', 250)

    I = imgs[idx]
    img = imread('%s/images/%s/%s'%(dataDir,dataType,I['file_name']))
    og_size = img.shape
    img = resize(img, (299,299)) 
    X = (img - 0.5)*2 # Inception pre-processing
    
    # perform LIME
    superpixels = quickshift(img, kernel_size=5, max_dist=300, ratio=0.2)
    N = np.unique(superpixels).shape[0]
    num_perturb = kwargs['num_perturb']
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, N))
    all_perturbations = np.zeros((num_perturb, 299, 299, 3))
    for i, pert in enumerate(perturbations):
        perturbed_img = perturb_image(X, pert, superpixels)
        all_perturbations[i,:,:,:] = perturbed_img
    predictions = model.predict(all_perturbations)
    
    # weights
    original_image = np.ones(N)[np.newaxis,:]
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:, imagenet_class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_

    num_top_features = 3
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(N)
    mask[top_features]= True
    og_img = resize(perturb_image(img, mask, superpixels), og_size)
    
    # load COCO annotation
    annIds = coco.getAnnIds(imgIds=I['id'], catIds=coco.getCatIds(catNms=label))
    anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)
    
    # compute score
    img_mask = resize(perturb_image(np.ones((img.shape[0],img.shape[1],1)), mask, superpixels), (og_size[0], og_size[1])).squeeze().astype('bool')
    if len(anns) < 1:
        return None
    ann_scores = [calculate_score(img_mask, coco.annToMask(ann).astype('bool')) for ann in anns]
    score = max(ann_scores)
    best_ann = np.argmax(ann_scores)
    area_ratio = np.sum(coco.annToMask(anns[best_ann])) / (og_size[0] * og_size[1])

    if False:
        high_score = score
        plt.figure(figsize=(10,20))
        plt.subplot(1,2,1)
        plt.title('score = %.2f, area_ratio = %.4f, idx = %d, label = %s' % (score, area_ratio, idx, label))
        plt.imshow(og_img)
        coco.showAnns([anns[best_ann]])
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(resize(img, og_size))
        plt.axis('off')
        plt.show()
        
    return { 'score': score, 'label': label, 'area_ratio': area_ratio, 'idx': idx }
