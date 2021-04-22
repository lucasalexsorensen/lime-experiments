import copy
import sklearn.metrics
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from PIL import Image
from skimage.io import imread
import sklearn.metrics
from skimage.transform import resize
from skimage.segmentation import quickshift, felzenszwalb
import random
import numpy as np
import pandas as pd
from consts import dataDir, dataType

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
def evaluate_set(val_df: pd.DataFrame, model, coco, **kwargs):
    """
    Keyword arguments:
    n       -- amount of perturbations to generate for every image (integer)
    seg     -- image segmentation algorithm ('quickshift', 'felzenszwalb')
    reg     -- regression model ('linear', 'lasso', 'ridge')
    sel     -- superpixel feature selection (3,'auto')
    weights -- sample_weight enabled (true, false)
    """
    kwargs.setdefault('n', 200)
    kwargs.setdefault('seg', 'quickshift')
    kwargs.setdefault('reg', 'linear')
    kwargs.setdefault('sel', 'auto')
    kwargs.setdefault('weights', 'true')

    result = val_df.progress_apply(lambda r: evaluate_image(r, model, coco, **kwargs), axis=1, result_type='expand')
    return (result, kwargs)

# evaluate a single image (i.e. row of the DF)
def evaluate_image(row, model, coco, **kwargs):
    idx, label, imagenet_class_to_explain = row.idx, row.label, row.imagenet_class_to_explain
    I = coco.loadImgs(row.imgId)[0]

    img = imread('%s/images/%s/%s' % (dataDir,dataType,I['file_name']))
    og_size = img.shape
    img = resize(img, (299,299)) 
    X = (img - 0.5)*2 # Inception pre-processing
    
    # perform LIME
    seg_algs = {
        'quickshift': (quickshift, { 'kernel_size': 5, 'max_dist': 300, 'ratio': 0.2 }),
        'felzenszwalb': (felzenszwalb, { 'scale': 100, 'sigma': 0.5, 'min_size': 200 })
    }
    seg_alg, seg_args = seg_algs.get(kwargs['seg'])
    superpixels = seg_alg(img, **seg_args) # , kernel_size=5, max_dist=300, ratio=0.2)
    N = np.unique(superpixels).shape[0]
    num_perturb = int(kwargs['n'])
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
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) # Kernel function
    
    reg_algs = {
        'linear': LinearRegression(),
        'lasso': Lasso(),
        'ridge': Ridge()
    }
    simpler_model = reg_algs.get(kwargs['reg'])

    fit_args = {}
    if kwargs.get('weights') == 'true':
        fit_args['sample_weight'] = weights

    simpler_model.fit(X=perturbations, y=predictions[:, imagenet_class_to_explain], **fit_args)
    coeff = simpler_model.coef_

    if kwargs.get('sel') == 'auto':
        lag = np.diff(-np.sort(-coeff), n=2)
        num_top_features = np.where(lag > 0)[0][0] + 1
    else:
        num_top_features = int(kwargs.get('sel'))

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
        return { 'score': None, 'label': None, 'area_ratio': None, 'idx': None }
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
