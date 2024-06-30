import numpy as np

def non_falsified(supp_cont,classes, class_lengths, alpha=0., randomize=False):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    preds = np.full(supp_cont[0].shape[1], -1.)
    for j in range(len(classes)):
        criter[j] = (supp_cont[j][1] <= ccl[j] * alpha).sum(axis=-1)
    if randomize:
        criter = criter.T / supp_cont[0].shape[-1]
    else:
        criter = criter.T / class_lengths
    pred_mask = (np.max(criter,axis=-1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def non_falsified_support(supp_cont, classes, class_lengths, alpha=0., randomize=False):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    preds = np.full(supp_cont[0].shape[1], -1.)
    for j in range(len(classes)):
        criter[j] = (supp_cont[j][0]*(supp_cont[j][1] <= ccl[j] * alpha)).sum(axis=-1)
    if randomize:
        criter = criter.T / (supp_cont[0].shape[-1]*class_lengths)
    else:
        criter = criter.T / class_lengths**2
    pred_mask = (np.max(criter,axis=-1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def ratio_support(supp_cont, classes, class_lengths, alpha=1., randomize=False):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    preds = np.full(supp_cont[0].shape[1], -1.)
    for j in range(len(classes)):
        sup = (supp_cont[j][0]*(supp_cont[j][1]/ccl[j] * alpha <= supp_cont[j][0]/class_lengths[j])).sum(axis=-1)
        cont = (supp_cont[j][1]*(supp_cont[j][1]/ccl[j] * alpha <= supp_cont[j][0]/class_lengths[j])).sum(axis=-1)+1e-6
        criter[j] = (ccl[j]*sup) / (cont*class_lengths[j])
    criter = criter.T
    pred_mask = (np.max(criter,axis=-1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def proximity_based(proximity, classes):
    preds = np.full(proximity[0].shape[0], -1.)
    criter = np.zeros(shape=(len(classes),proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = proximity[j].mean(axis=1)
    criter = criter.T
    pred_mask = (np.max(criter,axis=1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def proximity_non_falsified(proximity, support, classes, class_lengths, alpha):
    ccl = class_lengths.sum() - class_lengths
    preds = np.full(proximity[0].shape[0], -1.)
    criter = np.zeros(shape=(len(classes),proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = (proximity[j]*(support[j][1] <= ccl[j] * alpha)).mean(axis=1)
    criter = criter.T
    pred_mask = (np.max(criter,axis=1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def proximity_support(proximity, support, classes, class_lengths, alpha):
    ccl = class_lengths.sum() - class_lengths
    preds = np.full(proximity[0].shape[0], -1.)
    criter = np.zeros(shape=(len(classes),proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = (support[j][0]*proximity[j]*(support[j][1] <= ccl[j] * alpha)).sum(axis=1)
    criter = criter.T / class_lengths
    pred_mask = (np.max(criter,axis=1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds