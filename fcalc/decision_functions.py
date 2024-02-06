import numpy as np

def alpha_weak(supp_cont,classes, class_lengths, alpha=0.):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    for j in range(len(classes)):
        criter[j] = (supp_cont[j][1] <= ccl[j] * alpha).sum(axis=-1)
    criter = criter.T / class_lengths
    return classes[np.argmax(criter, axis=-1)]

def alpha_weak_support(supp_cont, classes, class_lengths, alpha=0.):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    for j in range(len(classes)):
        criter[j] = (supp_cont[j][0]*(supp_cont[j][1] <= ccl[j] * alpha)).sum(axis=-1)
    criter = criter.T / class_lengths**2
    return classes[np.argmax(criter, axis=-1)]

def ratio_support(supp_cont, classes, class_lengths, alpha=1.):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(len(class_lengths),supp_cont[0].shape[1]))
    for j in range(len(classes)):
        sup = (supp_cont[j][0]*(supp_cont[j][1]/ccl[j] * alpha < supp_cont[j][0]/class_lengths[j])).sum(axis=-1)
        cont = (supp_cont[j][1]*(supp_cont[j][1]/ccl[j] * alpha < supp_cont[j][0]/class_lengths[j])).sum(axis=-1)+1e-6
        criter[j] = sup / cont * (ccl[j]/class_lengths[j])
    return classes[np.argmax(criter.T, axis=-1)]