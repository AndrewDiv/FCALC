import numpy as np

def alpha_weak(supp_cont, class_lengths, alpha=0.):
    ccl = class_lengths.sum() - class_lengths
    criter = np.zeros(shape=(class_lengths,supp_cont[0].shape[1]))
    for j in range(len(class_lengths)):
        criter[j] = (supp_cont[j][1] <= ccl[j] * alpha).sum(axis=-1)
    
    criter /= class_lengths
    # if (criter == max(criter)).sum() > 1:
    #     return -1, criter

    return np.argmax(criter, axis=-1) #, criter