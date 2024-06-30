from . import patterns
from . import decision_functions 
import numpy as np
from math import ceil
from numbers import Integral, Real

class FcaClassifier:
    '''
    A base class of support based FCA slassifiers
 
    Attributes
    ----------
    context : list, numpy.ndarray
        Features of objects with a known class labels
    labels : list, numpy.ndarray
        Labels of the objects
    support : None, numpy.ndarray
        Precomputed support or None
    '''
    def __init__(self, context, labels, support = None):
        '''
        Initializes FcaClassifier object
 
        Parameters
        ----------
        context : list, numpy.ndarray
            Features of objects with a known class labels
        labels : list, numpy.ndarray
            Labels of the objects
        support : None, numpy.ndarray
            Precomputed support or None
        '''
        if not isinstance(context, (list, np.ndarray)):
            raise TypeError('Context should be of type list or numpy.ndarray')
        
        if isinstance(context, np.ndarray):
            self.context = context
        else:
            self.context = np.asanyarray(context)
        
        if isinstance(labels, np.ndarray):
            self.labels = labels
        else:
            self.labels = np.asanyarray(labels)
        
        if support is None:
            self.support = []
        else:
            self.support = support
                        
class BinarizedClassifier(FcaClassifier):
    '''
    FCA support based classifier for classification of binarized data
 
    Attributes
    ----------
    context : list, numpy.ndarray
        Binarized features of objects with a known class labels
    labels : list, numpy.ndarray
        Labels of the objects
    support : None, numpy.ndarray
        Precomputed support or None
    method : str
        Name of classification method
    alpha : float 
        Hyperparameter of the method
    classes : numpy.ndarray
        Array of possible classes
    class_lengths : numpy.ndarray
        Array of sizes of each class
    '''

    def __init__(self, context, labels, support=None, method="standard", alpha=0.):
        '''
        Initializes BinarizedClassifier object
 
        Parameters
        ----------
        context : list, numpy.ndarray: 
            Features of objects with a known class labels
        labels : list, numpy.ndarray 
            Labels of the objects
        support : None, numpy.ndarray
            Precomputed support or None
        method : str
            Name of classification method
        alpha : float
            Hyperparameter of the method
        '''      
        super().__init__(context, labels, support)
        self.classes = np.unique(labels)
        self.class_lengths = np.array([len(self.context[self.labels == c]) for c in self.classes])
        self.method = method
        self.alpha = alpha

    def compute_support(self, test):
        '''
        Computes support for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description (binarized)
        '''
        for c in self.classes:
            train_pos = self.context[self.labels == c]
            train_neg = self.context[self.labels != c]

            positive_support = np.zeros(shape=(len(test), len(train_pos)))
            positive_counter = np.zeros(shape=(len(test), len(train_pos)))

            for i in range(len(test)):
                intsec_pos = test[i].reshape(1, -1) & train_pos
                n_support_pos = ((intsec_pos @ (~train_pos.T)) == 0).sum(axis=1)
                n_counter_pos = ((intsec_pos @ (~train_neg.T)) == 0).sum(axis=1)

                positive_support[i] = n_support_pos
                positive_counter[i] = n_counter_pos

            self.support.append(np.array((positive_support, positive_counter)))

    def predict(self, test):
        '''
        Predicts the class labels for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description (binarized)
        '''
        if not self.support:
            self.compute_support(test)

        if self.method == "standard":
            self.predictions = decision_functions.alpha_weak(self.support, self.classes, 
                                                             self.class_lengths, self.alpha)
        elif self.method == "standard-support":
            self.predictions = decision_functions.alpha_weak_support(self.support, self.classes, 
                                                                     self.class_lengths, self.alpha)
        elif self.method == "ratio-support":
            self.predictions = decision_functions.ratio_support(self.support, self.classes, 
                                                                self.class_lengths, self.alpha)

class PatternClassifier(FcaClassifier):
    '''
    FCA support based classifier for classification using pattern structures
 
    Attributes
    ----------
    context : list, numpy.ndarray
        Binarized features of objects with a known class labels
    labels : list, numpy.ndarray
        Labels of the objects
    support : None, numpy.ndarray
        Precomputed support or None
    categorical : list
        list of indices of columns with categorical features
    method : str
        Name of classification method
    alpha : float
        Hyperparameter of the method
    randomize : bool
        Whether use randomiztion or not
    seed : int
        Seed for the randomization
    num_iters : int
        Number of subsamples to drow
    subsample_size : float
        Size of single subsample
    '''
    def __init__(self, context, labels, support=None, categorical=None, method="standard", alpha=0.,
                 randomize=False, seed=42, num_iters=10, subsample_size=1e-2):
        '''
        Initializes PatternBinaryClassifier object
 
        Parameters
        ----------
        context : list, numpy.ndarray
            Binarized features of objects with a known class labels
        labels : list, numpy.ndarray
            Labels of the objects
        support : None, numpy.ndarray
            Precomputed support or None
        categorical : list
            List of indices of columns with categorical features
        method : str
            Name of classification method
        alpha : float
            Hyperparameter of the method
        randomize : bool
            Whether use randomization or not
        seed : int
            Seed for the randomization
        num_iters : int
            Number of subsamples to drow
        subsample_size : float
            Size of single subsample
        '''      

        super().__init__(context, labels, support)
        self.classes = np.unique(labels)
        self.class_lengths = np.array([len(self.context[self.labels == c]) for c in self.classes])
        self.method = method
        self.alpha = alpha
        if categorical is None:
            self.categorical = []
        else: 
            self.categorical = categorical
        if self.method in ["proximity", "proximity-non-falsified", "proximity-support"] and self.categorical:
            raise TypeError(f'Method {self.method} can be used with numerical data only')
        self.randomize = randomize
        self.seed = seed
        self.num_iters = num_iters
        self.subsample_size = subsample_size
        self.intersections = []

    def compute_support(self, test):
        '''
        Computes support for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description
        '''
        if self.randomize:
            for c in self.classes:
                train_pos = self.context[self.labels == c]
                train_neg = self.context[self.labels != c]

                positive_support = np.zeros(shape=(len(test), self.num_iters))
                positive_counter = np.zeros(shape=(len(test), self.num_iters)) 
                
                intsecs = [[] for _ in range(len(test))]

                rng = np.random.default_rng(seed=self.seed)

                if isinstance(self.subsample_size, Integral):
                    train_pos_sampled = np.zeros(shape=(self.num_iters,
                                                        self.subsample_size,
                                                        self.context.shape[1]),
                                                dtype=self.context.dtype)
                    for j in range(self.num_iters):
                        train_pos_sampled[j] = rng.choice(train_pos, size=self.subsample_size,
                                                          replace=False, shuffle=True)
                elif isinstance(self.subsample_size, Real):
                    samp_size_pos = ceil(self.subsample_size * train_pos.shape[0])
                    train_pos_sampled = np.zeros(shape=(self.num_iters,
                                                        samp_size_pos,
                                                        self.context.shape[1]),
                                                        dtype=self.context.dtype)
                    for j in range(self.num_iters):
                        train_pos_sampled[j] = rng.choice(train_pos, size=samp_size_pos,
                                                          replace=False, shuffle=False)
                else:
                    raise TypeError(f'Subsample size should be of type int or float, not {type(self.subsample_size)}')
                
                if len(self.categorical) == 0:
                    for i in range(len(test)):
                        for j in range(len(train_pos_sampled)):
                            
                            low = np.minimum(test[i], np.min(train_pos_sampled[j], axis=0))
                            high = np.maximum(test[i], np.max(train_pos_sampled[j], axis=0))
                            positive_support[i][j] = ((~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0).sum()
                            positive_counter[i][j] = ((~((low <= train_neg) & (train_neg <= high))).sum(axis=1) == 0).sum()
                            intsecs[i].append({'intent':np.vstack((low, high)).T})
                
                elif len(self.categorical) == self.context.shape[1]:
                    for i in range(len(test)):
                        for j in range(len(train_pos_sampled)):
                            mask = (test[i]==train_pos_sampled[j]).all(axis=0)
                            vals = test[i][mask]
                            positive_support[i][j] = sum((~(train_pos[:,mask] == vals)).sum(axis=1)==0)
                            positive_counter[i][j] = sum((~(train_neg[:,mask] == vals)).sum(axis=1)==0)
                            intsecs[i].append({'mask': mask, 'values': vals})

                else:
                    train_pos_cat =  train_pos[:,self.categorical]
                    train_pos_num = np.delete(train_pos, self.categorical, axis=1)
                    train_neg_cat =  train_neg[:,self.categorical]
                    train_neg_num = np.delete(train_neg, self.categorical, axis=1)

                    train_pos_cat_sampled =  train_pos_sampled[:,:,self.categorical]
                    train_pos_num_sampled = np.delete(train_pos_sampled, self.categorical, axis=2)

                    numeric_cols = np.delete(np.arange(self.context.shape[1]), self.categorical)

                    for i in range(len(test)):
                        for j in range(len(train_pos_sampled)):
                            mask = (test[i][self.categorical]==train_pos_cat_sampled[j]).all(axis=0)
                            vals = test[i][self.categorical][mask]
                            low = np.minimum(test[i][numeric_cols], np.min(train_pos_num_sampled[j], axis=0))
                            high = np.maximum(test[i][numeric_cols], np.max(train_pos_num_sampled[j], axis=0))

                            positive_support[i][j] = sum(((~((low <= train_pos_num) * (train_pos_num <= high))).sum(axis=1) == 0) * 
                                                         ((~(train_pos_cat[:,mask] == vals)).sum(axis=1)==0))
                            positive_counter[i][j] = sum(((~((low <= train_neg_num) * (train_neg_num <= high))).sum(axis=1) == 0) * 
                                                         ((~(train_neg_cat[:,mask] == vals)).sum(axis=1)==0))
                            intsecs[i].append({'numerical': list(zip(low, high)), 'categorical': {'mask': mask, 'values': vals}})

                self.support.append(np.array((positive_support, positive_counter)))
                self.intersections.append(intsecs)

        else:
            for c in self.classes:
                train_pos = self.context[self.labels == c]
                train_neg = self.context[self.labels != c]

                positive_support = np.zeros(shape=(len(test), len(train_pos)))
                positive_counter = np.zeros(shape=(len(test), len(train_pos))) 

                intsecs = [[] for _ in range(len(test))]

                if len(self.categorical) == 0:
                    for i in range(len(test)):
                        for j in range(len(train_pos)):
                            low = np.minimum(test[i],train_pos[j])
                            high = np.maximum(test[i],train_pos[j])
                            positive_support[i][j] = ((~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0).sum()
                            positive_counter[i][j] = ((~((low <= train_neg) & (train_neg <= high))).sum(axis=1) == 0).sum()
                            intsecs[i].append({'intent':np.vstack((low, high)).T})

                elif len(self.categorical) == test.shape[1]:
                    for i in range(len(test)):
                        for j in range(len(train_pos)):
                            intsec = patterns.CategoricalPattern(test[i], train_pos[j])
                            positive_support[i][j] = sum((~(train_pos[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)
                            positive_counter[i][j] = sum((~(train_neg[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)
                            intsecs[i].append({'mask': intsec.mask, 'values': intsec.vals})
                else:
                    train_pos_cat =  train_pos[:,self.categorical]
                    train_pos_num = np.delete(train_pos, self.categorical, axis=1)
                    train_neg_cat =  train_neg[:,self.categorical]
                    train_neg_num = np.delete(train_neg, self.categorical, axis=1)

                    for i in range(len(test)):
                        for j in range(len(train_pos)):

                            intsec_cat = patterns.CategoricalPattern(test[i][self.categorical], train_pos_cat[j])
                            intsec_num = patterns.IntervalPattern(np.delete(test[i], self.categorical), train_pos_num[j])

                            positive_support[i][j] = sum(((~((intsec_num.low <= train_pos_num) * (train_pos_num <= intsec_num.high))).sum(axis=1) == 0) * 
                                                         ((~(train_pos_cat[:,intsec_cat.mask] == intsec_cat.vals)).sum(axis=1)==0))
                            positive_counter[i][j] = sum(((~((intsec_num.low <= train_neg_num) * (train_neg_num <= intsec_num.high))).sum(axis=1) == 0) * 
                                                         ((~(train_neg_cat[:,intsec_cat.mask] == intsec_cat.vals)).sum(axis=1)==0))
                            intsecs[i].append({'numerical': list(zip(intsec_num.low, intsec_num.high)),
                                               'categorical': {'mask': intsec_cat.mask, 'values': intsec_cat.vals}})

                self.support.append(np.array((positive_support, positive_counter)))
                self.intersections.append(intsecs)

    def compute_proximity(self, test):
        self.proximity = []
        for ind in range(len(self.classes)):
            train_pos = self.context[self.labels == self.classes[ind]]
            pos_dists = np.zeros(shape=(len(test),len(self.intersections[ind][0])))
            for i in range(len(test)):
                for j in range(len(self.intersections[ind][i])):
                    pos_mask = (~((self.intersections[ind][i][j]['intent'][:,0] <= train_pos) & 
                                  (train_pos <= self.intersections[ind][i][j]['intent'][:,1]))).sum(axis=1) == 0
                    pos_dists[i][j] = 1-np.linalg.norm(train_pos[pos_mask]-test[i], axis=1).mean() / np.sqrt(self.context.shape[1])
            self.proximity.append(pos_dists)
            

    def predict(self, test):
        '''
        Predicts the class labels for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description (binarized)
        '''
        if not self.support:
            self.compute_support(test)

        if self.method == "standard":
            self.predictions = decision_functions.non_falsified(self.support, self.classes, 
                                                                self.class_lengths, self.alpha,
                                                                self.randomize)
        elif self.method == "standard-support":
            self.predictions = decision_functions.non_falsified_support(self.support, self.classes, 
                                                                        self.class_lengths, self.alpha,
                                                                        self.randomize)
        elif self.method == "ratio-support":
            self.predictions = decision_functions.ratio_support(self.support, self.classes, 
                                                                self.class_lengths, self.alpha,
                                                                self.randomize)
        elif self.method == "proximity":
            self.compute_proximity(test)
            self.predictions = decision_functions.proximity_based(self.proximity, self.classes)
        elif self.method == "proximity-non-falsified":
            self.compute_proximity(test)
            self.predictions = decision_functions.proximity_non_falsified(self.proximity, self.support,
                                                                          self.classes, self.class_lengths,
                                                                          self.alpha)
        elif self.method == "proximity-support":
            self.compute_proximity(test)
            self.predictions = decision_functions.proximity_support(self.proximity, self.support,
                                                                    self.classes, self.class_lengths,
                                                                    self.alpha)