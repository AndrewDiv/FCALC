from . import patterns
from . import decision_functions 
import numpy as np
from sklearn.neighbors import KernelDensity

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
        list of indixes of columns with categorical features
    method : str
        Name of classification method
    alpha : float
        Hyperparameter of the method
    '''
    def __init__(self, context, labels, support=None, categorical=None, method="standard", alpha=0.,
                 kde_bandwidth=1.0, kde_kernel='gaussian', kde_leaf_size=40, kde_classwise=False):
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
            list of indixes of columns with categorical features
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
        if categorical is None:
            self.categorical = []
        else: 
            self.categorical = categorical
        self.kde_bandwidth = kde_bandwidth
        self.kde_kernel = kde_kernel
        self.kde_leaf_size = kde_leaf_size
        self.kde_classwise = kde_classwise
        self.density = []

    def estimate_density(self):
        '''
        Calculates probability density for context using 
        KernelDensity estimator from sklearn library 
        '''
        self.density = []
        if self.kde_classwise:
            for c in self.classes:
                kde = KernelDensity(bandwidth=self.kde_bandwidth,
                                    kernel=self.kde_kernel, 
                                    leaf_size=self.kde_leaf_size).fit(self.context[self.labels == c])
                self.density.append(np.exp(kde.score_samples(self.context[self.labels == c])))
        else:
            kde = KernelDensity(bandwidth=self.kde_bandwidth,
                                kernel=self.kde_kernel, 
                                leaf_size=self.kde_leaf_size).fit(self.context)
            for c in self.classes:
                self.density.append(np.exp(kde.score_samples(self.context[self.labels == c])))
            #self.density = np.exp(kde.score_samples(self.context))

    def compute_support(self, test):
        '''
        Computes support for the given test objects
 
        Parameters
        ----------
        test : list, numpy.ndarray
            Test objects description
        '''

        for c in self.classes:
            train_pos = self.context[self.labels == c]
            train_neg = self.context[self.labels != c]

            positive_support = np.zeros(shape=(len(test), len(train_pos)))
            positive_counter = np.zeros(shape=(len(test), len(train_pos))) 

            if len(self.categorical) == 0:
                for i in range(len(test)):
                    for j in range(len(train_pos)):
                        # intsec = patterns.IntervalPattern(test[i],train_pos[j])
                        # positive_support[i][j] = sum((~((intsec.low <= train_pos) * (train_pos <= intsec.high))).sum(axis=1) == 0)
                        # positive_counter[i][j] = sum((~((intsec.low <= train_neg) * (train_neg <= intsec.high))).sum(axis=1) == 0)
                        low = np.minimum(test[i],train_pos[j])
                        high = np.maximum(test[i],train_pos[j])
                        positive_support[i][j] = ((~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0).sum()
                        positive_counter[i][j] = ((~((low <= train_neg) & (train_neg <= high))).sum(axis=1) == 0).sum()

            elif len(self.categorical) == test.shape[1]:
                for i in range(len(test)):
                    for j in range(len(train_pos)):
                        intsec = patterns.CategoricalPattern(test[i], train_pos[j])
                        positive_support[i][j] = sum((~(train_pos[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)
                        positive_counter[i][j] = sum((~(train_neg[:,intsec.mask] == intsec.vals)).sum(axis=1)==0)

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
        elif self.method == "density-based":
            if not self.density:
                self.estimate_density()
            
            scaled_density = list(map(lambda x: (x-x.min())/(x.max()-x.min()) if (x.min()!=x.max()) else x, self.density)) # (self.density-self.density.min()) / (self.density.max()-self.density.min())
            
            self.predictions = decision_functions.alpha_weak_density(self.support, self.classes, 
                                                                     self.class_lengths, scaled_density,
                                                                     self.alpha)