# FCALC
A Python module to solve a classification problem using lazy learning and pattern structures from FCA. This library allows to work with tabular data with numeric and categorical features.

## Github contents
* data_sets - Contains some examples of data
* fcalc - Python module with classification tools
    * classifier.py -  Defines classifier classes for binarized and non-binarized data
    * decision_functions.py - Contains decision functions
* exmp.ipynb - Detailed example of module usage

## Task description

Suppose, we have data $X$ and the corresponding labels $Y$ where each label $y \in Y$. The task is to classify object $x$ into one of the classes $c$.

## Algorithm

Assume that we want to make a prediction for object $x$ given the set of training examples $X_{train}$ and the labels $y_i$ corresponding to each $x_i \in X_{train}$.

First, we split $X_{train}$ into non-intersecting subsets $X_c$ : $\\{x_i \in X_c \mid y_i = c\\}$ 

To classify the object $x$ we follow the procedure:
1) For each subset $X_c$ and each $x_i \in X_c$ we compute the intersection $x \sqcap x_i$.
2) Then, we calculate the characteristics of each intersection (support, size, etc.)
3) We pass the obtained characteristic values into decision function and classify $x$ based on them.  

Also there is a randomized version of the algorithm. The randomization is defined by 2 parameters: subsample size and number of iterations. Subsample size defines the number of objects in each intersection and number of iterations defines how many intersections to sample from each class.

## Decision functions

### General
* Non-falsified (```method = "standard"```)
$$y = \underset{c}{\text{argmax}}\left(\dfrac{\sum\limits_{x_i \in X_{c}}[b_i = 0]}{|X_{c}|}\right)$$
* Non-falsified support (```method = "standard-support"```)
$$y = \underset{c}{\text{argmax}}\left(\dfrac{\sum\limits_{x_i \in X_{c}}a_i \cdot [b_i = 0]}{|X_{c}|^2}\right)$$
* Ratio-support (```method = "ratio-support"```)
$$y = \underset{c}{\text{argmax}} \left(\dfrac{|X_{train} \setminus X_c|\cdot \sum\limits_{x_i \in X_{c}} a_i \cdot [\frac{a_i}{|X_c|} \geq \gamma \frac{b_i}{|X_{train} \setminus X_{c}|}]}{|X_{c}|\cdot \sum\limits_{x_i \in X_c} b_i\cdot[\frac{a_i}{|X_c|} \geq \gamma \frac{b_i}{|X_{train} \setminus X_c|}]}  \right)$$
### Numeric-only data
* Proximity (```method = "proximity"```)
$$y = \underset{c}{\text{argmax}}\left(\dfrac{ \sum\limits_{x_i\in X_c}p_i}{|X_c|}\right)$$
* Proximity non-falsified (```method = "proximity-non-falsified"```)
$$y = \underset{c}{\text{argmax}}\left(\dfrac{ \sum\limits_{x_i\in X_c}p_i\cdot [b_i = 0]}{|X_c|}\right)$$
* Proximity-support (```method = "proximity-support"```)
$$y = \underset{c}{\text{argmax}}\left(\dfrac{ \sum\limits_{x_i\in X_c}a_i \cdot p_i \cdot [b_i = 0]}{|X_c|}\right)$$
Here $a_i$ is support in target class, $b_i$ - support in non-target class, and $p_i$ is proximity of intersection $x \sqcap x_i$

## Example
Let's start with importing the libraries
```python
import fcalc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
```
### Binarized data
I will use tic-tac-toe dataset as an example for the binarized data. Firstly we need to read our data and binarize it:
```python
column_names = [
        'top-left-square', 'top-middle-square', 'top-right-square',
        'middle-left-square', 'middle-middle-square', 'middle-right-square',
        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square',
        'Class'
    ]
df = pd.read_csv('data_sets/tic-tac-toe.data', names = column_names)
df['Class'] = [x == 'positive' for x in df['Class']]
X = pd.get_dummies(df[column_names[:-1]], prefix=column_names[:-1]).astype(bool)
y = df['Class']
```
Then we need to split our data into train and test:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Then we can initialize our classifier, to do so, you need to provide training data and training labels, both in numpy format:
```python
bin_cls = fcalc.classifier.BinarizedBinaryClassifier(X_train.values, y_train.to_numpy())
```
You can also specify the method and parameter for that method. Now you can predict the classes for your test data and evaluate the models:
```python
bin_cls.predict(X_test.values)
print(accuracy_score(y_test, bin_cls.predictions))
print(f1_score(y_test, bin_cls.predictions))
```
>0.9965

>0.9974

**For  more detailed example check exmp.ipynb**