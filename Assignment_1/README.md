To run the decision tree implemented: python decision_tree.py

To run the decision tree implemented using scikit-learn: python decision_tree_library.py

Decision tree can be implemented using the Scikit learn library. The sklearn.tree.DecisionTreeClassifier( ) method can be used to create a decision tree based on entropy or gini-index. The fit( ) method is used to generate the tree.

Drawback: This cant handle categorical attributes. This can be overcome using encoding. Encoding can be done using LabelEncoder or One-hot encoding. One-hot encoding has been used by creating dummy variables as follows:

```
import pandas as pd 
dum=pd.get_dummies(dat[head]).rename(columns=lambda x: head+'_' + str(x)) dat=pd.concat([dat, dum], axis=1)
dat=dat.drop([head], axis=1)
```

Improvements:
The termination criteria can be added to reduce the number of splits when there are larger number of values for an attribute. To avoid overfitting, prepruning can be done which is usually implemented in scikit DecisionTreeClassifier.


