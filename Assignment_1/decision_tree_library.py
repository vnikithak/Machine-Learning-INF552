""" Team members:

Nikitha Krishna Vemulapalli
Naureen Firdous
Vijayantika Inkulla

"""

from sklearn import tree
import sys
import pandas as pd
import numpy as np
import time

def featureExpansion(dat,head):
	#use encoding to convert a given column to deal with categorical attributes
	dum=pd.get_dummies(dat[head]).rename(columns=lambda x: head+'_' + str(x))
	dat=pd.concat([dat, dum], axis=1)
	dat=dat.drop([head], axis=1)
	return dat

def main():

        input_file=open("dt_data.txt")
        dat=[]
        header=True
        for line in input_file:
            a=line.strip()
            if not a:
                continue
            if header:
                a=a[1:len(a)-1]
                header=False
            else:
                a=a[3:]
            a=a.replace(" ","")
            dat.append(a.split(','))
        df=pd.DataFrame(np.array(dat[1:]),columns=dat[0])

       	cols=df.columns
       	train_data=df.iloc[:,:len(cols)-1]
       	class_label=df.iloc[:,len(cols)-1:]
       	attrs=cols[:len(cols)-1]
       	target=cols[len(cols)-1:]
       	for col in attrs:
            train_data=featureExpansion(train_data,col)

       	class_label=featureExpansion(class_label,target[0])

       	clf = tree.DecisionTreeClassifier(criterion='entropy')

        clf = clf.fit(train_data, class_label)

if __name__ == "__main__":
    main()
