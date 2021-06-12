""" Team members:

Nikitha Krishna Vemulapalli
Naureen Firdous
Vijayantika Inkulla
"""

import pandas as pd
import numpy as np
import re
small=np.finfo(float).eps
def findMax(df):
    max=0
    maxClassLabel=None
    Class=df.keys()[-1]
    vals=df[Class].unique()
    for val in vals:
        if df[Class].value_counts()[val]>max:
            max=df[Class].value_counts()[val]
            maxClassLabel=val
    return maxClassLabel
def calculate_entropy(df):
    #calculates the entropy for the class label
    entropy_total=0
    Class=df.keys()[-1]
    vals=df[Class].unique()
    for val in vals:
        frac=df[Class].value_counts()[val]/len(df[Class])
        entropy_total+=-frac*np.log2(frac)
    return entropy_total
def calculate_attribute_entropy(df,attr):
    #calculates entropy of a given attribute
    Class=df.keys()[-1]
    target_vals=df[Class].unique()
    attr_vals=df[attr].unique()
    entropy=0
    for attr_val in attr_vals:
        et=0
        for target_val in target_vals:
            num=len(df[attr][df[attr]==attr_val][df[Class]==target_val])
            den=len(df[attr][df[attr]==attr_val])
            frac=num/(den+small)
            et+=-frac*np.log2(frac+small)
        frac2=den/len(df)
        entropy+=-frac2*et
    return abs(entropy)
def find_max_gain(df):
    #finds the attribute with the maximum gain
    attr_gains=[]
    total_entropy=calculate_entropy(df)
    for key in df.keys()[:-1]:
        attr_gains.append(total_entropy-calculate_attribute_entropy(df,key))
    return df.keys()[:-1][np.argmax(attr_gains)]
def getDivision(df,attr,val):
    #creates a subset of data based on split
    return df[df[attr]==val].reset_index(drop=True)
def generate_tree(df,decision_tree):
    #generates a tree for a given data frame and add it to decision_tree dictionary
    Class=df.keys()[-1]
    max_node=find_max_gain(df)
    #print(max_node)
    max_node_values=np.unique(df[max_node])
    if decision_tree is None:
        decision_tree={}
        decision_tree[max_node]={}
    for val in max_node_values:
        div=getDivision(df,max_node,val)
        if len(df.keys()[:-1])==1:
            maxClass=None
            maxnum=0
            vals=df[Class].unique()
            for val in vals:
                num=df[Class].value_counts()[val]
                if num>maxnum:
                    maxnum=num
                    maxClass=val
            decision_tree[max_node][val] = maxClass
            continue
        div=div.drop(columns=max_node)
        clValue,counts = np.unique(div[Class],return_counts=True)
        if len(counts)==1:
            decision_tree[max_node][val] = clValue[0]
        else:
            decision_tree[max_node][val] = generate_tree(div,None)

    return decision_tree

def predict(df,dat,decision_tree):
    try:
        for key in decision_tree.keys():
            val=dat[key]
            decision_tree=decision_tree[key][val]
            class_label=0
            if type(decision_tree) is dict:
                class_label=predict(dat,decision_tree)
            else:
                class_label=decision_tree
                break
        return class_label
    except:
        return findMax(df)

def main():
    input_file=open("dt_data.txt")
    dat=[]
    header=True
    for line in input_file:
        a=line.strip()
        if not a:
            continue
        if header:
            a=a[1:len(a)-1] #to check and remove '('' and  ')' in header
            header=False
        else:
            a=a[3:len(a)-1]# to store data other than header
        a=a.replace(" ","")
        dat.append(a.split(','))
    df=pd.DataFrame(np.array(dat[1:]),columns=dat[0])
    decision_tree=generate_tree(df,None)
    print('Decision tree\n',decision_tree)
    output_file=open("testfile.txt")
    test=[]
    header=True

    test_data={}
    header=True
    for line in output_file:
        test=line.strip()
        if not test:
            continue
        if header:
            test=test[1:len(test)-1]
            for x in re.split(',()',test):
                k=x.strip()
                if k:
                    test_data[k]=None
            header=False
        else:
            test=test[3:len(test)]
            test=test.split(',')
            i=0
            for k in test_data.keys():
                test_data[k]=test[i].strip()
                i+=1
            print(predict(df,test_data,decision_tree))

if __name__=="__main__":
    main()
