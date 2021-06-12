"""
Group members:

Naureen Firdous
Nikitha Krishna Vemulapalli
Vijayantika Inkulla

"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class FastMap:
    def __init__(self,components=2):
        self.components=components

    def findMaxDistance(self,input_data,objs,dist_data):
        obj_a=random.randint(0,len(objs)-1)
        obj_b=None
        i=0
        maxDist=0
        while True:
            obj_b=np.argmax(dist_data[obj_a])
            maxDist=dist_data[obj_a][obj_b]
            if i==5:
                break
            i+=1
            obj_a=obj_b
        return obj_a,obj_b,maxDist

    def embedObjects(self,input_data,objs,dist_data):
        result=[]
        for j in range(self.components):
            st=[]
            obj_a,obj_b,d_a_b=self.findMaxDistance(input_data,objs,dist_data)
            for i in range(len(objs)):
                if i==obj_a:
                    st.append(0)
                elif i==obj_b:
                    st.append(d_a_b)
                else:
                    d_oi_oa=dist_data[i][obj_a]
                    d_oi_ob=dist_data[i][obj_b]
                    xi=(d_oi_oa**2+d_a_b**2-d_oi_ob**2)/(2*d_a_b)
                    st.append(xi)
            result.append(st)
            #calculate new distances
            if j!=self.components-1:
                for i in range(len(objs)):
                    for j in range(i+1,len(objs)):
                        dist_data[i][j]=np.sqrt(dist_data[i][j]**2-(st[i]-st[j])**2)
                        dist_data[j][i]=dist_data[i][j]
        rs=np.array(result)
        return rs

def main():
    input_data=np.loadtxt("fastmap_data.txt",dtype='int',delimiter="\t")
    objs=np.genfromtxt("fastmap_objs.txt",dtype='str')
    num_objs=len(objs)
    #initially store distances in an array so that access takes O(1)
    #storing it initially takes O(n)
    dist_data=np.ones((num_objs,num_objs))
    for i in range(len(input_data)):
        p1,p2,dist=input_data[i]
        #store in p1-1 and p2-1 for the sake of zero indexing
        dist_data[p1-1][p2-1]=dist
        dist_data[p2-1][p1-1]=dist
    fastMap=FastMap()
    rs=fastMap.embedObjects(input_data,objs,dist_data)
    fig, ax = plt.subplots()
    #print(rs)
    ax.scatter(rs[0,:],rs[1,:])
    for i, txt in enumerate(objs):
		    ax.annotate(txt, (rs[0,i],rs[1,i]))
    plt.show()



if __name__=="__main__":
    main()
