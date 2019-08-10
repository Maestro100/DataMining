#!/usr/bin/env python
# coding: utf-8

# In[107]:


'''
input
    filename - dataset to be imported
    min_sup - minimum support
    
output
    All frequent itemsets
    Total number of frequent itemsets
    Total number of k-frequent itemsets for all k
    
plots
    number of frequent itemsets v/s minimum support
    log(number of frequent itemsets) v/s minimum support
    execution time v/s minimum support
'''
#import modules
import collections
import itertools as itls
from matplotlib import pyplot as plt
import numpy as np
import math


# In[108]:


#input arguments
filename = 'forests.txt'                #database to use
min_sup=0.5                            #minimum-support


# In[109]:


#fucntion to read transactions from the dataset
def extract(fn):
    with open(fn,"r") as f:
        itss = [[it for it in line.strip().split()] for line in f]
    return itss


# In[110]:


#this function returns all the frequent itemsets
def fr_itss(itss,minsup):
    tree = build(itss,minsup)[0]                  #here we call the build function which builds the FP-Tree for a given itemset and minimum support
    for its in fpg(tree,minsup):                  #here we call the fpg function: FP-growth algorithm
        yield its


# In[111]:


#this function performs the FP-Growth algorithm
def fpg(tree,minsup):
    itms = tree.nodes.keys()
    if tree.p_ex_t:                                         #calls the path existence property of the FP-tree class
        for i in range(1,len(itms)+1):
            for its in itls.combinations(itms,i):
                yield tree.conditional_items + list(its)    
    else:
        for it in itms:
            yield tree.conditional_items + [it]
            ct = tree.cond_tree(it,minsup)
            for its in fpg(ct,minsup):                     #recursively calling fpg function
                yield its


# In[112]:


#class to define node in FP-tree
class fp_node(object):
    def __init__(self, it, cnt=1, prnt=None):
        self.it = it                                      #item in the node
        self.cnt=cnt                                      #frequency of the itemset
        self.prnt = prnt                                  #parent of the itemset
        self.children = collections.defaultdict(fp_node)  #children of the itemset, same logic as in the fp tree constructor, recursive definition
        if prnt!=None:
            prnt.children[it]=self
    def root_path(self):                                  #returns the path from root to the node in the FP-tree
        pt = []
        if self.it == None:
            return pt
        node = self.prnt
        while node.it!=None:
            pt.append(node.it)
            node = node.prnt
        pt.reverse()
        return pt


# In[113]:


#class to define the structure for FP-tree
class fp_tree(object):
    def __init__(self,mapping=None):
        self.root = fp_node(None)                   #node of fp_tree
        self.mapping = mapping                      #mapping to handle string datatype in itemset
        self.nodes = collections.defaultdict(list)  #works as a dictionary except that it will create a new list if entry is not found, for more information refer https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work
        self.conditional_items = []                 #conditional itemset
    def add_its(self,its,cnt=1):                    #function to add itemset in the FP-tree
        if len(its)==0:
            return
        id = 0
        node = self.root
        for it in its:
            if it in node.children:
                cd = node.children[it]
                cd.cnt+=cnt
                node =cd
                id+=1
            else:
                break
        for it in its[id:]:
            cd_node = fp_node(it,cnt,node)
            self.nodes[it].append(cd_node)
            node = cd_node
            
    '''
    function to make conditional tree given the FP-tree
    inputs: Fp-tree, conditional items on which we have to make the conditional tree and minimum support
    returns the conditional tree
    '''
    def cond_tree(self,conditional_items_to_add,minsup):    
        its_cnt = collections.defaultdict(int)                    #its_cnt stores the count of itemset
        child_paths = []                                          #to store all the root paths of the nodes
        for node in self.nodes[conditional_items_to_add]:         #loop to store all the root paths to the nodes on which conditional tree is being made
            child_path = node.root_path()                         
            child_paths.append(child_path)                      
            for element in child_path:                           #increase the count of element by their count in the node
                its_cnt[element] += node.cnt
        items = [it for it in its_cnt if its_cnt[it]>=minsup]    #taking out conditional frequent itemsets
        items.sort(key=its_cnt.get)
        mapping = {it:i for i,it in enumerate(items)}            #creating mapping for conditional frequent items
        cd_tree = fp_tree(mapping)                               #making conditional tree using the fp_tree
        for i,child_path in enumerate(child_paths):              #adding paths in conditional FP-tree
            child_path =sorted([it for it in child_path if it in mapping], key=mapping.get, reverse=True)
            cd_tree.add_its(child_path, self.nodes[conditional_items_to_add][i].cnt)
        cd_tree.conditional_items = self.conditional_items + [conditional_items_to_add]
        return cd_tree
    
    @property
    def p_ex_t(self):
        if len(self.root.children)>1:
            return False
        for i in self.nodes:
            if len(self.nodes[i])>1 or len(self.nodes[i][0].children)>1:
                return False
        return True


# In[114]:


#this function makes a FP-tree for the itemsets with a given minimum support
def build(itss,minsup):
    cnt = collections.defaultdict(int)
    for it in itls.chain.from_iterable([its for its in itss]):
        cnt[it]+=1
    itms = sorted([it for it in cnt if cnt[it]>=minsup],key=cnt.get) #1-frequent itemsets
    mapping = {it:i for i,it in enumerate(itms)}                     #mapping same as the FP-tree constructor
    itss = [[it for it in its if it in mapping] for its in itss]     #itemsets in the form of numbers
    
    build_tree = fp_tree(mapping)                                    #FP-tree function made using the mapping
    for its in itss:
        its.sort(key=mapping.get,reverse=True)
        build_tree.add_its(its)
    return build_tree,mapping                                        #return the FP-tree with the mapping


# In[115]:


#Main function - Here the main code begins which calls the above functions defined
itss = extract(filename)
f_itss = [its for its in fr_itss(itss,min_sup*(len(itss)))]  #all frequent itemsets are obtained here
f_itss.sort(key=len)      #sorting all frequent itemsets according to there lengths
print("All frequent itemsets are:")
sum_k=0
loop=0
pre_length=0
arr = []                 #stores the count of k-frequent itemsets for all k
for itms in f_itss:
    if(loop==0):
        sum_k=1
        pre_length=len(itms)
        loop=1
    elif(pre_length==len(itms)):
        sum_k+=1
    else:
        arr.append(sum_k)
        pre_length=len(itms)
        sum_k=1
    print(itms)
arr.append(sum_k)    
print("Total number of frequent itemsets = "+str(len(f_itss)))
for i in range(len(arr)):
    print("Total number of frequent "+str(i+1)+"-itemsets = "+str(arr[i]))


# In[121]:


x = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
itss = extract(filename)
def y(x):
    f_itss = [its for its in fr_itss(itss,x*(len(itss)))]
    return len(f_itss)
def hh(b):
    l=[]
    for i in b:
        l.append(int(y(i)))
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on fp-tree for '+filename)
plt.xlabel('minimum support')
plt.ylabel('number of frequent itemsets')
plt.show()


# In[117]:


x = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
itss = extract(filename)
def y(x):
    f_itss = [its for its in fr_itss(itss,x*(len(itss)))]
    if(len(f_itss)==0):
        return 0
    else:
        return math.log(len(f_itss))
def hh(b):
    l=[]
    for i in b:
        l.append(int(y(i)))
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on fp-tree for '+filename)
plt.xlabel('minimum support')
plt.ylabel('log(number of frequent itemsets)')
plt.show()


# In[122]:


x = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
itss = extract(filename)
def y(x):
    f_itss = [its for its in fr_itss(itss,x*(len(itss)))]
    if(len(f_itss)==0):
        return 0
    else:
        return math.log(len(f_itss))
def hh(b):
    l=[]
    for i in b:
        start = time.clock()
        tre = y(i)
        end = time.clock()
        l.append(end-start)
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on fp-tree for '+filename)
plt.xlabel('minimum support')
plt.ylabel('Time taken in seconds')
plt.show()


# In[ ]:




