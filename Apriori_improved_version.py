#!/usr/bin/env python
# coding: utf-8

# <b>Input :</b>
#     <br>Filename - dataset to be imported
#     <br>min_sup - minimum support
#     
# <b>Output :</b>
#     <br>All frequent itemsets;
#     <br>Total number of frequent itemsets;
#     <br>Total number of k-frequent itemsets for all k such that Freq_k is non-empty
#     
# <b>Plots :</b>
# <br>number of frequent itemsets v/s minimum support;
# <br>log(number of frequent itemsets) v/s minimum support;
# <br>execution time v/s minimum support

# <h3>Some of the improvements done to the original Apriori code are: </h3> 
# 
# <ul>1. We eleminated those elements from the transactions databse whose count is less than the support count, this was one of the most important improvements to the code. This reduced the time taken by a large factor</ul>
# 
# <ul>2. The third improvement we did was that in the loop of the apriori algorithm, if in any
# iteration the number of k-frequent itemsets becomes less than k we break, because for the k+1 itemset
# to be frequent, we need that all k+1 k subsets must be frequent.</ul> 
#  <ul>3. Similarly, if any transaction
# does not have atleast k+1 itemsets present in the hashtable we can delete the transaction from the
# database.</ul>
# <ul>4. We tried to tune the mod value such that it increases the branching factor of the hashtree. Initially we set it to a low value, 2, but increasing it to a value around 100 decreases the running time considerably, by around 4x<ul>

# In[77]:


#Importing Modules
import numpy as np
import sys
import csv
import pandas as pd
import itertools
import time
from matplotlib import pyplot as plt
import math


# In[78]:


#Global Variables
min_sup = 0.7
mod = 50
dataset='forests.txt'


# In[79]:


'''
Function to generate the hash tree, Given a set of candidate itemsets C, and the length k of  itemsets
Output -> Hashtree (A dictionary), and list with count of each candidate itemset
'''
def generateHashTree(C, k):
    hashtree = {}
    temp = {}
    temp = hashtree
    list_counts = []
    for itemset in C:
        temp = hashtree
        for i in range(0, k-1):
            if itemset[i]%mod in temp:
                temp = temp[itemset[i]%mod]
            else:
                temp[itemset[i]%mod] = {}
                temp = temp[itemset[i]%mod]
        if itemset[k-1]%mod in temp:
            a = [itemset, 0]
            temp[itemset[k-1]%mod].append(a)
            list_counts.append(a)
        else:
            temp[itemset[k-1]%mod] = []
            a = [itemset, 0]
            temp[itemset[k-1]%mod].append(a)
            list_counts.append(a)
    return hashtree, list_counts


# In[80]:


'''
# This function returns if the k-itemset is present in the hashtre
'''
def isContained(itemset, Hashtree, k):
    temp = Hashtree                       
    for i in range(0, k):
        if itemset[i]%mod in temp:         #If the subtree corresponding to hash of i'th element of itemset is present in the hashtree, recursively set temp = that subtree of temp
            temp = temp[itemset[i]%mod]
        else:
            return False
    for search_item in temp:               # Once all the previous k-1 elements have been confirmed to be in the tree, check if the last element is present in any itemset of the leaf
        if(itemset == search_item[0]):
            if(search_item[1]>min_sup * num_transactions):
                return True
            else:
                return False
    return False


# In[81]:


'''
This function generates subsets of size k of a given itemset
'''
def gen_ksubsets(itemset, k):
    subsets = []
    for i in itemset:
        s = []
    for j in itemset:
        if(j==i):
            continue
        else:
            s.append(j)
        subsets.append(s)
    return subsets


# In[82]:


'''
This function generates the candidate itemsets of length k+1 from frequent itemsets of lenght k
'''
def apriori_gen(F, k, Hashtree):
    # If k=1, create candidate 2 itemsets by using join. No pruning is required in this case
    if(k==1):
        C = []
        for i1 in range(0, len(F)):
            for i2 in range(i1+1, len(F)):
                Ck = []
                Ck.append(min(F[i1], F[i2]))
                Ck.append(max(F[i1], F[i2]))
                C.append(Ck)
                
        return C
    else:
        C = []
        #insertion into C_k using self_joining
        for i1 in range(0, len(F)):
            for i2 in range(i1+1, len(F)):
                j=0
                while(j<k-1):
                    if(F[i1][j]!=F[i2][j]):
                        break
                    j+=1
                if(j==k-1):
                    Ck = []
                    for j in range(0, k-1):
                        Ck.append(F[i1][j])
                    Ck.append(min(F[i1][k-1], F[i2][k-1]))
                    Ck.append(max(F[i1][k-1], F[i2][k-1]))
                    C.append(Ck)
    #print("no of candidates of length ",k+1," = ",len(C))
    #Pruning step
    for itemset in C:
        for ksubset in gen_ksubsets(itemset, k+1):
            if(isContained(ksubset, Hashtree, k)==False):
                C.remove(itemset)
                break
    #print("no of candidates of length  ",k+1," after pruning  = ",len(C))
    return C


# In[83]:


'''
Function to print the number of frequent-itemsets of respective sizes k
'''
def print_freq(A):
    tot=0
    for i in range(len(A)):
        print ("Frequent itemsets of size ",i+1," = ",len(A[i]))
        print("\n")
        tot+=len(A[i])
    print("Total number of itemsets = ",tot)


# In[84]:


#Function to update the counts of each 
def update_counts(transactions, hashtree, k):
    #generate all k subsets transactions and update counts
    new_transactions = []
    for transaction in transactions:
        count = 0
        k_subset_list = list(itertools.combinations(transaction, k))
        temp = hashtree
        for ksubset in k_subset_list:
            temp = hashtree
            i=0
            while(i<k):
                if(ksubset[i]%mod in temp):
                    temp = temp[ksubset[i]%mod]
                else:
                    break
                i+=1
            if(i==k):
                for search_item in temp:
                    if(list(ksubset)==search_item[0]):
                        search_item[1]+=1
                        count+=1
        if(count>=k+1):
            new_transactions.append(transaction)
    transactions = new_transactions
    return transactions


# In[85]:


def advanced(min_sup,dataset):
    transactions = []
    file_reader = open(dataset)
    for line in file_reader:
        l = [int(x) for x in line.split()]
        transactions.append(l)
    C = []
    F = []
    max_id = -1
    global num_transactions
    num_transactions = len(transactions)
    for transaction in transactions:
        transaction.sort()
        for id in transaction:
            if(id>max_id):
                max_id = id
    count = [0]*(max_id+1)
    for transaction in transactions:
        for id in transaction:
            count[id]+=1
    new_transactions=[]
    for ls in transactions:
        l=[]
        for item in ls:
            if(count[item]>=min_sup*num_transactions):
                l.append(item)
        new_transactions.append(l)
    transactions = new_transactions.copy()
    new_transactions.clear()

    F1=[] #Frequent 1-itemset
    C1 = []
    for i in range(0, len(count)):
        C1.append(i)
        if(count[i] >= min_sup* num_transactions):
            F1.append(i)
    F.append(F1)
    C.append(C1)
    Freq_k = F1
    k=2
    oldHashTree = {}
    for i in range(0, mod):
        oldHashTree[i]=[]
    for i in range(1, max_id+1):
        oldHashTree[i%mod].append([i, count[i]])
    while(len(Freq_k)!=0):
        Candidate_gen = apriori_gen(Freq_k, k-1, oldHashTree)
        #print(Candidate_gen)
        if(len(Candidate_gen)==0):
            break
        newHashTree, list_counts = generateHashTree(Candidate_gen, k)
        transactions = update_counts(transactions, newHashTree, k)
        Freq_k = []
        for item in list_counts:
            if(item[1]>=min_sup*num_transactions):
                Freq_k.append(item[0])
        #print(len(Freq_k))
        #print(*Freq_k, sep='\n')
        if(len(Freq_k)<k+1):
            break
        F.append(Freq_k)
        k+=1
        oldHashTree = newHashTree.copy()
        newHashTree.clear()
    return F


# In[86]:


#This function gives the total number of frequent items when the list of lists of Frequent itemsets passed as argument
def no_freq(g):
    a=0
    for i in range(len(g)):
        a+= (len(g[i]))
    return a


# In[87]:


get_ipython().run_cell_magic('time', '', "transactions = []\nfile_reader = open(dataset)\nfor line in file_reader:\n    l = [int(x) for x in line.split()]\n    transactions.append(l)\nC = []\nF = []\nmax_id = -1\nglobal num_transactions\nnum_transactions = len(transactions)\nfor transaction in transactions:\n    transaction.sort()\n    for id in transaction:\n        if(id>max_id):\n            max_id = id\ncount = [0]*(max_id+1)\nfor transaction in transactions:\n    for id in transaction:\n        count[id]+=1\nnew_transactions=[]\nfor ls in transactions:\n    l=[]\n    for item in ls:\n        if(count[item]>=min_sup*num_transactions):\n            l.append(item)\n    new_transactions.append(l)\ntransactions = new_transactions.copy()\nnew_transactions.clear()\n\nF1=[] #Frequent 1-itemset\nC1 = []\nfor i in range(0, len(count)):\n    C1.append(i)\n    if(count[i] >= min_sup* num_transactions):\n        F1.append(i)\nF.append(F1)\nC.append(C1)\nFreq_k = F1\nk=2\noldHashTree = {}\n#print(*F1,sep='\\n')\nfor i in range(0, mod):\n    oldHashTree[i]=[]\nfor i in range(1, max_id+1):\n    oldHashTree[i%mod].append([i, count[i]])\nwhile(len(Freq_k)!=0):\n    Candidate_gen = apriori_gen(Freq_k, k-1, oldHashTree)\n    #print(Candidate_gen)\n    if(len(Candidate_gen)==0):\n        break\n    newHashTree, list_counts = generateHashTree(Candidate_gen, k)\n    transactions = update_counts(transactions, newHashTree, k)\n    Freq_k = []\n    for item in list_counts:\n        if(item[1]>=min_sup*num_transactions):\n            Freq_k.append(item[0])\n    print(len(Freq_k))\n    print(*Freq_k, sep='\\n')\n    if(len(Freq_k)<k+1):\n        break\n    F.append(Freq_k)\n    k+=1\n    oldHashTree = newHashTree.copy()\n    newHashTree.clear()\nprint_freq(F)")


# In[88]:


x = [0.75,0.8,0.85,0.9]


def y(x,dataset):
    result=no_freq(advanced(x,dataset))
    return result
def hh(b):
    l=[]
    for i in b:
        l.append(int(y(i,dataset)))
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on Apriori for '+dataset)
plt.xlabel('minimum support')
plt.ylabel('number of frequent itemsets')
plt.show()


# In[89]:


x = [0.75,0.8,0.85,0.9]


def y(x):
    result=math.log(no_freq(advanced(x,dataset)))
    return result
def hh(b):
    l=[]
    for i in b:
        l.append(int(y(i)))
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on Apriori for '+dataset)
plt.xlabel('minimum support')
plt.ylabel('log(number of frequent itemsets)')
plt.show()


# In[90]:


x = [0.75,0.8,0.85,0.9]

def y(x):
    
    result=math.log(no_freq(advanced(x,dataset)))
    return result
def hh(b):
    l=[]
    for i in b:
        start = time.clock()
        tre = y(i)
        end = time.clock()
        l.append(end-start)
        
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on Apriori for '+dataset)
plt.xlabel('minimum support')
plt.ylabel('Time taken in seconds')
plt.show()


# In[ ]:




