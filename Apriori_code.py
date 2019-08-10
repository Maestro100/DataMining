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

# In[59]:


#Importing Libraries
from tqdm import tqdm_notebook as tqdm
import itertools
import time
import math
from matplotlib import pyplot as plt


# Importing Dataset

# In[60]:


dataset = 'forests.txt'       #dataset to be imported
transactions = []
f = open(dataset)


# Converting the data in to a list of transactions

# In[61]:


for line in f:
    val=line.split()
    l=[]
    for x in val:
        l.append(int(x))
    transactions.append(l)  
num_transactions = len(transactions)


# Generating subset of size k for an itemset

# In[62]:


'''
Inputs  : itemset  -> The itemset whose subsets are to be generated
          k        -> The size of the subsets of itemset to be created 

Ouutput : C_dash   -> A list of k subsets of itemset, where each subset is a list of items  
'''
def gen_ksubsets(itemset,k):
    subset_list = list(itertools.combinations(itemset, k))
    res=[]
    for subset in subset_list:
        res.append(list(subset))
    return res


# Function to check if a list is contained in another list
# <br>Eg. it checks [1,2,3] is contained in [1,2,3,4,5,6]
# <br> If it is a subset, it returns flag=1 else returns flag=0

# In[63]:


'''
Inputs  : x        -> The list which is checked to be contained in the other
          y        -> The list which could contain x 

Ouutput : flag     -> Returns 1 if x is contained in y, else 0  
'''
def isthere(x,y):
    flag = 0
    if(all(x in y for x in x)): 
        flag = 1
    return flag


# Function to generate all feasible candidate itemsets of size k+1 from frequent itemsets of size k,
# using self join and pruning based on Apriori principle

# In[64]:


'''
Inputs  : x    -> The size of the frequent itemsets to be joined
          ff   -> The set containing the frequent k itemsets

Ouutput : C_dash -> The set of candidate k+1 itemsets formed by joining itemsets from ff and then pruning 
'''

def apriori_gen(ff, k):
    Cand = []                             #Cand will be used to store all the k+1 size candidate itemsets
    for i1 in range(0, len(ff)):
        for i2 in range(i1+1, len(ff)):
            j=0
            while(j<k-1):
                if(ff[i1][j]!=ff[i2][j]): # If any of the corresponding first k-1 items are not common to both itemsets, then they are not joined to form a candidate
                    break
                j+=1
            if(j==k-1):                   # If corresponding first k-1 items are common to both itemsets
                Ck = []                   # Ck will be used to join the two frequent k itemsets 
                for p in range(0, k-1):
                    Ck.append(ff[i1][p])                 # Append the first k-1 items to Ck
                Ck.append(min(ff[i1][k-1], ff[i2][k-1])) # Append smaller of of the two final items to Ck 
                Ck.append(max(ff[i1][k-1], ff[i2][k-1])) # Append the larger one to Ck
                Cand.append(Ck)
    
    c_dash=[]                             #c_dash will contain itemsets that are not pruned
    for itemset in Cand:
        check=1
        subsets=gen_ksubsets(itemset, k)  #subsets contains all k subsets of itemset
        for ksubset in subsets:
            if ksubset not in ff:         #If any k subset of the itemset is not frequent, it is pruned as per apriori principle
                check=0
                break
        if (check==1):                    #If all k subsets of the candidate are frequent, then it is appended to c_dash
            c_dash.append(itemset)
    return c_dash


# Function to print the number of frequent-itemsets of respective sizes k

# In[65]:


def print_freq(A):
    tot=0
    for i in range(len(A)-1):
        print ("Frequent itemsets of size ",i+1," = ",len(A[i]))
        print (*A[i],sep="\n")
        print("\n")
        tot+=len(A[i])
    print("Total number of itemsets = ",tot)
        


# The below cell conatins the main_function of the program. 
# <br>Run it to generate the k_frequent itemsets

# In[66]:


def advanced(min_sup):
    F = []                              # List to conatain all the Frequent itemsets in the form of a list of list 
    k=1
    l=[]
    # The below for-loop figures out the largest item_id in the given database
    for transaction in transactions:
        big_id = max(transaction)
        l.append(big_id)
    max_id=max(l) 

    county = (max_id+1)*[0]             # Define a list to maintain counts of 1-itemsets

    for transaction in transactions:
        for id in transaction:
            county[id]+=1               # increment count of candidates in Ck+1 that are contained in transaction
    
    
    
    F1=[]                              # List to contain Frequent 1-itemsets   
    C1 = []                            # List to contain Candidate 1-itemsets = Dataset
    for i in range(0, len(county)):
        C1.append([i])                 # Append each singleton item as a list to C1 
        if(county[i] >= min_sup* num_transactions):
            F1.append([i])             # Append items with support > min_sup to F1
    F.append(F1)
    
    
    while(len(F[k-1])!=0):                       #While k-Frequent set is not empty  
        Candidate_gen = apriori_gen(F[k-1], k)   #Generate candidate itemsets of size k+1 from F_k
        if(len(Candidate_gen)==0):               #If no candidate item is left after pruning, then terminate the program
            break

        count_k=len(Candidate_gen)*[0]           #Define a list for keeping counts of candidate itemsets

        for i in range(len(Candidate_gen)):      #For each candidate k+1 itemset, increment it's support_count by going through the database      
            candy=Candidate_gen[i]
            for transaction in transactions:
                if (isthere(candy,transaction)):
                    count_k[i]+=1  

        #The below command creates a list Freq_k of frequent K+1 itemsets from among the itemsets in Candidate_gen
        Freq_k=[Candidate_gen[t]for t in range(len(Candidate_gen)) if count_k[t]>=min_sup* num_transactions]
        no_of_f=len(Freq_k)                      #no_of_f referes to the number of frequent k itemsets 
        F.append(Freq_k)
        k+=1
    return F
    print('program finished')


# Set the <b><u>minimum support</b></u> for generating frquent itemsets
# <br> Run this cell to generate the frequent itemsets

# In[67]:


#Main function to get frequent itemsets for a given support
min_sup = 0.7                       # Minimum Support
print_freq(advanced(min_sup))


# In[68]:


#This function gives the total number of frequent items when the list of lists of Frequent itemsets passed as argument
def no_freq(g):
    a=0
    for i in range(len(g)):
        a+= (len(g[i]))
    return a


# In[69]:


x = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

def y(x):
    result=no_freq(advanced(x))
    return result
def hh(b):
    l=[]
    for i in b:
        l.append(int(y(i)))
    return l
plt.plot(x,hh(x),'ro')
plt.title('Plot on Apriori for '+dataset)
plt.xlabel('minimum support')
plt.ylabel('number of frequent itemsets')
plt.show()


# In[70]:


x = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

def y(x):
    result=math.log(no_freq(advanced(x)))
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


# In[71]:


x = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

def y(x):
    
    result=math.log(no_freq(advanced(x)))
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





# In[ ]:




