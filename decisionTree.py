#!/usr/bin/env python





import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log





# dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
#        'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
#        'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
# 'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}





# df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])

df = pd.read_csv('dataset.csv')


print (df)




# entropy_node = 0
# values = df.Class.unique()
# for value in values:
#     fraction = df.Class.value_counts()[value]/len(df.Class)
# entropy_node += -fraction*np.log2(fraction)





# attribute = 'Taste'
# target_variables = df.Class.unique()
# variables = df[attribute].unique()
# entropy_attribute = 0
# for variable in variables:
#     entropy_each_feature = 0
#     for target_variable in target_variables:
#         num = len(df[attribute][df[attribute]==variable][df.Class ==target_variable])
#         den = len(df[attribute][df[attribute]==variable])
#         fraction = num/(den+eps)
#         entropy_each_feature += -fraction*log(fraction+eps)
#     fraction2 = den/len(df)
# entropy_attribute += -fraction2*entropy_each_feature
#




def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy


def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]
  target_variables = df[Class].unique()
  variables = df[attribute].unique()
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:

        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None):
    Class = df.keys()[-1]


    node = find_winner(df)

    attValue = np.unique(df[node])

    if tree is None:
        tree={}
        tree[node] = {}



    for value in attValue:

        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['Class'],return_counts=True)

        if len(counts)==1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)

    return tree





tree = buildTree(df)




print(tree)




import pprint
pprint.pprint(tree)


def predict(inst,tree):


    for nodes in tree.keys():

        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;

    return prediction

data = {'age':'youth','income':'low','student':'yes','credit_rating':'low'}
prediction = predict(data,tree)
print(prediction)

inst = df.iloc[7]
print(predict(inst, tree))
