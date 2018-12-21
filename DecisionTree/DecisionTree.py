import math
import pandas as pd
import argparse
import os

def calculate_entropy_list(values):
    entropy = 0 
    if len(values)==1:
        return entropy
    for i in range(len(values)):
        p=float(values[i]/sum(values))
        if p==0:
            continue
        entropy -= p*math.log(p,len(values))
    return entropy

def calculate_entropy_wrt_target(train_x,train_y):
    entropy_list=[]
    train_y_set = list(set(train_y))
    for k in range(len(list(train_x))):
        train_x_set = []
        train_x_set = list(set(train_x.iloc[:,k]))
        entropy_table_wrt_attribute= [[0 for x in range(len(train_y_set))] for y in range(len(train_x_set))]
        for i in range(len(train_y)):
            entropy_table_wrt_attribute[train_x_set.index(train_x.iloc[i,k])][train_y_set.index(train_y.iloc[i])]+=1
        entropy = 0
        for i in range(len(train_x_set)):
            entropy += sum(entropy_table_wrt_attribute[i])/len(train_y)*calculate_entropy_list(entropy_table_wrt_attribute[i])
        entropy_list.append(entropy)
    return entropy_list

def calculate_entropy_target(train_y1):
    train_y = list(train_y1)
    train_y_values_count=[]
    for item in train_y_set:
        train_y_values_count.append(train_y.count(item)) 
    return calculate_entropy_list(train_y_values_count)

def get_data():
    df = pd.read_csv(input_file,header=None)
    train_x=[]
    train_y=[]
    train_x=df.iloc[:,0:-1]
    header_list=[]
    for i in list(train_x):
        header_list.append("att"+str(i))
    train_x.columns=header_list
    train_y=df.iloc[:,-1]
    return train_x,train_y

def find_best_node_and_split(train_x,train_y):
    entropy_list = calculate_entropy_wrt_target(train_x,train_y)
    return entropy_list

def split_data_based_on_attribute_value(train_x,train_y,index_of_attribute_to_split,value):
    train_x1=train_x[train_x[str(list(train_x)[index_of_attribute_to_split])]==value]
    train_x1=train_x1.drop(train_x1.columns[[index_of_attribute_to_split]], axis=1)
    train_y1=train_y[train_x[str(list(train_x)[index_of_attribute_to_split])]==value]
    return train_x1,train_y1

def run_id3(train_x,train_y,level):
    entropy_list = calculate_entropy_wrt_target(train_x,train_y)
    index_of_attribute_to_split = entropy_list.index(min(entropy_list))
    for value in list(set(train_x.iloc[:,index_of_attribute_to_split])):
        train_x1,train_y1 = split_data_based_on_attribute_value(train_x,train_y,index_of_attribute_to_split,value)
        file_instance.write("<node "+" entropy= \""+str(calculate_entropy_target(train_y1))+"\" feature=\""+str(list(train_x)[index_of_attribute_to_split])+"\" value=\""+value+"\">")
        if(calculate_entropy_target(train_y1)!=0):
            run_id3(train_x1,train_y1,level+1)
        else:
            file_instance.write(train_y1.iloc[0])
        file_instance.write("</node>")


def check_location(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Data filename", type=check_location)
parser.add_argument("--output", help="Output filename")

args = parser.parse_args()
input_file = args.data
output_file = args.output

train_x,train_y = get_data()
train_y_set = list(set(train_y))
level=0
file_instance=open(output_file,"w+")
file_instance.write("<tree entropy = \""+str(calculate_entropy_target(train_y))+"\">")
run_id3(train_x,train_y,level)
file_instance.write("</tree>")
file_instance.close()
