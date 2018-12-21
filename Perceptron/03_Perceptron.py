import pandas as pd
import argparse
import os

def get_data():
    df = pd.read_csv(input_file,header=None,delimiter='\t')
    train_x=[]
    train_y=[]
    if(input_file=="Example.tsv"):
        train_x=df.iloc[:,1:-1]
    else:
        train_x=df.iloc[:,1:]
    header_list=[] 
    for i in list(train_x):
        header_list.append("att"+str(i))
    train_x.columns=header_list
    train_y=df.iloc[:,0]
    #print("Train_x is :"+str(train_x))
    return train_x,train_y

def encode_train_y(train_y):
    for i in range(len(list(train_y))):
        if(train_y[i]=="A"):
            train_y[i]=1
        else:
            train_y[i]=0
    return train_y

def get_error(train_x,train_y,weight_vector):
    error = 0
    gradient=[0.0 for i in range(len(weight_vector))]
    for i in range(len(list(train_y))):
        output = weight_vector[-1]*1
        for j in range(len(weight_vector)-1):
            output+=weight_vector[j]*train_x.iloc[i,j]
        #print("output for datapoint "+str(i)+" is : "+str(output))
        if output > 0:
            output = 1
        else:
            output = 0
        #print("output after activation is "+str(output))
        if output!=train_y[i]:
            error+=1
            gradient[-1]+=(train_y[i]-output)*1
            for j in range(len(list(weight_vector))-1):
                gradient[j]+=(train_y[i]-output)*train_x.iloc[i,j]
                #print("Error_x["+str(i)+"]["+str(j)+"] = "+str((output-train_y[i])*train_x.iloc[i,j]))
    return error,gradient

def update_weight_vector(weight_vector,learning_rate,gradient):
    for i in range(len(list(weight_vector))):
        weight_vector[i]+=learning_rate*gradient[i]
    return weight_vector

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
file_instance=open(output_file,"w+")

learning_rate=1.0
train_x,train_y = get_data()
train_y=encode_train_y(train_y)
weight_vector = [0.0 for i in range(len(train_x.columns)+1)]
#print("Constant Learning : ")
for i in range(1,102):
    #print("Weight Vector : "+str(weight_vector))
    error,gradient = get_error(train_x,train_y,weight_vector)
    file_instance.write(str(error)+"\t")
    weight_vector = update_weight_vector(weight_vector,learning_rate,gradient)
#print("Annealing Learning : ")
file_instance.write("\n")
weight_vector = [0.0 for i in range(len(train_x.columns)+1)]
for i in range(1,102):
    error,gradient = get_error(train_x,train_y,weight_vector)
    file_instance.write(str(error)+"\t")
    weight_vector = update_weight_vector(weight_vector,learning_rate/i,gradient)
file_instance.close()