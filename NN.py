from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time

class ClassificationNeuralNet:
    
    def __init__(self,input_size,layer_sizes,num_classes,act_fun,LR,momentum):
        self.weights=[]
        self.delta_weights=[]
        self.delta_biases=[]
        self.biases=[]
        self.LR=LR
        self.momentum=momentum
        self.act_fun=act_fun
        self.weights.append( np.random.rand(input_size,layer_sizes[0]) )
        self.biases.append( np.random.rand(layer_sizes[0]) )
        self.prev_delta_wt=[]
        self.prev_delta_bias=[]
        

        for i in range(len(layer_sizes)-1):
            self.weights.append( np.random.rand(layer_sizes[i],layer_sizes[i+1]) )
            self.biases.append( np.random.rand(layer_sizes[i+1]) )
        
        self.weights.append( np.random.rand(layer_sizes[-1],num_classes) )
        self.biases.append( np.random.rand(num_classes) )

        '''for i in range(len(layer_sizes)-1):
            self.weights.append( np.zeros((layer_sizes[i],layer_sizes[i+1])) )
            self.biases.append( np.zeros((layer_sizes[i+1])) )
        self.weights.append( np.zeros((layer_sizes[-1],num_classes)) )
        self.biases.append( np.zeros((num_classes)) )'''



        self.delta_weights=[0]*(len(self.weights))
        self.delta_biases=[0]*(len(self.biases))

    def sigmoid(self,A):
        return 1.0/(1+np.exp(-A))
    def tanh(self,A):
        return np.tanh(A)
    
    def softmax(self,A):
        sub = np.exp(A)
        return sub/(np.sum(sub,axis=1).reshape(-1,1))

    def cross_entropy_loss(self,preds,labels):
        return -np.sum(np.log(preds)*labels,axis=1)

    def evaluate(self,inputV):
        X = inputV
        mid_ops = [X]
        if self.act_fun=="sigmoid":
            activation_fn = self.sigmoid
        if self.act_fun=="tanh":
            activation_fn=self.tanh

        sub = activation_fn( np.dot(X,self.weights[0]) + self.biases[0] )  #op of layer0
        mid_ops.append(  sub  )
        for i in range(len(self.weights)-2):  #-2 cuz last weight is to make output, and we  need to softmax that
            sub = activation_fn( np.dot(mid_ops[i+1],self.weights[i+1]) + self.biases[i+1] )  #op of layer i+1
            mid_ops.append( sub )
            

        OUTPUT = self.softmax( np.dot(mid_ops[-1],self.weights[-1]) + self.biases[-1] )#finalOP
        mid_ops.append(OUTPUT)
        return mid_ops

    def train_delta(self,inputV,labels):#just delta rule
        mid_vals = self.evaluate(inputV)
        preds = mid_vals[-1]
        loss = self.cross_entropy_loss(preds,labels)

        delta_op = labels - preds
        delta_list = [0]*len(mid_vals)
        delta_list[-1] = delta_op

        if self.act_fun=="tanh":
            for i in range(len(mid_vals)-2,0,-1):
                delta_list[i] = ( np.dot(delta_list[i+1],self.weights[i].T)*(1-mid_vals[i]**2) )

        if self.act_fun=="sigmoid":
            for i in range(len(mid_vals)-2,0,-1):
                delta_list[i] = ( np.dot(delta_list[i+1],self.weights[i].T)*( (1-mid_vals[i])*mid_vals[i] ) ) 

        self.delta_weights=[0]*(len(self.weights))
        self.delta_biases=[0]*(len(self.biases))
        for i in range(len(self.delta_weights)):
            self.delta_weights[i] = -self.LR * np.dot(mid_vals[i].T,delta_list[i+1])
            self.delta_biases[i] = -self.LR * delta_list[i+1].reshape(-1)

        return loss
    
    def train_generalized_delta(self,inputV,labels):#generalized delta rule (with momentum scaling)
        self.prev_delta_wt=self.delta_weights
        self.prev_delta_bias=self.delta_biases

        mid_vals = self.evaluate(inputV)
        preds = mid_vals[-1]
        loss = self.cross_entropy_loss(preds,labels)

        delta_op = labels - preds
        delta_list = [0]*len(mid_vals)
        delta_list[-1] = delta_op

        if self.act_fun=="tanh":
            for i in range(len(mid_vals)-2,0,-1):
                delta_list[i] = ( np.dot(delta_list[i+1],self.weights[i].T)*(1-mid_vals[i]**2) )

        if self.act_fun=="sigmoid":
            for i in range(len(mid_vals)-2,0,-1):
                delta_list[i] = ( np.dot(delta_list[i+1],self.weights[i].T)*( (1-mid_vals[i])*mid_vals[i] ) ) 

        self.delta_weights=[0]*(len(self.weights))
        self.delta_biases=[0]*(len(self.biases))
        for i in range(len(self.delta_weights)):
            self.delta_weights[i] = -self.LR * np.dot(mid_vals[i].T,delta_list[i+1]) - self.momentum*self.prev_delta_wt[i]
            self.delta_biases[i] = -self.LR * delta_list[i+1].reshape(-1) - self.momentum*self.prev_delta_bias[i]
            #self.delta_weights[i] = -self.LR * np.dot(mid_vals[i].T,delta_list[i+1])
            #self.delta_biases[i] = -self.LR * delta_list[i+1].reshape(-1)
        #print(self.prev_delta_wt)

        return loss


    def update(self):
        for i in range(len(self.delta_weights)):
            self.weights[i]-=self.delta_weights[i]
            self.biases[i]-=self.delta_biases[i]
    
    

#reading data from file
f = open("traingroup1.csv",'r')
l = f.readlines()[1:]
f.close()
inputdata = []
inputlabels = []
yeet=[]
for i in l:
    a,b,c = i.split(',')
    inputdata.append([a,b])
    subl = [0,0,0]
    subl[int(float(c[:-1]))] = 1
    yeet.append(int(float(c[:-1])))
    inputlabels.append(subl)

inputdata = np.array(inputdata).astype('float64')
inputlabels = np.array(inputlabels).astype('float64')
yeet = np.array(yeet).astype('uint8')

NN = ClassificationNeuralNet(2,[6,6],3,"tanh",0.001,0.0001)



#Actual training
epoch=0
error=0
flag=1
while epoch < 1000:
    prev_error=error
    error=0
    for i in range(int(len(inputdata))):
        # if i == 10:
            # tic = time.time()
        loss = NN.train_delta(inputdata[i:i+1,:],inputlabels[i])
        # if i == 10:
            # toc = time.time()
            # print(toc - tic)
        NN.update()
        #print(loss)
    error+=loss**2
    if np.abs(error-prev_error)<.00001 and error<.01:
        flag=0
    if epoch%10 == 1:
        print("epoch",epoch,"done, error - ",error)
    epoch+=1


#testing
confusion = np.zeros((3,3))
predyeet=[]
for i in range(len(inputdata)):
    op = NN.evaluate(np.array([inputdata[i]]))
    predyeet.append(np.argmax(op[-1][0]))
    confusion[np.argmax(inputlabels[i])][np.argmax(op[-1][0])]+=1

print(confusion)
print(np.diag(confusion).sum()/confusion.sum())


cc=['r','g','b']
#for i in range(3):
#    plt.scatter(inputdata.T[0][yeet==i],inputdata.T[1][yeet==i],c=cc[i])
#plt.show()

for i in range(len(inputdata)):
    plt.scatter([inputdata[i][0]],[inputdata[i][1]],c = cc[predyeet[i]])

plt.show()











