import numpy as np

class NeuralNetwork:

    def __init__(self,inputLayer,hiddenLayer,eta,epochs):
        self.eta=eta
        self.epochs=epochs
        self.inputLayer=inputLayer
        self.hiddenLayer=hiddenLayer
        self.outputLayer=1
        #weights for Layer 1 that is hidden layer
        self.W1 = np.random.uniform(low=-0.001, high=0.001, size=(self.inputLayer,self.hiddenLayer))
        #weights for Layer 2 that is output layer
        self.W2 = np.random.uniform(low=-0.001, high=0.001, size=(self.hiddenLayer,self.outputLayer))

    def sigmoid(self,s):
        return 1.0/(1.0+np.exp(-s))

    def calcWeights(self,train_instance):
        self.X1=self.sigmoid(np.dot(train_instance,self.W1))
        self.X2=self.sigmoid(np.dot(self.X1,self.W2))
        return self.X2

    def predictDerivative(self,x):
        #return self.sigmoid(x)*(1-self.sigmoid(x))
        return x*(1-x)

    def calculateError(self,train_instance,train_label,predicted):
        errorDerivative=2*(predicted-train_label)
        self.delta2=np.multiply(errorDerivative,self.predictDerivative(predicted))
        self.delta1=np.multiply(np.dot(self.W2,self.delta2),self.predictDerivative(self.X1))
        self.W2-=self.eta*np.outer(self.X1,self.delta2)
        self.W1-=self.eta*np.outer(train_instance,self.delta1)

    def backPropagate(self,train_dat,train_labels):
        for i in range(self.epochs):
            for j in range(len(train_dat)):
                #print(j)
                predicted=self.calcWeights(train_dat[j])
                self.calculateError(train_dat[j],train_labels[j],predicted)

    def predict(self, x):

        # Allocate memory for the outputs
        y = np.zeros([x.shape[0],self.W2.shape[1]])

        # Loop the inputs
        for i in range(0,x.shape[0]):

            # Outputs
            y[i] = self.calcWeights(x[i])

        # Return the results
        return y

def read_img(img_name):
    f=open(img_name,'rb')
    f.readline()   # skip P5
    f.readline()   # skip the comment line
    width, height = [int(i) for i in f.readline().split()]  # size of the image
    bits = int(f.readline().strip())
    image = []
    for _ in range(width * height):
        image.append(f.read(1)[0] / bits)

    return image

def main():
    train_dat=[]
    train_labels=[]
    f=open("downgesture_train.list","r")
    for img_name in f.readlines():
        img_name=img_name.strip()
        train_dat.append(read_img(img_name))
        if 'down' in img_name:
            train_labels.append(1)  
        else:
            train_labels.append(0)
    train_dat=np.array(train_dat,dtype='float')
    train_labels=np.array(train_labels,dtype='float')
    #size of input layer=960 as the size of images is 30x32
    nn=NeuralNetwork(960,100,0.1,1000)
    nn.backPropagate(train_dat,train_labels)

    total = 0
    correct = 0
    f=open('downgesture_test.list','r')
    for img_name in f.readlines():
        total+=1
        img_name = img_name.strip()
        predictedLabel = nn.predict(np.array([read_img(img_name),]))
        predictedLabel=0 if predictedLabel <0.5 else 1
        print(img_name,":",predictedLabel)
        if (predictedLabel!=0)==('down' in img_name):
            correct += 1
    print("Accuracy:",(correct/total)*100)


if __name__ == "__main__":
    main()
