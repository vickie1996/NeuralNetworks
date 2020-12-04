import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('E:\\Projects\\neural network\\Neural Networks\\student_scores2.csv')
#df.plot.scatter(x= 'Hours', y='IQ', c= 'Pass', colormap = 'bwr')
df.describe()

x = df.drop(['Scores', 'Pass'],axis=1).values
y = df['Pass'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 40)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#function to give the Sigmoid of any thing
def sigmoid(s):
    
    return 1 / (1 + np.exp(-s))

def sigmoid_prime(s):
    return s * (1 - s)


#Neural network class
class NeuralNetwork(object):
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_units = 3
        self.w1= np.zeros((self.input_size, self.hidden_units))
        self.w2= np.zeros((self.hidden_units, self.output_size))
        
        
    def forward(x_train):
        
        self.output_1 = sigmoid(np.dot(x_train, self.w1))
        self.output_2 = sigmoid(np.dot(self.output_1, self.w2))
        return self.output_2
        
        
    def backward(self, x_train, y_train):
        
        delta_w2 = (self.output_2 - y_train) * sigmoid_prime(self.ouput_2)
        delta_w1 = delta_w2.dot(self.w2.T) * sigmoid_prime(self.output_1)
        
        self.w2 = self.w2 + self.output_1.T(delta_w2) * -1
        self.w1 = self.w1 + x_train.T.dot(delta_w1) * -1
        
        
        
    def fit(self, x_train, y_train):
        
        self.forward(x_train)
        self.backward(x_train, y_train)
        
    def predict(self, x_test):
        
        return y_prediction
    
    def loss(self, x, y_actual):
        
        prediction = self.forward(x)        
        return np.mean(np.square(y_actual - prediction))

    
    
    
    
    
nn = NeuralNetwork(input_size = 2, output_size =1)

train_loss = []
test_loss = []
for i in range(1000):
    nn.fit(x_train, y_train)
    train_loss.append(nn.loss(x_train, y_train))
    test_loss.append(nn.loss(x_test, y_test))
    
plt.plot(train_loss, 'r--')
plt.show()
