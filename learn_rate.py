import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weight_input_1 = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (784, 512))
        self.bias_1 = np.zeros((1,512))

        self.weight_1_2 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (512, 512))
        self.bias_2 = np.zeros((512,1)).T
        self.weight_2_3 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (512, 512))
        self.bias_3 = np.zeros((512,1)).T
        self.weight_3_4 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (512, 512))
        self.bias_4 = np.zeros((512,1)).T
        self.weight_4_5 = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (512, 10))
        self.bias_5 = np.zeros((10,1)).T
        self.lr = learning_rate
        
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x :(np.divide(1,1+np.exp(-x))) 
        self.softmax_function = lambda x : np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
        # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        delta = []
        n_records = features.shape[0]
        delta_weight_2 = np.zeros(self.weight_1_2.shape)
        delta_weight_3 = np.zeros(self.weight_2_3.shape)
        delta_weight_4 = np.zeros(self.weight_3_4.shape)
        delta_weight_5 = np.zeros(self.weight_4_5.shape)
        delta_weight_f1 = np.zeros(self.weight_input_1.shape)
        X=features
        y=targets    
        cache = self.forward_pass_train(X)  # Implement the forward pass function below
            
            # Implement the backproagation function below
        delta_bias,delta_weights = self.backpropagation(cache, X, y,delta_weight_f1,delta_weight_2,delta_weight_3,delta_weight_4,delta_weight_5)
        #delta1=sum(sum(delta_weights_i_h**2))
        #delta2 = sum(sum(delta_weights_h_o**2))
        #factor2 = delta1/delta2
        #factor = delta2/delta1
        
        self.update_weights(delta_bias,delta_weights, n_records)
        
        


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        z1 = np.dot(X,self.weight_input_1)+self.bias_1 # signals into hidden layer
        a1 = self.activation_function(z1) # signals from hidden layer
        z2 = np.dot(a1,self.weight_1_2)+self.bias_2
        a2 = self.activation_function(z2)
        z3 = np.dot(a2,self.weight_2_3)+self.bias_3
        a3 = self.activation_function(z3)
        z4 = np.dot(a3,self.weight_3_4)+self.bias_4
        a4 = self.activation_function(z4)
        z5 = np.dot(a3,self.weight_4_5)+self.bias_5
        output = self.softmax_function(z5)

       
        
        return (output,a4,a3,a2,a1)

    def backpropagation(self, cache, X, y, delta_weight_f1,delta_weight_2,delta_weight_3,delta_weight_4,delta_weight_5):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        output,a4,a3,a2,a1= cache

        # TODO: Output error - Replace this value with your calculations.
        #error = (-y/output) + (1-y)/(1-output) # Output layer error is the difference between desired target and actual output.
        delta_weight_5 = np.dot(a4.T,(output-y))
        delta_bias_5 = np.sum((output-y).T,axis=1,keepdims=True).T
        ###################################################
        delta_weight_4 = np.dot(np.dot(self.weight_4_5,(output-y).T),(a4*(1-a4)*a3))
        error4 = np.dot(self.weight_4_5,(output-y).T)  #shape (512,m)
        delta_bias_4= np.sum(np.dot(error4,(a4*(1-a4))),axis=1,keepdims=True).T
        ############################################################################
        error3 = np.dot(self.weight_3_4,error4)
        delta_weight_3 = np.dot(error3,(a4*(1-a4)*a3*(1-a3)*a2))
        delta_bias_3= np.sum(np.dot(error3,(a4*(1-a4)*a3*(1-a3))),axis=1,keepdims=True).T                        
        ############################################################################
        error2 = np.dot(self.weight_2_3,error3)
        delta_weight_2 = np.dot(error2,(a4*(1-a4)*a3*(1-a3)*a2*(1-a2)*a1))
        delta_bias_2 = np.sum(np.dot(error2,(a4*(1-a4)*a3*(1-a3)*a2*(1-a2))),axis=1,keepdims=True).T
        ##############################################################################
        error1 = np.dot(self.weight_1_2,error2)
        delta_weight_1 = error1.T*(a4*(1-a4)*a3*(1-a3)*a2*(1-a2)*a1*(1-a1))
        delta_weight_f1 = np.dot(delta_weight_1.T,X).T
        delta_bias_1 = np.sum(delta_weight_1.T , axis=1,keepdims=True).T
        
        ###################################################################################################################
        delta_bias = (delta_bias_5,delta_bias_4,delta_bias_3,delta_bias_2,delta_bias_1)
        delta_weights = (delta_weight_5,delta_weight_4,delta_weight_3,delta_weight_2,delta_weight_f1)
        
        return delta_bias,delta_weights

    def update_weights(self,delta_bias,delta_weights , n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        delta_weight_5,delta_weight_4,delta_weight_3,delta_weight_2,delta_weight_f1 = delta_weights
        delta_bias_5,delta_bias_4,delta_bias_3,delta_bias_2,delta_bias_1 = delta_bias
        
        delta_bias_distance_5 = np.linalg.norm(delta_bias_5)
        delta_bias_distance_4 = np.linalg.norm(delta_bias_4)
        delta_bias_distance_3 = np.linalg.norm(delta_bias_3)
        delta_bias_distance_2 = np.linalg.norm(delta_bias_2)
        delta_bias_distance_1 = np.linalg.norm(delta_bias_1)
        bias_factor_4 = delta_bias_distance_5/delta_bias_distance_4
        bias_factor_3 = delta_bias_distance_5/delta_bias_distance_3
        bias_factor_2 = delta_bias_distance_5/delta_bias_distance_2
        bias_factor_1 = delta_bias_distance_5/delta_bias_distance_1
        delta_weight_distance_5 = np.linalg.norm(delta_weight_5)
        delta_weight_distance_4 = np.linalg.norm(delta_weight_4)
        delta_weight_distance_3 = np.linalg.norm(delta_weight_3)
        delta_weight_distance_2 = np.linalg.norm(delta_weight_2)
        delta_weight_distance_1 = np.linalg.norm(delta_weight_f1)
        weight_factor_4 = delta_weight_distance_5/delta_weight_distance_4
        weight_factor_3 = delta_weight_distance_5/delta_weight_distance_3
        weight_factor_2 = delta_weight_distance_5/delta_weight_distance_2
        weight_factor_1 = delta_weight_distance_5/delta_weight_distance_1
        
        
        self.weight_input_1 += np.divide(np.multiply((self.lr*weight_factor_1),delta_weight_f1),n_records)
        self.bias_1 += np.divide(np.multiply((self.lr*bias_factor_1),delta_bias_1),n_records)
        
        self.weight_1_2 += np.divide(np.multiply((self.lr*weight_factor_2),delta_weight_2),n_records)
        self.bias_2 += np.divide(np.multiply((self.lr*bias_factor_2),delta_bias_2),n_records)                              
        self.weight_2_3 += np.divide(np.multiply((self.lr*weight_factor_3),delta_weight_3),n_records)
        self.bias_3 += np.divide(np.multiply((self.lr*bias_factor_3),delta_bias_3),n_records)                              
        self.weight_3_4 += np.divide(np.multiply((self.lr*weight_factor_4),delta_weight_4),n_records)
        self.bias_4 += np.divide(np.multiply((self.lr*bias_factor_4),delta_bias_4),n_records)                              
        self.weight_4_5 += np.divide(np.multiply(self.lr,delta_weight_5),n_records)
        self.bias_5 += np.divide(np.multiply(self.lr,delta_bias_5),n_records)
                                      
        
        #self.weights_input_to_hidden -= np.divide(np.multiply((self.lr*factor),delta_weights_i_h),n_records)# update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        X=features
        z1 = np.dot(X,self.weight_input_1)+self.bias_1 # signals into hidden layer
        a1 = self.activation_function(z1) # signals from hidden layer
        z2 = np.dot(a1,self.weight_1_2)+self.bias_2
        a2 = self.activation_function(z2)
        z3 = np.dot(a2,self.weight_2_3)+self.bias_3
        a3 = self.activation_function(z3)
        z4 = np.dot(a3,self.weight_3_4)+self.bias_4
        a4 = self.activation_function(z4)
        z5 = np.dot(a3,self.weight_4_5)+self.bias_5
        output = self.softmax_function(z5)

        
        return output


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 6
learning_rate = 0.01
hidden_nodes = 11
output_nodes = 1