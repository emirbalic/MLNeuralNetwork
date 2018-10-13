'''
Created on 18. sij 2016.

@author: User
'''

import numpy as np
from sympy.physics.unitsystems.systems.mks import momentum

''' Out of the class (global) functions - Transfer '''


def Sigmoid (x, Derivative=False):
    if not Derivative:
         return 1 / (1 + np.exp(-x))
    else:
        # Could it be one line of code?
         out = Sigmoid(x)
         return out*(1.0 - out)  
         
def Linear (x, Derivative=False):
    if not Derivative:  
        return x
    else:
        return 1.0
    
def Gaussian (x, Derivative=False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2*x*np.exp(-x**2)
    
def Tangent(x, Derivative= False):
    if not Derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2
    
        

''' Class '''


class Network_BackPropagate:
    ''' Class containing the back-propagation network '''
    
    
    'Variables - members of the class'
    
    count_layers = 0
    net_shape = None
    list_weights = []
    transfer_functions = []
    
    
    'Methods - functions native to the class'
    
    def __init__(self, size_layer, layer_funtions = None):
        
        'Initializing the network'
        
        ' Setting up layers: info and initiation'
        self.count_layers = len(size_layer) - 1
        self.net_shape = size_layer
        
        if layer_funtions is None:
            lay_funcs = []
            for i in range (self.count_layers):
                if i == self.count_layers - 1:
                    lay_funcs.append(Linear)
                else:
                    lay_funcs.append(Sigmoid)
        else:
            if len(size_layer) != len(layer_funtions):
                raise ValueError("List of transfer functions is not compatible!")
            elif layer_funtions[0] is not None:
                raise ValueError("No transfer function can be assigned to input layer!")
            else:
                lay_funcs = layer_funtions[1:]
                        
        self.transfer_functions = lay_funcs
                          
        ' Data structure for data propagating the network'
        self.input_layer = []
        self.output_layer = []
        self.past_weight_delta = []
        
        ' Weight arrays initialize '
        for (l1, l2) in zip(size_layer[:-1], size_layer[1:]):
            self.list_weights.append(np.random.normal(scale = 0.1, size = (l2, l1+1)))
            self.past_weight_delta.append(np.zeros((l2, l1+1)))
          
    
    """Run method"""
    
    def Network_Run(self, argument):
        'Method: Based on the argument data - running the network'
        
        cases_layer = argument.shape[0]
        
        ' Previous intermediate value lists - clear' 
        self.input_layer = []
        self.output_layer = []
        
        ' Running the network ' 
        for index in range(self.count_layers):
            # Determine layer argument
            if index == 0:
                layerInput = self.list_weights[0].dot(np.vstack([argument.T, np.ones([1, cases_layer])]))
            else:
                layerInput = self.list_weights[index].dot(np.vstack([self.output_layer[-1], np.ones([1, cases_layer])]))
            self.input_layer.append(layerInput)
            self.output_layer.append(self.transfer_functions[index](layerInput))
             
        return self.output_layer[-1].T


    
    
    " Training method " 
    
    def Epoch_Training (self, argument, goal_value, rate_training = 0.2, momentum = 0.5):
        'Method: One epoch'
        
        list_delta = []
        cases_layer = argument.shape[0]
        
        ' Argument to run the network:' 
        self.Network_Run(argument)
        
        ' For loop used to calculate deltas:'
        for index in reversed(range(self.count_layers)):
            if index == self.count_layers - 1:
                'Comparing to the target values to get the deltas:'
                output_delta = self.output_layer[index] - goal_value.T
                error = np.sum(output_delta**2)
                list_delta.append(output_delta * self.transfer_functions[index](self.input_layer[index], True))
            else:
                'Comparing to the following layers delta'
                pullback_delta = self.list_weights[index + 1].T.dot(list_delta[-1])
                list_delta.append(pullback_delta[:-1, :] * self.transfer_functions[index](self.input_layer[index], True))
        
        ' For loop used to compute deltas:'
        for index in range (self.count_layers):
            index_delta = self.count_layers - 1 - index
            
            if index == 0:
                layerOutput = np.vstack([argument.T, np.ones([1, cases_layer])])
            else:
                layerOutput = np.vstack([self.output_layer[index - 1], np.ones([1, self.output_layer[index - 1].shape[1]])])
            
            current_weight_delta = np.sum(\
                                 layerOutput[None, :, :].transpose(2, 0, 1) * list_delta[index_delta][None, :, :].transpose(2, 1, 0)\
                                 , axis = 0)
            weight_delta = rate_training * current_weight_delta + momentum * self.past_weight_delta[index]
            
            self.list_weights[index]-= weight_delta
            
            self.past_weight_delta[index] = weight_delta
        return error   

         

"create a test object:"
                 
if __name__ == "__main__":
    
    "For the purpose of testing I left all varieties of the code commented, so it can be easily uncommented and tested on various parameters"
    
    'Three layers'
    
    ' Functions'
    
    # AND function
    input_args = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    target_network = np.array([[0.05], [0.95], [0.05], [0.05]])
    
    # XOR function
    #input_args = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    #target_network = np.array([[0.05], [0.05], [0.95], [0.95]])
   
    # OR function
    #input_args = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    #target_network = np.array([[0.00], [1.00], [1.00], [1.00]])

    # NOR function
    #input_args = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    #target_network = np.array([[0.90], [0.05], [0.05], [0.05]])
    
    'Three layers - Passing in parameters'
    # Passing in parameters = Sigmoid three layers
    #lay_funcs = [None, Sigmoid, Linear]

    # Passing in parameters = Gausian three layers
    lay_funcs = [None, Gaussian, Linear]

    # Passing in parameters = Tangent three layers
    #lay_funcs = [None, Tangent, Linear]
    
    'Four layers - Passing in parameters'
    # Passing in parameters = Sigmoid four layers
    #lay_funcs = [None, Sigmoid, Sigmoid, Linear]
    
    # Passing in parameters = Gausian four layers
    # lay_funcs = [None, Gaussian, Gaussian, Linear]

    # Passing in parameters = Tangent four layers
    #lay_funcs = [None, Tangent, Tangent,  Linear]
    
    'Three layer function call'
    bp_network = Network_BackPropagate((2,2,1), lay_funcs)

    'Four layer function call'
    #bp_network = Network_BackPropagate((2,2,2,1), lay_funcs)
    
    print("The shape of the network is: {0}\n".format(bp_network.net_shape))
    print("\nRandom list of weights: {0}\n".format(bp_network.list_weights))

    max_length = 100000
    error_length = 1e-5
    for i in range(max_length+1):
        # I can optionally add momentum_coeficient in this call of the function for testing
        # it is now really high for development purposes
        error = bp_network.Epoch_Training(input_args, target_network, momentum = 0.05)
        if i % 25000 == 0:
            print("Iteration nr. {0}\tError: {1:0.6f}".format(i, error))
        if error <= error_length:
            print("Minimum error is reached at the iteration nr. {0}\n".format(i))
            break

    
    network_output = bp_network.Network_Run(input_args)
    for i in range (input_args.shape[0]):
        print("Input: {0}\tOutput: {1}".format(input_args[i], network_output[i]))
    
    
    
           
