import numpy as np 

class Layer_Dense:
    '''
        Implementazione del Layer Dense di una Rete Neurale. 
        Tale Layer potrà essere combinato con altri Layer Dense per creare una rete 
        a più strati. 
    '''
    def __init__(self, n_inputs, n_neurons):
        '''
            Inizializzazione del Layer Dense.

            Keyword arguments:
            n_inputs -- Numero di features in input
            n_neurons -- Numero di Features in output

        '''
        # Nel costruttore vengono inizializzati i pesi e i bias
        # - Nel nostro caso i pesi vengono inizializzati random.
        # - La backpropagation correggerà i pesi
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) 
        
    def forward(self, inputs):
        '''
            Implementazione dell'operazione di forward.

            Keyword arguments:
            inputs -- Features in input 
        '''

        self.output = np.dot(inputs, self.weights) + self.biases
        # Vengono salvati gli input così da essere utilizzati nello step di backpropagation
        self.inputs = inputs
        
    def backward ( self , dvalues ):
        '''
            Implementazione dell'operazione di backward.

            Keyword arguments:
            dvalues -- Gradiente del layer successivo 
        '''

        # Calcolo del gradiente.
        # dvalues è il gradiente del livello successivo
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0 , keepdims = True ) 
        self.dinputs = np.dot(dvalues, self.weights.T)