import numpy as np 

class Optimizer_SGD :

    def __init__ ( self , learning_rate = 0.1):
        self.learning_rate = learning_rate
      
    
    def update_params ( self , layer ):
        """
            Update dei pesi e dei bias della rete rispetto al gradiente valutato 
            nella fase di backward.

            Keyword arguments:
            layer -- layer per la quale deve essere eseguito l'update
        """

        # Update dei parametri per una frazione (Learning Rate) del gradiente
        layer.weights += - self.learning_rate * layer.dweights
        layer.biases += - self.learning_rate * layer.dbiases
