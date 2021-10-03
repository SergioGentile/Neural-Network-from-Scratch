import numpy as np 

class Activation_Softmax_Loss_CategoricalCrossentropy():
    '''
        Classe che implementa la combinazione tra Softmax e Categorical Cross Entropy.
        Il vantaggio di tale classe è ridurre il costo computazionale 
        dell'operazione di backward.
    '''
    def __init__ ( self ):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    # Forward pass
    def forward ( self , inputs , y_true ):
        '''
            Implementazione dell'operazione di forward.

            Keyword arguments:
            inputs -- Features in input
            y_true -- Valore della ground truth
        '''

        # Forward pass per la softmax
        self.activation.forward(inputs)
        
        self.output = self.activation.output
        
        # Calcolo della loss e ritorno del valore
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward ( self , dvalues , y_true ):
        '''
            Implementazione dell'operazione di backward.

            Keyword arguments:
            dvalues -- Output della funzione Softmax
            y_true -- Valore della ground truth
        '''

        # Numero di esempi
        samples = len (dvalues)
        
        # Se si ha un encoding one hot, questo viene tradotto come etichetta di classe:
        # [0, 1, 0] -> 1
        if len (y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis = 1 )
        # Copia per evitare la copia by reference
        self.dinputs = dvalues.copy()
        
        # Calcolo del gradiente
        # Come avviente?
        # Viene sottratto 1 al valore della softmax relativo alla ground truth
        self.dinputs[ range (samples), y_true] -= 1
        
        # Normalizzazione del gradiente
        self.dinputs = self.dinputs / samples
        

class Loss: 
    '''
        Classe che implementa il metodo per il calcolo della Loss.
    '''
    def calculate(self, output, y):
        '''
            Calcola il valore della Loss come media di tutte le loss legate al batch in input. 

            Keyword arguments:
            output -- Output della funzione Softmax
            y -- Valore della ground truth
        '''
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) # Calcolo della media
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    '''
        Classe che implementa la funzione di Cateogrical Cross Entropy. 
    '''
    def forward(self, y_pred, y_true): 
        '''
            Implementazione dell'operazione di forward.

            Keyword arguments:
            y_pred -- Valori predetti dal modello
            y_true -- Valore della ground truth
        '''
        samples = len(y_pred)
        # per evitare il valore della loss pari a infinito
        # si ricorda che log(0) = -inf. 
        # Per evitare questo, si mettono eventuali valori a 0 della y_pred in un valore molto vicino a 0 ma non 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)   

        if len(y_true.shape) == 1: # calcolo se la y_true non è one-hot encoded
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # calcolo se la y_true è one-hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # Calcolo della negative loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward ( self, dvalues, y_true ):
        '''
            Implementazione dell'operazione di backward.

            Keyword arguments:
            dvalues -- Gradiente del layer successivo
            y_true -- Valore della ground truth
        '''

        # Non implementato. 
        # Verrà utilizzata la classe Activation_Softmax_Loss_CategoricalCrossentropy
        pass

class Activation_Softmax:
    def forward(self, inputs):
        '''
            Implementazione dell'operazione di forward.

            Keyword arguments:
            inputs -- Features in input alla funzione softmax
        '''

        # calcolo della Softmax
        # Si noti come viene sottratto ad ogni valore in input il massimo valore presente 
        # all'interno del vettore di input.
        # Perchè? 
        # Questo viene fatto per evitare l'overflow (e^n può causare overflow per n grande)
        # Questa operazione non incide minimamente sul calcolo della softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
    # Backward pass
    def backward ( self , dvalues ):
        '''
            Implementazione dell'operazione di backward.

            Keyword arguments:
            dvalues -- Gradiente del layer successivo
        '''

        # Non implementato. 
        # Verrà utilizzata la classe Activation_Softmax_Loss_CategoricalCrossentropy
        pass

class Activation_ReLU:
    '''
        Classe che implementa la funzione d'attivazione ReLU. 
    '''

    def forward(self, inputs):
        '''
            Implementazione dell'operazione di forward.

            Keyword arguments:
            inputs -- Features in input alla funzione ReLU
        '''

        self.output = np.maximum(0, inputs)
        # Vengono salvati gli input così da essere utilizzati nello step di backpropagation
        self.inputs = inputs
        
    # Backward pass
    def backward ( self , dvalues ):
        '''
            Implementazione dell'operazione di backward.

            Keyword arguments:
            dvalues -- Gradiente del layer successivo
        '''

        # dvalues è il gradiente del livello successivo
        # Viene effettuata una copia così da non modificare i valori 
        # presenti in dvalues (si ricorda che = fa una copia by reference)
        self.dinputs = dvalues.copy()
        # Calcolo del gradiente
        self.dinputs[self.inputs <= 0 ] = 0

