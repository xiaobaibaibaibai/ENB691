import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        sd = 0.0001
        self.params['w1'] = np.random.randn(inputDim, hiddenDim) * sd
        self.params['b1'] = np.random.randn(hiddenDim) * sd
        self.params['w2'] = np.random.randn(hiddenDim, outputDim) * sd
        self.params['b2'] = np.random.randn(outputDim) * sd

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################
        
        # add bias to x and calculate Hin
        x_b = np.hstack([x, np.ones([x.shape[0], 1])])
        u_b = np.vstack((self.params['w1'], self.params['b1']))
        Hin = x_b.dot(u_b)
        # calculate Hout with bias by RELU
        Hout = np.maximum(Hin*0.01, Hin)
        Hout_b = np.append(Hout, np.ones([Hout.shape[0], 1]), axis=1)
        # calculate scores
        w_b = np.vstack((self.params['w2'], self.params['b2']))
        sc = Hout_b.dot(w_b)
        s = np.maximum(sc*0.01, sc)
        s = s - np.max(s, axis=1, keepdims=True)
        # caclulate correction probability
        exp_s = np.exp(s)
        sum_x = np.sum(exp_s, axis=1, keepdims=True)
        prob = exp_s / sum_x
        prob_correct = prob[np.arange(x.shape[0]), y]
        # cacluate loss iwht L2 regularization
        loss = - np.log(prob_correct)
        loss = np.sum(loss) / x.shape[0]
        loss += 0.5 * reg * (np.sum(self.params['w1']**2) + np.sum(self.params['w2']**2))

        # Do backward pass
        ind = np.zeros_like(prob)
        ind[np.arange(x.shape[0]), y] = 1
        ds = prob - ind
        dw = Hout.T.dot(ds) / x.shape[0]
        # calculate derivatives for w2 and b2 
        grads['w2'] = dw + 2 * reg * self.params['w2']
        grads['b2'] = np.sum(ds, axis=0) / x.shape[0]
        # calculate derivatives for Hout and Hin 
        dHout = ds.dot(self.params['w2'].T)
        dHin = dHout * (dHout >= 0)
        dHin = dHin + dHout * (dHout < 0) * 0.01
        # calculate derivatives for w2 and b2 
        grads['w1'] = x.T.dot(dHin) / x.shape[0] + reg * self.params['w1']
        grads['b1'] = np.sum(dHin, axis=0) / x.shape[0]
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            
            # creat xBatch and y Batch
            D = x.shape[0]
            index = np.random.choice(D, batchSize, replace=True)
            xBatch = x[index]
            yBatch = y[index]
            # calculate and save loss
            loss, grads = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)
            # update weights and bias
            self.params['w1'] = self.params['w1'] - lr * grads['w1']
            self.params['b1'] = self.params['b1'] - lr * grads['b1']
            self.params['w2'] = self.params['w2'] - lr * grads['w2']
            self.params['b2'] = self.params['b2'] - lr * grads['b2']
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        # calculate x to hidden layer
        x_b = np.hstack([x, np.ones([x.shape[0], 1])])
        u_b = np.vstack((self.params['w1'], self.params['b1']))
        Hin = x_b.dot(u_b)
        # calculate hidden layer to output
        Hout = np.maximum(0, Hin)
        Hout_b = np.append(Hout, np.ones([Hout.shape[0], 1]), axis=1)
        w_b = np.vstack((self.params['w2'], self.params['b2']))
        s = Hout_b.dot(w_b)
        yPred = np.argmax(s, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        yPred = self.predict(x)
        comp = np.sum(yPred == y) 
        acc = (comp / float(x.shape[0])) * 100.0

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



