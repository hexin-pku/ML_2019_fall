import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

"""@package docstring
Support Vector Machine: ML teamwork
@ author:
    1) Xin He
"""

class svm:
    """Support Vector Machine:
        This is a basic SVM class, with some enhenced algorithm to reduce the number of support vectors
        some references:
        1) [Ho Gi Jung, and Gahyun Kim, Support Vector Number Reduction: Survey and Experimental
        Evaluations, IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL.15, NO.2, APRIL
        2014](https://ieeexplore.ieee.org/document/6623200)
            : On summary of different reduced methods
        2) [P. M. L. Drezet and R. F. Harrison, “A new method for sparsity control in support vector
        classification and regression,” Pattern Recognit., vol.34, no.1, pp. 111–125, Jan. 2001](
        https://www.sciencedirect.com/science/article/pii/S0031320399002034)
            :
        3) [O. Dekel and Y. Singer, “Support vector machines on a budget,” in Proc. NIPS Found., 2006,
        pp.1–8](https://ieeexplore.ieee.org/book/6267330)
        4) [S.-Y. Chiu, L.-S. Lan, and Y.-C. Hwang, “Two sparsity-controlled schemes for 1-norm support
        vector classification,” in Proc. IEEE Northeast Workshop Circuits Syst., Montreal, QC, Canada,
        Aug. 5–8, 2007,pp. 337–340](https://ieeexplore.ieee.org/document/4487961?arnumber=4487961)
        5) []()
    """

    def __init__(self,**kwargs):
        """
        kwargs:
            1) select: feature selection before SVM-fit, in ['origin','random','cluster']
            2) selectargs: dictionary arguments for select feature

            3) kernel: in ['linear', 'poly', 'gauss'], (last two should with kernalargs)
            4) kernelargs: dictionary arguments for kernel

            5) C: penalty of unseperable data, with constraint 0 < alpha < C
            6) gamma: penalty for support vector number

            7) postselect:
            8) postconstruct:

            9) eps: tolerance of numerical error, default 1e-5
            10) maxiter: max iteration times, default 2|D|
        """

        self.select = kwargs.get('select', 'origin')
        self.selectargs = kwargs.get('selectargs', None)

        self.kernel = kwargs.get('kernel', 'linear')
        self.kernelargs = kwargs.get('kernelargs', None)

        self.C = kwargs.get('C', 1)
        self.gamma = kwargs.get('gamma', 1)

        # postselect # TODO
        # postconstruct # TODO

        self.eps = kwargs.get('eps', 1e-10)
        self.maxiter = kwargs.get('maxiter', 0) # laterly will auto determined as 2|D|

    def _Select(self,X,y):
        """
            select feature/or sampling of data
            1) origin
            2) random
            3) cluster
        """

        if self.select == 'origin':
            self.Ns = np.shape(X)[0]
            return X,y

        elif self.select == 'random':
            self.Ns = self.selectargs.get('N',100)
            if self.Ns < np.shape(X)[0]:
                Xs = np.zeros((self.Ns, np.shape(X)[1]))
                ys = np.zeros((self.Ns))
                idx = np.random.choice(np.shape(X)[0], self.Ns, replace=False)
                for i in range(len(_idx)):
                    Xs[i,:] = X[idx[i],:]
                    ys[i] = y[idx[i]]
                return Xs,ys
            raise ValueError('random N exceeds oringal data')

        elif self.select == 'cluster':
            #TODO
            # k = self.selectargs.get('k',200)
            # kmean = kmean(X)
            # return kmean.kavgs
            return X,y

    def _Kernel(self,X,Xp):
        """construct kernel
            when X=Xp, are both training data, it construct an inner kenel
            when X is training data, and Xp is a test data, it construct a crossing kernel
        """
        # HERE WE USE EINSTEIN'S SUMMATION RULE FOR IMPLEMENT
        # linear kernel: Kij = Xik Xjk
        if self.kernel == 'linear':
            return np.dot(X, Xp.T)

        # polynormial kernel: Kij = (Xik Xjk + 1)**P
        elif self.kernel == 'poly':
            P = self.kernelargs.get('polyorder',2)
            return (1+np.dot(X,Xp.T))**P

        # gauss kernel: Kij = exp(-(Xik-Xjk)(Xik-Xjk)/(2*sigma**2))
            """@note
                (Xik-Xjk)(Xik-Xjk) = [Xik Xik + Xjk Xjk] - 2*Xik Xjk
            """
        elif self.kernel == 'gauss':
            _N2 = np.zeros((X.shape[0]))
            _I = np.ones((X.shape[0]))
            _Np2 = np.zeros((Xp.shape[0]))
            _Ip = np.ones((Xp.shape[0]))

            for i in range(X.shape[0]):
                _N2[i] = np.norm(X[i,:])**2
            for i in range(Xp.shape[0]):
                _Np2[i] = np.norm(Xp[i,:])**2

            norm2 = np.outer(_I, _Np2) + np.outer(_N2, _Ip) - 2*np.dot(X,Xp.T)
            sigma = self.kernelargs.get('sigma', 1 )

            return np.exp( - norm2 / (2*sigma**2) )
        else:
            raise ValueError('kernel error')

    def _AlphaCor(self):
        zero_idx = np.argwhere(self.alpha < self.eps).reshape(1,-1)[0]
        zero_len = len(zero_idx)
        if zero_len < self.Ns:
            correct = np.sum(self.alpha[zero_idx])/(self.Ns - zero_len)
            self.alpha += correct
        self.alpha[zero_idx] = 0

        self.support_idx = np.argsort(-self.alpha)

    def _KKTBias(self):
        """Calculate and sort of KKT bias
        a=0    <=>  yi (gi-yi) >= 0, first tyope
        0<a<C  <=>  yi (gi-yi) == 0, second type
        a=C    <=>  yi (gi-yi) <= 0, third type
        """
        for i in range(self.Ns):
            if self.alpha[i] < self.eps:
                self.kktbias[i] = max(1-self.eps - self.y[i]*self.predict[i], 0)
            elif self.alpha[i] > self.C - self.eps:
                self.kktbias[i] = max(-(1+self.eps) + self.y[i]*self.predict[i], 0)
            else:
                self.kktbias[i] = max( abs( self.y[i]*self.predict[i] - 1 ) - self.eps, 0 )
        if len(np.argwhere(self.kktbias > self.eps).reshape(1,-1)[0]) > 0:
            self.kkt_hold = False
        else:
            self.kkt_hold = True

        self.kktbias_idx = np.argsort(-self.kktbias)

    def fit(self,X,y):
        """fitting SVM:
            train data X and label y, with in SVM
        """

        # sampling and determin Ns
        self.X, self.y = self._Select(X,y)
        if self.maxiter == 0:
            self.maxiter = 10*self.Ns

        # inner kernel
        self.K = self._Kernel(self.X, self.X)

        # svm training variables, initial with zero, s.t. \sum alpha_i y_i = 0
        self.alpha = np.zeros((self.Ns))
        self.b = 0

        # Prediction and error
        self.predict = np.dot(self.alpha*self.y, self.K) + self.b
        self.error = self.predict - self.y

        # Bias from KKT condition
        self.kktbias = np.zeros((self.Ns))
        self.kkt_hold = False
        self.kktbias_idx = np.zeros((self.Ns),dtype='int') # sorted indice of kktbias
        self.support_idx = np.zeros((self.Ns),dtype='int') # sorted indice of kktbias
        self._KKTBias() # initial calculate and sort

        #print(self.predict, self.error, self.kktbias, self.kktbias_idx)

        self.SMO_sequential()

        return np.dot(self.alpha*self.y,self.X), self.b

    def SMO_sequential(self):
        """SMO algoithm, for choosing sequential pair (i,j) for optimization
            heuristical idea:
                i choose with max KKTBias
                j choose with max |error(i) - error(j)|
            clumsy idea: traversing the whole set
        """
        nowiter = 0
        while nowiter < self.maxiter:
            nowiter += 1
            changed = False

            for i in self.kktbias_idx:
                # find j, for maximize |Ei-Ej|
                if(self.error[i] > 0):
                    err_list = np.argsort(self.error)
                else:
                    err_list = np.argsort(-self.error)

                for j in err_list:
                    changed = self.SMO_minimal(i,j)
                    if changed or self.error[i] * self.error[j]>0:
                        break

                # laterly find j from support vectors
                if not changed:
                    for j in self.support_idx:
                        if j==i:
                            continue
                        changed = self.SMO_minimal(i,j)
                        if changed:
                            break
                if changed:
                    break

            if changed or not self.kkt_hold:
                continue # for we must update kktbias_idx
            else: # we use full i and j, without any change, the converged result
                break
        if nowiter == self.maxiter:
            print('warning: result may not converged!')

    def SMO_minimal(self,i,j):
        """
            try solve optimization problem for pair (i,j)
        """

        #print(i,j)

        #print(self.kktbias)
        #print(self.kktbias[self.kktbias_idx])
        #print(self.kktbias[i], self.kktbias[j])

        if i==j:
            return False

        # save old ai, aj
        ai = self.alpha[i]
        aj = self.alpha[j]

        L = max(0,aj-ai) if(self.y[i]!=self.y[j]) else max(0,aj+ai-self.C)
        H = min(self.C,self.C+aj-ai) if(self.y[i]!=self.y[j]) else min(self.C,aj+ai)
        if L >= H:
            return False # failed

        eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j] #《statistic learning method》p127(7.107)
        if eta >= 0:
            return False # failed, for eta should be positive

        # solve new aj
        self.alpha[j] -= self.y[j]*(self.error[i] - self.error[j] + (1-self.gamma)*(self.y[i] - self.y[j]) ) / eta #《statistic learning method》p127(7.106)
        self.alpha[j] = max(min(self.alpha[j],H),L) #《statistic learning method》p127(7.108)

        D_aj = self.alpha[j] - aj
        if (abs(D_aj) < self.eps):
            self.alpha[j] = aj # reget old value, might without reget old value?
            return False # move not enough

        self.alpha[i] += self.y[j]*self.y[i]*(aj - self.alpha[j])#《statistic learning method》p127(7.109)
        D_ai = self.alpha[i] - ai

        # update b and Ek
        b1 = self.b - self.error[i] - self.y[i]*D_ai*self.K[i,i] - self.y[j]*D_aj*self.K[i,j]
        b2 = self.b - self.error[j] - self.y[i]*D_ai*self.K[i,j] - self.y[j]*D_aj*self.K[j,j]
        if (0 < self.alpha[i] < self.C):
            self.b = b1
        elif (0 < self.alpha[j] < self.C):
            self.b = b2
        else:
            self.b = (b1 + b2)/2.0

        self._AlphaCor()
        self.predict = np.dot(self.alpha*self.y, self.K) + self.b
        self.error = self.predict - self.y
        self._KKTBias()
        return True

    def summary(self):
        accuracy = self.test_accuracy(self.X,self.y)
        w = np.dot(self.alpha,self.X)
        svidx = np.nonzero(self.alpha)[0]
        print('accuracy is %f'%accuracy )
        print('\nHyper surface:\n\tw =\n',w,'\n\tb =\n',self.b)
        print('\nSupport vector number:\t%d'%(len(svidx)))
        print('\nSupport vector f(x):\n', self.predict[svidx])

    def test_accuracy(self,Xt,yt):
        Ns_test, _tmp = np.shape(Xt)
        TP=0;FP=0;TN=0;FN=0

        Kt = self._Kernel(self.X,Xt)
        f = np.dot(self.alpha*self.y, self.K) + self.b
        TP = ( np.heaviside(np.sign(f),0)*np.heaviside(yt,0) ).sum()
        FP = ( np.heaviside(np.sign(f),0)*np.heaviside(-yt,0) ).sum()
        TN = ( np.heaviside(-np.sign(f),0)*np.heaviside(-yt,0) ).sum()
        FN = ( np.heaviside(-np.sign(f),0)*np.heaviside(yt,0) ).sum()

        return (TP+TN)/(TP+TN+FP+FN)

if __name__=='__main__':
    N = 1000
    r = np.random.randint(2,size=N)
    X = np.zeros((N,2))
    y = np.zeros((N))
    X[:,0] = np.random.normal(r,0.5)
    X[:,1] = np.random.normal(r,0.5)
    y = 2*r-1

    clf = svm(kernel='linear', C=1, maxiter=np.float('Inf'), gamma=1)
    w,b = clf.fit(X, y)

    clf.summary()

    for i in range(len(y)):
        plt.plot(X[i,0],X[i,1],'.',c=(0.5+0.2*y[i],0,0.5-0.2*y[i]))

    x1s = np.linspace(-1,2,201)
    x2s = -w[0]/w[1]*x1s - b/w[1]
    plt.plot(x1s,x2s,'k-')
    x2s = -w[0]/w[1]*x1s - (b+1)/w[1]
    plt.plot(x1s,x2s,'k-')
    x2s = -w[0]/w[1]*x1s - (b-1)/w[1]
    plt.plot(x1s,x2s,'k-')
    plt.show()