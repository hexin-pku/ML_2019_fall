# Filename: svm.py
import numpy as np
import matplotlib.pyplot as plt

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
            
            7) postselect: Support SSCA 
            8) postconstruct: Support IPA
            
            9) eps: tolerance of numerical error, default 1e-5
            10) maxiter: max iteration times, default 2|D|
        """
        
        self.select = kwargs.get('select', 'origin')
        self.selectargs = kwargs.get('selectargs', {})
        
        self.kernel = kwargs.get('kernel', 'linear')
        self.kernelargs = kwargs.get('kernelargs', {})
        
        self.C = kwargs.get('C', 1)
        self.gamma = kwargs.get('gamma', 1)
        
        
        self.postselect = kwargs.get('postselect','SSCA')
        self.postconstruct = kwargs.get('postconstruct', 'IPA')
        self.postargs = kwargs.get('postargs', {})
        
        self.eps = kwargs.get('eps', 1e-10)
        self.epslow = kwargs.get('epslow', 1e-10)
        self.maxiter = kwargs.get('maxiter', 0) # laterly will auto determined as 2|D|
    
    def _Select(self,X,y):
        """
            select feature/or sampling of data
            1) origin
            2) random
            3) cluster
        """
        
        self.nfeature = np.shape(X)[1]

        # utilize the original X as the sample of the data
        if self.select == 'origin':
            self.Ns = np.shape(X)[0]
            return X,y
        # utilize numpy.random.choice to sample a specific percentage of data, then return the data
        elif self.select == 'random':
            sample_percentage = self.selectargs.get('per', 0.1)
            self.Ns = int(np.shape(X)[0] * sample_percentage)
            sample_id = np.random.choice(np.shape(X)[0], self.Ns, replace = False)
            Xs = np.zeros((self.Ns, np.shape(X)[1]))
            ys = np.zeros(self.Ns)
            for i in range(len(sample_id)):
                Xs[i] = X[sample_id[i]]
                ys[i] = y[sample_id[i]]
            return Xs,ys

        # implement 4 clustering methods to cluster all data
        elif self.select == 'cluster':
            method = self.selectargs.get('method', 'kmeans')
            # utilize K-Means to cluster different data, and calculate the sum of y value of each cluster, 
            # if the sum is positive, then the y value of the cluster is +1, else if the sum is negative,
            # then the y value of the cluster is -1, if the sum is zero, we put all data belonging to the
            # cluster back to the sample set.
            if method == 'kmeans':
                clusters = self.selectargs.get('clusters', 10)
                Xs = np.zeros((clusters, np.shape(X)[1]))
                ys = np.zeros(clusters)
                kmeans = KMeans(n_clusters = clusters, random_state=0).fit(X)
                Xs = kmeans.cluster_centers_
                for i in range(np.shape(X)[0]):
                    ys[kmeans.labels_[i]] += y[i]
                ys = np.sign(ys)
                zero_indices = np.where(ys == 0)
                Xs = np.delete(Xs, zero_indices, 0)
                ys = np.delete(ys, zero_indices)
                for i in range(np.shape(X)[0]):
                    if kmeans.labels_[i] in zero_indices[0]:
                        Xs = np.append(Xs, [X[i]], 0)
                        ys = np.append(ys, y[i])
                self.Ns = np.shape(Xs)[0]
                return Xs,ys

            # utilize Mean-shift to cluster different data, and calculate the sum of y value of each cluster, 
            # if the sum is positive, then the y value of the cluster is +1, else if the sum is negative,
            # then the y value of the cluster is -1, if the sum is zero, we put all data belonging to the
            # cluster back to the sample set.
            elif method == 'meanshift':
                bw = self.selectargs.get('bandwith', 0.1)
                meanshift = MeanShift(bandwidth = bw).fit(X)
                clusters = np.unique(meanshift.labels_).shape[0]
                print(clusters)
                Xs = meanshift.cluster_centers_
                ys = np.zeros(clusters)
                for i in range(np.shape(X)[0]):
                    ys[meanshift.labels_[i]] += y[i]
                ys = np.sign(ys)
                zero_indices = np.where(ys == 0)
                Xs = np.delete(Xs, zero_indices, 0)
                ys = np.delete(ys, zero_indices)
                for i in range(np.shape(X)[0]):
                    if meanshift.labels_[i] in zero_indices[0]:
                        Xs = np.append(Xs, [X[i]], 0)
                        ys = np.append(ys, y[i])
                self.Ns = np.shape(Xs)[0]
                return Xs,ys

            # utilize DBSCAN to cluster different data, and calculate the sum of y value of each cluster, 
            # if the sum is positive, then the y value of the cluster is +1, else if the sum is negative,
            # then the y value of the cluster is -1, if the sum is zero, we put all data belonging to the
            # cluster back to the sample set
            # finally, we put all nosie data back to the sample set.
            elif method == 'dbscan':
                argeps = self.selectargs.get('eps', 0.1)
                argmin_samples = self.selectargs.get('min_samples', 10)
                dbscan = DBSCAN(eps = argeps, min_samples = argmin_samples).fit(X)
                clusters = 0
                if -1 in np.unique(dbscan.labels_):
                    clusters = np.unique(dbscan.labels_).shape[0] - 1 + np.where(dbscan.labels_ == -1)[0].shape[0]
                else:
                    clusters = np.unique(dbscan.labels_).shape[0]
                Xs = np.zeros((0, np.shape(X)[1]))
                ys = np.zeros(0)
                for cnum in np.unique(dbscan.labels_):
                    if cnum == -1:
                        indices = np.where(dbscan.labels_ == -1)
                        Xs = X[indices]
                        ys = y[indices]
                    else:
                        indices = np.where(dbscan.labels_ == cnum)
                        tempX = X[indices]
                        tempy = y[indices]
                        yvalue = np.sum(tempy)
                        yvalue = np.sign(yvalue)
                        if yvalue == 0:
                            Xs = np.append(Xs, tempX, axis = 0)
                            ys = np.append(ys, tempy)
                        else:
                            Xs = np.append(Xs, [np.mean(tempX, axis = 0)], axis = 0)
                            ys = np.append(ys, yvalue)
                self.Ns = np.shape(Xs)[0]
                return Xs,ys

            # utilize Gaussian Mixture Model to cluster different data, and calculate the sum of y value of each cluster, 
            # if the sum is positive, then the y value of the cluster is +1, else if the sum is negative,
            # then the y value of the cluster is -1, if the sum is zero, we put all data belonging to the
            # cluster back to the sample set
            elif method == 'gmm':
                components = self.selectargs.get('n_components', 100)
                print(components)
                gmm = GaussianMixture(n_components = components).fit(X)
                Xs = gmm.means_
                ys = np.zeros(components)
                label = gmm.predict(X)
                for i in range(len(label)):
                    ys[label[i]] += y[i]
                ys = np.sign(ys)
                zero_indices = np.where(ys == 0)
                Xs = np.delete(Xs, zero_indices, 0)
                ys = np.delete(ys, zero_indices)
                for i in range(np.shape(X)[0]):
                    if label[i] in zero_indices[0]:
                        Xs = np.append(Xs, [X[i]], 0)
                        ys = np.append(ys, y[i])
                self.Ns = np.shape(Xs)[0]
                return Xs,ys
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
                _N2[i] = np.linalg.norm(X[i,:])**2
            for i in range(Xp.shape[0]):
                _Np2[i] = np.linalg.norm(Xp[i,:])**2
            
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
        
        self.accuracy = self.test_accuracy(self.X,self.y)
        self.w = np.dot(self.alpha,self.X)
        self.svidx = np.nonzero(self.alpha)[0]
        self.nsv = len(self.svidx)
        
        return self.w, self.b
        
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
        
    def PostSample(self,threshold):
        if self.postselect == 'SSCA':
            flag = self.y * ( np.dot(self.alpha*self.y, self.K) + self.b )
            samp_list = np.argwhere(flag > threshold).reshape(1,-1)[0]
            Nsamp = len(samp_list)
            k = len(self.X[0,:])
            Xs = np.zeros((Nsamp,k))
            ys = np.zeros((Nsamp))
            for i in range(Nsamp):
                Xs[i,:] = self.X[samp_list[i],:]
                ys[i] = self.y[samp_list[i]]
            return Xs, ys
        else:
            raise ValueError('post sample error')
    
    def preimg_fixedpoint(self, Xc, beta, Nt):
        # RBFPREIMG RBF pre-image by Schoelkopf's fixed-point algorithm.
        
        Z = np.dot( 2*np.random.rand(1,Nt)-np.ones((1,Nt)), Xc[0:Nt,:])
        Z_old = Z + np.float('Inf')
            
        nowiter = 1    
        while np.linalg.norm(Z-Z_old) > self.eps and nowiter < 10e6:
            nowiter +=1
            Z_old = Z
            
            KB = self._Kernel(Z, Xc[0:Nt,:]) * beta[0:Nt]
            costZ = np.sum(KB)
            
            if costZ > 0:
                Z = np.dot(KB, Xc[0:Nt,:]) / costZ;
            else:
                Z = np.dot( 2*np.random.rand(1,Nt)-np.ones((1,Nt)), Xc[0:Nt,:])
            
        Xc[Nt,:] = Z
        Kzz = self._Kernel(Xc[self.nsv:Nt+1,:],Xc[self.nsv:Nt+1,:])
        Kzx = self._Kernel(Xc[self.nsv:Nt+1,:], self.Xsv)
        beta[self.nsv:Nt+1] = - np.dot( np.dot(np.linalg.inv(Kzz), Kzx), beta[0:self.nsv] )  
        return Xc, beta
            
    def preimg_gradient(self, Xc, beta, Nt):
        # RBFPREIMG2 RBF pre-image problem by Gradient optimization.
     
        def cost(Zt):
            Kz1xNt = self._Kernel(Zt, Xc[0:Nt,:])
            return np.dot(Kz1xNt, beta[0:Nt])[0]
        
        def grad(Zt):
            Kz1xNt = self._Kernel(Zt, Xc[0:Nt,:])
            g1 = Zt * np.sum( np.dot(Kz1xNt, beta[0:Nt] ))
            g2 = np.dot(Kz1xNt[0,:]*beta[0:Nt], Xc[0:Nt,:])
            g1[0,:] -= g2
            return g1
        
        Z = np.zeros((1,self.nfeature))
        Z_old = Z.copy()
        
        for i in range(50):
            idx = np.random.randint(0,Nt-1)
            if cost(Xc[idx:idx+1,:]) > cost(Z):
                Z = Xc[idx:idx+1,:]
            
        while np.linalg.norm(Z-Z_old) > self.epslow:
            Z_old = Z
            g = grad(Z) # find max of cost, along the deriviative
            print('g',g)
            print('cz', cost(Z))
            print('cz+', cost(Z+0.001*g))
            print('cz-', cost(Z-0.001*g))
            
            t = 1.618
            Za = Z; Zb = Za+g; Zc = Zb + t*g;

            costa = cost(Za); costb = cost(Zb); costc = cost(Zc)
            print(costa, costb, costc)
            
            return Xc, beta
            
            while costa < costb and costb < costc:
                Ztemp = Zc
                Zc = Zc + t*(Zc-Zb)
                Za = Zb
                Zb = Ztemp
                costa = cost(Za); costb = cost(Zb); costc = cost(Zc)
                
            while costa > costb and costb > costc:
                Ztemp = Za
                Za = Za + (Za-Zb)/t
                Zc = Zb
                Zb = Ztemp
                costa = cost(Za); costb = cost(Zb); costc = cost(Zc)
                
            #print(costa, costb, costc)    
            #print('find convex')
            
            Zd = Z; right = True
            while True:
                #print('cost Zb', costb)
                if right:
                    Zd = (1.618*Zb + Zc)/2.618
                    costd = cost(Zd);
                    if(costd-costb < self.epslow and costd-costb > - self.epslow):
                        break
                    if costd-costb > 0:
                        Za = Zb
                        Zb = Zd
                        right = True
                    else:
                        Zc = Zd
                        right = False
                else:
                    Zd = (1.618*Zb + Za)/2.618
                    costd = cost(Zd);
                    if(costd-costb < self.epslow and costd-costb > - self.epslow):
                        break
                    if costd-costb >0:
                        Zc = Zb
                        Zb = Zd
                        right = False
                    else:
                        Za = Zd
                        right = True
                costa = cost(Za); costb = cost(Zb); costc = cost(Zc)
            Z = Zd
            print('done Z once')
            
        Xc[Nt,:] = Z
        Kzz = self._Kernel(Xc[self.nsv:Nt+1,:],Xc[self.nsv:Nt+1,:])
        Kzx = self._Kernel(Xc[self.nsv:Nt+1,:], self.X)
        beta[self.nsv:Nt+1] = - np.dot( np.dot(np.linalg.inv(Kzz), Kzx), self.alpha*self.y )  
        return Xc, beta
            
            
    def PostConstruct(self, Nc): # Nc is the target number of support vector
        self.svidx = np.nonzero(self.alpha)[0]
        self.nsv = len(self.svidx)
        self.Xsv = self.X[self.svidx,:]
        self.Asv = self.alpha[self.svidx]
        self.ysv = self.y[self.svidx]
        
        if self.postconstruct == 'IPA' and self.kernel=='gauss': # only support for gauss kernel
            Xc = np.zeros((self.nsv+Nc,self.nfeature ))
            Xc[0:self.nsv,:] = self.Xsv
            beta = np.zeros((self.nsv+Nc))
            beta[0:self.nsv] = self.Asv*self.ysv # in this section, alpha=>alpha*y; beta => beta*y

            for i in range(Nc):
                Nt = self.nsv + i
                if self.postargs.get('preimg', 'grad') == 'grad':
                    Xc, beta = self.preimg_gradient(Xc, beta, Nt)
                elif self.postargs.get('preimg', 'grad') == 'fixed':
                    Xc, beta = self.preimg_fixedpoint(Xc, beta, Nt)
                print("xxx",i)
                
            return Xc[self.nsv:,:], 2*np.heaviside(- beta[self.nsv:],0)-1
        else:
            raise ValueError('post sample error')
            
    def summary(self):
        print('accuracy is %f'%self.accuracy )
        print('\nHyper surface:\n\tw =\n',self.w,'\n\tb =\n',self.b)
        print('\nSupport vector number:\t%d'%self.nsv)
        
    def plot_surface(self, xmin, xmax, ycolors=['k','r'],surfcolors=['b','g']):
        if(self.nfeature!=2):
            raise ValueError('plot_surface is only for nfeature=2 case')
            return
            
        for i in range(self.Ns):
            if(self.y[i] > 0):
                plt.plot(self.X[i,0],self.X[i,1],'.',c=ycolors[0])
            else:
                plt.plot(self.X[i,0],self.X[i,1],'.',c=ycolors[1])
    
        w = np.dot(self.alpha*self.y,self.X)
        b = self.b
        x1s = np.linspace(xmin,xmax,201)
        x2s = -w[0]/w[1]*x1s - b/w[1]
        plt.plot(x1s,x2s,c=surfcolors[0],label='Divide Surface')
        
        x2s = -w[0]/w[1]*x1s - (b+1)/w[1]
        plt.plot(x1s,x2s,c=surfcolors[1], label='Support Surface')
        x2s = -w[0]/w[1]*x1s - (b-1)/w[1]
        plt.plot(x1s,x2s,c=surfcolors[1], label='Support Surface')
        
    def test_accuracy(self,Xt,yt):
        Ns_test, _tmp = np.shape(Xt)
        TP=0;FP=0;TN=0;FN=0
        
        Kt = self._Kernel(self.X,Xt)
        f = np.dot(self.alpha*self.y, Kt) + self.b
        
        TP = ( np.heaviside(np.sign(f),0)*np.heaviside(yt,0) ).sum()
        FP = ( np.heaviside(np.sign(f),0)*np.heaviside(-yt,0) ).sum()
        TN = ( np.heaviside(-np.sign(f),0)*np.heaviside(-yt,0) ).sum()
        FN = ( np.heaviside(-np.sign(f),0)*np.heaviside(yt,0) ).sum()

        return (TP+TN)/(TP+TN+FP+FN)
    