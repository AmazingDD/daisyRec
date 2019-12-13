cimport numpy as np
import numpy as np
from collections import defaultdict

class RSVD(object):
    def __init__(self, user_num, item_num, n_factors=96, n_epochs=20, version=2, init_mean=0, init_std_dev=.1, 
                 lr=.001, reg=.02, reg2=.05, random_state=None, verbose=True):
        self.user_num = user_num
        self.item_num = item_num

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.version = version
        self.lr = lr
        self.reg = reg
        self.reg2 = reg2
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, train_set):
        cdef np.ndarray[np.double_t] ci
        cdef np.ndarray[np.double_t] dj
        cdef np.ndarray[np.double_t, ndim=2] ui
        cdef np.ndarray[np.double_t, ndim=2] vj
        cdef int i, j, k
        cdef double r, err, dot, uik, vjk, cii, djj
        cdef double global_mean = train_set.rating.mean()

        cdef double lr = self.lr
        cdef double reg = self.reg
        cdef double reg2 = self.reg2

        ci = np.zeros(self.user_num, np.double)
        dj = np.zeros(self.item_num, np.double)

        ui = np.random.normal(self.init_mean, self.init_std_dev, size=(self.user_num, self.n_factors))
        vj = np.random.normal(self.init_mean, self.init_std_dev, size=(self.item_num, self.n_factors))

        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f'Processing epoch {epoch + 1}')
                for _, row in train_set.iterrows():
                    i, j, r = row['user'], row['item'], row['rating']
                    dot = 0
                    for k in range(self.n_factors):
                        dot += ui[i, k] * vj[j, k]
                    err = r - (ci[i] + dj[j] + dot)

                    if self.version == 2:
                        cii = ci[i]
                        djj = dj[j]
                        ci[i] += lr * (err - reg2 * (cii + djj - global_mean))
                        dj[j] += lr * (err - reg2 * (cii + djj - global_mean))
                    
                    for k in range(self.n_factors):
                        uik = ui[i, k]
                        vjk = vj[j, k]
                        ui[i, k] += lr * (err * vjk - reg * uik)
                        vj[j, k] += lr * (err * uik - reg * vjk)
        
        self.ci = ci
        self.dj = dj
        self.ui = ui
        self.vj = vj

    def predict(self, i, j):
        if i >= self.user_num:
            raise ValueError('Invalid user code')
        if j >= self.item_num:
            raise ValueError('Invalid item code')
        if self.version == 2:
            est = self.ci[i] + self.dj[j] + np.dot(self.ui[i], self.vj[j])
        elif self.version == 1:
            est = np.dot(self.ui[i], self.vj[j])

        return est


class SVD(object):
    def __init__(self, user_num, item_num, n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=.1, 
                 lr_all=.005, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, 
                 reg_pu=None, reg_qi=None, random_state=None, verbose=True):
        self.user_num = user_num
        self.item_num = item_num

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, train_set):
        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef int u, i, f
        cdef double r, err, dot, puf, qif, global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        bu = np.zeros(self.user_num)
        bi = np.zeros(self.item_num)
        pu = np.random.normal(self.init_mean, self.init_std_dev, size=(self.user_num, self.n_factors))
        qi = np.random.normal(self.init_mean, self.init_std_dev, size=(self.item_num, self.n_factors))

        global_mean = train_set.rating.mean()
        if not self.biased:
            global_mean = 0
        self.global_mean = global_mean

        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f'Processing epoch {epoch + 1}')
            for _, row in train_set.iterrows():
                u, i, r = row['user'], row['item'], row['rating']
                dot = 0
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (self.global_mean + bu[u] + bi[i] + dot)

                # update bias
                if self.biased:
                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])
                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                    qi[i, f] += self.lr_qi * (err * puf - self.reg_qi * qif)
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def predict(self, u, i):
        if u >= self.user_num:
            raise ValueError('Invalid user code')
        if i >= self.item_num:
            raise ValueError('Invalid item code')
        if self.biased:
            est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.qi[i], self.pu[u])
        else:
            est = np.dot(self.qi[i], self.pu[u])

        return est

class SVDpp(object):
    def __init__(self, user_num, item_num, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None, 
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None, random_state=None, verbose=True):
        self.user_num = user_num
        self.item_num = item_num
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, train_set):
        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef np.ndarray[np.double_t, ndim=2] yj

        cdef int u, i, j, f
        cdef double r, err, dot, puf, qif, sqrt_Iu, _, 
        cdef double global_mean = train_set.rating.mean()
        cdef np.ndarray[np.double_t] u_impl_fdb

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_yj = self.lr_yj

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj

        bu = np.zeros(self.user_num, np.double)
        bi = np.zeros(self.item_num, np.double)
        pu = np.random.normal(self.init_mean, self.init_std_dev, size=(self.user_num, self.n_factors))
        qi = np.random.normal(self.init_mean, self.init_std_dev, size=(self.item_num, self.n_factors))

        yj = np.random.normal(self.init_mean, self.init_std_dev, size=(self.item_num, self.n_factors))
        u_impl_fdb = np.zeros(self.n_factors)

        self.global_mean = global_mean

        ur = defaultdict(list)
        for _, row in train_set.iterrows():
            u, i, r = row['user'], row['item'], row['rating']
            ur[u].append((i, r))

        for epoch in range(self.n_epochs):
            if self.verbose:
                print(f'processing epoch {epoch + 1}')
            for _, row in train_set.iterrows():
                u, i, r = row['user'], row['item'], row['rating']
                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))
                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu
                # compute current error
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])
                
                err = r - (self.global_mean + bu[u] + bi[i] + dot)
                
                # update biases
                bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += self.lr_pu * (err * qif - self.reg_pu * puf)
                    qi[i, f] += self.lr_qi * (err * (puf + u_impl_fdb[f]) - self.reg_qi * qif)
                    for j in Iu:
                        yj[j, f] += self.lr_yj * (err * qif / sqrt_Iu - self.reg_yj * yj[j, f])

        self.ur = ur

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def predict(self, u, i):
        est = self.global_mean
        if u >= self.user_num:
            raise ValueError('Invalid user code')
        if i >= self.item_num:
            raise ValueError('Invalid item code')
        est += self.bu[u] + self.bi[i]
        Iu = len(self.ur[u]) # # No. of items rated by u
        if Iu == 0:
            u_impl_feedback = 0
        else:
            u_impl_feedback = (sum(self.yj[j] for (j, _) in self.ur[u]) / np.sqrt(Iu))
        est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est
