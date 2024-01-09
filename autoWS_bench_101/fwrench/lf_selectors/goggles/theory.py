import math
import numpy as np
from scipy.stats import beta

def n_given_sum(n_class,n_sum,n_max,x_0):
    n_given_sum_list = [[None for _ in range(n_sum+1)] for _ in range(n_class+1)]
    def _n_given_sum_(n_class,n_sum,n_max):
        if n_given_sum_list[n_class][n_sum] is not None:
            return n_given_sum_list[n_class][n_sum]
        if n_class == 1:
            if n_max < n_sum:
                n_given_sum_list[n_class][n_sum] = 0
                return 0
            else:
                n_given_sum_list[n_class][n_sum] = 1/math.factorial(x_0 - n_sum-1)
                return n_given_sum_list[n_class][n_sum]
        if n_class < 1 and n_sum!=0:
            return 0
        n = 0
        for i in range(min(n_max,n_sum)+1):
            n += 1/math.factorial(x_0-i-1)*_n_given_sum_(n_class-1,n_sum-i,n_max)
        n_given_sum_list[n_class][n_sum] = n
        return n
    return _n_given_sum_(n_class,n_sum,n_max)

class DevSetTheory:
    """
    development set theory
    """
    def __init__(self,d_matrix):
        """
        :param d_matrix: a 2-d numpy array, D_matrix[i,j] is the number of instances
        of the ith class that falls into the jth cluster
        """
        self.D_matrix = d_matrix
        self.K = d_matrix.shape[0]
        self.n=100
        self.alpha_list, self.p_alpha_list= self.p_alphas()
        self.d_alpha = self.alpha_list[1]-self.alpha_list[0]

    def log_likelihood(self,acc):
        log_L = 0
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    log_L += self.D_matrix[i,j]*np.log(acc)
                else:
                    log_L += self.D_matrix[i,j]*np.log(1-acc)
        return log_L

    def P_d(self):
        prior = beta(1,1)
        P = 0
        for acc in np.linspace(1e-6, 1-1e-6,self.n):
            P+=prior.pdf(acc)*np.exp(self.log_likelihood(acc))*( 1 / self.n)
        return P

    def p_alphas(self):
        P_d_matrix = self.P_d()
        prior = beta(1, 1)
        p_alpha_list = []
        acc_list = []
        for acc in np.linspace(1e-6, 1-1e-6, self.n):
            p = prior.pdf(acc) * np.exp(self.log_likelihood(acc)) / P_d_matrix
            p_alpha_list.append(p)
            acc_list.append(acc)
        return acc_list,p_alpha_list

    def p_acc_greater(self,target_acc):
        acc_greater = 0
        for i in range(len(self.alpha_list)):
            acc = self.alpha_list[i]
            if acc > target_acc:
                acc_greater+=self.p_alpha_list[i]*self.d_alpha
        return acc_greater

    def feasibility_test(self,epsilon = 0.7):
        """
        The probability of the task being feasible
        :param epsilon: threshold of the estimated accuracy that make the task considered to be feasible
        :return: the probability of feasibility
        """
        acc_greater = self.p_acc_greater(epsilon)
        acc_greater = np.clip(acc_greater, 0, 1)
        return acc_greater


    def p_one_dim(self,acc,i_dim):
        dev_size = int(np.sum(self.D_matrix[i_dim, :]))
        x_0_min = int(math.ceil((dev_size + self.K - 1) / self.K))

        def p(x_0):
            y_sum = self.K * x_0 - dev_size - (self.K - 1)

            return math.factorial(dev_size) / math.factorial(x_0) * math.exp(
                math.log(acc) * x_0 + math.log((1 - acc) / (self.K - 1)) * (dev_size - x_0)) * \
                   n_given_sum(self.K - 1, y_sum, x_0 - 1, x_0)

        prob = 0
        for x_0 in range(x_0_min, dev_size + 1):
            prob += p(x_0)
        return prob

    def dev_set_sufficiency_test(self):
        """
        test whether the current dev set is sufficient.
        This offers a guidance on whether you should use a larger dev set.
        :return: the lower-bound probability whether the dev set is sufficiency enough
         to obtain the correct class-cluster mapping
        """
        p = 0
        for i in range(len(self.alpha_list)):
            alpha = self.alpha_list[i]
            p_alpha = self.p_alpha_list[i]
            pl = 1
            for i_dim in range(self.K):
                pl*=self.p_one_dim(alpha, i_dim)
            p = p + pl*p_alpha*self.d_alpha
        p = np.clip(p,0,1)
        return p
