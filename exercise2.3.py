"""
    Author:Tiki
    Date:6/15/2019
"""
import random as rd
import numpy as np
import matplotlib.pyplot as plt

def bandit(k,t,means,varis):
    """
        basic bandit model
        params:
            k: int,the k-armed bandit
            t: int,the run step
            means : list,mean of the k-armed bandit
            varis : list,variance of the k-armed bandit
        return:
            reward: the reward of the k-armed bandit
            means: new value of mean
            varis: new value of variance
    """
    if t%50 == 0:
        means = np.random.uniform(0.5,12,10)
        varis = [ 0.5 for i in range(10)]
    return rd.gauss(means[k],varis[k]),means,varis


def basic_algorithm(epsilon,weight,steps):
    """
        simple reinforcement learning algorithm
        params:
            epsilon: float,epsilon-greedy
            weight: str,sample averages or others
            steps: int,total periods
        return:
            ans: list, average reward of each step 
    """
    Q = [0 for i in range(10)]
    N = [0 for i in range(10)]
    fix_means = [0 for i in range(10)]
    fix_varis = [0 for i in range(10)]
    rec_reward = 0
    ans = []
    for t in range(steps):
        explore_will = rd.random()
        A,R = 0,0
        if explore_will > epsilon:
            A = Q.index(max(Q))
            R,fix_means,fix_varis = bandit(A,t,fix_means,fix_varis)
        else:
            A = rd.randint(0,9)
            R,fix_means,fix_varis = bandit(A,t,fix_means,fix_varis)
        N[A] = N[A] + 1
        if weight == "sample averages":
            Q[A] = Q[A] + (1/N[A])*(R-Q[A])
        elif weight == "constant step size":
            Q[A] = Q[A] + 0.1*(R-Q[A])
        rec_reward = rec_reward + R
        ans.append(rec_reward/(t+1))
    return ans

if __name__ == '__main__':
    ans_m1 = basic_algorithm(0.1,"sample averages",10000)
    ans_m2 = basic_algorithm(0.1,"constant step size",10000)
    x = np.linspace(0,10000,10000)
    plt.plot(x,ans_m1,linestyle='--',color ='green',label="sample averages")
    plt.plot(x,ans_m2,linestyle='--',color ='red',label="constant step size")
    plt.legend(loc='upper right')
    plt.show()
    
