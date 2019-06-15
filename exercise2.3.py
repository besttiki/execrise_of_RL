"""
    Author:Tiki
    Date:6/15/2019
"""
import random as rd
import numpy as np
import matplotlib.pyplot as plt

def bandit(k,t,means,varis):
    if t%50 == 0:
        means = np.random.uniform(0.5,12,10)
        varis = [ 0.5 for i in range(10)]
    return rd.gauss(means[k],varis[k]),means,varis


def basic_algorithm(epsilon,weight,steps):
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
        if weight == "method_1":
            Q[A] = Q[A] + (1/N[A])*(R-Q[A])
        elif weight == "method_2":
            Q[A] = Q[A] + 0.1*(R-Q[A])
        rec_reward = rec_reward + R
        ans.append(rec_reward/(t+1))
    return ans
                #             N[A] = N[A] + 1
    #             Q[A] = Q[A] + 1/N[A]*(R - Q[A])
if __name__ == '__main__':
    ans_m1 = basic_algorithm(0.1,"method_1",10000)
    ans_m2 = basic_algorithm(0.1,"method_2",10000)
    x = np.linspace(0,10000,10000)
    plt.plot(x,ans_m1,linestyle='--',color ='green',label="sample averages")
    plt.plot(x,ans_m2,linestyle='--',color ='red',label="constant step size")
    plt.legend(loc='upper right')
    plt.show()
    # plt.show()
    # print(31%30)
    # #method 1
    # iteration = 0
    # Q = [0,0,0,0,0,0,0,0,0,0]
    # N = [0,0,0,0,0,0,0,0,0,0]
    # V1 = []
    # OAR1 = []
    # while (iteration < 1000):
    #     if iteration < 0 :
    #         A = rd.randint(0,9)
    #         R = bandit(A)
    #         N[A] = N[A] + 1
    #         Q[A] = Q[A] + 1/N[A]*(R - Q[A])
    #         iteration = iteration + 1
    #         V1.append(R)
    #         rate = N[8]/iteration
    #         OAR1.append(rate)
    #     else:
    #         explore_rate =  rd.random()
    #         if explore_rate < 0.15:
    #             A = rd.randint(0,9)
    #             R = bandit(A)
    #             N[A] = N[A] + 1
    #             Q[A] = Q[A] + 1/N[A]*(R - Q[A])
    #             iteration = iteration + 1
    #             V1.append(R)
    #             rate = N[8]/iteration
    #             OAR1.append(rate)
    #         else:
    #             A = Q.index(max(Q))
    #             R = bandit(A)
    #             N[A] = N[A] + 1
    #             Q[A] = Q[A] + 1/N[A]*(R - Q[A])
    #             iteration = iteration + 1
    #             V1.append(R)
    #             rate = N[8]/iteration
    #             OAR1.append(rate)

    # #method 2
    # iteration = 0
    # Q = [5,5,5,5,5,5,5,5,5,5]
    # N = [0,0,0,0,0,0,0,0,0,0]
    # V2 = []
    # OAR2 = []
    # while (iteration < 1000):
    #     if iteration < 0 :
    #         A = rd.randint(0,9)
    #         R = bandit(A)
    #         N[A] = N[A] + 1
    #         Q[A] = Q[A] + 1/N[A]*(R - Q[A])
    #         iteration = iteration + 1
    #         V2.append(R)
    #         rate = N[8]/iteration
    #         OAR2.append(rate)
    #     else:
    #         explore_rate =  rd.random()
    #         if explore_rate < 0.00:
    #             A = rd.randint(0,9)
    #             R = bandit(A)
    #             N[A] = N[A] + 1
    #             Q[A] = Q[A] + 1/N[A]*(R - Q[A])
    #             iteration = iteration + 1
    #             V2.append(R)
    #             rate = N[8]/iteration
    #             OAR2.append(rate)
    #         else:
    #             A = Q.index(max(Q))
    #             R = bandit(A)
    #             N[A] = N[A] + 1
    #             Q[A] = Q[A] + 1/N[A]*(R - Q[A])
    #             iteration = iteration + 1
    #             V2.append(R)
    #             rate = N[8]/iteration
    #             OAR2.append(rate)
    # print(Q)
    # #plt.figure(figsize=(8,4))
    # #plt.figure(figsize=(8,4))

    # x = np.linspace(0,1000,1000)
    # plt.plot(x,OAR1,linestyle='--',color ='green')
    # plt.plot(x,OAR2,linestyle='--',color ='red')
    # plt.show()
    # print(V1)
    
