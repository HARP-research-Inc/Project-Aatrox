from simulated_bifurcation.core import QuadraticPolynomial
from simulated_bifurcation.models import ABCModel
import simulated_bifurcation as sb
from typing import Union, Optional
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools
import random

class portfolio_opt_qubo():
        def __init__(
                self,
                theta,
                R,
                price,
                data,
                K
        ):
                self.thet = theta
                self.ret = R
                self.df = pd.DataFrame(data)
                self.pf = pd.DataFrame(price)
                self.cov_matrix = self.df.cov()
                self.monte_carlo_M = tools.monte_carlo_mcal(self.df,R,self.cov_matrix,N=5000)


                self.N = len(price)
                self.ganurity = K
                self.lb = None

        def first_stage(self):
                # self.monte_carlo_M = tools.monte_carlo_mcal(self.df,R,self.cov_matrix,N=5000)
                # M1 = torch.tensor((self.thet*self.cov_matrix + self.monte_carlo_M*(self.pf.T @ self.pf)).to_numpy())
                # V1 = torch.tensor((-2*self.monte_carlo_M*self.ret*(self.pf)).to_numpy()[0])
                # C1 = 1*self.monte_carlo_M*self.ret*self.ret
                M1 = torch.tensor(((self.thet*self.cov_matrix) - (self.monte_carlo_M*(self.pf.T @ self.pf))).to_numpy())
                V1 = torch.tensor((2*self.monte_carlo_M*self.ret*(self.pf)).to_numpy()[0])
                C1 = -1*self.monte_carlo_M*self.ret*self.ret

                binary_value, binary_vector = sb.maximize(M1, V1, C1, domain='binary')

                print(binary_value,binary_vector)
                self.lb = np.array(binary_value.numpy())
        
        def second_stage(self):
                T1 = np.zeros((self.N, self.N*self.ganurity))
                for i in range(self.N):
                        for k in range(self.ganurity):
                                T1[i][(i)*self.ganurity + k] = pow(2,k)/pow(2,self.ganurity)


                M2 = torch.tensor(((self.thet*(T1.T @ self.cov_matrix @ T1)) - (self.monte_carlo_M*T1.T@(self.pf.T @ self.pf)@T1)).to_numpy())
                V2 = torch.tensor((2*self.monte_carlo_M*self.ret*(self.pf @ T1)).to_numpy()[0])
                C2 = -1*self.monte_carlo_M*self.ret*self.ret + (random.randint(0, 5000)/10000*self.monte_carlo_M)
                H = self.stage2_refine_QUBO()
                binary_value, binary_vector = sb.maximize(M2, V2, C2+H, domain='binary')
                binary_value = binary_value.numpy().reshape(1,self.N*self.ganurity)

                matrix = M2.numpy()
                vector = V2.numpy().reshape(self.N*self.ganurity,1)
                
                result = binary_value @ matrix @ binary_value.T + binary_value @ vector + C2 + H
                print(binary_value,binary_vector)
                if(result < 0):
                        print("FAILURE, classifies Lable of stage two as Lb_k_without")
                else:
                        print("SUCCESS, classifies Lable of stage two as Lb_k_with")

                return binary_value, binary_vector

        def digit_to_binary(self,num):
                return np.array([int(i) for i in bin(num)[2:].zfill(self.ganurity)])
        
        def convert_binary(self,x_list):
                return np.array([self.digit_to_binary(int(x)) for x in x_list]).reshape(self.N*self.ganurity,)
                #returns array
        
        def stage2_refine_QUBO(self):
                array = (self.convert_binary(self.lb))
                K = self.ganurity
                H = 0
                for e in self.lb:
                        temp = 0
                        for i in range(K):
                                temp -= array[int(e*K + i)]
                                if i != K-1:
                                        for cross_term in range(i+1, K):
                                                temp += array[int(e*K + i)] * array[int(e*K + cross_term)]
                        temp += 1
                        H += np.random.uniform(0.0000001, 0.5) * self.monte_carlo_M * temp
                return H


class helper(portfolio_opt_qubo):
        def __init__(
                self, 
                theta, #scaling factor
                R, #targeted return
                returns, #dictionary of all stocks containing the price of stock at given time
                current, #dictionary of all stocks prices currently
                K=None, #ganurity
        ):
                self.scaling_factor = theta
                self.target = R
                self.ganurity = K
                self.expected = returns
                self.actual = current
                self.N = len(returns)
                super().__init__(theta, R, current, returns, K)

        def helper_optimize(self):
                super().first_stage()
                weights, _ = super().second_stage()
                self.visualization(weights)

        def visualization(self,weights):
                wei = np.array(weights).reshape(self.N,self.ganurity)
                lab = []
                w = []
                for i in range(len(wei)):
                        str_bin = ''.join(str(int(x)) for x in wei[i])
                        w.append(int(str_bin,2))
                        lab.append("Stock #" + str(i+1))
                plt.pie(np.array(w), labels=np.array(lab),autopct='%1.1f%%')
                plt.legend()
                plt.show()
                
        
        