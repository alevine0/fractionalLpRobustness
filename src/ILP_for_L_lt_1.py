import cvxpy as cp
import numpy as np
import math
import torch
import sys
import os


from argparse import ArgumentParser
argparser = ArgumentParser()
argparser.add_argument("--inverse_p", default=2., type=float)
argparser.add_argument("--scale", default=1., type=float)
argparser.add_argument("--budget", default=1000, type=int)
argparser.add_argument("--approx_gap", default=0.02, type=float)
args = argparser.parse_args()




quantization = 255

func = lambda x: min(math.pow(x/args.scale, 1./args.inverse_p),1)
func_description ='p=1_over_'+str(args.inverse_p)+',scale='+str(args.scale)
C_base = np.zeros([quantization,quantization])





for i in range(1, quantization+1):
	for j in range(1, quantization+1):
		C_base[i-1,j-1] = 1.*i if i<j else j
v = np.arange(1,quantization + 1)
fn = np.array([func(x/quantization) for x in range(1,quantization + 1)])


xs = [0] + list(v/quantization)
ys_fun = [func(0)] + list(fn)
err_fake =  open(os.devnull, 'w')
import matplotlib.pyplot as plt 

C = C_base/float(args.budget)
x = cp.Variable(quantization,  integer = True)
y = cp.Variable(1)

constraints = [ x >= np.zeros(quantization), C@x <= fn, v.T@x <= args.budget, fn - C@x <= y*np.ones(quantization), y <= args.approx_gap]
objective = 0
prob = cp.Problem(cp.Minimize(objective), constraints)

prob.solve( solver=cp.SCIP, verbose=True)
if (prob.status != 'infeasible'):
	print('Solution:')
	best_y = y.value
	print("\t L_inf approximation gap: " + str(best_y))
	print("\t Budget: " + str(args.budget))
	print("\t Solution: " + str(x.value), flush=True)
	torch.save({'budget': args.budget, 'solution': x.value}, func_description+'_uniform_budget_'+str(args.budget)+ '_approx_gap_'+str(args.approx_gap)+'.pth')

	ys_approx = [0] + list((C@x).value)

	plt.figure()
	plt.plot(xs,ys_fun, label='True function', color='red', linestyle = '--')
	plt.plot(xs,ys_approx, label='Approximated function', color='blue')
	plt.legend()
	plt.ylim(0,1.)

	plt.xlim(0,1.)
	plt.savefig(func_description+'_uniform_budget_'+str(args.budget)+ '_approx_gap_'+str(args.approx_gap)+'.png', dpi=400)
	plt.close()
else:
	print('No solution found.')