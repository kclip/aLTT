import numpy as np
from utils_stat import aLTT
import pickle 

'''Script to compare aLTT and LTT'''

'''Load log of collected data (for faster testing)'''
with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
	res = pickle.load(fp)
loss_vals=res['loss_vals'][1:]
n_cal=50000
loss_vals=[l[:n_cal] for l in loss_vals]
alpha=0.57		#Target Reliability
delta=0.1		#Error Tolerance Level
T=5000			#n° of Calibratin Rounds
for CONTROL in ['FDR','FWER']:	#Type of error control

	'''Adaptive Testing (alTT)'''
	for eps in [0.25,0.5,0.75,0.95]:	#epsilon parameter for epsilon-greegy exploration
		for exp_pol in ['GREEDY_E']:	#Type of Exploration
			for bet_pol in ['AGRAPA']:		#Betting strategy to test
				file_name='aLTT_'+bet_pol+'_'+exp_pol+'_eps_'+str(eps)+'_delta_'+str(delta)+'_alpha_'+str(alpha)+'_'+CONTROL+'.pkl'
				SET_T,WS_T,N_T=[],[],[]
				n_evals_tot = np.zeros(len(loss_vals))
				while(np.max(n_evals_tot)<len(loss_vals[0])-T):
					data=[l[int(id_s):int(id_s)+T] for l,id_s in zip(loss_vals,n_evals_tot)]
					Ws_t,N_t,Set_t=aLTT(data,alpha,delta,T,bet_policy=bet_pol,exp_policy=exp_pol,eps=eps,control=CONTROL)
					SET_T.append(np.uint8(Set_t))
					N_T.append(np.uint8(N_t))
					WS_T.append(np.float32(Ws_t))
					n_evals_tot=n_evals_tot+N_t[-1,:]
				dictionary = {
					'alpha': alpha,
					'SET_T': SET_T,
					'N_T': N_T,
					'WS_T': WS_T
				}
				with open('logs/results/'+file_name, 'wb') as f:
					pickle.dump(dictionary, f)

	'''Non-adaptive Testing Benchmark (LTT)'''
	eps=0  #Zero exploitation
	for exp_pol in ['UNIFORM']: #Uniform Exploration
		for bet_pol in ['AGRAPA']:	#Betting strategy to test
			file_name='aLTT_'+bet_pol+'_'+exp_pol+'_eps_'+str(eps)+'_delta_'+str(delta)+'_alpha_'+str(alpha)+'_'+CONTROL+'.pkl'
			SET_T,WS_T,N_T=[],[],[]
			n_evals_tot = np.zeros(len(loss_vals))
			while(np.max(n_evals_tot)<len(loss_vals[0])-T):
				data = [l[int(id_s):int(id_s) + T] for l, id_s in zip(loss_vals, n_evals_tot)]
				Ws_t,N_t,Set_t=aLTT(data,alpha,delta,T,bet_policy=bet_pol,exp_policy=exp_pol,eps=eps,control=CONTROL)
				SET_T.append(np.uint8(Set_t))
				N_T.append(np.uint8(N_t))
				WS_T.append(np.float32(Ws_t))
				n_evals_tot = n_evals_tot + N_t[-1, :]
			dictionary = {
				'alpha': alpha,
				'SET_T': SET_T,
				'N_T': N_T,
				'WS_T': WS_T
			}
			with open('logs/results/'+file_name, 'wb') as f:
				pickle.dump(dictionary, f)