import numpy as np
from utils_stat import aLTT
import pickle 
import matplotlib.pyplot as plt


with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
	res = pickle.load(fp)
TARGET_EEs=np.arange(0.025,1,0.05)  #Energy Efficiency Requirements to Test
for target_EE in TARGET_EEs:
	'''Remapping energy-efficiency of high-priority users into a loss bounded by 1'''
	EE_max = 1
	target_EE = -target_EE + EE_max
	EE = np.mean(res['avg_tx_bits'], axis=2) / np.mean(res['avg_tx_powers'], axis=2)/1e6
	EE[np.isnan(EE)] = 0
	EE = -EE + EE_max
	delta=0.1	#Error Tolerance Level
	T=1500			#Calibration Horizon
	for CONTROL in ['FDR','FWER']:	#Type of error control
		'''Adaptive Testing'''
		for eps in [0.25,0.5,0.75,0.95]:	#epsilon parameter for epsilon-greegy exploration
			for exp_pol in ['GREEDY_E']:	#Type of Exploration
				for bet_pol in ['AGRAPA']:	#Betting strategy to test
					file_name='aLTT_'+bet_pol+'_'+exp_pol+'_eps_'+str(eps)+'_delta_'+str(delta)+'_alpha_ee_'+str(target_EE)+'_'+CONTROL+'.pkl'
					print('Testing '+file_name)
					SET_T,WS_T,N_T=[],[],[]
					n_evals_tot = np.zeros(len(EE))
					while (np.max(n_evals_tot) < len(EE[0]) - T):
						data = [l[int(id_s):int(id_s) + T] for l, id_s in zip(EE, n_evals_tot)]
						Ws_t, N_t, Set_t = aLTT(data, target_EE, delta, T, bet_policy=bet_pol, exp_policy=exp_pol, eps=eps, control=CONTROL)
						SET_T.append(np.uint8(Set_t[::25, :]))
						N_T.append(np.int32(N_t[::25, :]))
						WS_T.append(np.float32(Ws_t[::25, :]))
						n_evals_tot = n_evals_tot + N_t[-1]
					dictionary = {
						'alpha': target_EE,
						'SET_T': SET_T,
						'N_T': N_T,
						'WS_T': WS_T
					}
					with open('logs/results/' + file_name, 'wb') as f:
						pickle.dump(dictionary, f)
		'''Non-adaptive Testing'''
		eps=0
		for exp_pol in ['UNIFORM']:
			for bet_pol in ['AGRAPA']:
				file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_ee_' + str(target_EE) + '_' + CONTROL + '.pkl'
				print('Testing ' + file_name)
				SET_T, WS_T, N_T = [], [], []
				n_evals_tot = np.zeros(len(EE))
				while (np.max(n_evals_tot) < len(EE[0]) - T):
					data = [l[int(id_s):int(id_s) + T] for l, id_s in zip(EE, n_evals_tot)]
					Ws_t, N_t, Set_t = aLTT(data, target_EE, delta, T, bet_policy=bet_pol, exp_policy=exp_pol, eps=eps, control=CONTROL)
					SET_T.append(np.uint8(Set_t[::25, :]))
					N_T.append(np.int32(N_t[::25, :]))
					WS_T.append(np.float32(Ws_t[::25, :]))
					n_evals_tot = n_evals_tot + N_t[-1]
				dictionary = {
					'alpha': target_EE,
					'SET_T': SET_T,
					'N_T': N_T,
					'WS_T': WS_T
				}
				with open('logs/results/' + file_name, 'wb') as f:
					pickle.dump(dictionary, f)
