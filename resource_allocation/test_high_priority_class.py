import numpy as np
from utils_stat import aLTT,aLTT_multiple_risks
import pickle
import matplotlib.pyplot as plt

with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
	res = pickle.load(fp)
URLLC=np.arange(16,32)  #High Priority Class
DATA_TR=np.arange(0,16)  #Low Priority Class
TARGET_Qs=np.arange(0.10,0.3,0.01)
for target_queue_urllc in TARGET_Qs:
	'''Remapping energy-efficiency of high-priority users into a loss bounded by 1'''
	EE_max = 1
	target_ee_urllc = 0.01  #Energy Efficiency Req.
	target_ee_urllc = -target_ee_urllc + EE_max
	ee_urllc = np.mean(res['avg_tx_bits'][:, :, URLLC], axis=2) / np.mean(res['avg_tx_powers'][:, :, URLLC], axis=2) / 1e6
	ee_urllc[np.isnan(ee_urllc)] = 0
	ee_urllc=-ee_urllc+EE_max
	'''Normalized Queue Size of high-priority users'''
	queue_urllc = np.mean(res['queue_pkts'][:,:,URLLC],axis=2)/8
	ALPHAs=[target_queue_urllc,target_ee_urllc]
	LOSSEs=[queue_urllc,ee_urllc]
	delta=0.1  #Error Tolerance Level
	CONTROL='FDR'
	T=5000 		#Calibration Horizon
	for CONTROL in ['FDR','FWER']:		#Type of error control
		'''Adaptive Testing'''
		for eps in [0.25,0.5,0.95]:		#epsilon parameter for epsilon-greegy exploration
			for exp_pol in ['GREEDY_E']:	#Type of Exploration
				for bet_pol in ['AGRAPA']:	#Betting strategy to test
					file_name='aLTT_'+bet_pol+'_'+exp_pol+'_eps_'+str(eps)+'_delta_'+str(delta)+'_alpha_q_occ_'+str(target_queue_urllc)+'_alpha_ee_'+str(target_ee_urllc)+'_'+CONTROL+'.pkl'
					print('Testing '+file_name)
					SET_T,WS_T,N_T=[],[],[]
					n_evals_tot = np.zeros(len(LOSSEs[0]))
					while(np.max(n_evals_tot)<LOSSEs[0].shape[1]-T):
						DATAs=[[l[int(id_s):int(id_s)+T] for l,id_s in zip(loss_vals,n_evals_tot)] for loss_vals in LOSSEs]
						Ws_t,N_t,Set_t=aLTT_multiple_risks(DATAs,ALPHAs,delta,T,bet_policy=bet_pol,exp_policy=exp_pol,eps=eps,control=CONTROL)
						SET_T.append(np.uint8(Set_t[::25, :]))
						N_T.append(np.int32(N_t[::25, :]))
						WS_T.append(np.float32(Ws_t[::25, :]))
						n_evals_tot=n_evals_tot+N_t[-1]
					dictionary = {
						'alpha': ALPHAs,
						'SET_T': SET_T,
						'N_T': N_T,
						'WS_T': WS_T
					}
					with open('logs/results/'+file_name, 'wb') as f:
						pickle.dump(dictionary, f)
		eps=0
		'''Non-Adaptive Testing'''
		for exp_pol in ['UNIFORM']: #Non-adaptive Exploration
			for bet_pol in ['AGRAPA']:
				file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_q_occ_' + str(target_queue_urllc) + '_alpha_ee_' + str(
					target_ee_urllc) + '_' + CONTROL + '.pkl'
				print('Testing ' + file_name)
				SET_T, WS_T, N_T = [], [], []
				n_evals_tot = np.zeros((len(ALPHAs), len(LOSSEs[0])))
				while (np.max(n_evals_tot) < LOSSEs[0].shape[1] - T):
					DATAs = [[l[int(id_s):int(id_s) + T] for l, id_s in zip(loss_vals, n)] for loss_vals, n in zip(LOSSEs, n_evals_tot)]
					Ws_t, N_t, Set_t = aLTT_multiple_risks(DATAs, ALPHAs, delta, T, bet_policy=bet_pol, exp_policy=exp_pol, eps=eps, control=CONTROL)
					SET_T.append(np.uint8(Set_t[::25, :]))
					N_T.append(np.int32(N_t[::25, :]))
					WS_T.append(np.float32(Ws_t[::25, :]))
					n_evals_tot = n_evals_tot + N_t[-1]
				dictionary = {
					'alpha': ALPHAs,
					'SET_T': SET_T,
					'N_T': N_T,
					'WS_T': WS_T
				}
				with open('logs/results/' + file_name, 'wb') as f:
					pickle.dump(dictionary, f)
