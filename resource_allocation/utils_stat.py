from scipy.stats import binom
import numpy as np


def HB_p_value(alpha,l_hat,n):
	'''HB p value'''
	t1=np.exp(-n*kl(np.min(np.array([l_hat,alpha])),alpha))
	t2=np.e*binom.cdf(np.ceil(n*l_hat),n,alpha)
	return np.min(np.array([t1,t2]))

def BH(p_vals,delta):
	id_sort=np.argsort(p_vals)
	p_vals_ord=p_vals[id_sort]
	th=delta*((np.arange(0,len(p_vals_ord))+1))/len(p_vals)
	id_rej = np.where(p_vals_ord < th)
	return id_sort[id_rej]

def fixed_sequence_testing(p_vals,delta,starts):
	delta_p=delta/len(starts)
	ids_rej=[]
	for start in starts:
		for i in range(len(p_vals)):
			id_te=int((start+i)%len(p_vals))
			if p_vals[id_te]<delta_p and id_te not in ids_rej:
				ids_rej.append(id_te)
			if p_vals[id_te]>delta_p:
				break
	return ids_rej

def kl(a,b):
	'''Binary KL divergence'''
	return a*np.log(a/b)+(1.-a)*np.log((1.-a)/(1.-b))

''' Betting Policies'''

def aGRAPA(log,alpha,c_min=0.1,c_max=None):
	if c_max is None:
		c_max=1./(1-alpha)
	if(len(log)==0):
		bet=1.
	else:
		mean = (np.sum(log)) / (len(log))
		var = (np.sum((np.asarray(log) - mean) ** 2)) / (len(log))
		bet=np.min([np.max([c_min,(alpha-mean)/(var+(mean-alpha)**2)]),c_max])
	return bet

def LBOW(log,alpha,c_min=0.1,c_max=None):
	if c_max is None:
		c_max=1./(1-alpha)
	if(len(log)==0):
		bet=1.
	else:
		mean=(np.sum(log))/(len(log))
		var=(np.sum((np.asarray(log)-mean)**2))/(len(log))
		if alpha-mean<0:
			omega=1-alpha
		else:
			omega=alpha
		bet=np.min([np.max([c_min,(alpha-mean)/(omega*np.abs(mean-alpha)+var+(mean-alpha)**2)]),c_max])
	return bet

def ONS(log,lambda_o,alpha,c_min=0.1,c_max=None):
	if c_max is None:
		c_max=1./(1-alpha)
	if(len(log)==0):
		lambda_o=1.
	else:
		Ys=np.asarray(log)-alpha
		Zs=Ys/(1.-Ys*lambda_o)
		A=1+np.sum(Zs**2)
		lambda_o=np.min([np.max([c_min,lambda_o-2/(2+np.log(3))*Zs[-1]/A]),c_max])
	return lambda_o


'''Adaptive LTT'''
def aLTT(loss_vals,alpha,delta,n_samples,bet_policy=None,exp_policy=None,eps=0,control='FDR'):
	E_p=np.ones(len(loss_vals))*1.
	N_evals=np.zeros(len(loss_vals))
	L_hats=np.zeros(len(loss_vals))
	max_Ws=np.ones(len(loss_vals))
	Ws_t,N_t,Set_t,rej_ids=[],[],[],[]
	lambda_os=np.ones(len(loss_vals))
	LOG_l_hats=[[] for _ in range(len(loss_vals))]
	for t in range(0,n_samples):
		if t%(n_samples)==0 and t>0:
			np.set_printoptions(precision=3)
			print('Iteration: '+str(int(t))+'/'+str(n_samples))
			print('E-process: '+str(E_p))
			print('Bets: '+str(lambda_os))
			print('Emp. loss:'+str(np.asarray([np.mean(l) for l in LOG_l_hats])))
			print('N eval: '+str(N_evals))
		set_zeros=np.ones(len(E_p))
		set_zeros[rej_ids]=0
		if exp_policy=='GREEDY_E':
			if np.min(N_evals) > 0:
				if np.random.random() > eps:
					id_te = np.argmax((E_p) * set_zeros)
				else:
					id_te = np.random.choice(np.arange(len(loss_vals))[set_zeros == 1])
			else:
				id_te = t % len(loss_vals)
		elif exp_policy=='UCB_R':
			if np.min(N_evals)>1:
				p=0.1
				id_te = np.argmax((1 - L_hats / N_evals +np.sqrt(-np.log(p / 2.) / (2 * N_evals))) * set_zeros)
			else:
				id_te=t%len(loss_vals)
		elif exp_policy=='UNIFORM':
			id_te = t % len(loss_vals)
		l_hat=loss_vals[id_te][int(N_evals[int(id_te)])]
		L_hats[id_te]=L_hats[id_te]+l_hat
		N_evals[id_te] =N_evals[id_te] + 1.
		LOG_l_hats[id_te].append(l_hat)
		if bet_policy=='AGRAPA':
			bet=aGRAPA(LOG_l_hats[id_te],alpha)
			lambda_os[id_te]=bet
		elif bet_policy=='LBOW':
			bet=LBOW(LOG_l_hats[id_te],alpha)
			lambda_os[id_te]=bet
		elif bet_policy=='ONS':
			lambda_os[id_te]=ONS(LOG_l_hats[id_te],lambda_os[id_te],alpha)
			bet=lambda_os[id_te]
		elif bet_policy=='MAX':
			bet=1./alpha
		elif bet_policy=='UNIT':
			bet=1./(2*alpha)
		E_p[id_te] =E_p[id_te]*(1.+bet*(alpha-l_hat))
		Ws_t.append(E_p * 1)
		N_t.append(N_evals * 1)
		Set_t.append((1 - set_zeros) * 1)
		max_Ws[max_Ws<E_p]=E_p[max_Ws<E_p]
		if control=='FDR':
			rej_ids=BH(1./max_Ws,delta)
		elif control=='FWER':
			#rej_ids=fixed_sequence_testing(1./np.max(Ws_t,axis=0),delta,[0,3])
			rej_ids=np.where(1./max_Ws < delta/len(loss_vals))
	return np.asarray(Ws_t),np.asarray(N_t),np.asarray(Set_t)


def aLTT_multiple_risks(loss_vals,alpha,delta,n_samples,bet_policy=None,exp_policy=None,eps=0,control='FDR'):

	n_risks=len(loss_vals)
	n_h=len(loss_vals[0])
	E_p=np.ones((n_risks,n_h))*1.
	N_evals=np.zeros(n_h)
	L_hats=np.zeros((n_risks,n_h))
	max_Ws=np.ones((n_risks,n_h))
	Ws_t,N_t,Set_t,rej_ids=[],[],[],[]
	lambda_os=np.ones((len(loss_vals),len(loss_vals[0])))
	LOG_l_hats=[[[] for _ in range(n_h)] for _ in range(n_risks)]
	loss_vals=np.asarray(loss_vals)
	for t in range(0,n_samples):
		if t%2000==0 and t>0:
			np.set_printoptions(precision=3)
			print('Iteration: '+str(int(t))+'/'+str(n_samples))
			print('E-process: '+str(E_p))
			print('Bets: '+str(lambda_os))
			#print('Emp. loss:'+str(np.asarray([np.mean(l) for l in LOG_l_hats])))
			print('N eval: '+str(N_evals))
		set_zeros = np.ones(len(E_p[0]))
		set_zeros[rej_ids] = 0
		if exp_policy == 'GREEDY_E':
			if np.min(N_evals) > 0:
				if np.random.random() > eps:
					id_te = np.argmax((np.min(E_p, axis=0)) * set_zeros)
				# id_te = np.argmax(E_p[i] * set_zeros)
				else:
					id_te = np.random.choice(np.arange(n_h)[set_zeros == 1])
			else:
				id_te = t % n_h
		elif exp_policy == 'UNIFORM':
			id_te = t %  n_h
		for i in range(0,len(loss_vals)):
			l_hat=loss_vals[i,id_te][int(N_evals[int(id_te)])]
			L_hats[i,id_te]=L_hats[i,id_te]+l_hat
			LOG_l_hats[i][id_te].append(l_hat)
			if bet_policy=='AGRAPA':
				bet=aGRAPA(LOG_l_hats[i][id_te],alpha[i])
				lambda_os[i,id_te]=bet
			elif bet_policy=='LBOW':
				bet=LBOW(LOG_l_hats[i][id_te],alpha[i])
				lambda_os[i,id_te]=bet
			elif bet_policy=='ONS':
				lambda_os[i,id_te]=ONS(LOG_l_hats[i][id_te],lambda_os[i,id_te],alpha[i])
				bet=lambda_os[i,id_te]
			elif bet_policy=='MAX':
				bet=1./alpha
			elif bet_policy=='UNIT':
				bet=1./(2*alpha)
			E_p[i,id_te] =E_p[i,id_te]*(1.+bet*(alpha[i]-l_hat))
		N_evals[id_te] = N_evals[id_te] + 1.
		Ws_t.append(E_p * 1)
		N_t.append(N_evals * 1)
		Set_t.append((1 - set_zeros) * 1)
		max_Ws[max_Ws<E_p]=E_p[max_Ws<E_p]
		if control=='FDR':
			rej_ids=BH(1./np.min(max_Ws,axis=0),delta)
		elif control=='FWER':
			#rej_ids=fixed_sequence_testing(1./np.max(Ws_t,axis=0),delta,[0,3])
			rej_ids=np.where(1./np.min(max_Ws,axis=0) < delta/n_h)
	return np.asarray(Ws_t),np.asarray(N_t),np.asarray(Set_t)
