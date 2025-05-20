import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.rcParams['figure.figsize'] = [5, 3.75]
alpha=0.57
with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
	res = pickle.load(fp)
loss_vals=res['loss_vals'][1:]
n_cal=50000
loss_vals=[l[n_cal:] for l in loss_vals]
mean_loss=np.mean(loss_vals,axis=1)
N=len(loss_vals[0])
H_1 = mean_loss < alpha
H_0 = mean_loss >= alpha
delta=0.1
plt.rcParams['figure.figsize'] = [8/1.1, 5/1.1]
fig, axs = plt.subplots(1,2)
colors = ['tab:blue', 'tab:grey', 'tab:orange', 'tab:green', 'k']
LABELS = ['aGRAPA','LBOW','ONS','Max Bet','Unit Bet']
MARKERS=['s','^','v','o','+','x']
LS=['-','--','-.','-','-']
i=0
exp_pol='GREEDY_E'
T=4999
eps=0.25
k=0
for CONTROL  in ['FWER','FDR']:
    i = 0
    for bet_pol in ['AGRAPA','LBOW','ONS','MAX','UNIT']:
        file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_' + str(alpha) + '_' + CONTROL + '.pkl'
        with  open('logs/results/' + file_name, 'rb') as fp:
            res = pickle.load(fp)
        TR_D = np.sum(np.asarray(res['SET_T']) * H_1, axis=2)
        TPR = np.mean(TR_D / np.sum(H_1), axis=0)
        TPR = TPR[:T + 1]
        FA_D = np.sum(np.asarray(res['SET_T']) * H_0, axis=2)
        FDR = FA_D / np.sum(np.asarray(res['SET_T']), axis=2)
        FDR[np.isnan(FDR)] = 0
        FDR = np.mean(FDR, axis=0)
        FWER = FA_D > 0
        FWER = np.mean(FWER, axis=0)
        axs[k].plot(TPR, color=colors[i], marker=MARKERS[i], markevery=399, label=LABELS[i])
        i=i+1
    axs[k].legend()
    axs[k].grid()
    axs[k].set_xlim([0,4000])
    axs[k].set_ylim([-0.05,1.05])
    axs[k].set_xlabel('Round ($t$)')
    axs[k].set_ylabel('True Positive Rate')
    if CONTROL == 'FWER':
        axs[k].set_ylabel('True Positive Rate')
        axs[k].set_title('FWER Control')
    else:
        axs[k].set_title('FDR Control')
    k = k + 1
plt.tight_layout()
plt.savefig('logs/images/Fig3_betting.pdf')
plt.show()
