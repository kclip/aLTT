

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        width=20
        if orig_handle==0:
            l1 = plt.Line2D([x0,y0+width], [-0.25*height,-0.25*height],
                                                    marker='s',markersize=0, color='tab:green')
            l2 = plt.Line2D([x0,y0+width], [0.25*height,0.25*height],marker='^',markersize=0, color='mediumaquamarine')
            l3 = plt.Line2D([x0, y0 + width], [0.75 * height, 0.75 * height], marker='v',markersize=0, color='skyblue')
            l4 = plt.Line2D([x0, y0 + width], [ 1.25*height,  1.25*height], marker='o',markersize=0, color='tab:blue')
            out=[l1, l2,l3,l4]
        elif orig_handle==1:
            l1=plt.Line2D([x0,y0+width], [0.5*height,0.5*height],linestyle='-.', color='tab:grey')
            out=[l1]
        elif orig_handle == 2:
            l1 = plt.Line2D([x0, y0 + width], [0.5 * height, 0.5 * height], linestyle=':', color='k')
            out = [l1]
        return out

plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({
    'font.size': 15,
    'text.usetex': True,
})
with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
	res = pickle.load(fp)
loss_vals=res['loss_vals'][1:]
n_cal=50000
loss_vals=[l[n_cal:] for l in loss_vals]
alpha=0.57
'''Use all data to estimate reliable models'''
mean_loss=np.mean(loss_vals,axis=1)
H_1 = mean_loss < alpha
H_0 = mean_loss >= alpha
colors = ['tab:green', 'mediumaquamarine','skyblue','tab:blue','tab:grey', 'k']
LABELS = ['LBOW', 'ONS', 'AGRAPA', 'Max Bet', 'Unit Bet']
MARKERS = ['s', '^', 'v','o']
plt.rcParams['figure.figsize'] = [8/1.1, 5/1.1]
fig, axs = plt.subplots(1,2)
delta = 0.1
T = 4999
k=0
for CONTROL in ['FWER','FDR']:
    i=0
    exp_pol = 'GREEDY_E'
    bet_pol = 'AGRAPA'
    '''EPSILON-GREEDY'''
    for eps in [0.25,0.5,0.75,.95]:
        file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_' + str(alpha) + '_' + CONTROL + '.pkl'
        with  open('logs/results/'+file_name, 'rb') as fp:
            res = pickle.load(fp)
        TR_D = np.sum(np.asarray(res['SET_T'])*H_1,axis=2)
        TPR= np.mean(TR_D/np.sum(H_1),axis=0)
        TPR = TPR[:T+1]
        FA_D = np.sum(np.asarray(res['SET_T'])*H_0,axis=2)
        FDR=FA_D/np.sum(np.asarray(res['SET_T']),axis=2)
        FDR[np.isnan(FDR)] = 0
        FDR=np.mean(FDR,axis=0)
        FWER=FA_D>0
        FWER=np.mean(FWER,axis=0)
        axs[k].plot(TPR, color=colors[i],marker=MARKERS[i],markersize=0,markevery=399,  label=r'aLTT $\epsilon$-Greedy $\epsilon$='+str(eps))
        i=i+1
    if CONTROL=='FDR':
        axs[k].text(2000, 0.63, r'$\epsilon$=0.25', color='tab:green', rotation=55)
        axs[k].text(2500, 0.6, r'$\epsilon$=0.5', color='mediumaquamarine', rotation=53)
        axs[k].text(3000, 0.46, r'$\epsilon$=0.75', color='skyblue', rotation=50)
        axs[k].text(3599, 0.25, r'$\epsilon$=0.95', color='tab:blue', rotation=52)
    elif CONTROL=='FWER':
        axs[k].text(2000, 0.52, r'$\epsilon$=0.25', color='tab:green', rotation=45)
        axs[k].text(2500, 0.48, r'$\epsilon$=0.5', color='mediumaquamarine', rotation=40)
        axs[k].text(3000, 0.39, r'$\epsilon$=0.75', color='skyblue', rotation=30)
        axs[k].text(3599, 0.18, r'$\epsilon$=0.95', color='tab:blue', rotation=30)
    '''NON ADAPTIVE'''
    exp_pol = 'UNIFORM'
    eps=0
    file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_' + str(alpha) + '_' + CONTROL + '.pkl'
    with  open('logs/results/'+file_name, 'rb') as fp:
        res = pickle.load(fp)
    TR_D = np.sum(np.asarray(res['SET_T'])*H_1,axis=2)
    TPR= np.mean(TR_D/np.sum(H_1),axis=0)
    TPR = TPR[:T + 1]
    FA_D = np.sum(np.asarray(res['SET_T'])*H_0,axis=2)
    FDR=FA_D/np.sum(np.asarray(res['SET_T']),axis=2)
    FDR[np.isnan(FDR)] = 0
    FDR=np.mean(FDR,axis=0)
    FWER=FA_D>0
    FWER=np.mean(FWER,axis=0)
    axs[k].plot(TPR, color=colors[i],  label=r'aLTT w/ non adap. acq.',linestyle='-.')
    i=i+1

    '''LTT'''
    TPR[0:T]=0
    TPR = TPR[:T + 1]
    axs[k].scatter(T,TPR[T],color=colors[i])
    axs[k].plot(TPR, color=colors[i],  label=r'LTT',linestyle=':')
    if CONTROL=='FWER':
        axs[k].legend([0,1,2], [r'aLTT, $\epsilon$-greedy acq.','aLTT, non-adap. acq.','LTT'],
                   handler_map={object: AnyObjectHandler()},handletextpad=-0.1,borderaxespad=0.1)
    axs[k].set_xlim([0,T+100])
    axs[k].set_ylim([-0.02,1.05])
    axs[k].set_xlabel('Round ($t$)')
    if CONTROL=='FWER':
        axs[k].set_ylabel('True Positive Rate')
        axs[k].set_title('FWER Control')
    else:
        axs[k].set_title('FDR Control')
    axs[k].grid()
    k=k+1
plt.tight_layout()
plt.savefig('logs/images/Fig1_eps.pdf')
plt.show()


