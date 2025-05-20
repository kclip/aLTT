import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        width = 20
        if orig_handle==0:
            l1 = plt.Line2D([x0,y0+width], [-0.25*height,-0.25*height],
                                                    marker='s',markersize=0, color='tab:green')
            l2 = plt.Line2D([x0,y0+width], [0.25*height,0.25*height],marker='^',markersize=0, color='mediumaquamarine')
            l3 = plt.Line2D([x0, y0 + width], [0.75 * height, 0.75 * height], marker='v',markersize=0, color='skyblue')
            l4 = plt.Line2D([x0, y0 + width], [ 1.25*height,  1.25*height], marker='o',markersize=0, color='tab:blue')
            out=[l1, l2,l3,l4]
        elif orig_handle==1:
            l1=plt.Line2D([x0,y0+width], [0.5*height,0.5*height],linestyle='-.', color='k')
            out=[l1]
        return out

''' Plotting Parameters '''
plt.rc('font', family='serif', serif='Computer Modern Roman')
plt.rcParams.update({
    'font.size': 15,
    'text.usetex': True,
})

plt.rcParams['figure.figsize'] = [8/1.1, 5/1.1]
EE_max=5
alpha=1
delta=0.1
colors = ['tab:green', 'mediumaquamarine','tab:blue', 'k','tab:grey']
LABELS = ['aGRAPA ABet','LBOW Bet','ONS Bet','Max Bet','Unit Bet']
MARKERS=['s','^','v','o','+','x']
LS=['-','--','-.',':','-']
i=0
exp_pol='GREEDY_E'
bet_pol='AGRAPA'
T=int(3999/25)
eps=0.25
CONTROL='FWER'
TARGET_Qs=np.arange(0.10,0.3,0.01)
URLLC=np.arange(16,32)
DATA_TR=np.arange(0,16)
fig, axs = plt.subplots(1,2)
k=0
for CONTROL in ['FWER','FDR']:
    i=0
    exp_pol = 'GREEDY_E'
    bet_pol = 'AGRAPA'
    for eps in [0.25,0.5,0.95]:
        BEST_EE_LAT, BEST_EE_LAT_STD = [], []
        for target_Q in TARGET_Qs:
            EE_max = 1
            target_ee_urllc = 0.01
            target_ee_urllc = -target_ee_urllc + EE_max
            with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
                res = pickle.load(fp)
            ee_bf =  np.mean(res['avg_tx_powers'][:, :, DATA_TR], axis=2)*0.01
            ee_bf[np.isnan(ee_bf)] = 0
            delay_bf=  np.mean(res['delays'][:, :, DATA_TR], axis=2)*0.01
            mean_ee = np.mean(ee_bf, axis=1)
            mean_delay=np.mean(delay_bf,axis=1)
            mean_target = mean_delay*mean_ee
            file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_q_occ_' + str(target_Q) + '_alpha_ee_' + str(target_ee_urllc) + '_' + CONTROL + '.pkl'
            with  open('logs/results/' + file_name, 'rb') as fp:
                res = pickle.load(fp)
            sets=[s[T] for s in res['SET_T']]
            id = np.zeros(len(ee_bf)).astype(int)
            ns = [s[T] for s in res['N_T']]
            ws = [np.min(s[T],axis=0) for s in res['WS_T']]
            best_pol=[]
            for n,set,w in zip(ns,sets,ws):
                trgt_eval = np.asarray([np.mean(delay_bf[i, id[i]:id[i] + n[i]], axis=0)*np.mean(ee_bf[i, id[i]:id[i] + n[i]], axis=0) for i in range(0,len(mean_ee))])* 1/set
                if len(np.where(set>0)[0])>0:
                    best_pol.append(np.argmin(trgt_eval))
                id = id + n
            if len(best_pol) == 0:
                BEST_EE_LAT.append(np.nan)
            else:
                BEST_EE_LAT.append(np.mean(mean_target[best_pol]))

            BEST_EE_LAT_STD.append(np.std(mean_target[best_pol])/np.sqrt(len(best_pol)))
        axs[k].plot(TARGET_Qs, BEST_EE_LAT,color=colors[i], marker=MARKERS[i],label='aLTT')
        i=i+1
    exp_pol='UNIFORM'
    eps=0
    BEST_EE_LAT, BEST_EE_LAT_STD = [], []
    for target_Q in TARGET_Qs:
        target_ee_urllc = 0.01
        target_ee_urllc = -target_ee_urllc + EE_max
        with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
            res = pickle.load(fp)
        ee_bf = np.mean(res['avg_tx_powers'][:, :, DATA_TR], axis=2)*0.01 #mJoule
        ee_bf[np.isnan(ee_bf)] = 0
        delay_bf = np.mean(res['delays'][:, :, DATA_TR], axis=2)*0.01 #ms
        mean_ee = np.mean(ee_bf, axis=1)
        mean_delay = np.mean(delay_bf, axis=1)
        mean_target = mean_delay * mean_ee
        file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_q_occ_' + str(target_Q) + '_alpha_ee_' + str(target_ee_urllc) + '_' + CONTROL + '.pkl'
        with  open('logs/results/' + file_name, 'rb') as fp:
            res = pickle.load(fp)
        sets = [s[T] for s in res['SET_T']]
        id = np.zeros(len(ee_bf)).astype(int)
        ns = [s[T] for s in res['N_T']]
        ws = [np.min(s[T], axis=0) for s in res['WS_T']]
        best_pol = []
        for n, set, w in zip(ns, sets, ws):
            trgt_eval = np.asarray([np.mean(delay_bf[i, id[i]:id[i] + n[i]], axis=0) * np.mean(ee_bf[i, id[i]:id[i] + n[i]], axis=0) for i in range(0, len(mean_ee))]) * 1 / set
            if len(np.where(set > 0)[0]) > 0:
                best_pol.append(np.argmin(trgt_eval))
            id = id + n
        if len(best_pol) == 0:
            BEST_EE_LAT.append(np.nan)
        else:
            BEST_EE_LAT.append(np.mean(mean_target[best_pol]))
        BEST_EE_LAT_STD.append(np.std(mean_target[best_pol]) / np.sqrt(len(best_pol)))

    axs[k].plot(TARGET_Qs, BEST_EE_LAT,color=colors[i], marker=MARKERS[i],linestyle='-.',label='LTT')
    axs[k].set_xlabel(r'Avg. Queue Occupancy')
    axs[k].set_ylim([-0.01*1e-5,1.3*1e-5])
    axs[k].set_xlim([0.15, 0.295])
    if CONTROL=='FDR':
        axs[k].set_title('FDR Control')
        axs[k].legend([0,1], [r'aLTT, $\epsilon$-greedy acq.','LTT'],
                       handler_map={object: AnyObjectHandler()},loc='upper right',handletextpad=-0.1,borderaxespad=0.1)
        axs[k].yaxis.set_tick_params(labelleft=False)
    if CONTROL=='FWER':
        axs[k].set_title('FWER Control')
        axs[k].set_ylabel(r'Energy-Delay Product [mJ$\times$ ms]')

    if CONTROL == 'FDR':
        axs[k].text(0.148, 0, r'$\epsilon$=0.25', color='tab:green', rotation=-60)
        axs[k].text(0.165, 1.e-6, r'$\epsilon$=0.75', color='mediumaquamarine', rotation=-50)
        axs[k].text(0.19, 3.89e-6, r'$\epsilon$=0.95', color='tab:blue', rotation=-70)
    if CONTROL == 'FWER':
        axs[k].text(0.15, 0.3e-6, r'$\epsilon$=0.25', color='tab:green', rotation=-60)
        axs[k].text(0.17, 1.5e-6, r'$\epsilon$=0.75', color='mediumaquamarine', rotation=-60)
        axs[k].text(0.19, 3.89e-6, r'$\epsilon$=0.95', color='tab:blue', rotation=-70)

    axs[k].grid()
    k=k+1
plt.tight_layout()
plt.savefig('logs/images/EEvsEDP.pdf')
plt.show()

