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
EE_max=1
alpha=1
delta=0.1
colors = ['tab:green', 'mediumaquamarine','tab:blue', 'k','tab:grey']
LABELS = ['aGRAPA ABet','LBOW Bet','ONS Bet','Max Bet','Unit Bet']
MARKERS=['s','^','v','o','+','x']
LS=['-','--','-.',':','-']

T=int(999/25)
fig, axs = plt.subplots(1,2)

k=0
TARGET_EEs=np.arange(0.025,1,0.05)
for CONTROL in ['FWER','FDR']:
    i = 0
    exp_pol = 'GREEDY_E'
    bet_pol = 'AGRAPA'
    for eps in [0.25,0.75,0.95]:
        BEST_Q_OCC, BEST_Q_OCC_STD = [], []
        for target_EE in TARGET_EEs:
            with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
                res = pickle.load(fp)
            target_EE = -target_EE + EE_max
            q_occ = np.mean(res['delays'], axis=2)
            mean_q_occ = np.mean(q_occ, axis=1)
            file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_ee_' + str(target_EE) + '_' + CONTROL + '.pkl'
            with  open('logs/results/' + file_name, 'rb') as fp:
                res = pickle.load(fp)
            accept = [np.where(s[T] > 0)[0] for s in res['SET_T']]
            nt = [s[T] for s in res['N_T']]
            id = np.zeros(len(q_occ)).astype(int)
            q_occ_acc = []
            best_acc = []
            for acc,ns in zip(accept,nt):
                if len(acc) > 0:
                    best_acc.append(acc[np.argmin(np.asarray([np.mean(q_occ[a, id[a]:id[a] + ns[a]], axis=0) for a in acc]))])
                id = id + ns
            if len(best_acc) == 0:
                BEST_Q_OCC.append(np.nan)
            else:
                BEST_Q_OCC.append(np.mean(mean_q_occ[best_acc]))
            BEST_Q_OCC_STD.append(np.std(mean_q_occ[best_acc])/np.sqrt(len(best_acc)))
        axs[k].plot(TARGET_EEs, BEST_Q_OCC,color=colors[i], marker=MARKERS[i],label='aLTT')
        i=i+1


    exp_pol='UNIFORM'
    eps=0
    BEST_Q_OCC=[]
    for target_EE in TARGET_EEs:
        with open('logs/data/collected_data_aggregate.pkl', 'rb') as fp:
            res = pickle.load(fp)
        target_EE = -target_EE + EE_max
        q_occ = np.mean(res['delays'], axis=2)
        mean_q_occ = np.mean(q_occ, axis=1)
        file_name = 'aLTT_' + bet_pol + '_' + exp_pol + '_eps_' + str(eps) + '_delta_' + str(delta) + '_alpha_ee_' + str(target_EE) + '_' + CONTROL + '.pkl'
        with  open('logs/results/' + file_name, 'rb') as fp:
            res = pickle.load(fp)
        accept = [np.where(s[T] > 0)[0] for s in res['SET_T']]
        nt = [s[T] for s in res['N_T']]
        id = np.zeros(len(q_occ)).astype(int)
        q_occ_acc = []
        best_acc=[]
        for acc, ns in zip(accept, nt):
            if len(acc)>0:
                best_acc.append(acc[np.argmin(np.asarray([np.mean(q_occ[a, id[a]:id[a] + ns[a]], axis=0) for a in acc]))])
            id = id + ns
        if len(best_acc)==0:
            BEST_Q_OCC.append(np.nan)
        else:
            BEST_Q_OCC.append(np.mean(mean_q_occ[best_acc]))
    axs[k].plot(TARGET_EEs, BEST_Q_OCC,color=colors[i], marker=MARKERS[i],linestyle='-.',label='LTT')
    axs[k].set_xlabel(r'Energy Eff. Req. [Mbit/Joule]')
    if CONTROL == 'FWER':
        axs[k].set_title('FWER Control')
        axs[k].set_ylabel(r'Average Delay [ms]')
    if CONTROL=='FDR':
        axs[k].set_title('FDR Control')
        axs[k].legend([0,1], [r'aLTT w/ $\epsilon$-greedy acq.','LTT'],
                       handler_map={object: AnyObjectHandler()},loc='upper left',handletextpad=-0.1,borderaxespad=0.1)

    if CONTROL=='FDR':
        axs[k].annotate(r'$\epsilon$=0.25', color='tab:green', xy=(0.44, 26.8), xytext=(0.47, 25), arrowprops=dict(arrowstyle='-|>', fc='tab:green', ec='tab:green', lw=1.5),
                        bbox=dict(facecolor="none", edgecolor="none"), )
        axs[k].annotate(r'$\epsilon$=0.75', color='mediumaquamarine', xy=(0.61, 28.5), xytext=(0.4, 31), arrowprops=dict(arrowstyle='-|>', fc='mediumaquamarine', ec='mediumaquamarine', lw=1.5),
                        bbox=dict(facecolor="none", edgecolor="none"), )
        axs[k].annotate(r'$\epsilon$=0.95', color='tab:blue', xy=(0.88, 31), xytext=(0.6, 33), arrowprops=dict(arrowstyle='-|>', fc='tab:blue', ec='tab:blue', lw=1.5),
                        bbox=dict(facecolor="none", edgecolor="none"), )
    if CONTROL == 'FWER':
        axs[k].annotate( r'$\epsilon$=0.25',color='tab:green', xy=(0.44, 26.8), xytext=(0.47, 25),arrowprops=dict(arrowstyle='-|>', fc='tab:green', ec='tab:green', lw=1.5),
                bbox=dict(facecolor="none", edgecolor="none"),)
        axs[k].annotate(r'$\epsilon$=0.75', color='mediumaquamarine', xy=(0.61, 29), xytext=(0.4, 31), arrowprops=dict(arrowstyle='-|>', fc='mediumaquamarine', ec='mediumaquamarine', lw=1.5),
                        bbox=dict(facecolor="none", edgecolor="none"), )
        axs[k].annotate(r'$\epsilon$=0.95', color='tab:blue', xy=(0.88, 31.45), xytext=(0.6,33), arrowprops=dict(arrowstyle='-|>', fc='tab:blue', ec='tab:blue', lw=1.5),
                        bbox=dict(facecolor="none", edgecolor="none"), )
    axs[k].set_xlim([0.05,0.95])
    axs[k].set_ylim([22.5, 38])
    axs[k].grid()
    k=k+1
plt.tight_layout()
plt.savefig('logs/images/EEvsDelay.pdf')
plt.show()

