import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse


def change_to_quan(score_list, q=0.0):
    mean_vec = np.zeros(len(score_list))
    for i in range(len(score_list)):
        mean_vec[i] = np.quantile(np.array(score_list[i]),q)
    return mean_vec

def change_to_mean(score_list):
    mean_vec = np.zeros(len(score_list))
    for i in range(len(score_list)):
        mean_vec[i] = np.mean(np.array(score_list[i]))
    return mean_vec

def get_stat_from_dict_list(test_dict, mode, name):
    info = test_dict[mode][name]
    print('info', info, info.shape)
    return np.mean(info, axis=0), 1.96*np.std(info, axis=0)/np.sqrt(info.shape[0])

def plot(test_dict, task, risk_control_mode):
    #print(test_dict)
    plt.rcParams["font.family"] = "Times New Roman"
    #fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(3,12))
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(3,3))
    colors = ['#529F3F', '#80CBAC', '#97CCE8', '#3B75AF', 'k', 'k', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
    lines = ['-','-', '-','-', ':', '-.','-','-','--', '-', '--', '-', '--']
    markers = ['.','.', '.', '.', '.', '.', '.', '.', 'o','.','.', 'o', '^', 'v', '.', 'v', '^', 'o']
    ind_mode = 0
    #print('test_dict', test_dict)
    for mode in test_dict.keys():
        if mode == '': #mode == 'e_APE_epsilon_greedy_0.75' or mode == 'e_APE_epsilon_greedy_0.5':
            pass
        else:
            if mode == 'p_APE':
                mode_name = 'LTT'
            elif mode == 'n_e_APE':
                mode_name = 'aLTT, non-adap. acq.'
            else:
                mode_name = 'aLTT, e-greedy acq.'
            ind_cost = 0
            # axs[0].plot(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'fdp')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
            # axs[0].fill_between(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'fdp')[0]-get_stat_from_dict_list(test_dict, mode, 'fdp')[1], get_stat_from_dict_list(test_dict, mode, 'fdp')[0]+get_stat_from_dict_list(test_dict, mode, 'fdp')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])

            # axs[1].plot(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'fwer')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
            # axs[1].fill_between(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'fwer')[0]-get_stat_from_dict_list(test_dict, mode, 'fwer')[1], get_stat_from_dict_list(test_dict, mode, 'fwer')[0]+get_stat_from_dict_list(test_dict, mode, 'fwer')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])


            # axs[2].plot(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'power')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
            # axs[2].fill_between(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'power')[0]-get_stat_from_dict_list(test_dict, mode, 'power')[1], get_stat_from_dict_list(test_dict, mode, 'power')[0]+get_stat_from_dict_list(test_dict, mode, 'power')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])

            axs.plot(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'power')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
            axs.fill_between(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'power')[0]-get_stat_from_dict_list(test_dict, mode, 'power')[1], get_stat_from_dict_list(test_dict, mode, 'power')[0]+get_stat_from_dict_list(test_dict, mode, 'power')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])



            # axs[3].plot(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'min_len')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode],label=mode_name)
            # axs[3].fill_between(get_stat_from_dict_list(test_dict, mode, 'cost')[0], get_stat_from_dict_list(test_dict, mode, 'min_len')[0]-get_stat_from_dict_list(test_dict, mode, 'min_len')[1], get_stat_from_dict_list(test_dict, mode, 'min_len')[0]+get_stat_from_dict_list(test_dict, mode, 'min_len')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])


            # axs[0].plot(test_dict[mode]['cost'], test_dict[mode]['fdp'], color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode],label=mode)
            # axs[1].plot(test_dict[mode]['cost'], test_dict[mode]['power'], color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode],label=mode)
            # axs[2].plot(test_dict[mode]['cost'], change_to_mean(test_dict[mode]['score']), color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode], alpha=1.0, label=mode)
            ind_mode += 1
    
    # axs[0].legend()
    axs.set_xscale('log')
    # axs.set_yscale('log')
    # axs[0].grid()
    # axs[1].set_xscale('log')
    # axs[1].grid()
    # axs[2].set_xscale('log')
    # axs[2].grid()
    axs.set_ylim(bottom=0.0, top=1.02)
    # axs[1].set_ylim(bottom=0.0, top=0.2)
    # axs[2].set_ylim(bottom=0.00, top=1.02)
    xmin = get_stat_from_dict_list(test_dict, mode, 'cost')[0][0]
    xmax = get_stat_from_dict_list(test_dict, mode, 'cost')[0][-1]

    axs.get_xaxis().set_ticks([])
    axs.get_yaxis().set_ticks([])

    # axs[0].hlines(0.1, xmin, xmax, color='y', linestyle=':')
    # axs[0].set_xlim(left=xmin, right=xmax)
    # axs[1].hlines(0.1, xmin, xmax, color='y', linestyle=':')
    # axs[1].set_xlim(left=xmin, right=xmax)
    #axs[1].set_xlim(left=1000, right=100000)
    #axs[1].set_ylim(bottom=0.0, top=0.8)
    #axs[1].legend()
    plt.tight_layout() 
    path = Path('./vis_results/ICML_LLM/fig1/appendix/all_tasks/' + str(risk_control_mode) +'/' + str(alpha)+ '/' + task + '.png')
    path.parent.mkdir(parents=True, exist_ok=True) 
    plt.savefig(path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='e_prompting')
    parser.add_argument('--task', type=str, default='antonyms', help='...')
    parser.add_argument('--risk_control_mode', type=str, default='FDR', help='...')
    parser.add_argument('--alpha', type=float, default=0.2, help='...')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #fire.Fire(run)
    args = parse_args()


    total_tasks = [ 'diff', 'first_word_letter',
            'larger_animal', 'letters_list', 'taxonomy_animal', 'num_to_verbal',
            'rhymes',
            'sum', 'translation_en-de', 'translation_en-es']
    


    for task in total_tasks:
        test_dict_per_alpha = {}
        alpha = args.alpha
        #alpha = 0.3
        delta = 0.1
        #task = 'word_in_context'
        save_dir_name = f'experiments/ICML_real_final/all_tasks/for_plotting/instruction_induction/{task}/{args.risk_control_mode}/'
        #test_dict = np.load(save_dir_name+'test_scores.npy', allow_pickle=True)
        with open(save_dir_name + str(alpha)+str(delta) + 'per_task_per_budget.pkl', 'rb') as f:
            test_dict = pickle.load(f)
        for keys in test_dict.keys():
            print('test_dict', np.mean(test_dict[keys]['fwer'], axis=0))
        print('test_dict', test_dict)
        plot(test_dict, task, args.risk_control_mode)


