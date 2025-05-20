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
    # print('info', np.std(info, axis=0))
    # return np.nanmean(info, axis=0), 1.96*np.nanstd(info, axis=0)/np.sqrt(info.shape[0])
    # return np.nanmean(info, axis=0), 1.96*np.nanstd(info, axis=0)/np.sqrt(info.shape[0])
    # print(np.mean(info, axis=0), 1.96*np.std(info, axis=0)/np.sqrt(info.shape[0]))
    return np.mean(info, axis=0), 1.96*np.std(info, axis=0)/np.sqrt(info.shape[0])

def plot(test_dict, task, risk_control_mode):
    #print(test_dict)
    plt.rcParams["font.family"] = "Times New Roman"
    #fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(3,12))
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(2.5,3))
    colors = ['#529F3F', '#80CBAC', '#97CCE8', '#3B75AF', 'k', 'k', '#009051', 'k', '#A62B17', '#A62B17', '#3274B5', '#3274B5', '#EF5FA7','#EF5FA7']
    lines = ['-','-', '-','-', ':', '-.','-','-','--', '-', '--', '-', '--']
    markers = ['.','.', '.', '.', '.', '.', '.', '.', 'o','.','.', 'o', '^', 'v', '.', 'v', '^', 'o']
    ind_mode = 0
    
    #print('test_dict', test_dict)
    for mode in test_dict.keys():
        print('mode', mode)
        if mode == 'p_APE':
            mode_name = 'LTT'
        elif mode == 'n_e_APE':
            mode_name = 'aLTT, non-adap. acq.'
        else:
            mode_name = 'aLTT, e-greedy acq.'
        ind_cost = 0
        # axs[0].plot(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'fdp')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
        # axs[0].fill_between(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'fdp')[0]-get_stat_from_dict_list(test_dict, mode, 'fdp')[1], get_stat_from_dict_list(test_dict, mode, 'fdp')[0]+get_stat_from_dict_list(test_dict, mode, 'fdp')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])

        # axs[1].plot(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'fwer')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
        # axs[1].fill_between(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'fwer')[0]-get_stat_from_dict_list(test_dict, mode, 'fwer')[1], get_stat_from_dict_list(test_dict, mode, 'fwer')[0]+get_stat_from_dict_list(test_dict, mode, 'fwer')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])


        # axs[2].plot(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'power')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
        # axs[2].fill_between(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'power')[0]-get_stat_from_dict_list(test_dict, mode, 'power')[1], get_stat_from_dict_list(test_dict, mode, 'power')[0]+get_stat_from_dict_list(test_dict, mode, 'power')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])
        print(mode, get_stat_from_dict_list(test_dict, mode, 'min_len')[0], get_stat_from_dict_list(test_dict, mode, 'min_len')[1])
        axs.plot(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'min_len')[0], color=colors[ind_mode], marker=markers[ind_mode], markersize=0, linestyle=lines[ind_mode],label=mode_name)
        axs.fill_between(get_stat_from_dict_list(test_dict, mode, 'alpha')[0], get_stat_from_dict_list(test_dict, mode, 'min_len')[0]-get_stat_from_dict_list(test_dict, mode, 'min_len')[1], get_stat_from_dict_list(test_dict, mode, 'min_len')[0]+get_stat_from_dict_list(test_dict, mode, 'min_len')[1], alpha=0.2, facecolor=colors[ind_mode],linestyle=lines[ind_mode])
        # axs[0].plot(test_dict[mode]['alpha'], test_dict[mode]['fdp'], color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode],label=mode)
        # axs[1].plot(test_dict[mode]['alpha'], test_dict[mode]['power'], color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode],label=mode)
        # axs[2].plot(test_dict[mode]['alpha'], change_to_mean(test_dict[mode]['score']), color=colors[ind_mode], marker=markers[ind_mode], markersize=5, linestyle=lines[ind_mode], alpha=1.0, label=mode)
        ind_mode += 1

    # axs[0].legend()
    # axs.set_xscale('log')
    axs.grid()
    # 'larger_animal', 'diff', 'first_word_letter', 'letters_list', 'taxonomy_animal', 'num_to_verbal',
    #     'active_to_passive',  'rhymes', 'sentiment', 
    #     'sum',  'translation_en-es', 'singular_to_plural'
    if task == 'larger_animal':
        axs.set_ylim(bottom=24.5, top=42.5)
    elif task == 'letters_list':
        axs.set_ylim(bottom=21.5, top=29)
    elif task == 'num_to_verbal':
        axs.set_ylim(bottom=24.5, top=45)
    elif task == 'rhymes':
        axs.set_ylim(bottom=15, top=35)
    elif task == 'singular_to_plural':
        axs.set_ylim(bottom=40, top=70)
    elif task == 'taxonomy_animal':
        axs.set_ylim(bottom=28, top=42)
    elif task == 'translation_en-es':
        axs.set_ylim(bottom=23.5, top=36)
    else:
        pass
    #axs.set_ylim(bottom=0.0, top=1.02)
    # axs[1].set_xscale('log')
    # axs[1].grid()
    # axs[2].set_xscale('log')
    # axs[2].grid()
    # axs[0].set_ylim(bottom=0.0, top=0.2)
    # axs[1].set_ylim(bottom=0.0, top=0.2)
    # axs[2].set_ylim(bottom=0.00, top=1.02)
    #axs[3].set_ylim(bottom=29, top=34.5)
    #xmin = get_stat_from_dict_list(test_dict, mode, 'cost')[0][0]
    #xmax = get_stat_from_dict_list(test_dict, mode, 'cost')[0][-1]
    # xmin = get_stat_from_dict_list(test_dict, mode, 'alpha')[0][0]
    # xmax = get_stat_from_dict_list(test_dict, mode, 'alpha')[0][-1]
    # axs[0].hlines(0.1, xmin, xmax, color='y', linestyle=':')
    # axs[0].set_xlim(left=xmin, right=xmax)
    # axs[1].hlines(0.1, xmin, xmax, color='y', linestyle=':')
    # axs[1].set_xlim(left=xmin, right=xmax)
    #axs[1].set_xlim(left=1000, right=100000)
    #axs[1].set_ylim(bottom=0.0, top=0.8)
    #axs[1].legend()
    plt.tight_layout() 
    path = Path('./vis_results/ICML_LLM/fig2/appendix/task/per_alpha_' + str(risk_control_mode) +'/' + str(alpha)+str(delta)+ task + '.png')
    path.parent.mkdir(parents=True, exist_ok=True) 
    plt.savefig(path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='e_prompting')
    parser.add_argument('--task', type=str, default='antonyms', help='...')
    parser.add_argument('--risk_control_mode', type=str, default='FDR', help='...')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #fire.Fire(run)
    #sub_tasks = [ 'diff', 'antonyms', 'first_word_letter', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',  'second_word_letter', 'sentiment',   'sum', 'translation_en-de', 'translation_en-es', 'translation_en-fr', 'word_in_context']
    num_repetition = 10
    args = parse_args()
    task = args.task #'antonyms'


    total_tasks = [ 'diff', 'first_word_letter',
            'larger_animal', 'letters_list', 'taxonomy_animal', 'num_to_verbal',
            'rhymes',
            'sum', 'translation_en-de', 'translation_en-es']
    
    for task in total_tasks:
        tasks = [task]
        test_dict_per_alpha = {}
        alpha_list = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]#, 0.1, 0.05]
        ind_alpha = 0
        ind_budget = 8
        num_cand = 10

        total_budget_list = [10,20,50,100,200,500,1000,2000]
        ind_budget = 8


        for alpha in alpha_list:
            #alpha = 0.5
            delta = 0.1
            # save_dir_name = f'experiments/ICML_real_final/all_tasks/for_plotting/instruction_induction/{task}/{args.risk_control_mode}/'

            num_repetition = 10
            test_dict_per_task = {}
            for mode in ['e_APE_epsilon_greedy_0.25','e_APE_epsilon_greedy_0.5','e_APE_epsilon_greedy_0.75','e_APE_epsilon_greedy_0.95','p_APE','n_e_APE']: #, 'standard_APE', 'bandit_APE','e_APE_epsilon_greedy_0.0','e_APE_epsilon_greedy_0.1', 'e_APE_epsilon_greedy_0.25','e_APE_epsilon_greedy_0.5','e_APE_epsilon_greedy_0.75','e_APE_epsilon_greedy_0.95','e_APE_LCB']:
                test_dict_per_task[mode] = {}
                test_dict_per_task[mode]['cost'] = np.zeros([num_repetition*len(tasks),  len(total_budget_list)])
                test_dict_per_task[mode]['fdp'] = np.zeros([num_repetition*len(tasks),  len(total_budget_list)])
                test_dict_per_task[mode]['fwer'] = np.zeros([num_repetition*len(tasks), len(total_budget_list)])
                test_dict_per_task[mode]['power'] = np.zeros([num_repetition*len(tasks), len(total_budget_list)])
                test_dict_per_task[mode]['min_len'] = np.zeros([num_repetition*len(tasks),  len(total_budget_list)])
                ind_task = 0
                for task in tasks:
                    save_dir_name = f'experiments/ICML_real_final/all_tasks/for_plotting/instruction_induction/{task}/{args.risk_control_mode}/'
                    #test_dict = np.load(save_dir_name+'test_scores.npy', allow_pickle=True)
                    with open(save_dir_name + str(alpha)+str(delta) + 'per_task_per_budget.pkl', 'rb') as f:
                        test_dict = pickle.load(f)
                    test_dict_per_task[mode]['power'][ind_task*num_repetition:(ind_task+1)*num_repetition] = test_dict[mode]['power']
                    test_dict_per_task[mode]['cost'][ind_task*num_repetition:(ind_task+1)*num_repetition] = test_dict[mode]['cost']
                    test_dict_per_task[mode]['min_len'][ind_task*num_repetition:(ind_task+1)*num_repetition] = test_dict[mode]['min_len']
                    ind_task += 1



            for mode in test_dict_per_task.keys():
                if mode in test_dict_per_alpha.keys():
                    pass
                else:
                    test_dict_per_alpha[mode] = {}
                    test_dict_per_alpha[mode]['alpha'] = np.zeros([num_repetition*len(tasks), len(alpha_list)])
                    # test_dict_per_alpha[mode]['score'] = np.zeros([num_repetition, len(alpha_list)])
                    test_dict_per_alpha[mode]['fdp'] = np.zeros([num_repetition*len(tasks), len(alpha_list)])
                    test_dict_per_alpha[mode]['fwer'] = np.zeros([num_repetition*len(tasks), len(alpha_list)])
                    test_dict_per_alpha[mode]['power'] = np.zeros([num_repetition*len(tasks), len(alpha_list)])
                    test_dict_per_alpha[mode]['min_len'] = np.zeros([num_repetition*len(tasks), len(alpha_list)])

                test_dict_per_alpha[mode]['alpha'][:, ind_alpha] = alpha
                # test_dict_per_alpha[mode]['score'][:, ind_alpha] = test_dict[mode]['score'][:, 0]
                test_dict_per_alpha[mode]['fdp'][:, ind_alpha] = test_dict_per_task[mode]['fdp'][:, ind_budget]
                test_dict_per_alpha[mode]['fwer'][:, ind_alpha] = test_dict_per_task[mode]['fwer'][:, ind_budget]
                test_dict_per_alpha[mode]['power'][:, ind_alpha] = test_dict_per_task[mode]['power'][:, ind_budget]
                tmp = test_dict_per_task[mode]['min_len'][:, ind_budget]
                # print(tmp)
                tmp[tmp==999]=np.nan
                # print(tmp)
                test_dict_per_alpha[mode]['min_len'][:, ind_alpha] = tmp
            ind_alpha += 1
        print('test_dict', test_dict_per_alpha)
        # sdfsdf
        plot(test_dict_per_alpha, task, args.risk_control_mode)


