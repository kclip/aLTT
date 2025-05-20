import random
from tqdm import tqdm
import fire

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from automatic_prompt_engineer import ape, data
from automatic_prompt_engineer.altogether_ape import ALTOGETHER_APE
from experiments.data.instruction_induction.load_data import load_data, tasks
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from pathlib import Path
import numpy as np
import pickle 
import argparse

IF_SAVE_ALL = False
WARMUP_RATIO = 0.0
MAX_EVAL_NUM = 2000

sub_tasks = ['antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']

def parse_args():
    parser = argparse.ArgumentParser(description='e_prompting')
    parser.add_argument('--ind_experiment', type=int, default=0, help='index of independent experiments')
    parser.add_argument('--task', type=str, default='antonyms', help='...')
    parser.add_argument('--risk_control_mode', type=str, default='FDR', help='...') 
    parser.add_argument('--alpha', type=float, default=0.5, help='')
    parser.add_argument('--delta', type=float, default=0.1, help='')
    args = parser.parse_args()
    return args

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def random_split(ind_experiment, task):
    MAX_EVAL_NUM = 2000
    # getting the size
    induce_data, test_data = load_data('induce', task), load_data('eval', task)
    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)
    test_num = len(test_data[0])
    cal_num = min(MAX_EVAL_NUM, len(eval_data[0]))
    try:
        te_loss_table = np.load('./cache/'+str(ind_experiment) +'/' + task + '/total_cand_prompts_testing_loss_table_num_te_' + str(test_num) + '.npy')
        cal_loss_table = np.load('./cache/'+str(ind_experiment) +'/' + task + '/total_cand_prompts_loss_table_num_' + str(cal_num) + '.npy')
    except:
        te_loss_table = np.load('./cache_original/' + task + '/total_cand_prompts_testing_loss_table_num_te_' + str(test_num) + '.npy')
        cal_loss_table = np.load('./cache_original/' + task + '/total_cand_prompts_loss_table_num_' + str(cal_num) + '.npy')
        cal_te_loss_table = np.zeros([te_loss_table.shape[0], test_num+cal_num])
        cal_te_loss_table[:, :cal_num] = cal_loss_table
        cal_te_loss_table[:, cal_num:] = te_loss_table
        cal_te_loss_table = shuffle_along_axis(cal_te_loss_table, 1)
        #np.squeeze(cal_te_loss_table[:, shuffle])
        # sdfsfd?
        #print('cal_te_loss_table', cal_te_loss_table)
        os.makedirs('./cache/' +str(ind_experiment) +'/'+task +'/', exist_ok=True)
        np.save('./cache/' +str(ind_experiment) +'/' + task + '/total_cand_prompts_testing_loss_table_num_te_' + str(test_num) + '.npy', cal_te_loss_table[:, cal_num:])
        np.save('./cache/' +str(ind_experiment) +'/' + task + '/total_cand_prompts_loss_table_num_' + str(cal_num) + '.npy', cal_te_loss_table[:, :cal_num])
    # print(te_loss_table)

def run(ind_experiment, task, num_val, alpha, delta, mode, total_budget, if_allow_set):
    #np.random.seed(seed)
    #mode = 'e_APE'
    #mode = 'standard_APE'
    assert task in tasks, 'Task not found!'
    induce_data, test_data = load_data('induce', task), load_data('eval', task)

    # Get size of the induce data
    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]
    #print('prompt_gen_data', prompt_gen_data) # if output has more than 1, randomly choose 1. Simply speaking, just use the first split for choosing exemplers
    # : (no need reasonong and explanations) 
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
    #prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced " \
                        #   "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]" 
    # prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced " \
    #                       "the following input-output pairs:\n\n[full_DEMO]\n\nCan you guess what the instruction was? I just need your guess without further explanatinos/reasoning/notes. [APE]" 
    prompt_gen_template = "I gave a friend a instruction. Based on the instruction they produced " \
                          "the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]" 
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"

    base_config = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': 3, #3
            'num_demos': 5, #5
            'num_prompts_per_subsample': 30, #30
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(num_val, len(eval_data[0])), # this is the number of eval data used to estimate the performance for each prompt -- per prompt, randomly sample
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        }
    }
    
    try:
        #print('load candidate prompts!')
        total_prompts = np.load('./cache_original/' + task + '/total_cand_prompts.npy')
    except:
        raise NotImplementedError
        print('generate candidate prompts!')
        total_prompts, _ = ape.find_prompts(cand_prompts=None, loss_table=None, eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=None,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)
        os.makedirs('./cache/' + task, exist_ok=True)
        total_prompts = np.array(total_prompts)
        np.save('./cache/' + task + '/total_cand_prompts.npy', total_prompts)

    
    #loss_table = np.load('./cache/' + task + '/total_cand_prompts_loss_table_num_' + str(min(2000, len(eval_data[0]))) + '.npy')
    try:
        loss_table = np.load('./cache/' + str(ind_experiment) +'/' + task + '/total_cand_prompts_loss_table_num_' + str(min(2000, len(eval_data[0]))) + '.npy')
    except:
        #print('------------------ generate loss table !!!')
        loss_table = None


    cost_single_eval = 1
    if mode == 'standard_APE':
        conf['evaluation']['num_samples'] = total_budget//(cost_single_eval*len(total_prompts))
        res, demo_fn = ape.find_prompts(cand_prompts=total_prompts, loss_table=loss_table, eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=eval_data,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)
        num_eval_samples = conf['evaluation']['num_samples']
        #print(loss_table.shape)
        loss_table = loss_table[:, :num_eval_samples]
        #print(loss_table.shape, num_eval_samples)
        loss_mean_vec = np.mean(loss_table, axis=1)
        num_total_cand = len(total_prompts)
        #print(num_total_cand)
        actual_cost = cost_single_eval*num_total_cand*num_eval_samples
        prompts = [total_prompts[np.argmin(loss_mean_vec)]]
        prompts_ind = [np.argmin(loss_mean_vec)]
    elif mode == 'bandit_APE':
        def _SC(B, score_table, if_allow_set=False): # S_k: index subset
            actual_cost = 0
            n = score_table.shape[0]
            total_rounds = np.ceil(np.log2(n))
            r_0 = np.floor(B/(n*np.ceil(np.log2(n))))
            if r_0 < 1:
                if if_allow_set:
                    total_rounds = B//n
                else:
                    pass
            S_k = np.arange(n) # init
            S_k_compl = []
            S_k_size = len(S_k)
            #R_k = np.zeros(n) # number of pulling
            R_k = 0
            for k in range(int(total_rounds)):
                r_k = int(np.floor(B/(S_k_size*total_rounds)))
                R_k += r_k
                curr_score_vec = np.mean(score_table[:, :int(R_k)], axis=1)
                actual_cost += r_k * S_k_size
                curr_score_vec[S_k_compl] = -1 # block
                S_k = np.argsort(-curr_score_vec)[:S_k_size//2]
                S_k_compl = np.argsort(-curr_score_vec)[S_k_size//2:]
                S_k_size = len(S_k)
                if S_k_size == 1:
                    break
                #print('k', k, '------', curr_score_vec, S_k)
            assert  actual_cost <= B
            #print(actual_cost, B, S_k)
            return S_k, actual_cost
        
        ### here ...
        res, demo_fn = ape.find_prompts(cand_prompts=total_prompts, loss_table=loss_table, eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data,
                                    eval_data=eval_data,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)
        if res is None:
            score_table = 1-loss_table
        else:
            score_table = res.scores
        #### do !
        selected_indexes, actual_cost = _SC(total_budget//cost_single_eval, score_table, if_allow_set=if_allow_set)
        actual_cost *= cost_single_eval
        prompts = total_prompts[selected_indexes]
        prompts_ind = selected_indexes
    else:
        warmup_iters = int(total_budget*WARMUP_RATIO //  (cost_single_eval* len(total_prompts)) ) 
        if 'LCB' in mode:
            explore_mode = 'LCB'
            epsilon = None
        elif 'greedy' in mode:
            explore_mode = 'epsilon_greedy'
            epsilon = float(mode[21:]) # mode = e_APE_epsilon_greeay_0.3
        else:
            explore_mode = 'RR'
            epsilon = None
        e_ape = ALTOGETHER_APE(ind_experiment, setting, task, alpha, delta, total_prompts, eval_data, eval_template, demos_template, prompt_gen_data, conf, base_config, cost_single_eval, e_mode='cost_first', explore_mode=explore_mode, epsilon=epsilon)
        prompts, prompts_ind = e_ape.forward(total_budget, warmup_iters)
        sorted_arg = np.argsort(-e_ape.total_capitals)
        actual_cost = e_ape.total_cost
    # Evaluate on test data
    #print('Evaluating on test data...')

    test_conf = {
        'generation': {
            'num_subsamples': 3, # how many exemplers
            'num_demos': 5,      # length of each exempler
            'num_prompts_per_subsample': 30, # how many promp gen. per exempler 
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(100, len(test_data[0])),
            'model': {
                'gpt_config': {
                    # 'model': 'text-ada-001'
                }
            }
        }
    }

    def _single_testing(prompt, ind=None, total_cost=None, alpha=None, bet_up=None, save_result=True):
        test_res = ape.evaluate_prompts(prompts=[prompt],
                                        eval_template=eval_template,
                                        eval_data=test_data,
                                        few_shot_data=prompt_gen_data,
                                        demos_template=demos_template,
                                        conf=test_conf,
                                        base_conf=base_config)

        test_score = test_res.sorted()[1][0]
        if save_result:
            if alpha is None:
                target_score = None
            else:
                target_score = 1-alpha
            mode_details = mode + str(total_budget)
            save_dir_name = f'experiments/16Jan/{mode_details}/instruction_induction/{task}/'
            Path(save_dir_name).mkdir(parents=True, exist_ok=True)
            # Save a text file to experiments/results/instruction_induction/task.txt with the best prompt and test score
            with open(save_dir_name + f'{ind}.txt', 'w') as f:
                f.write(f'total cost: {total_cost} target score: {target_score}\n')
                f.write(f'Test score: {test_score}\n')
                f.write(f'Prompt: {prompt}\n')
        else:
            save_dir_name = None
        return test_score, save_dir_name
    if IF_SAVE_ALL:
        test_scores = []
        ind_sel_prompt = 0
        for prompt in prompts:
            print('------prompt--------', prompt)
            test_score, save_dir_name = _single_testing(prompt, ind=ind_sel_prompt, total_cost=actual_cost)
            test_scores.append(test_score)
            ind_sel_prompt += 1
        test_scores = np.array(test_scores)
        if len(prompts) > 0:
            np.save(save_dir_name+'test_scores.npy', test_scores)
            print('--------------------------------------------------')
            print(task, ': test scores: ', np.load(save_dir_name+'test_scores.npy'))
            print('--------------------------------------------------')
    else:
        pass

    ## FDP and power
    def _FDP_and_power(ind_experiment, prompts, prompts_ind, target_score, total_prompts): # target_score = 1-alpha
        def _testing(ind_experiment, prompts, prompts_ind, ind=None, total_cost=None, alpha=None, bet_up=None, save_result=True):
        
            try: 
                #print('load entire testing loss table')
                te_loss_table = np.load('./cache/' + str(ind_experiment) + '/' + task + '/total_cand_prompts_testing_loss_table_num_te_' + str(len(test_data[0])) + '.npy')
            except:
                raise NotImplementedError
                print('evaluate the entire testing loss table for academic purpose (not to be used in real-word system!)')
                #self.loss_table = np.zeros([self.num_total_cand, len(self.eval_inputs)])
                #self.conf['evaluation']['num_samples'] = 'entire'
                #print('entire number of eval data examples: ', self.eval_num)
                # res = evaluate.evalute_prompts(self.cand_hyperparams, self.eval_template, (self.eval_inputs, self.eval_outputs), self.demos_template, self.few_shot_data,
                #                             self.conf['evaluation']['method'], self.conf['evaluation'])
                test_res = ape.evaluate_prompts(prompts=total_prompts,
                                            eval_template=eval_template,
                                            eval_data=test_data,
                                            few_shot_data=prompt_gen_data,
                                            demos_template=demos_template,
                                            conf=test_conf,
                                            base_conf=base_config)
                te_loss_table = 1-test_res.scores
                os.makedirs('./cache/' + task, exist_ok=True)
                np.save('./cache/' + task + '/total_cand_prompts_testing_loss_table_num_te_' + str(len(test_data[0])) + '.npy', te_loss_table)
            te_scores_entire = 1-te_loss_table
            te_scores = te_scores_entire[prompts_ind]
            return np.mean(te_scores, axis=1)
        entire_test_scores = _testing(ind_experiment, total_prompts, np.arange(len(total_prompts)), save_result=False)
        selected_test_scores = _testing(ind_experiment, prompts, prompts_ind, save_result=False)
        fdp = np.sum(selected_test_scores < target_score) / max(len(selected_test_scores), 1)
        if np.sum(selected_test_scores < target_score) > 0:
            fwer = 1
        else:
            fwer = 0
        # print('fwer', fwer)
        if len(selected_test_scores) == 0:
            fdp = np.nan
            fwer = np.nan
        power = np.sum(selected_test_scores >= target_score) / max(np.sum(entire_test_scores >= target_score), 1)
        if np.sum(entire_test_scores >= target_score) == 0:
            #power = 1.0 #np.nan
            power = np.nan
        return fdp, fwer, power, selected_test_scores
    fdp, fwer, power, test_scores = _FDP_and_power(ind_experiment, prompts, prompts_ind, 1-alpha, total_prompts)

    min_len = 999999
    for promt in prompts:
        if len(promt) < min_len:
            min_len = len(promt)
            # print('min len: ', min_len, promt)
        else:
            pass
    if len(prompts) == 0:
        min_len = np.nan
    return test_scores, actual_cost, fdp, fwer, power, min_len

if __name__ == '__main__':
    args = parse_args()
    sub_tasks = ['antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']

    delta = args.delta
    num_repetition = 20
    for task in tqdm(sub_tasks):
        alpha_list = [args.alpha] #[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for alpha in alpha_list:
            test_dict_per_task = {}
            print('task: ', task)
            for ind_experiment in range(num_repetition):
                random_split(ind_experiment, task)
                # print('ind_experiment', ind_experiment)
                setting = {}
                num_cand = 1 #110
                budget_int_list = [10,15,20,50,100,150,200,500,1000,1500,2000]#[10,20,50,100,200,500,1000,2000,5000] #[2000]
                total_budget_list = [num_cand*i for i in budget_int_list]
                ind_budget = 0
                for total_budget in total_budget_list:
                    for mode in ['e_APE_epsilon_greedy_0.25','e_APE_epsilon_greedy_0.5','e_APE_epsilon_greedy_0.75','e_APE_epsilon_greedy_0.95','p_APE','n_e_APE']: #, 'standard_APE', 'bandit_APE','e_APE_epsilon_greedy_0.0','e_APE_epsilon_greedy_0.1', 'e_APE_epsilon_greedy_0.25','e_APE_epsilon_greedy_0.5','e_APE_epsilon_greedy_0.75','e_APE_epsilon_greedy_0.95','e_APE_LCB']:
                        if mode == 'p_APE':
                            setting['HO'] = 'LTT'
                        elif mode == 'n_e_APE':
                            setting['HO'] = 'a-LTT-nada'
                        else:
                            setting['HO'] = 'a-LTT-ada'
                        setting['risk'] = args.risk_control_mode # either 'FDR', 'FWER'
                        test_scores, actual_cost, fdp, fwer, power, min_len = run(ind_experiment, task, 2000, alpha, delta, mode, total_budget, False) 
                        if mode in test_dict_per_task.keys():
                            pass
                        else:
                            test_dict_per_task[mode] = {}
                            test_dict_per_task[mode]['cost'] = np.zeros([num_repetition,  len(total_budget_list)])
                            #test_dict_per_task[mode]['score'] = np.zeros([num_repetition, len(total_budget_list)])
                            test_dict_per_task[mode]['fdp'] = np.zeros([num_repetition,  len(total_budget_list)])
                            test_dict_per_task[mode]['fwer'] = np.zeros([num_repetition, len(total_budget_list)])
                            test_dict_per_task[mode]['power'] = np.zeros([num_repetition, len(total_budget_list)])
                            test_dict_per_task[mode]['min_len'] = np.zeros([num_repetition,  len(total_budget_list)])

                        test_dict_per_task[mode]['cost'][ind_experiment, ind_budget] =actual_cost
                        test_dict_per_task[mode]['fdp'][ind_experiment, ind_budget] = fdp
                        test_dict_per_task[mode]['fwer'][ind_experiment, ind_budget] = fwer
                        test_dict_per_task[mode]['power'][ind_experiment, ind_budget] = power
                        test_dict_per_task[mode]['min_len'][ind_experiment, ind_budget] = min_len
                    ind_budget += 1
            save_dir_name = f'experiments/ICML_definitive/all_tasks/for_plotting/instruction_induction/{task}/{args.risk_control_mode}/'
            os.makedirs(save_dir_name, exist_ok=True)
            if len(budget_int_list) == 1:
                with open(save_dir_name +  str(alpha) +str(delta) + 'per_task_per_alpha.pkl', 'wb') as f:
                    pickle.dump(test_dict_per_task, f)
            else:
                with open(save_dir_name +  str(alpha) +str(delta) + 'per_task_per_budget.pkl', 'wb') as f:
                    pickle.dump(test_dict_per_task, f)