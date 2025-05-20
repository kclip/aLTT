import numpy as np
from automatic_prompt_engineer import evaluate, config, template
import os

# we can get the entire loss table first as before, and then stop as before ...
# then we first generate loss table
# then we do the same 

EPS = 0.0000000001
INF = 99999999
#TOL = 1.5

class ALTOGETHER_APE:
    def __init__(self, ind_experiment, setting, task, alpha, delta, cand_hyperparams, eval_data, eval_template, demos_template, few_shot_data, conf, base_conf, cost_single_eval, e_mode='cost_first', max_eval_num=2000, if_loss_compute_at_once=True, explore_mode='LCB', epsilon=0.25): 
        self.setting = setting
        # setting: {risk: [FDR, FWER]; HO: [LTT, a-LTT-ada, a-LTT-nada] }
        self.alpha = alpha
        self.delta = delta
        self.e_mode = e_mode # e_mode: cost_first or perf_first
        self.cand_hyperparams = cand_hyperparams
        self.num_total_cand = len(cand_hyperparams)
        # self.selected_hyperparams = []
        self.cost_single_eval = cost_single_eval
        # self.total_cost = 0
        self.rho = 0.0
        self.bet_up = 3/4 #3/4 #1/2
        self.explore_mode = explore_mode
        self.epsilon = epsilon
        eval_template = template.EvalTemplate(eval_template)
        demos_template = template.DemosTemplate(demos_template)
        if few_shot_data is None:
            few_shot_data = prompt_gen_data
        self.eval_inputs, self.eval_outputs = eval_data
        self.eval_inputs = self.eval_inputs[:max_eval_num]
        self.eval_outputs = self.eval_outputs[:max_eval_num]
        self.eval_num = len(self.eval_inputs)
        self.eval_template = eval_template
        self.demos_template = demos_template
        self.few_shot_data = few_shot_data
        conf = config.update_config(conf, base_conf)
        self.conf = conf
        self.if_loss_compute_at_once = if_loss_compute_at_once # this is for academic purpose to make experiments faster -- not to be used in practice
        if self.if_loss_compute_at_once:
            try: 
                #print('load entire loss table')
                self.loss_table = np.load('./cache/' + str(ind_experiment)+ '/'+ task + '/total_cand_prompts_loss_table_num_' + str(len(self.eval_inputs)) + '.npy')
            except:
                raise NotImplementedError
                print('evaluate the entire loss table for academic purpose (not to be used in real-word system!)')
                #self.loss_table = np.zeros([self.num_total_cand, len(self.eval_inputs)])
                self.conf['evaluation']['num_samples'] = 'entire'
                print('entire number of eval data examples: ', self.eval_num)
                res = evaluate.evalute_prompts(self.cand_hyperparams, self.eval_template, (self.eval_inputs, self.eval_outputs), self.demos_template, self.few_shot_data,
                                            self.conf['evaluation']['method'], self.conf['evaluation'])
                self.loss_table = 1-res.scores
                print('loss_table', self.loss_table, self.loss_table.shape, 'should be 90 * 20')
                os.makedirs('./cache/' + task, exist_ok=True)
                np.save('./cache/' + task + '/total_cand_prompts_loss_table_num_' + str(self.eval_num) + '.npy', self.loss_table)
        else:
            pass
        # rng = np.random.default_rng()
        # self.loss_table = rng.permuted(self.loss_table, axis=1)

        
    @staticmethod
    def argmax(b):
        return np.random.choice(np.flatnonzero(b == b.max()))
    @staticmethod
    def argmin(b):
        return np.random.choice(np.flatnonzero(b == b.min()))
    def reset(self):
        self.selected_hyperparams = []
        self.success_indices = []
        self.running_out_sample = []
        self.remaining_indices = np.arange(self.num_total_cand)
        self.total_cost = 0
        self.estimated_more_betting_rounds = np.ones(self.num_total_cand) * self.eval_num
        #self.remaining_val = np.ones(self.num_total_cand) * self.eval_num
        self.lcb = np.zeros(self.num_total_cand)
        self.epsilon_greedy = np.zeros(self.num_total_cand)
        self.total_capitals = np.ones(self.num_total_cand)
        self.best_e = np.ones(self.num_total_cand)
        self.num_bets = np.zeros(self.num_total_cand, dtype=int)
        self.hyperparams_dict = {}
        for ind_cand in range(self.num_total_cand):
            #self.hyperparams_dict['num_bet_'+str(ind_cand)] = 0
            self.hyperparams_dict['curr_bet_'+str(ind_cand)] = 1
            self.hyperparams_dict['loss_list_'+str(ind_cand)] = np.array([])
            self.hyperparams_dict['z_i_sq_sum_'+str(ind_cand)] = 0
            self.hyperparams_dict['e_'+str(ind_cand)] = []
            
    def loss_compute(self, ind_hyperparam, ind_sample):
        if self.if_loss_compute_at_once:
            self.total_cost += self.cost_single_eval
            return self.loss_table[ind_hyperparam][ind_sample]
        else:
            curr_eval_example = ([self.eval_inputs[ind_sample]], [self.eval_outputs[ind_sample]])
            self.conf['evaluation']['num_samples'] = 1
            res = evaluate.evalute_prompts([self.cand_hyperparams[ind_hyperparam]], self.eval_template, curr_eval_example, self.demos_template, self.few_shot_data,
                                        self.conf['evaluation']['method'], self.conf['evaluation'])
            _, score = res.sorted()
            self.total_cost += self.cost_single_eval
            #print('score should have length 1!', score)
            assert len(score) == 1
            assert 0 <= score[0] <= 1 # if not, we need to normalize
            return 1-score[0] # score: higher the better; loss: lower the better
    def test_martingale(self, loss, betting):
        assert 0 <= betting <= 1/(1-self.alpha) # ensuring positive supermartingale
        return 1 - betting*(loss-self.alpha)
    @staticmethod
    def update_betting_strategy(nu_t, rhos, X_t, m, z_i_sq_sum, c): # c = 1/2 or 3/4 by default, ONS
        y_t = X_t - m
        z_t = y_t/( 1-y_t*nu_t + EPS)
        z_i_sq_sum += np.square(z_t)
        A_t = 1 + z_i_sq_sum
        lr_t = (2/(2-np.log(3))) * (1/A_t) 
        next_nu = np.minimum(np.maximum(nu_t - lr_t * z_t, 0), c/(1+rhos-m))
        #print('next_nu', next_nu)
        return next_nu, z_i_sq_sum

    def exploit(self, ind_hyperparam):
        self.num_bets[ind_hyperparam] += 1
        if self.num_bets[ind_hyperparam] >= self.eval_num:
            self.running_out_sample.append(ind_hyperparam)
        #ind_sample = self.num_bets[ind_hyperparam] - 1 # python indexing from 0
        ind_sample = self.t - 1 
        # get loss
        loss = self.loss_compute(ind_hyperparam, ind_sample) 
        self.hyperparams_dict['loss_list_'+str(ind_hyperparam)] = np.append(self.hyperparams_dict['loss_list_'+str(ind_hyperparam)], loss)
        # get current capital
        curr_capital = self.test_martingale(loss, self.hyperparams_dict['curr_bet_'+str(ind_hyperparam)])
        # update total capital
        self.total_capitals[ind_hyperparam] *= curr_capital
        self.hyperparams_dict['e_'+str(ind_hyperparam)].append(self.total_capitals[ind_hyperparam])
        self.best_e[ind_hyperparam] = max(self.hyperparams_dict['e_'+str(ind_hyperparam)])
        # update betting strategy
        self.hyperparams_dict['curr_bet_'+str(ind_hyperparam)], self.hyperparams_dict['z_i_sq_sum_'+str(ind_hyperparam)] = self.update_betting_strategy(self.hyperparams_dict['curr_bet_'+str(ind_hyperparam)], self.rho, loss, self.alpha, self.hyperparams_dict['z_i_sq_sum_'+str(ind_hyperparam)], self.bet_up)
        if self.explore_mode == 'epsilon_greedy':
            self.epsilon_greedy[ind_hyperparam] = self.total_capitals[ind_hyperparam]
        else:
            pass


    def ensure_stopping_time(self, cand_set, cand_remaining_set):
        # cand_set = descending_argsort[:first_ind_negative]
        # cand_remaining_set = descending_argsort[first_ind_negative:]
        self.success_indices = cand_set # 0, 1, .., first_ind_negative-1
        self.remaining_indices = cand_remaining_set
        self.selected_hyperparams = self.cand_hyperparams[self.success_indices]

    def ensure_stopping_time_prev(self, cand_set, cand_remaining_set):
        #cand_set = descending_argsort[:first_ind_negative]
        #cand_remaining_set = descending_argsort[first_ind_negative:]
        if len(self.success_indices) < len(cand_set):
            new_added_cand = np.setdiff1d(cand_set, self.success_indices)
            if len(new_added_cand) > 1: # priotize choosing the one that can be immediately selected
                if len(self.num_bets[self.success_indices]) == 0:
                    new_added_cand = [new_added_cand[0]]
                else:
                    diff_to_max =  max(self.num_bets[self.success_indices]) - self.num_bets[new_added_cand]
                    new_added_cand = new_added_cand[np.argmin(diff_to_max)]
                #new_added_cand = [new_added_cand[0]]
            if len(self.num_bets[self.success_indices]) == 0:
                self.success_indices = cand_set # 0, 1, .., first_ind_negative-1
                self.remaining_indices = cand_remaining_set
                self.selected_hyperparams = self.cand_hyperparams[self.success_indices]
            else:
                # print(self.num_bets[new_added_cand])
                # print(self.num_bets[self.success_indices])
                if self.num_bets[new_added_cand] < max(self.num_bets[self.success_indices]):
                    #print('no!!!!!', self.num_bets[new_added_cand], self.num_bets[self.success_indices])
                    pass
                else:
                    # print(self.num_bets[new_added_cand])
                    self.success_indices = cand_set # 0, 1, .., first_ind_negative-1
                    self.remaining_indices = cand_remaining_set
                    self.selected_hyperparams = self.cand_hyperparams[self.success_indices]
        else:
            pass

    def select(self):
        if self.setting['risk'] == 'FDR':
            if self.setting['HO'] == 'LTT':
                curr_e_status = self.total_capitals # last
                curr_p_status = 1/(curr_e_status+EPS)
                ## apply BY
                def _get_corr_BY(num_hyper):
                    corr = 0
                    for i in range(num_hyper):
                        corr += 1/(i+1)
                    return corr
                #corr_BY = _get_corr_BY(self.num_total_cand)
                ascending_argsort = np.argsort(curr_p_status) 
                ascending_sorted = curr_p_status[ascending_argsort]
                #ref_line = (self.delta*(np.arange(self.num_total_cand)+1))/(self.num_total_cand*corr_BY)
                ref_line = (self.delta*(np.arange(self.num_total_cand)+1))/(self.num_total_cand)
                first_ind_negative = np.argwhere(ref_line - ascending_sorted < 0)[0][0] # first element, np.argwhere gives list
                if first_ind_negative == 0:
                    self.remaining_indices = ascending_argsort[:]
                else:
                    self.success_indices = ascending_argsort[:first_ind_negative]
                    self.remaining_indices = ascending_argsort[first_ind_negative:]
                    self.selected_hyperparams = self.cand_hyperparams[self.success_indices]
            else:
                assert 'a-LTT' in self.setting['HO']
                if self.setting['HO'] == 'a-LTT-ada':
                    curr_e_status = self.total_capitals
                elif self.setting['HO'] == 'a-LTT-nada':
                    curr_e_status = self.best_e
                else:
                    raise NotImplementedError
                # descending_argsort = np.argsort(-self.total_capitals) # descending order
                # descending_sorted = self.total_capitals[descending_argsort]
                descending_argsort = np.argsort(-curr_e_status) # descending order
                descending_sorted = curr_e_status[descending_argsort]
                ref_line = self.num_total_cand/(self.delta*(np.arange(self.num_total_cand)+1))
                first_ind_negative = np.argwhere(descending_sorted - ref_line < 0)[0][0] # first element, np.argwhere gives list
                if first_ind_negative == 0:
                    self.remaining_indices = descending_argsort[:]
                else:
                    self.ensure_stopping_time(descending_argsort[:first_ind_negative], descending_argsort[first_ind_negative:])
                
        elif self.setting['risk'] == 'FWER':
            if self.setting['HO'] == 'LTT':
                curr_e_status = self.total_capitals # last
            elif self.setting['HO'] == 'a-LTT-nada':
                curr_e_status = self.best_e
            elif self.setting['HO'] == 'a-LTT-ada':
                curr_e_status = self.total_capitals
            curr_p_status = 1/(curr_e_status+EPS)
            ref_line = self.delta/self.num_total_cand # Bonferroni
            self.ensure_stopping_time(np.squeeze(np.argwhere( ref_line - curr_p_status >= 0 ), axis=1), np.squeeze(np.argwhere( ref_line - curr_p_status < 0 ), axis=1))
            #print(self.num_bets[self.selected_hyperparams])
            # self.success_indices = 
            # self.selected_hyperparams = self.cand_hyperparams[self.success_indices]
            # self.remaining_indices = 
            # print('curr_p_status', curr_p_status, 'ref', ref_line, 'success', self.success_indices)
        else:
            raise NotImplementedError

    def explore(self):
        if self.explore_mode == 'epsilon_greedy':
            self.epsilon_greedy[self.success_indices] = -1
            self.epsilon_greedy[self.running_out_sample] = -1 # as we do not have samples anymore, cannot explore anymore
            exploring_indices = [i for i in self.remaining_indices if i not in self.running_out_sample]
            if len(exploring_indices) == 0:
                next_ind = None
            else:
                if np.random.uniform(0, 1) < self.epsilon:
                    next_ind_tmp = np.random.randint(0, len(exploring_indices))
                    next_ind = exploring_indices[next_ind_tmp]
                else:
                    next_ind = self.argmax(self.epsilon_greedy)
        elif self.explore_mode == 'RR':
            next_ind = self.t % self.num_total_cand
        else:
            raise NotImplementedError
        return next_ind

    def forward(self, total_budget, warmup_iters):
        self.reset()
        if self.setting['HO'] == 'LTT' or self.setting['HO'] == 'a-LTT-nada':
            self.explore_mode == 'RR'
        else:
            pass
        self.total_budget = total_budget
        self.t = 0
        while self.total_cost < total_budget:
            # explore phase
            ind_hyperparam = self.explore()
            self.t += 1
            if self.eval_num < self.t:
                return self.selected_hyperparams, self.success_indices
            #print('ind_hyperparam', ind_hyperparam)
            if ind_hyperparam is None or self.num_bets[ind_hyperparam] >= self.eval_num:
            #if ind_hyperparam is None:
                return self.selected_hyperparams, self.success_indices
            else:
                self.exploit(ind_hyperparam)
                self.select()
        return self.selected_hyperparams, self.success_indices
