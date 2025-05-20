"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import gym
import json
import pickle
from sacred import Experiment
import matplotlib.pyplot as plt
from wireless.agents.bosch_agent import BoschAgent
from wireless.agents.time_freq_resource_allocation_v0.round_robin_agent import *
from wireless.agents.time_freq_resource_allocation_v0.proportional_fair import *
from wireless.agents.noma_ul_time_freq_resource_allocation_v0.noma_ul_proportional_fair import *
import time

print(np.__version__)
# Load agent parameters
with open('config/config_agent.json') as f:
    ac = json.load(f)

# Configure experiment
with open('config/config_sacred.json') as f:
    sc = json.load(f)   # Sacred Configuration
    ns = sc["sacred"]["n_metrics_points"]  # Number of points per episode to log in Sacred
    ex = Experiment(ac["agent"]["agent_type"], save_git_info=False)
    ex.add_config(sc)
    ex.add_config(ac)
mongo_db_url = f'mongodb://{sc["sacred"]["sacred_user"]}:{sc["sacred"]["sacred_pwd"]}@' +\
               f'{sc["sacred"]["sacred_host"]}:{sc["sacred"]["sacred_port"]}/{sc["sacred"]["sacred_db"]}'
# ex.observers.append(MongoObserver(url=mongo_db_url, db_name=sc["sacred"]["sacred_db"]))  # Uncomment to save to DB

# Load environment parameters
with open('config/config_environment.json') as f:
    ec = json.load(f)
    ex.add_config(ec)



def collect_data(scheduler,alpha_pc=1.,p_max=23,p_min=-70,n_eps=1,t_max=5000,seed=0):
    # Simulate
    n_ues=32
    n_prbs=16
    avg_tx_pow, avg_ee, avg_delay, avg_tx_bits,avg_queue_pkts = [], [], [], [], []
    for ep in range(n_eps):  # Run episodes
        env = gym.make('TimeFreqResourceAllocation-v0', n_ues=n_ues,
                       n_prbs=n_prbs, buffer_max_size=8,p_max=p_max,p_min=p_min,
                       alpha=alpha_pc, f_carrier_mhz=2655,
                       max_pkt_size_bits=41250,
                       it=10)  # Init environment
        env.seed(seed=seed+ep)
        if scheduler == "random":
            agent = RandomAgent(env.action_space)
            agent.seed(seed=seed+ep)
        elif scheduler == "round robin":
            agent = RoundRobinAgent(env.action_space, env.K, env.L)
        elif scheduler == "round robin iftraffic":
            agent = RoundRobinIfTrafficAgent(env.action_space, env.K, env.L)
        elif scheduler == "proportional fair":
            agent = ProportionalFairAgent(env.action_space, env.K, env.L)
        elif scheduler == "proportional fair channel aware":
            agent = ProportionalFairChannelAwareAgent(env.action_space, env.K, env.L)
        elif scheduler == "knapsack":
            agent = Knapsackagent(env.action_space, env.K, env.L, env.Nf)
        elif scheduler == "Bosch":
            agent = BoschAgent(env.action_space, env.K, env.L, env.max_pkt_size_bits)
        else:
            raise NotImplemented
        reward = 0
        done = False
        state = env.reset()
        ra_ee=0
        ra_tx_pow=0
        ra_tx_bits=0
        ra_queues_pkts = 0
        ra_delays = 0
        start = time.time()
        t_ee=np.zeros(n_ues)
        t_delays=np.zeros(n_ues)
        actions=np.zeros(n_ues)
        for _ in range(t_max):  # Run one episode
            action = agent.act(state, reward, done)
            actions[action]=actions[action]+1
            state, reward, done, dict = env.step(action)
            if dict['tx_bits'][action]>0:
                ra_ee=ra_ee+reward
                t_ee=t_ee+(reward>0)
            ra_tx_pow=ra_tx_pow+dict['tx_power']/t_max
            ra_tx_bits=ra_tx_bits+dict['tx_bits']/t_max
            t_delays=t_delays+(dict['queues_delay']>0)
            ra_delays=ra_delays+dict['queues_delay']
            ra_queues_pkts = ra_queues_pkts + dict['queues_pkts'] / t_max
            if done:
                break
        ra_ee=ra_ee/t_ee
        ra_ee[np.isnan(ra_ee)] = 0
        ra_delays=ra_delays/t_delays
        ra_delays[np.isnan(ra_delays)] = 0
        ra_ee[np.isnan(ra_ee)] = 0
        end = time.time()
        if ep%100==0:
            print('Finished ep:'+str(ep))
            print(end - start)
        env.close()
        avg_ee.append(ra_ee)
        avg_tx_pow.append(ra_tx_pow)
        avg_tx_bits.append(ra_tx_bits)
        avg_delay.append(ra_delays)
        avg_queue_pkts.append(ra_queues_pkts)
    return np.asarray(avg_ee),np.asarray(avg_tx_pow),np.asarray(avg_tx_bits), np.asarray(avg_delay), np.asarray(avg_queue_pkts)

T_MAX=2500
N_EPS=500
SCHEDULERs=['Bosch','proportional fair channel aware','knapsack']
ALPHAs=np.arange(0.75,1.01,0.025)
print(ALPHAs)
for seed in range(20,25):
    print(seed)
    seed_init=seed*len(SCHEDULERs)*len(ALPHAs)*N_EPS
    log_avg_ee,log_avg_pow,log_avg_bits,log_delays,log_queue_pkts,models_tag = [],[],[],[],[],[]
    for scheduler in SCHEDULERs:
        for alpha in ALPHAs:
            print(scheduler+'_ALPHA_'+str(alpha))
            avg_ee,avg_pow,avg_bits,avg_delays,avg_queue_pkts=collect_data(scheduler=scheduler,alpha_pc=alpha,n_eps=N_EPS,t_max=T_MAX,seed=seed_init)
            seed_init=seed_init+N_EPS
            log_avg_ee.append(avg_ee)
            log_avg_pow.append(avg_pow)
            log_avg_bits.append(avg_bits)
            log_delays.append(avg_delays)
            log_queue_pkts.append(avg_queue_pkts)
            '''
            plt.plot(np.mean(log_delays[0],axis=0),label='delay')
            plt.show()
            plt.plot(np.mean(log_avg_bits[0], axis=0), label='bits')
            plt.show()
            te=np.mean(log_avg_bits[0], axis=0)/np.mean(log_avg_pow[0], axis=0)/1e6
            plt.plot(np.mean(log_avg_bits[0], axis=0)/np.mean(log_avg_pow[0], axis=0)/1e6, label='ee')
            plt.plot(np.mean(log_avg_ee[0], axis=0) , label='ee ra')
            plt.legend()
            plt.show()'''
            models_tag.append(scheduler+'_ALPHA_'+str(alpha))
    log_avg_ee = np.asarray(log_avg_ee)
    log_avg_pow = np.asarray(log_avg_pow)
    log_avg_bits = np.asarray(log_avg_bits)
    log_delays = np.asarray(log_delays)
    log_queue_pkts = np.asarray(log_queue_pkts)
    '''
    plt.plot(np.mean(log_avg_ee,axis=1))
    plt.plot(np.mean(log_avg_bits/log_avg_pow/1e6, axis=1))
    plt.show()'''
    dict={'avg_buffers':log_avg_ee,
          'avg_tx_powers':log_avg_pow,
          'avg_tx_bits': log_avg_bits,
          'queue_bits': log_delays,
          'queue_pkts': log_queue_pkts,
          'model_tags':models_tag}
    with open('logs/data/collected_data_seed_' + str(seed)+'.pkl', 'wb') as f:
        pickle.dump(dict, f)