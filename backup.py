#!/usr/bin/python
import gym
import numpy as np 
import matplotlib.pyplot as plt 
import json
import csv



env = gym.make('LunarLander-v2')
env._max_episode_steps=5000
bucketNum=9
MAXSTATES = (10**6)
GAMMA = 0.9
ALPHA = 0.01




def max_dict(d):
	max_v = float('-inf')
	for key, val in d.items():
		if val > max_v:
			max_v = val
			max_key = key
	return max_key, max_v

def create_bins():
	# obs[0] -> cart position --- -4.8 - 4.8
	# obs[1] -> cart velocity --- -inf - inf
	# obs[2] -> pole angle    --- -41.8 - 41.8
	# obs[3] -> pole velocity --- -inf - inf
        print('Bins Made')	
	
        bins=np.zeros((6,bucketNum))
        discreteBins=np.zeros((2,2))
	bins[0] = np.linspace(-1.1, 1.1, bucketNum)
	bins[1] = np.linspace(-0.4, 1.6, bucketNum)
	bins[2] = np.linspace(-1, 1, bucketNum)
	bins[3] = np.linspace(-1.5, 0.5, bucketNum)
        bins[4] = np.linspace(-1.4, 0.5, bucketNum)
        bins[5] = np.linspace(-1.3, 0.5, bucketNum)
        #discreteBins[0] = np.linspace(0,1,2)
        #discreteBins[1] = np.linspace(0,1,2)
        return bins,discreteBins



def saveQ(Q):
    w=csv.writer(open("array.csv","w"))
    for key, val in Q.items():
        w.writerow([key,val])




def assign_bins(observation, bins,discreteBins):
        state = np.zeros(6)
	
        for i in range(6):
            if i<=5:
	        state[i] = np.digitize(observation[i], bins[i])
            else:
                
                state[i]=np.digitize(observation[i],discreteBins[i%2])
	return state

def get_state_as_string(state):
        string_state = ''.join(str(int(e)) for e in state)
        #print(string_state) 
        return string_state

def get_all_states_as_string():
	print('Getting all states as strings')
        states = []
        cnt=0
	for i in range(MAXSTATES):
            cnt+=1
            states.append(str(i).zfill(6))    
        print(cnt)
        return states

def initialize_Q():
    
    Q = {}
    
    try:
        print('Looking for a Q array')
        np.load('QArray.npy').item()
    
        return Q
    except:
        print('No Array Found, Creating One now')
	all_states = get_all_states_as_string()
	for state in all_states:
                Q[state] = {}
		
                for action in range(env.action_space.n):
			Q[state][action] = 0
        print('Initializing done')
        return Q

def play_one_game(bins,discreteBins, Q,render, eps):
        observation = env.reset()
	done = False
	cnt = 0 # number of moves in an episode
	state = get_state_as_string(assign_bins(observation, bins,discreteBins))
	total_reward = 0
        #print('Playing 1 game')        
	while not done:
                if render==1:
                    env.render()

                cnt += 1
		# np.random.randn() seems to yield a random action 50% of the time ?
		if np.random.uniform() < eps:
			act = env.action_space.sample() # epsilon greedy
		else:			
                    act = max_dict(Q[state])[0]
		
		observation, reward, done, _ = env.step(act)
                 
		total_reward += reward
                state_new=assign_bins(observation,bins,discreteBins)
                state_new=get_state_as_string(state_new)
		#state_new = get_state_as_string(assign_bins(observation, bins,discreteBins))
                #print(state_new,'Given observation',observation)
		#print((state_new,bucketNum))
                a1, max_q_s1a1 = max_dict(Q[state_new])
                Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
                state,act = state_new,a1					
	return total_reward, cnt

def play_many_games(bins,discreteBins, N=10):
         
	#Q = initialize_Q()
        print('Starting to play many games')
	length = []
	reward = []
	for n in range(N):

		#eps=0.5/(1+n*10e-3)
		#eps = 1.0 / (np.sqrt(n+1)*0.5)
                eps=0.1
		episode_reward, episode_length= play_one_game(bins,discreteBins, Q,0, eps)
                print(n, '%.4f' % eps, episode_reward)
		length.append(episode_length)
		reward.append(episode_reward)
        #plot_running_avg(reward)
        saveQ(Q)
        while True:
            play_one_game(bins,discreteBins,Q,1,eps)
	return length, reward


def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

Q = initialize_Q()
if __name__ == '__main__':
	bins,discreteBins = create_bins()
	episode_lengths, episode_rewards = play_many_games(bins,discreteBins)
