import random
from PPO import PPO
def rand_value():
    return random.random() - 0.5 * 100


if __name__== '__main__':
    action_dim = [1]
    state_dim = [4]



    max_ep_len = 400  # max timesteps in one episode
    max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(2e4)  # save model frequency (in num timesteps)

    action_std = None

    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################

    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 40  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network
    has_continuous_action_space = True

    random_seed = 0  # set random seed if required (0 = no random seed)


    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # params = ai.parameters()
    # for p in params:
        # print(p)



    # plt.plot(range(epochs),rewards_history)
    # plt.show()