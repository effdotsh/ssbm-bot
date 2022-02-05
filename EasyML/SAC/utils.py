import torch

def save(args, save_name, model, wandb=None, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        if wandb is not None:
            wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        if wandb is not None:
            wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random_old(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    for _ in range(num_samples):
        dataset.add(state, action, reward, next_state, done)

