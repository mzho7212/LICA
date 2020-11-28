from collections import defaultdict
import multiprocessing as mp

import torch
import numpy as np


iterations = 60


def worker_run(index, k=5, lrfac=2):
    agent_id = np.array([[0], [1]], dtype=np.float32)
    combined_actions = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float32)     # joint actions
    rewards = np.array([[0], [1], [1], [0]], dtype=np.float32)                          # corresponding shared rewards

    counter_factual_actions = np.array([
        # Counterfactual actions for agent 1
        [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0],
        [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        # Counterfactual actions for agent 2
        [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0],
        [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]
    ], dtype=np.float32)

    critic_model = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )

    policy_model = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )

    critic_optimiser = torch.optim.Adam(critic_model.parameters(), lr=(1e-2 * lrfac))
    policy_optimiser = torch.optim.Adam(policy_model.parameters(), lr=1e-2)

    policy_outputs = []

    for i in range(iterations):
        for c in range(k):
            prediction = critic_model(torch.tensor(combined_actions))
            criterion = torch.nn.MSELoss()
            loss = criterion(prediction, torch.tensor(rewards))
            critic_optimiser.zero_grad()
            loss.backward()
            critic_optimiser.step()

        advantages = critic_model(torch.tensor(combined_actions)).repeat(4, 1) - critic_model(torch.tensor(counter_factual_actions))

        out = policy_model(torch.tensor(agent_id))
        probs = torch.sigmoid(out)

        probs_out = tuple(probs.view(-1).clone().detach().numpy())
        policy_outputs.append(probs_out)

        out = torch.sigmoid(out.repeat(1,8).view(16,1))

        # NOTE: Sampling 15 out of 16 possible training examples to create noise
        loss = (-1.0 * advantages * torch.log(out))[torch.randperm(16)[:15]].mean()

        policy_optimiser.zero_grad()
        loss.backward()
        policy_optimiser.step()

    return tuple(np.rint(probs_out)), policy_outputs


def parallel_run(repeat=100, k=1, lrfac=1):
    policy_outputs_lists = defaultdict(list)
    print(f'COMA running with repeat={repeat}, k={k}, lrfac={lrfac}')
    with mp.Pool() as pool:
        results = []
        for i in range(repeat):
            results.append(pool.apply_async(worker_run, (i, k, lrfac)))

        for res in results:
            outcome, policy_outputs = res.get()
            policy_outputs_lists[outcome].append(policy_outputs)

    return policy_outputs_lists


if __name__ == '__main__':
    policy_outputs_lists = parallel_run(100)
    print('Coma output:')
    for outcome in ((1,0), (0,1), (0,0), (1,1)):
        print(outcome, len(policy_outputs_lists[outcome]))

