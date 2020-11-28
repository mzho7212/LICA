from collections import defaultdict
import multiprocessing as mp

import torch
import numpy as np


iterations = 60


def worker_run(index, k=5, lrfac=2):
    agent_id = np.array([[0], [1]], dtype=np.float32)
    combined_actions = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    rewards = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

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
            prediction = critic_model(torch.from_numpy(combined_actions))
            criterion = torch.nn.MSELoss()
            loss = criterion(prediction, torch.from_numpy(rewards))
            critic_optimiser.zero_grad()
            loss.backward()
            critic_optimiser.step()

        probs = torch.sigmoid(policy_model(torch.from_numpy(agent_id)))
        probs = torch.cat((probs[0], probs[1]), 0)

        probs_out = tuple(probs.view(-1).clone().detach().numpy())
        policy_outputs.append(probs_out)

        # concatenated `probs` -> joint action (distribution parameters)
        loss = - 1.0 * critic_model(probs)

        policy_optimiser.zero_grad()
        loss.backward()
        policy_optimiser.step()

    return tuple(np.rint(probs_out)), policy_outputs


def parallel_run(repeat=100, k=1, lrfac=1):
    print(f'LICA running with repeat={repeat}, k={k}, lrfac={lrfac}')
    policy_outputs_lists = defaultdict(list)
    with mp.Pool() as pool:
        results = []
        for i in range(repeat):
            results.append(pool.apply_async(worker_run, (i, k, lrfac)))

        for res in results:
            outcome, policy_outputs = res.get()
            policy_outputs_lists[outcome].append(policy_outputs)

    return policy_outputs_lists


if __name__ == '__main__':
    policy_outputs_lists = run(300)
    print('Mix output:')
    for outcome in ((1,0), (0,1), (0,0), (1,1)):
        print(outcome, len(policy_outputs_lists[outcome]))
