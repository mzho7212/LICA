import json
import os
import re
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from coma import parallel_run as run_coma
from mix import parallel_run as run_mix

plt.style.use(['science'])
plt.rc('grid', linestyle=':', color='gray')
plt.grid(which='major')


def show(i, ax, means, stds, legend, x=None, scale='linear', alpha=1):
    means = np.array(means) * 100
    stds = np.array(stds) * 100

    if x is None:
        x = list(range(1, len(means)+1))
        ax.plot(x, means, label=legend, alpha=alpha)
        ax.fill_between(x, means-(stds), means+(stds), alpha=0.1, edgecolor='face')

    ax.set_xlim([1, 60])
    ax.set_ylim([0, 100])

    if i == 0:
        ax.set_ylabel('Move Probability (\%)')
    else:
        ax.set_yticklabels([])

    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel('Steps', fontsize=13)
    ax.yaxis.set_ticks([0,25,50,75,100])


if __name__ == '__main__':
    repeat = int(sys.argv[1])
    # default values
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    lrfac = int(sys.argv[3]) if len(sys.argv) >= 4 else 2

    print(f'repeat={repeat}, k={k}, lrfac={lrfac}')

    mix_outs = run_mix(repeat, k, lrfac)
    coma_outs = run_coma(repeat, k, lrfac)

    outcomes = ((1,0), (0,1), (0,0), (1,1))
    print('Outcomes:', outcomes)
    print('Mix outcome counts:', [len(mix_outs[o]) for o in outcomes])
    print('Coma outcome counts:', [len(coma_outs[o]) for o in outcomes])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5), constrained_layout=True)

    success_outcomes = ((1,0), (0,1))
    for i, (ax, outcome) in enumerate(zip((ax1, ax2), success_outcomes)):

        print('Success Outcome', outcome)

        mix_runs = mix_outs[outcome]
        coma_runs = coma_outs[outcome]

        a1_mix_means = []
        a2_mix_means = []
        a1_mix_stds = []
        a2_mix_stds = []

        for step_vals in zip(*mix_runs):
            a1_stepvals, a2_stepvals = zip(*step_vals)

            a1_mix_means.append(np.mean(a1_stepvals))
            a1_mix_stds.append(np.std(a1_stepvals))

            a2_mix_means.append(np.mean(a2_stepvals))
            a2_mix_stds.append(np.std(a2_stepvals))

        a1_coma_means = []
        a2_coma_means = []
        a1_coma_stds = []
        a2_coma_stds = []

        for step_vals in zip(*coma_runs):
            a1_stepvals, a2_stepvals = zip(*step_vals)

            a1_coma_means.append(np.mean(a1_stepvals))
            a1_coma_stds.append(np.std(a1_stepvals))

            a2_coma_means.append(np.mean(a2_stepvals))
            a2_coma_stds.append(np.std(a2_stepvals))

        show(i, ax, a1_mix_means, a1_mix_stds, 'LICA 1')
        show(i, ax, a2_mix_means, a2_mix_stds, 'LICA 2')

        show(i, ax, a1_coma_means, a1_coma_stds, 'COMA 1')
        show(i, ax, a2_coma_means, a2_coma_stds, 'COMA 2')

        ax.grid()

    rand = random.randint(0,1000)
    filename = f'outcome-subplot-{repeat}-{k}-{lrfac}-{rand:03}.pdf'
    plt.savefig(filename)
    print(f'Runs with {repeat} repeats saved to {filename}')
