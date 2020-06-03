import pandas as pd
import numpy as np

from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns

import tensortools as tt


# Load df for all units from fc experiments
fc_units_df = pd.read_csv(
    '/home/brian/data/ecephys_project_cache/functional_connectivity_analysis_metrics.csv')


def load_responses(response_filepath, unit_filepath, filter=True, sigma=10):
    # Load responses and units df
    responses = np.load(response_filepath)
    units = pd.read_csv(unit_filepath)

    # Filter units based on valid, on-screen RFs
    if filter:
        idx = units[units.unit_id.isin(fc_units_df[(fc_units_df.p_value_rf < 0.05) & (
            fc_units_df.on_screen_rf == True)].ecephys_unit_id)].index.values
        responses = responses[:, :, idx]

    # Smooth responses with Gaussian
    if sigma:
        # Save shape of responses
        num_trials, num_stim, num_neurons = responses.shape

        # Split tensor in half and filter
        responses = gaussian_filter(responses.reshape(
            2, -1, num_neurons).astype('float'), sigma=(0, sigma, 0), mode='constant')
        responses = responses.reshape(num_trials, num_stim, num_neurons)

    # Reshape responses (neurons x stim x trials)
    responses = np.transpose(responses, (2, 1, 0))

    return responses


def fit(responses, ranks=[1, 2, 3, 4, 5], N=20):
    results = []

    # Loop through ranks and N
    for r in ranks:
        for n in range(N):
            # Do decomposition and record rank, err and factors
            U = tt.ncp_hals(responses, rank=r, verbose=False)
            results.append(
                {'rank': r, 'err': U.obj, 'sim': np.nan, 'factors': U.factors})

    # Create results dataframe
    results = pd.DataFrame(results)

    # Find U with min error
    U_min = results.loc[results.groupby('rank')['err'].idxmin(), ['factors']]
    U_min.index = ranks
    U_min = U_min.to_dict()['factors']

    for i, row in results.iterrows():
        U = U_min[row['rank']]
        V = row['factors']

        # Compute similarity
        results.loc[i, 'sim'] = tt.kruskal_align(U, V)

    return results, U_min


def plot_err_sim(results, savename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Reconstruction error
    sns.pointplot(x='rank', y='err', data=results,
                  color='red', ci=None, markers='.', ax=axes[0])
    sns.stripplot(x='rank', y='err', data=results, color='black', ax=axes[0])
    axes[0].set_xlabel('# components', fontsize=22)
    axes[0].set_ylabel('error', fontsize=22)
    axes[0].set_ylim([0, 1.1])

    # Similarity
    sns.pointplot(x='rank', y='sim', data=results,
                  color='red', ci=None, markers='.', ax=axes[1])
    sns.stripplot(x='rank', y='sim', data=results, color='black', ax=axes[1])
    axes[1].set_xlabel('# components', fontsize=22)
    axes[1].set_ylabel('similarity', fontsize=22)
    axes[1].set_ylim([0, 1.1])

    # Global defaults
    [plt.setp(ax.get_xticklabels(), fontsize=18) for ax in axes]
    [plt.setp(ax.get_yticklabels(), fontsize=18) for ax in axes]

    plt.tight_layout()

    # Save figure
    fig.savefig(savename, dpi=200, bbox_inches='tight')


def permute_U(U):
    # Argmax across neuron factors
    factor_id = np.argmax(U[0], axis=1)

    neuron_idx = []
    for i in range(U.rank):
        idx = np.argwhere(factor_id == i).squeeze()
        selected = np.argsort(U[0][idx, i])[::-1]
        neuron_idx.append(idx[selected])

    factor_idx = np.argsort([len(i) for i in neuron_idx])[::-1]
    neuron_idx = [neuron_idx[f] for f in factor_idx]
    neuron_idx = np.concatenate(neuron_idx)

    # Resort U by neuron and factor indices
    U[0] = U[0][neuron_idx]
    U.factors = [u[:, factor_idx] for u in U.factors]

    return U


def plot_results(U, savename):
    fig, axes = plt.subplots(3, 3, figsize=(16, 6))
    colors = ['red', 'blue', 'green']

    for i in range(U.rank):
        # Neuron factors
        axes[i, 0].bar(np.arange(1, U[0].shape[0]+1), U[0]
                       [:, i], color='black', width=1)
        axes[i, 0].set_xlim(0, U[0].shape[0]+1)
        axes[i, 0].set_ylim([0, U[0].max()+0.1])
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 0].spines['right'].set_visible(False)

        # Stimulus factors
        # Divide by 30 (Hz) to get into units of seconds
        axes[i, 1].plot(np.arange(1, U[1].shape[0]+1)/30, U[1]
                        [:, i], color=colors[i], linewidth=3)
        axes[i, 1].set_xlim(0, U[1].shape[0]/30)
        axes[i, 1].set_ylim([0, U[1].max()+0.1])
        axes[i, 1].spines['top'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)

        # Trial factors
        axes[i, 2].axvline(31, color='black', linewidth=3, zorder=1)
        axes[i, 2].axvspan(31, U[2].shape[0]+1, alpha=0.25,
                           color='gray', zorder=2)
        axes[i, 2].scatter(np.arange(1, U[2].shape[0]+1), U[2]
                           [:, i], color=colors[i], zorder=10)
        axes[i, 2].set_xlim(0, U[2].shape[0])
        axes[i, 2].set_ylim([0, U[2].max()+0.1])
        axes[i, 2].spines['top'].set_visible(False)
        axes[i, 2].spines['right'].set_visible(False)

        # Remove unneeded x/y-ticks
        if i != U.ndim-1:
            [plt.setp(ax.get_xticklabels(), visible=False) for ax in axes[i]]
            [plt.setp(ax.get_yticklabels(), fontsize=18) for ax in axes[i]]
        else:
            [plt.setp(ax.get_xticklabels(), fontsize=18) for ax in axes[i]]
            [plt.setp(ax.get_yticklabels(), fontsize=18) for ax in axes[i]]
            [a.set_xlabel(c, size=22) for a, c in zip(
                axes[i], ['neurons', 'time (s)', 'trials'])]

    # Format y-ticks
    for r in range(U.rank):
        for i in range(U.ndim):
            # only two labels
            ymin, ymax = np.round(axes[r, i].get_ylim(), 2)
            axes[r, i].set_ylim((ymin, ymax))

            # remove decimals from labels
            if ymin.is_integer():
                ymin = int(ymin)
            if ymax.is_integer():
                ymax = int(ymax)

            # update plot
            axes[r, i].set_yticks([ymin, ymax])

    # Format y-labels
    [a.set_ylabel(c, size=24, rotation=0)
     for a, c in zip(axes[:, 0], ['#1', '#2', '#3'])]
    [a.yaxis.set_label_coords(-0.2, 0.4) for a in axes[:, 0]]

    plt.tight_layout()

    # Save figure
    fig.savefig(savename, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    # Loop through experiments
    for e in ['835479236', '839068429', '839557629', '840012044', '847657808']:
        response_filepath = './processed/'+e+'_repeat_frame_cell.npy'
        unit_filepath = './processed/'+e+'_units.csv'

        # Load responses and filter
        responses = load_responses(response_filepath, unit_filepath)

        # Fit nonnegative CPD
        results, U_min = fit(responses)

        # Plot error and similarity vs. rank
        plot_err_sim(results, savename='./results/'+e+'_err_sim.png')

        # Permute U for plotting results
        U_permuted = permute_U(U_min[3].copy())

        # Plot results
        plot_results(U_permuted, savename='./results/'+e+'_tca.png')
