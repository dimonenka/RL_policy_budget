import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
import numpy as np
from plot.colors import Neon
from plot import plot_config
from scipy.stats import ttest_rel


def table_and_plot_sw_from_csv():
    df = None
    dirs = filter(os.path.isdir, os.listdir('.'))
    for dir in dirs:
        try:
            N_AGENTS, N_POLICIES = int(dir.split('_')[0]), int(dir.split('_')[2])
        except:
            continue
        df_ = pd.read_csv(dir + '/csv/' + TAG_SW)
        col = df_.iloc[:, -1].name
        df_ = df_.groupby(['env', 'task', 'algorithm', 'seed'])[col].mean()
        df_ = df_.rename('social_welfare').to_frame()
        df_['k'] = N_POLICIES
        if df is None:
            df = df_.copy()
        else:
            df = pd.concat([df, df_], axis=0)

    df.reset_index(inplace=True)
    df.loc[df['task'] == 'resource', 'social_welfare'] *= 100
    df = df[df['algorithm'].isin(NAMES.keys())]

    df.loc[(df['task'] == 'resource') & (df['k'] == 25) & (df['algorithm'] != 'em'), 'social_welfare'] = np.NaN
    df.dropna(inplace=True)
    for env in df.env.unique():
        df_ = df[(df['env'] == env) & ((df['k'] == 1) | (df['k'] == 25))]
        for algorithm in df.loc[df['env'] == env, 'algorithm'].unique():
            if algorithm != 'em':
                df_.loc[:, 'algorithm'] = algorithm
                df = pd.concat([df, df_])

    for name, new_name in NAMES.items():
        df.loc[df['algorithm'] == name, 'algorithm'] = new_name

    def plot_sw(df, directory='images/'):
        if not os.path.exists(directory): os.mkdir(directory)
        palette = sns.color_palette()
        for env in df['env'].unique():
            df_ = df[df['env'] == env].drop(columns=['seed', 'env', 'task'])

            ax = sns.barplot(data=df_, x='k', y='social_welfare', hue='algorithm', errorbar='se',
                             palette={'EM': palette[0], 'random': palette[1], 'end-to-end': palette[2], 'cluster': palette[3]},
                             hue_order=('EM', 'end-to-end', 'cluster', 'random')[:df_.algorithm.nunique()])
            ax.set_ylabel(None)
            if env.startswith('resource'):
                ax.set_ylim(70, 95)
                ax.axhline(93.6, ls='--', c='k', lw=2)

                sns.move_legend(
                    ax, "lower center",
                    bbox_to_anchor=(.5, 1), ncol=df_.algorithm.nunique(), title=None, frameon=False, fontsize=16
                )
            else:
                ax.legend([], [], frameon=False)

            ax.set_xlabel('$k$ policies', fontdict={'weight': 'bold', 'size': 20})
            ax.tick_params(axis='y', labelsize=20, direction='out', pad=5)
            ax.tick_params(axis='x', labelsize=20)
            for label in (ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels()):
                label.set_fontweight('bold')
            plot_config.save(directory + env + '_bars.png')
            ax.clear()

    plot_sw(df)

    def _ttest(group, p_value=0.01):
        best_algorithm = group.groupby('algorithm')['social_welfare'].mean().idxmax()
        t = {best_algorithm: 1.}
        for algorithm in group['algorithm'].unique():
            if algorithm == best_algorithm:
                continue
            t[algorithm] = ttest_rel(group.loc[group['algorithm'] == algorithm, 'social_welfare'].values,
                                     group.loc[group['algorithm'] == best_algorithm, 'social_welfare'].values,
                                     alternative='less')[1]
        group.drop(columns=['k', 'env', 'task', 'social_welfare', 'seed'], inplace=True)
        group.drop_duplicates(inplace=True)
        group['bold'] = False
        for algorithm in group['algorithm'].unique():
            group.loc[group['algorithm'] == algorithm, 'bold'] = t[algorithm] >= p_value
        return group

    df['env'] = df['env'].apply(lambda x: x.split('-')[0])

    df_bold = df.groupby(['k', 'env', 'task']).apply(_ttest)
    df_bold = df_bold.reset_index().set_index(['k', 'env', 'task', 'algorithm'])['bold']
    df = df.groupby(['k', 'env', 'task', 'algorithm'])['social_welfare'].mean().round(1)
    df = df.astype(str)
    df[df_bold] = df[df_bold].apply(lambda x: f'\\textbf{{{x}}}')
    df = df.reset_index()

    df_ = df[df['task'] == 'speed']
    df_ = df_.pivot_table(index=['env', 'algorithm'], columns=['k'], values='social_welfare', aggfunc=lambda x: ' '.join(x))
    df_ = df_[[1, 2, 5, 10, 50]]
    df_ = df_.loc[[('HalfCheetah', 'EM'),
                   ('HalfCheetah', 'end-to-end'),
                   ('HalfCheetah', 'cluster'),
                   ('HalfCheetah', 'random'),
                   ('Hopper', 'EM'),
                   ('Hopper', 'end-to-end'),
                   ('Hopper', 'cluster'),
                   ('Hopper', 'random'),
                   ('Walker2d', 'EM'),
                   ('Walker2d', 'end-to-end'),
                   ('Walker2d', 'cluster'),
                   ('Walker2d', 'random'),
                   ('Ant', 'EM'),
                   ('Ant', 'end-to-end'),
                   ('Ant', 'cluster'),
                   ('Ant', 'random'),
                   ]]
    print(df_)
    df_.to_csv('df_mujoco.csv')

    df_ = df[df['task'] == 'resource']
    df_ = df_.pivot_table(index='algorithm', columns=['k'], values='social_welfare', aggfunc=lambda x: ' '.join(x))
    df_ = df_.loc[['EM', 'end-to-end', 'cluster'], [1, 2, 3, 5, 10, 25]]
    print(df_)
    df_.to_csv('df_resource.csv')


def plot_from_csv():
    tag = TAG_SW
    PIC_DIR = PATH + tag[:-4] + '/'
    if not os.path.exists(PIC_DIR): os.mkdir(PIC_DIR)

    df = pd.read_csv(PATH + tag)
    df = df[df['algorithm'].isin(NAMES.keys())]
    if tag.startswith('n_flips'):
        df = df[df['algorithm'].isin(['em', 'diff'])]

    for env in df['env'].unique():
        for task in df['task'].unique():
            df_ = df[(df['env'] == env) & (df['task'] == task)].drop(columns=['seed', 'env', 'task'])
            df_mean = df_.groupby('algorithm').mean().T
            if tag == 'n_flips.csv':
                df_min = df_.groupby('algorithm').min().T
                df_max = df_.groupby('algorithm').max().T
            else:
                df_std = df_.groupby('algorithm').sem().T
                df_min = df_mean - df_std
                df_max = df_mean + df_std

            inds = np.arange(0, 1000, 50) + 50
            xticks = np.arange(0, 1001, 200)
            if env.startswith('Ant'):
                inds *= 2
                xticks *= 2

            ax = None
            for algorithm in df_['algorithm'].unique():
                ax = plot_config.plot_with_intervals(
                    df_min[algorithm].values,
                    df_max[algorithm].values,
                    df_mean[algorithm].values,
                    inds=inds,
                    label=algorithm,
                    c=COLORS[algorithm],
                    lw=2,
                    linestyle=LINESTYLES[algorithm],
                    ax=ax)

            ax.legend([], [], frameon=False)
            ylim = None
            if tag == 'n_flips.csv':
                ylim = (0, N_AGENTS)
            plot_config.process_axes([ax], xticks=xticks, ylim=ylim)
            plot_config.save(PIC_DIR + env + '_' + task + '.png')
            ax.clear()


def legend_from_csv():
    colors = [[k, v] for k, v in COLORS.items()]
    f = lambda m, c: plt.plot([], [], color=c, ls=m)[0]
    handles = [f(LINESTYLES[colors[i][0]], colors[i][1]) for i in range(len(colors))]
    labels = [NAMES[colors[i][0]] for i in range(len(colors))]
    legend = plt.legend(handles, labels, ncol=4, loc=3, framealpha=1, frameon=True)

    def export_legend(legend, filename="legend.png"):
        fig = legend.figure
        fig.canvas.draw()
        legend.get_window_extent()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi=500, bbox_inches=bbox)

    plt.grid(False)
    export_legend(legend, 'legend.png')
    plt.grid(True)


if __name__ == '__main__':
    os.chdir('../runs/')

    N_AGENTS, N_POLICIES = 100, 5
    PATH = f'{N_AGENTS}_agents_{N_POLICIES}_policies/csv/'
    TAG_SW = 'social_welfare.csv'
    NAMES = {'em': 'EM', 'diff': 'end-to-end', 'cluster': 'cluster', 'random': 'random'}
    COLORS = {'em': Neon.BLUE.norm, 'diff': Neon.GREEN.norm, 'cluster': Neon.PURPLE.norm, 'random': Neon.RED.norm}
    LINESTYLES = {'em': '-', 'diff': '-', 'cluster': '--', 'random': ':'}

    # legend_from_csv()

    sns.set_theme()
    # plot_from_csv()  # learning curves for a particular N_AGENTS, N_POLICIES
    table_and_plot_sw_from_csv()  # summarized results for all folders {N_AGENTS}_agents_{N_POLICIES}_policies/
