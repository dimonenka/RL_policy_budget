import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-pastel')


def draw_paths(frame, verts, path='./', filename='gym_paths.png', lw=2.5, ms=12):
    plt.style.use('seaborn')
    colors = ['magenta', 'blue', 'red', 'cyan', 'purple']
    fig, ax = plt.subplots(figsize=(frame.shape[1] / 36, frame.shape[0] / 36), dpi=100)
    plt.imshow(frame)
    plt.axis('off')

    intervals = np.linspace(0.2, 0.2 + 0.06 * len(verts), len(verts), endpoint=True)
    for i, vert in enumerate(verts):
        lam = lambda lst: [(x + intervals[i]) * 64 for x in lst]
        ys, xs = zip(*lam(vert))
        ys, xs = list(ys), list(xs)

        offset = (intervals[-1] - intervals[0] + intervals[1] - intervals[0]) * 64 if len(intervals) > 1 else 0.1 * 64
        for j in range(1, len(xs)-1):
            if xs[j-1] == xs[j+1] and ys[j-1] == ys[j+1]:
                xs.insert(j+1, xs[j] + offset), ys.insert(j+1, ys[j] + offset)
                for jj in range(j+2, len(xs)):
                    xs[jj] += offset
                    ys[jj] += offset
                break

        if xs[1] > xs[0]:
            xs[0] = (xs[0] // 64 + 1.1) * 64
            ax.plot(xs[:1], ys[:1], '>', lw=0, ms=ms, color=colors[i % len(colors)])
        elif xs[1] < xs[0]:
            xs[0] = (xs[0] // 64 - 0.1) * 64
            ax.plot(xs[:1], ys[:1], '<', lw=0, ms=ms, color=colors[i % len(colors)])
        elif ys[1] > ys[0]:
            ys[0] = (ys[0] // 64 + 1.1) * 64
            ax.plot(xs[:1], ys[:1], 'v', lw=0, ms=ms, color=colors[i % len(colors)])
        elif ys[1] < ys[0]:
            ys[0] = (ys[0] // 64 - 0.1) * 64
            ax.plot(xs[:1], ys[:1], '^', lw=0, ms=ms, color=colors[i % len(colors)])

        if xs[-2] > xs[-1]:
            xs[-1] = (xs[-1] // 64 + 1.1) * 64
            ax.plot(xs[-1:], ys[-1:], '<', lw=0, ms=ms, color=colors[i % len(colors)])
        elif xs[-2] < xs[-1]:
            xs[-1] = (xs[-1] // 64 - 0.1) * 64
            ax.plot(xs[-1:], ys[-1:], '>', lw=0, ms=ms, color=colors[i % len(colors)])
        elif ys[-2] > ys[-1]:
            ys[-1] = (ys[-1] // 64 + 1.1) * 64
            ax.plot(xs[-1:], ys[-1:], '^', lw=0, ms=ms, color=colors[i % len(colors)])
        elif ys[-2] < ys[-1]:
            ys[-1] = (ys[-1] // 64 - 0.1) * 64
            ax.plot(xs[-1:], ys[-1:], 'v', lw=0, ms=ms, color=colors[i % len(colors)])

        ax.plot(xs, ys, '--', lw=lw, dashes=(4, 2), color=colors[i % len(colors)])

    plt.savefig(path+filename, bbox_inches='tight', dpi=100)
    plt.close()


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    """
    From https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 36, frames[0].shape[0] / 36), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1000)
    anim.save(path + filename, fps=5)
    plt.close()


def clusters_histogram(assignment_mask, targets, low, high, path='./', filename='clusters_histogram.png', n_ticks=2):
    if not os.path.exists(path):
        os.makedirs(path)

    bins = np.linspace(low, high, num=26, endpoint=True)
    idx = assignment_mask.mean(1).argsort()
    assignment_mask = assignment_mask[idx]
    # for mask in assignment_mask:
    #     plt.hist(targets, bins=bins, weights=mask, alpha=0.6, edgecolor='k', density=True)
    targets = np.expand_dims(targets, 0)
    targets = np.repeat(targets, assignment_mask.shape[0], axis=0)
    plt.hist(targets.T, bins=bins, weights=assignment_mask.T, alpha=0.75, edgecolor='k', density=True, stacked=True)
    plt.xlim(low, high)
    ticks = np.linspace(low, high, n_ticks)
    plt.xticks(ticks, fontsize=20)
    plt.yticks([])
    plt.xlabel('target speed', fontsize=20)
    plt.savefig(path + filename, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # assignment_mask = np.array([[0, 1, 0, 0, 1], [1, 0, .4, 0, 0], [0, 0, .6, 1, 0], [0, 0, .6, 1, 0], [0, 0, .6, 1, 0]])
    # targets = np.array([0.7, 0.2, 0.8, 0.9, 0.1])
    assignment_mask = np.random.randint(2, size=100)
    targets = np.random.uniform(0, 0.6, size=100)
    targets[assignment_mask.astype(bool)] = 1 - targets[assignment_mask.astype(bool)]
    assignment_mask = np.expand_dims(assignment_mask, 0)
    assignment_mask = np.concatenate([assignment_mask, 1-assignment_mask])
    low, high = 0, 1
    clusters_histogram(assignment_mask, targets, low, high, n_ticks=6)
