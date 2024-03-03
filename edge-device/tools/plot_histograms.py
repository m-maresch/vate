import itertools

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

frq_faster_rcnn_sync, edges_faster_rcnn_sync = (
    np.array([28, 139, 325, 426, 262, 146, 9, 0, 2, 2, 11, 10, 19,
              33, 39, 61, 109, 145, 108, 111, 156, 175, 171, 250, 200, 224,
              238, 255, 222, 224, 245, 312, 386, 623, 227, 742]),
    np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
              13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
              25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., np.inf])
)

frq_faster_rcnn_async, edges_faster_rcnn_async = (
    np.array([21, 16, 20, 26, 31, 17, 21, 22, 16, 40, 57,
              70, 94, 87, 134, 153, 223, 201, 173, 142, 186, 212,
              248, 322, 383, 465, 405, 476, 1080, 494, 800]),
    np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
              13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
              25., 26., 27., 28., 29., 30., np.inf])
)
frq_swin_transformer_sync, edges_swin_transformer_sync = (
    np.array([33, 167, 328, 391, 275, 128, 12, 1, 0, 3, 1, 8, 9,
              32, 24, 49, 80, 150, 134, 98, 132, 154, 160, 202,
              236, 270, 224, 188, 197, 261, 235, 318, 420, 692, 268, 755]),
    np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
              13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
              24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., np.inf])
)

frq_swin_transformer_async, edges_swin_transformer_async = (
    np.array([18, 8, 4, 9, 14, 10, 12, 26, 34, 45, 65,
              72, 120, 129, 159, 210, 218, 199, 150, 185, 178,
              197, 241, 248, 317, 408, 410, 482, 1156, 476, 835]),
    np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
              13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
              24., np.inf])
)

fig, (
    (ax_faster_rcnn_sync, ax_swin_transformer_sync),
    (ax_faster_rcnn_async, ax_swin_transformer_async),
    (ax_faster_rcnn_diff, ax_swin_transformer_diff),
    (ax_std, _),
) = (plt.subplots(4, 2, layout="constrained"))


def preprocess(frq, edges):
    last = sum(frq[24:])
    frq[24] = last

    frq = frq[:25]
    frq = np.append(frq, [0])

    edges = edges[:25]
    edges = np.append(edges, [edges[-1] + 1])

    return frq, edges


def create_histogram(ax, frq, edges):
    bars = ax.bar(edges, frq, width=1, edgecolor="white", align="edge")
    bars = bars[:-1]

    bars[-1].set_facecolor("purple")

    labels = itertools.chain(map(int, edges[:-1]), ["inf"])
    ax.set_xticks(edges, labels=labels)

    ax.set_xlabel('FPS')


def var_std(n, bins):
    mids = 0.5 * (bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    var = np.average((mids - mean) ** 2, weights=n)
    std = np.sqrt(var)
    return var, std


def create_var_std_plot(ax, sync_frq, async_frq, edges):
    measures = ("Variance", "Standard deviation")

    sync_var, sync_std = var_std(sync_frq[:-1], edges)
    async_var, async_std = var_std(async_frq[:-1], edges)

    modes = {
        'Synchronous mode': (sync_var, sync_std),
        'Asynchronous mode': (async_var, async_std),
    }

    x = np.arange(len(measures))
    width = 0.4

    ax.set_prop_cycle(color=['firebrick', 'salmon'])

    multiplier = 0
    for mode, measurement in modes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=mode)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xticks(x + width / 2, measures)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel('FPS')


def create_std_plot(ax, sync_frq_1, async_frq_1, edges_1, label_1, sync_frq_2, async_frq_2, edges_2, label_2):
    _, sync_std_1 = var_std(sync_frq_1[:-1], edges_1)
    _, async_std_1 = var_std(async_frq_1[:-1], edges_1)

    _, sync_std_2 = var_std(sync_frq_2[:-1], edges_2)
    _, async_std_2 = var_std(async_frq_2[:-1], edges_2)

    modes = {
        'Synchronous mode': (sync_std_1, sync_std_2),
        'Asynchronous mode': (async_std_1, async_std_2),
    }

    x = np.arange(2)
    width = 0.4

    ax.set_prop_cycle(color=['firebrick', 'salmon'])

    multiplier = 0
    for mode, measurement in modes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=mode)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xticks(x + width / 2, [label_1, label_2])
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 15)
    ax.set_ylabel('FPS')


ax_faster_rcnn_sync.set_title('Synchronous mode histogram - Faster RCNN')
ax_faster_rcnn_async.set_title('Asynchronous mode histogram - Faster RCNN')
ax_faster_rcnn_diff.set_title('Comparison of modes - Faster RCNN')

ax_swin_transformer_sync.set_title('Synchronous mode histogram - Swin transformer')
ax_swin_transformer_async.set_title('Asynchronous mode histogram - Swin transformer')
ax_swin_transformer_diff.set_title('Comparison of modes - Swin transformer')

ax_std.set_title('FPS Std Dev by combination of mode and cloud model')

frq_faster_rcnn_sync, edges_faster_rcnn_sync = preprocess(frq_faster_rcnn_sync, edges_faster_rcnn_sync)
frq_faster_rcnn_async, edges_faster_rcnn_async = preprocess(frq_faster_rcnn_async, edges_faster_rcnn_async)
frq_swin_transformer_sync, edges_swin_transformer_sync = (
    preprocess(frq_swin_transformer_sync, edges_swin_transformer_sync))
frq_swin_transformer_async, edges_swin_transformer_async = (
    preprocess(frq_swin_transformer_async, edges_swin_transformer_async))

create_histogram(ax_faster_rcnn_sync, frq_faster_rcnn_sync, edges_faster_rcnn_sync)
create_histogram(ax_faster_rcnn_async, frq_faster_rcnn_async, edges_faster_rcnn_async)
create_histogram(ax_swin_transformer_sync, frq_swin_transformer_sync, edges_swin_transformer_sync)
create_histogram(ax_swin_transformer_async, frq_swin_transformer_async, edges_swin_transformer_async)

create_var_std_plot(ax_faster_rcnn_diff, frq_faster_rcnn_sync, frq_faster_rcnn_async, edges_faster_rcnn_sync)
create_var_std_plot(ax_swin_transformer_diff, frq_swin_transformer_sync, frq_swin_transformer_async,
                    edges_swin_transformer_sync)

create_std_plot(ax_std, frq_faster_rcnn_sync, frq_faster_rcnn_async, edges_faster_rcnn_sync, "Faster RCNN",
                frq_swin_transformer_sync, frq_swin_transformer_async, edges_swin_transformer_sync, "Swin transformer")

engine = fig.get_layout_engine()
engine.set(rect=(0.1, 0.1, 0.8, 0.8))

plt.show()
