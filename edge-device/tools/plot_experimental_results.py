import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})

cloud_models = ("Faster RCNN", "Swin transformer")

faster_rcnn_async = 0.053
faster_rcnn_sync = 0.0812
faster_rcnn_sync_original_fusion = 0.0662

swin_transformer_async = 0.0607
swin_transformer_sync = 0.0978
swin_transformer_sync_original_fusion = 0.0767

modes = {
    'Synchronous mode': (faster_rcnn_sync, swin_transformer_sync),
    'Asynchronous mode': (faster_rcnn_async, swin_transformer_async),
}

fusions = {
    'Original edge-cloud fusion': (faster_rcnn_sync_original_fusion, swin_transformer_sync_original_fusion),
    'Enhanced edge-cloud fusion': (faster_rcnn_sync, swin_transformer_sync),
}

fig, (accuracy_ax, fusion_ax) = plt.subplots(1, 2)

x = np.arange(len(cloud_models))
width = 0.4


def create_accuracy_plot(ax):
    ax.set_prop_cycle(color=['firebrick', 'salmon'])

    multiplier = 0
    for mode, measurement in modes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=mode)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('mAP@50')
    ax.set_title('Accuracy by combination of mode and cloud model')
    ax.set_xticks(x + width / 2, cloud_models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 0.12)


def create_fusion_plot(ax):
    ax.set_prop_cycle(color=['lightsteelblue', 'royalblue'])

    multiplier = 0
    for fusion, measurement in fusions.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=fusion)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('mAP@50')
    ax.set_title('Comparison of edge-cloud fusion algorithms')
    ax.set_xticks(x + width / 2, cloud_models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 0.12)


create_accuracy_plot(accuracy_ax)
create_fusion_plot(fusion_ax)

plt.show()
