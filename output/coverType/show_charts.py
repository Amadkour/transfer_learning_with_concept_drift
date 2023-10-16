
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import rcParams
names = ["score"]
# Set plot params
rcParams["font.family"] = "monospace"
colors = [(0, 0.7, 0), (0, 0, 0.9),  (0.9, 0, 0)]
ls = ["--", "-", ":"]
lw = [1, 1,  1]
methods = ["AW", "weights", "ADWIN"]
metrics = ["Balanced accuracy", "G-mean", 'f1_score', "precision", "recall"]
clfs = ["Cover Type"]


# print(scores.shape)

def plot_runs(
        clfs, metrics, selected_scores, methods, mean_scores, dependency, what
):
    fig = plt.figure(figsize=(4.5, 3))
    ax = plt.axes()
    for z, (value, label, mean) in enumerate(
            zip(selected_scores, methods, mean_scores)
    ):
        label = "\n{0:.3f}".format(mean)

        val = gaussian_filter1d(value, sigma=3, mode="nearest")
        # val = medfilt(value, 3)
        print(value)
        # plt.plot(value, label=label, c=colors[z], ls=ls[z])

        plt.plot(val, label=label, c=colors[z], ls=ls[z], lw=lw[z])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(
        loc=8,
        bbox_to_anchor=(0.5, 0.97),
        fancybox=False,
        shadow=True,
        ncol=5,
        fontsize=7,
        frameon=False,
    )

    plt.grid(ls=":", c=(0.7, 0.7, 0.7))
    plt.xlim(0, 200)
    axx = plt.gca()
    axx.spines["right"].set_visible(False)
    axx.spines["top"].set_visible(False)

    # plt.title(
    #     "%s %s\n%s" % (clfs[j], dependency[k][:], metrics[i]),
    #     fontfamily="serif",
    #     y=1.04,
    #     fontsize=8,
    # )
    plt.ylim(0.0, 1.0)
    plt.xticks(fontfamily="serif")
    plt.yticks(fontfamily="serif")
    plt.ylabel(metrics[i], fontfamily="serif", fontsize=6)
    plt.xlabel("chunks", fontfamily="serif", fontsize=6)
    plt.tight_layout()
    if metrics[i] == "G-mean":
        # plt.show()
        plt.savefig(
            "%s_%s_%s.png" % (
                clfs[j], metrics[i], dependency),
            bbox_inches='tight', dpi=2000, pad_inches=0.0)

    plt.close()


def plot_radars(
        methods, metrics, table, classifier_name, parameter_name, what
):
    """
    Strach.
    """
    columns = ["group"] + methods
    df = pd.DataFrame(columns=columns)
    for i in range(len(table)):
        df.loc[i] = table[i]
    df = pd.DataFrame()
    df["group"] = methods
    for i in range(len(metrics)):
        df[table[i][0]] = table[i][1:]
    groups = list(df)[1:]

    N = len(groups)

    # nie ma nic wspolnego z plotem, zapisywanie do txt texa
    # print(df.to_latex(index=False), file=open("tables/%s.tex" % (filename), "w"))

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # No shitty border
    ax.spines["polar"].set_visible(False)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:len(metrics)], metrics)

    # Adding plots
    for i in range(len(methods)):
        values = df.loc[i].drop("group").values.flatten().tolist()
        values += values[:1]
        values = [float(i) for i in values]
        ax.plot(
            angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
        )

    # Add legend
    plt.legend(
        loc="lower center",
        ncol=3,
        columnspacing=1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.25),
        fontsize=6,
    )

    # Add a grid
    plt.grid(ls=":", c=(0.7, 0.7, 0.7))

    # Add a title
    plt.title("%s %s" % (clfs[j], parameter_name), size=8, y=1.08, fontfamily="serif")
    plt.tight_layout()

    # Draw labels
    a = np.linspace(0, 1, 6)
    plt.yticks(a[1:], ["%.1f" % f for f in a[1:]], fontsize=6, rotation=90)
    plt.ylim(0.0, 1.0)
    plt.gcf().set_size_inches(4, 3.5)
    plt.gcf().canvas.draw()
    angles = np.rad2deg(angles)

    ax.set_rlabel_position((angles[0] + angles[1]) / 2)

    har = [(a >= 90) * (a <= 270) for a in angles]

    for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
        x, y = label.get_position()
        print(label, angle)
        lab = ax.text(
            x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
        )
        lab.set_rotation(angle)

        if har[z]:
            lab.set_rotation(180 - angle)
        else:
            lab.set_rotation(-angle)
        lab.set_verticalalignment("center")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
        x, y = label.get_position()
        # print(label, angle)
        lab = ax.text(
            x,
            y,
            label.get_text(),
            transform=label.get_transform(),
            fontsize=4,
            c=(0.7, 0.7, 0.7),
        )
        lab.set_rotation(-(angles[0] + angles[1]) / 2)

        lab.set_verticalalignment("bottom")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # plt.show()
    plt.savefig(
        "radar_%s_%s.png" % (classifier_name, parameter_name),
        bbox_inches='tight', dpi=2000, pad_inches=0.0)
    plt.close()


for j, name in enumerate(names):
    print("\n---\n--- %s\n---\n" % (name))
    scores = np.load("%s.npy" % name)

    for j, clf in enumerate(clfs):
        print("\n###\n### %s\n###\n" % (clf))
        for i, metric in enumerate(metrics):
            print("\n---\n--- %s\n---\n" % (metric))
            sub_scores = scores[:, :,i]
            reduced_scores = np.mean(sub_scores, axis=0)

            plot_runs(clfs, metrics, sub_scores, methods, reduced_scores, name,"")

            table = []
            header = ["LN"] + methods

for j, name in enumerate(names):
    # print("\n---\n--- %s\n---\n" % (name))
    scores = np.load("%s.npy" % name)

    # Metryka
    # drifttype, LABELNOISE, METHOD, CHUNK, METRYKA
    print(scores[0][0])
    reduced_scores2 = np.mean(scores, axis=1)
    reduced_scores = np.mean(reduced_scores2, axis=1)
    table = []
    header = ["Metric"] + clfs
    # print(reduced_scores2)
    for i, metric in enumerate(metrics):
        # METHOD, CHUNK, Metryka
        selected_scores = np.transpose(reduced_scores2)[i]

        table.append([metric] + ["%.3f" % score for score in selected_scores])
    print(table)
    plot_radars(methods, metrics, table, clfs[0], name, name)