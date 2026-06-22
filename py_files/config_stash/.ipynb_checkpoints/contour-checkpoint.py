# HELPER FUNCTIONS TO MAKE CORNER PLOTS
def get_mean(dist):
    return np.mean(dist, axis=0)

def get_range(dist):
    range_arr = np.array([np.min(dist, axis=0),
                          np.max(dist, axis=0)]).T
    final = [tuple(i) for i in range_arr]
    return final

CORNER_KWARGS = dict(
    smooth = 0.9,
    label_kwargs=dict(fontsize=40),
    
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    show_titles=True,
    max_n_ticks=3,
    bins=20
)

def make_contour(list_of_dists, labels, categories, colors, range_for_bin=False):
    cat_to_col = dict(zip(categories, colors))
    legend_elements = []
    for cat in categories:
        legend_elements.append(Patch(facecolor="w", edgecolor=cat_to_col[cat], label=cat))
    
    exemplar_dist = list_of_dists[0]
    if range_for_bin:
        bin_range = get_range(exemplar_dist)
    else:
        bin_range=None
        
    fig,ax = plt.subplots(exemplar_dist.shape[1],exemplar_dist.shape[1],figsize=(20,20))
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20)
    CORNER_KWARGS['title_kwargs'].update(color=colors[-1])
    i = 0
    alpha = 0.3
    for dist in list_of_dists:
        means = get_mean(dist)
        # new_labels = []
        # print(labels)
        # for l in range(len(labels)):
        #     label = labels[l]
        #     new_labels.append(label + ' = ' + str(np.round(means[l])))
        corner.corner(
            data=dist,
            labels=labels,
            color=colors[i],
            truths= means,
            hist_kwargs=dict(density=True,lw=3, color=colors[i], range=bin_range),
            levels=[0.68,0.95],
            truth_color=colors[i],
            **CORNER_KWARGS,
            title_fmt = '.1f',
            fig=fig,
            alpha=alpha
            );
        i+=1
        alpha = alpha + len(list_of_dists)/10
        alpha = max(1, alpha)
    fig.legend(handles=legend_elements, frameon=False, ncol=1 ,loc=(0.6, 0.8), fontsize=30)
    return fig