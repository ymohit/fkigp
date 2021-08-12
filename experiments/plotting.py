import numpy as np
from fkigp.configs import GridSizeFunc
from matplotlib import container
from matplotlib import pyplot as plt
from collections import defaultdict


attributes = ['#iters', "time/iter", 'total', "time-preprocess"]
methods = ['cg', 'scg', 'fcg']


def plot_error_vs_time(dumps, x_logscale=False, y_logscale=False, x_label=None, y_label=None, save_path=None):
    MS = 10
    methods = ['kissgp', 'gsgp']

    use_seconds = True

    fig, axs = plt.subplots(1, figsize=(10, 8))
    ax = axs

    ms = []
    for dump_key in dumps.keys():
        ms += dump_key[0],
    ms = np.array(sorted(ms))

    # Collecting inference error
    Ys = defaultdict(list)
    attribute = 'smae'
    for dump_keys, dump_value in dumps.items():
        if dump_keys[-1] == 'kissgp':
            Ys[(dump_keys[0], 'kissgp')] += dump_value.get_att(attribute),
        else:
            Ys[(dump_keys[0], 'gsgp')] += dump_value.get_att(attribute),

    # Collecting inference time
    Xs = defaultdict(list)
    attribute = 'total'
    for dump_keys, dump_value in dumps.items():
        if dump_keys[-1] == 'kissgp':
            Xs[(dump_keys[0], 'kissgp')] += dump_value.get_att(attribute),
        else:
            Xs[(dump_keys[0], 'gsgp')] += dump_value.get_att(attribute),

    # Plotting data for grid_size_f for all methods seperately
    for method in methods:

        # Collecting data for specific method
        Ys_method = dict()
        for key, value in Ys.items():
            if key[1] != method:
                continue
            Ys_method[key[0]] = value

        # Computed errors
        Ys_mean = np.array([np.mean(Ys_method[_ms]) for _ms in ms])
        Ys_std = np.array([np.std(Ys_method[_ms]) for _ms in ms])

        # Collecting data for specific method
        Xs_method = dict()
        for key, value in Xs.items():
            if key[1] != method:
                continue
            Xs_method[key[0]] = value
        Xs_mean = np.array([np.mean(Xs_method[_ms]) for _ms in ms])
        Xs_std = (1.96 / np.sqrt(20)) * np.array([np.std(Xs_method[_ms]) for _ms in ms])

        if use_seconds:
            Xs_mean = Xs_mean / 1e3
            Xs_std = Xs_std / 1e3

        if method == "kissgp":
            lc = "b"
            fmt = lc + "-"
            marker = "P"
        elif method == "gsgp":
            lc = "k"
            fmt = lc + '-'  # dotted line style
            marker = "^"

        ax.errorbar(Xs_mean, Ys_mean, yerr=Ys_std, xerr=Xs_std, fmt=fmt, marker=marker,
                    markersize=MS, label="SKI" if method == 'kissgp' else "GSGP",
                    elinewidth=3, linewidth=3,
                    uplims=True, lolims=True, xuplims=True, xlolims=True)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, fontsize=25)

    if x_logscale:
        ax.set_xscale('log')
    if y_logscale:
        ax.set_yscale('log')

    if x_label is not None:
        plt.xlabel(x_label, fontsize=35)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=35)

    params = {'legend.fontsize': 30, 'axes.linewidth': 5}
    plt.rcParams.update(params)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=2400)


def plot_attribute_gs(dumps, attribute='#iters', x_logscale=False, y_logscale=False, print_values=False,
                   x_label=None, y_label=None, save_path=None, set_zero_min_y_limit=False,
                   yticks_vals=None, max_limit=None, set_y_limit=None, show_legends=False, set_y_limit_upper=None):
    MS = 12
    methods = ['kissgp', 'gsgp']

    use_seconds = True

    fig, axs = plt.subplots(1, figsize=(10, 8))
    ax = axs

    Xs = []
    for dump_key in dumps.keys():
        Xs += dump_key[0],
    Xs = np.array(sorted(list(set(Xs))))
    M_values = Xs

    correct_order = np.argsort(M_values)
    M_values = np.array([M_values[i] for i in correct_order])
    Xs = np.array([Xs[i] for i in correct_order])

    # Plotting data values for each method seperately
    Ys = defaultdict(list)
    for dump_keys, dump_value in dumps.items():

        if dump_keys[-1] == 'kissgp':
            Ys[(dump_keys[0], 'kissgp')] += dump_value.get_att(attribute),
        else:
            Ys[(dump_keys[0], 'gsgp')] += dump_value.get_att(attribute),

    # Plotting data for grid_size_f for all methods seperately
    for method in methods:

        # Collecting data for specific method
        Ys_method = dict()
        for key, value in Ys.items():
            if key[1] != method:
                continue
            Ys_method[key[0]] = value

        Ys_mean = np.array([np.mean(Ys_method[_xs]) for _xs in Xs])
        Ys_std = (1.96 / np.sqrt(30)) * np.array([np.std(Ys_method[_xs]) for _xs in Xs])

        if use_seconds and ("time" in attribute or "total" == attribute):
            Ys_mean = Ys_mean / 1e3
            Ys_std = Ys_std / 1e3

        if method == "kissgp":
            lc = "b"
            fmt = lc + "-"
            marker = "P"
        elif method == "gsgp":
            lc = "k"
            fmt = lc + '-'  # dotted line style
            marker = "^"

        ax.errorbar(M_values, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker,
                    markersize=MS, label="SKI" if method == 'kissgp' else "GSGP", elinewidth=3,
                    linewidth=3, uplims=True, lolims=True, )

        if print_values:
            print("method:", method)
            for _m_value, _Xs_mean, _Xs_std in zip(M_values, Ys_mean, Ys_std):
                print("M: ", _m_value, "inf_time: ", _Xs_mean, " +/- ", _Xs_std)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    if show_legends:
        ax.legend(handles, labels, fontsize=28)  # ,  loc='lower right')

    if x_logscale:
        ax.set_xscale('log')
    if y_logscale:
        ax.set_yscale('log')

    if x_label is not None:
        plt.xlabel(x_label, fontsize=35)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=35)

    if yticks_vals is not None:
        plt.yticks(yticks_vals)

    if set_y_limit is not None:
        if not y_logscale:
            plt.gca().set_ylim(bottom=set_y_limit)
            if set_y_limit_upper is not None:
                plt.gca().set_ylim(top=set_y_limit_upper)
        else:
            plt.gca().set_ylim(bottom=1e0)
            plt.gca().set_ylim(top=1e3)

    params = {'legend.fontsize': 30, 'axes.linewidth': 5}
    plt.rcParams.update(params)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=2400)


def plot_attribute(dumps, Gsfs, ignore_methods=[], attribute='#iters', log_scale=True,
                   all_together=False, x_label=None, y_label=None, save_path=None, x_scale_log=False,
                   show_legends=True, methods=['cg', 'fcg']):
    MS = 3
    use_seconds = True

    ax = None
    for gsf in Gsfs:

        if ax is None:
            fig, axs = plt.subplots(1, figsize=(15, 10))
            ax = axs
        elif not all_together:
            fig, axs = plt.subplots(1, figsize=(15, 10))
            ax = axs

        Xs = []
        for dump_key in dumps.keys():
            Xs += dump_key[1],
        Xs = np.array(sorted(Xs))

        Ys = defaultdict(list)

        for dump_keys, dump_value in dumps.items():
            if dump_keys[0] != gsf:
                continue

            if dump_keys[-1] == 'cg':
                Ys[(dump_keys[1], 'cg')] += dump_value.get_att(attribute),
            else:
                Ys[(dump_keys[1], 'fcg')] += dump_value.get_att(attribute),

        # Plotting data for grid_size_f for all methods seperately
        for method in methods:

            if method in ignore_methods:
                continue

                # Collecting data for specific method
            Ys_method = dict()
            for key, value in Ys.items():
                if key[1] != method:
                    continue
                Ys_method[key[0]] = value

            Ys_mean = np.array([np.mean(Ys_method[_xs]) for _xs in Xs])
            Ys_std = (1.96 / np.sqrt(8)) * np.array([np.std(Ys_method[_xs]) for _xs in Xs])

            # temporary mapping
            m_map = {'cg': "kissgp", 'fcg': "gsgp"}
            fmt, marker = get_fmt(m_map[method], gsf)

            if use_seconds and ("time" in attribute or "total" == attribute):
                Ys_mean = Ys_mean / 1e3
                Ys_std = Ys_std / 1e3

            if gsf == GridSizeFunc.SQRT_N:
                if m_map[method] == 'kissgp':
                    ax.errorbar(Xs, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker, markersize=MS,
                                label=r'$SKI \,\,\, m=\sqrt{n}$', elinewidth=3, linewidth=3,
                                uplims=True, lolims=True)
                else:
                    ax.errorbar(Xs, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker, markersize=MS,
                                label=r'$GSGP \,\,\, m=\sqrt{n}$', elinewidth=3, linewidth=3,
                                uplims=True, lolims=True)
            elif gsf == GridSizeFunc.N_BY_16:
                if m_map[method] == 'kissgp':
                    ax.errorbar(Xs, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker, markersize=MS,
                                label=r'$SKI \,\,\, m=\frac{n}{16}$', elinewidth=3, linewidth=3,
                                uplims=True, lolims=True)
                else:
                    ax.errorbar(Xs, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker, markersize=MS,
                                label=r'$GSGP \,\,\, m=\frac{n}{16}$', elinewidth=3, linewidth=3,
                                uplims=True, lolims=True)
            elif gsf == GridSizeFunc.N:

                if m_map[method] == 'kissgp':
                    ax.errorbar(Xs, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker, markersize=MS,
                                label=r'$SKI \,\,\, m=n$', elinewidth=3, linewidth=4,
                                uplims=True, lolims=True)
                else:
                    ax.errorbar(Xs, Ys_mean, yerr=Ys_std, fmt=fmt, marker=marker, markersize=MS,
                                label=r'$GSGP \,\,\, m=n$', elinewidth=3, linewidth=4,
                                uplims=True, lolims=True)

            else:
                raise NotImplementedError

        if not all_together:
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
            ax.legend(handles, labels)

            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            if ("time" in attribute or "total" == attribute):
                if use_seconds:
                    plt.ylabel("Inference time (seconds)")
                else:
                    plt.ylabel("Inference time (miliseconds)")

            plt.xlabel("Number of samples")
            plt.show()

    if all_together:
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
        if show_legends:
            ax.legend(handles, labels, fontsize=30)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        if x_scale_log:
            ax.set_xscale('log')

        params = {'legend.fontsize': 15, 'axes.linewidth': 2}
        plt.rcParams.update(params)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Number of Samples" if x_label is None else x_label, fontsize=25)
        plt.ylabel("Inference Time" if y_label is None else y_label, fontsize=30)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=2500)
        plt.show()


def M_rep(tt):
    if tt == GridSizeFunc.N:
        return "M=N"
    elif tt == GridSizeFunc.N_BY_2:
        return "M=N/2"
    elif tt == GridSizeFunc.N_BY_4:
        return "M=N/4"
    elif tt == GridSizeFunc.N_BY_16:
        return r'M=N/16'
    elif tt == GridSizeFunc.SQRT_N:
        return r'M=\sqrt(N)'
    elif tt == GridSizeFunc.CONST_N:
        return "M=Const"
    else:
        raise NotImplementedError


def get_fmt(method, tt=GridSizeFunc.N):

    if tt == GridSizeFunc.N:
        marker = "H"
        lc = "b"
    elif tt == GridSizeFunc.N_BY_2:
        marker = "H"
        lc = "b"
    elif tt == GridSizeFunc.N_BY_2:
        marker = "H"
        lc = "b"
    elif tt == GridSizeFunc.N_BY_4:
        marker = "p"
        lc = "r"
    elif tt == GridSizeFunc.N_BY_16:
        marker = "s"
        lc = "g"
    elif tt == GridSizeFunc.SQRT_N:
        marker = "^"
        lc = "c"
    elif tt == GridSizeFunc.CONST_N:
        marker = "P"
        lc = "k"
    else:
        raise NotImplementedError

    if method == "cg":
        fmt = lc + "-."
    elif method == "kissgp":
        fmt = lc + "-"
    elif method == "gsgp":
        fmt = lc + '-.'  # dotted line style
    else:
        raise NotImplementedError

    return fmt, marker


def comp_arr(a1, a2):
    return np.max(np.abs(a1 - a2))



