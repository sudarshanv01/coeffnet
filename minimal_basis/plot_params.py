def get_plot_params():
    """Create the plot parameters used in the plotting
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    COLOR = "k"
    plt.rcParams["text.color"] = COLOR
    plt.rcParams["axes.labelcolor"] = COLOR
    plt.rcParams["xtick.color"] = COLOR
    plt.rcParams["ytick.color"] = COLOR
