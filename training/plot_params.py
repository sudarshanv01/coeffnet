def get_plot_params():
    """Create the plot parameters used in the plotting
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc

    import scienceplots

    plt.style.use(["science", "nature"])

    COLOR = "k"
    plt.rcParams["text.color"] = COLOR
    plt.rcParams["axes.labelcolor"] = COLOR
    plt.rcParams["xtick.color"] = COLOR
    plt.rcParams["ytick.color"] = COLOR
