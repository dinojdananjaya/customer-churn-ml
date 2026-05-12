import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style() -> None:
    """
    Apply a consistent visual style for all project figures.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def save_figure(path) -> None:
    """
    Save the active matplotlib figure in a report-ready format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()