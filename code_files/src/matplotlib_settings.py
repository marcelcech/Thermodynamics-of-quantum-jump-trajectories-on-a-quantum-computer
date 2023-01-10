from .package_requirements import *

# use correct LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Latin Modern'], 'size': 10})
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{dsfont}"

max_fig_width = 3.3
