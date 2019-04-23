import os
from matplotlib import pyplot as plt
import seaborn as sns


def plot(metrics, key):
  fig, ax = plt.subplots()
  ax = sns.lineplot(x=metrics['episodes'], y=metrics[key], ax=ax)
  fig.savefig(os.path.join('results', key + '.png'))
  plt.close(fig)
