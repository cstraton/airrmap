# Query Engine performance figure
# (query time vs number of sequences).
# Manual values collected from use of application.

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Timings, mac mini 2018, limited to 4 CPUs and 4.25GB RAM.
data = [
    # dict(recs=329, secs=0.722), # Test set
    # dict(recs=86201, secs=0.887), # Bashford-Rogers
    dict(recs=462587, secs=1.629),  # Eliyahu
    dict(recs=1463436, secs=2.965),  # Joyce
    dict(recs=2855352, secs=6.634),  # Gupta
    dict(recs=4574536, secs=16.961)  # Schultheiss (csv)
]

# %% Plot
df: pd.DataFrame = pd.DataFrame.from_records(
    data=data
)

# %% Plot
sns.set_style('white')
sns.set_context(
    'paper',  # scale: paper, notebook, talk, poster.
    font_scale=1.5,
)
fig = sns.lineplot(
    data=df,
    x='secs',
    y='recs',
    linewidth=3,
    marker='o',
)
fig.axes.set_xlim(0, 18)
plt.xlabel('Query time (seconds)')
plt.ylabel('Sequences (millions)')
#plt.title('Query time vs dataset size')
plt.tight_layout()
sns.despine()

# %% Show/Save
plt
#plt.savefig('./results/img/fig_query_time.svg', dpi=300)
