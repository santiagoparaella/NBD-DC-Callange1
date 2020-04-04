import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils
import csv

# Our values:

nodes = range(10, 270, 10)
degree = 4
lambd = 1.4

# Generate a csv to save data

PATH_R = '/Users/Dario/Google Drive/DS/First Year - Secon Semester/NBD/DC_Homeworks/HW_1/results.csv'
results = open(PATH_R, 'w')
write_r = csv.writer(results)
write_r.writerow(['nodes', 'graph_type' ,'is_connected', 'bfs_time', 'adj_time', 'laplacian_time'])
results.close()

# Work on a regular graph:

utils.check_connectivity("r-graph", nodes, degree, PATH_R)

# Work on a er_graph:

utils.check_connectivity("er-graph", nodes, degree, PATH_R, lambd)


# Generate DFs:

results_df = pd.read_csv(PATH_R)


df_melt_er = pd.melt(results_df, id_vars = ['nodes', 'graph_type'], 
        value_vars = ['bfs_time', 'adj_time', 'laplacian_time']).query("graph_type == 'er-graph'")

df_melt_r = pd.melt(results_df, id_vars = ['nodes', 'graph_type'], 
        value_vars = ['bfs_time', 'adj_time', 'laplacian_time']).query("graph_type == 'r-graph'")

# Plot results:

plt.figure(1)
sns.lineplot(x = 'nodes', y = 'value', hue = 'variable', data = df_melt_r)
plt.title("Regular-Graph Connectivity Checking")
plt.ylabel("Time")
plt.legend(title = '', labels=['BFS', 'Adjacency', 'Laplacian'])


plt.figure(2)
sns.lineplot(x = 'nodes', y = 'value', hue = 'variable', data = df_melt_er)
plt.title("ErdosRenyi-Graph-Graph Connectivity Checking")
plt.ylabel("Time")
plt.legend(title = '', labels=['BFS', 'Adjacency', 'Laplacian'])







