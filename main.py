import seaborn as sns
import utils
import csv

# Our values:

nodes = range(100, 1000, 100)
degree = 4
lambd = 1.1

# Generate a csv to save data

PATH_R = '/Users/Dario/Google Drive/DS/First Year - Secon Semester/NBD/DC_Homeworks/HW_1/HW_1.results.csv'
results = open(PATH_R, 'w')
write_r = csv.writer(results)
write_r.writerow(['nodes', 'graph_type' ,'is_connected', 'bfs_time', 'adj_time', 'laplacian_time'])

# Work on a regular graph:

utils.Check_Connectivity("r-graph", nodes, degree, write_r)

# Work on a er_graph:

utils.Check_Connectivity("er-graph", nodes, degree, write_r, lambd)

# Save the csv:

results.close()

# Draw the graphs:

sns.lineplot()