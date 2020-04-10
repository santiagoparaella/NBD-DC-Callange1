#  Library To manipulate data:  
import pandas as pd
import utils
import csv

PATH_R = '/Users/Dario/Google Drive/DS/First Year - Secon Semester/NBD/DC_Homeworks/HW_1/results.csv'

''' RUN THIS ONLY THE FIRST TIME TO CREATE THE CSV FILE
# Generate a csv to save data

results = open(PATH_R, 'w')
write_r = csv.writer(results)
write_r.writerow(['nodes', 'graph_type', 'bfs', 'irreducibility', 'laplacian'])
results.close()

# Work on a regular graph:

utils.check_connectivity(PATH_R, boot = 15, g_type = 'r_graph', nodes = range(10, 1060, 50))

# Work on a Erdon graph:

utils.check_connectivity(PATH_R, boot = 15, g_type = 'er_graph', nodes = range(10, 1060, 50))
'''

# Generate DFs:

results_df = pd.read_csv(PATH_R)
results_df.sort_values(['nodes'], inplace = True)

# Plot:
    
utils.plot_results(results_df, "Regular")
utils.plot_results(results_df, "Erdos-Renyi")

# Study variance:
    
utils.variability_check(n = 150, boot = 5000) # variability among methods and graph
utils.connected_vs_disconnected(n = 150, boot = 1000) # Variability in connected and dsconnected graphs
                                                        # It takes time (few minutes)
utils.laplacian_variability(n = 1010) # Check variability in Laplacian in function
 
    
        