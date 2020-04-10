# Library to work with graphs
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from networkx.linalg.algebraicconnectivity import algebraic_connectivity

# Library to manipulate data:
from math import log
from statistics import median
from statistics import stdev
import numpy as np
import pandas as pd
import csv

# Library to use multiprocessing:
import time
import multiprocessing
from functools import partial

# Library to plot and adjust the curves:
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

'''
This function takes as INPUT a graph and the nuomber of nodes that it has. It apply the DFS
algorithm and check if the number nodes visited is the same of number of nodes of the graph.
It returns the time (in seconds) needed to do this process. If check_result is True, it returns
a list of two elements [Time needed for the process, graph status (connecyed, disconnected)].
'''

def bfs_check(graph, n, check_result = False):

    start = time.time()
    
    # Explore the graph DFS:
        
    traversed_n = list(dfs_preorder_nodes(graph, 0))
    
    # Check connectivity:
        
    check = True if len(traversed_n) == n else False

    tot_time = time.time() - start
    
    if check_result:
        
        return [tot_time, 'Connected' if check else 'Disconnected']
    
    else:
    
        return tot_time

'''
This function takes as INPUT a graph and the nuomber of nodes that it has. 
It uses dynamic programming to sum and calculate the power of the adjacency matrix.
The aim is to ceck if the following:
    check --> I + A^1 + A^2 + ... + A^(n-1) > 0
It returns the time (in seconds) needed to do this process. If check_result is True, it returns
a list of two elements [Time needed for the process, graph status (connecyed, disconnected)].
'''

def adjacency_check(g, n, check_result = False):
    
    # Generate the variables:

    adj_matr = nx.to_numpy_matrix(g)
    
    identity = np.identity(n)
    old_prod = adj_matr # It contains the dot product of the adj to itself n-1 times
    
    start = time.time()
    
    # Implement the calcolous:
    
    sum_matr = np.add(identity, old_prod) # It is the sum of the identity and the powers of the adj
    
    for _ in range(2, n):
        
        old_prod = old_prod.dot(adj_matr) # each time I use the old dot product for a new one with the adj
                                            # EX: A^3 = A^2.dotproduct(A)
        sum_matr = np.add(sum_matr, old_prod) # Each time I add the new values to the previous ones
        
    # Check connectivity:
        
    check = np.all(sum_matr)
    
    tot_time = time.time() - start
    
    if check_result:
        
        return [tot_time, 'Connected' if check else 'Disconnected']
    
    else:
    
        return tot_time

'''
It takes as input a graph and apply the laplacian method 
(already implemented and optimized in netwirkx).
It returns the time (in seconds) needed to do this process. If check_result is True, it returns
a list of two elements [Time needed for the process, graph status (connecyed, disconnected)].
'''
    
def laplacian_check(g, check_result = False):

    start = time.time()

    # Find the second eingenvalue:
    ## tracemin_lu is a setting to find solution in almost linear time
    
    secondo_autovalore = algebraic_connectivity(g, method = 'tracemin_lu') 
    
    check = True if secondo_autovalore > 0 else False
    
    tot_time = time.time() - start
    
    if check_result:
        
        return [tot_time, 'Connected' if check else 'Disconnected']
    
    else:
    
        return tot_time             

'''
INPUT:
    path: the path to the csv to write
    boot: number of graph to test for each n
    g_type: kind of graph on which we do the test
    nodes: number of nodes in the graph
    r: number of edges per node (degree), only for regular graph
    lam: parameter to calcualte the optimal p 
        (defaul 1: namely graph can be either connecte or not with the same probability)

The function generates m = boot graph for each value of n and executes all the three methods
to check connectivity for each graph and takes the median time for each method 
(to do this we it uses multiprocessing). Finally, it writes the results in a csv file.
Each row of the csv contains the follow info:
    [nodes, graph_type, bfs_median_time, irreducibility_median_time, laplacian_median_time]
'''


def check_connectivity(path, boot = 10, g_type = 'r_graph', nodes = 100, r = 4, lam = 1):
    
    start = time.time()
    
    for n in nodes:  

        # Generate m different graphs:
        
        graphs = []
        if g_type == 'er_graph':
            for _ in range(boot):
                p = lam * log(n)/n # See MIT slides
                g = nx.gnp_random_graph(n, p)
                graphs.append(g)
        elif g_type == 'r_graph':
            for _ in range(boot):
                g = nx.random_regular_graph(r, n)
                graphs.append(g)
        else:
            raise ValueError("Insert a proper value for g_type: r_graph or er_graph")
            
        # Calculate time for each graph for each method:
        
        with multiprocessing.Pool(processes = 4) as pool:
            
            prod_bfs = partial(bfs_check, n = n) # prod_x has only one argument x (y is fixed to 10)
            bfs_result = pool.map(prod_bfs, graphs)
            
            prod_adj = partial(adjacency_check, n = n)
            adj_result = pool.map(prod_adj, graphs)
            
            lap_result = pool.map(laplacian_check, graphs)
        
        # Calculate median result:
        
        result = [n, g_type]
        bfs_med = median(bfs_result)
        result.append(bfs_med)
        adj_med = median(adj_result)
        result.append(adj_med)
        lap_med = median(lap_result)
        result.append(lap_med)
        
        # Write the csv:
        
        file = open(path, 'a+') 
        write = csv.writer(file)
        write.writerow(result)
        file.close()
    
    end = time.time() - start
    
    print("Process completed in {} minutes!".format(round(end/60, 2)))

''''
INPUT:
    df: the dataframe which contains the times for the check
    g_type: a string can be: 'Regular' or 'Erdos-Renyi'

It returns the graph of the choosen data as lines plot.
'''

def plot_results(df, g_type):
    
    # Generate the DF with the needed data:
    
    if g_type == 'Regular':
        
        new_df = df[df["graph_type"] == 'r_graph'].copy()
        title = " ".join([g_type, 'Graph Connectivity Check (log)'])
        
    elif g_type == 'Erdos-Renyi':
        
        new_df = df[df["graph_type"] == 'er_graph'].copy()
        title = " ".join([g_type, 'Graph Connectivity Check (log)'])
        
    plt.figure(1, figsize = (12, 8))
    plt.plot(new_df['nodes'], savgol_filter(new_df["bfs"], 7, 1), '.', color = 'orange', markersize = 20)
    plt.plot(new_df['nodes'], savgol_filter(new_df["laplacian"], 9, 1), '.', color = 'blue', markersize = 20)
    plt.plot(new_df['nodes'], new_df["irreducibility"], '.', color = 'green', markersize = 20)
    plt.ylabel("log Time (s)", fontsize = 20, labelpad = 10)
    plt.yticks(fontsize = 16)
    plt.xlabel("Nodes", fontsize = 20, labelpad = 10)
    plt.xticks(fontsize = 16)
    plt.yscale("log")
    plt.title(title, pad = 16, fontsize = 22)
    plt.legend(title = '', labels=['BFS', 'Laplacian', 'Irreducibility'], fontsize = 16)
    plt.show()
        
'''
INPUT:
    n (int): number of nodes of the graph 
    boot (int): number of graph to check

It generates a number of graph quals to boot (both erdos and regular) and on each graph
it checks the connectivity thanks to the three different method and measures the time for the check.
After doing that it calculates the standard deviation for each method for each kind of graph
and plot the results.
'''


def variability_check(n, boot):

    rgs = []
    ergs = []
    for _ in range(boot):
        
        # Generate Erdon random graph:
        
        p = log(n)/n 
        erg = nx.gnp_random_graph(n, p)
        ergs.append(erg)
        
        # Generate Regular Graph:
            
        rg = nx.random_regular_graph(4, n)
        rgs.append(rg)
        
    # Dictionary to save results:
        
    results = {'bfs': {'regular': None, "erdos-renyi": None},
               'laplacian': {'regular': None, "erdos-renyi": None},
               'irreducibility': {'regular': None, "erdos-renyi": None}
               }
    
    # Execute the different checks:
    
    with multiprocessing.Pool(processes = 4) as pool:
        
        # Regular Graph:
        
        prod_bfs = partial(bfs_check, n = n) 
        bfs_result = pool.map(prod_bfs, rgs)
        results['bfs']['regular'] = stdev(bfs_result)
        
        prod_adj = partial(adjacency_check, n = n)
        adj_result = pool.map(prod_adj, rgs)
        results['irreducibility']['regular'] = stdev(adj_result)
        
        lap_result = pool.map(laplacian_check, rgs)
        results['laplacian']['regular'] = stdev(lap_result)
        
        # Erdon graph:
        
        prod_bfs = partial(bfs_check, n = n) 
        bfs_result = pool.map(prod_bfs, ergs)
        results['bfs']['erdos-renyi'] = stdev(bfs_result)
        
        prod_adj = partial(adjacency_check, n = n)
        adj_result = pool.map(prod_adj, ergs)
        results['irreducibility']['erdos-renyi'] = stdev(adj_result)
        
        lap_result = pool.map(laplacian_check, ergs)
        results['laplacian']['erdos-renyi'] = stdev(lap_result)
            
    # Generate the DataFrame:
        
    lst = []
    for k1, v1 in results.items():
        for k2, v2 in v1.items():
            lst.append((k2, k1, v2))
    
    sd_df = pd.DataFrame(lst, columns = ['graph', 'method', 'stdev'])
    sd_df.sort_values(["stdev"], ascending = False , inplace = True)
    
    # Plot the result:
    
    plt.figure(2, figsize = (12, 8))
    sns.barplot(x = 'stdev', y = 'method', hue = 'graph', data = sd_df, orient = 'h', edgecolor=".1")
    plt.ylabel("", fontsize = 22, labelpad = 10)
    plt.yticks(fontsize = 22)
    plt.xlabel("StDev", fontsize = 22, labelpad = 10)
    plt.xticks(fontsize = 20)
    plt.legend(loc = 4, fontsize = 24)
    plt.title('Analysis of Variability (SD)', pad = 16, fontsize = 26)
    plt.show()

'''
It generates m = boot graphs and check for each method each graph and calculate the time needed.
After that it plots a bar plot for each algorithm and each kind of plot to show
how the SD behave in the different situations.
'''

def connected_vs_disconnected(n, boot):
    
    # Generate connected graph:
    
    c_graphs = []
    p = 1.6 * log(n)/n
    for _ in range(boot):
    
        connected = False
        while not connected:

            g = nx.gnp_random_graph(n, p)
            connected = nx.is_connected(g)
            
            if connected:
                c_graphs.append(g)
    
    
    # Run methods on connected graph:
        
    lst_for_df = []
    with multiprocessing.Pool(processes = 4) as pool:
        
        prod_bfs = partial(bfs_check, n = n) 
        bfs_result = pool.map(prod_bfs, c_graphs)
        b = ( 'connected', 'bfs', stdev(bfs_result) )
        lst_for_df.append(b)
        
        prod_adj = partial(adjacency_check, n = n)
        adj_result = pool.map(prod_adj, c_graphs)
        a = ( 'connected', 'irreducibility', stdev(adj_result) )
        lst_for_df.append(a)
        
        lap_result = pool.map(laplacian_check, c_graphs)
        l = ( 'connected', 'laplacian', stdev(lap_result) )
        lst_for_df.append(l)
                
    # Generate disconnected graph:
    
    c_graphs = []
    p = 0.2 * log(n)/n
    for _ in range(boot):
    
        connected = True
        while connected:

            g = nx.gnp_random_graph(n, p)
            connected = nx.is_connected(g)
            
            if not connected:
                c_graphs.append(g)
    
    
    # Run methods for disconnetced graphs:
        
    with multiprocessing.Pool(processes = 4) as pool:

        
        prod_bfs = partial(bfs_check, n = n) 
        bfs_result = pool.map(prod_bfs, c_graphs)
        b = ( 'disconnected', 'bfs', stdev(bfs_result) )
        lst_for_df.append(b)
        
        prod_adj = partial(adjacency_check, n = n)
        adj_result = pool.map(prod_adj, c_graphs)
        a = ( 'disconnected', 'irreducibility', stdev(adj_result) )
        lst_for_df.append(a)
        
        lap_result = pool.map(laplacian_check, c_graphs)
        l = ( 'disconnected', 'laplacian', stdev(lap_result) )
        lst_for_df.append(l)   
    
    # Generate DataFrame:
    
    sd_df = pd.DataFrame(lst_for_df, columns = ['graph', 'method', 'stdev'])
    sd_df.sort_values(["stdev"], ascending = False , inplace = True)
    
    # Plot results:
        
    plt.figure(3, figsize = (12, 8))
    sns.barplot(x = 'stdev', y = 'method', hue = 'graph', data = sd_df, orient = 'h', edgecolor=".1")
    plt.ylabel("", fontsize = 22, labelpad = 10)
    plt.yticks(fontsize = 22)
    plt.xlabel("StDev", fontsize = 22, labelpad = 10)
    plt.xticks(fontsize = 20)
    plt.legend(loc = 4, fontsize = 24)
    plt.title('Variability, Connected vs Disconected graph (SD)', pad = 16, fontsize = 26)
    plt.show()
    
'''
It studies the variability of time in laplacian method due to the fact that 
a graph is connected or not.
It generates graphs with different number of nodes (n) and for each it runs the laplacian method
and save the time and the type of pragph.

It returns a grpah with the time depending on the n of nodes and the average time
for the check for connected and disconencted graphs.
'''

def laplacian_variability(n):
    
    lst_to_df = []
    for n in range(10, n, 10):
        p = 1 * log(n)/n
        g = nx.gnp_random_graph(n, p)
        res = laplacian_check(g, True)
        lst_to_df.append((n, res[1], res[0]))
        
    # Generate DataFrame:
        
    df = pd.DataFrame(lst_to_df, columns = ['nodes', 'is_connected', 'time'])
    
    # Plot:
        
    plt.figure(4, figsize = (14, 8))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(121)
    plt.plot(df['nodes'], df['time'], '-')
    plt.ylabel("Time", fontsize = 22, labelpad = 10)
    plt.yticks(fontsize = 22)
    plt.xlabel("Nodes", fontsize = 22, labelpad = 10)
    plt.xticks(fontsize = 20)
    plt.legend(loc = 4, fontsize = 24)
    plt.title('Laplacian Method', pad = 16, fontsize = 26)
    
    groups = df.groupby(["is_connected"]).mean()
    groups.sort_values(by = ["time"], ascending = False, inplace = True)
    
    plt.subplot(122)
    sns.barplot(groups.index.astype(str), groups['time'], edgecolor = ".1") 
    plt.ylabel("Mean", fontsize = 22, labelpad = 10)
    plt.yticks(fontsize = 22)
    plt.xlabel("Status", fontsize = 22, labelpad = 10)
    plt.xticks(fontsize = 20)
    plt.legend(loc = 4, fontsize = 24)
    plt.title('Anlysis of time mean', pad = 16, fontsize = 26)
    plt.show()

    
        
        
        

        
        
        
        
        

    
           
        
        
        
            
        
        
        
        
    
    
    
    
        

    
