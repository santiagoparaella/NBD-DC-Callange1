import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from networkx.linalg.algebraicconnectivity import algebraic_connectivity

import time
from math import log
import numpy as np
import csv
    
def bfs_check(graph, n):

    start = time.time()
    
    traversed_n = list(dfs_preorder_nodes(graph, 0))
    check = True if len(traversed_n) == n else False
    
    tot_time = time.time() - start
    
    return [tot_time, check]

def adjacency_check(adj_matr, n):
    
    identity = np.identity(n)
    old_prod = adj_matr
    
    start = time.time()
    
    sum_matr = np.add(identity, old_prod)
    
    for _ in range(2, n):
        
        old_prod = old_prod.dot(adj_matr)
        sum_matr = np.add(sum_matr, old_prod)
        
    check = np.all(sum_matr)
    
    tot_time = time.time() - start
    
    return [tot_time, True if check else False]
    
def laplacian_check(graph):

    start = time.time()

    # Find the second eingenvalue:
    
    secondo_autovalore = algebraic_connectivity(graph)
    
    check = True if secondo_autovalore > 0 else False
    
    tot_time = time.time() - start
    
    return [tot_time, check]    
    
def check_connectivity(g_type, n_range, r, path, lam = None):
    
    if g_type == "r-graph":
        
        for n in n_range:
            
            result = [n, g_type]
                    
            # Generate Graph:
            
            g = nx.random_regular_graph(r, n, seed = 42)
            
            # Check Connectivity by BFS:
            
            bfs_result = bfs_check(g, n)
            result.append(bfs_result[0])
            
            # Check connectivity by Adjacency matrix:
            
            adj_r = nx.to_numpy_matrix(g)
            adj_result = adjacency_check(adj_r, n)
            result.append(adj_result[0])
            
            # Check connectivity by Laplacian:
            
            lap_result = laplacian_check(g)
            result.append(lap_result[0])
            
            # Check connectivity:
            
            if bfs_result[1] and adj_result[1] and lap_result[1]:
                
                result.insert(2, "Connected")
                
            elif not bfs_result[1] and not adj_result[1] and not lap_result[1]:
                
                result.insert(2, "Disconnected")
                
            else:
                
                raise ValueError('The three results are not the same')
                
            
            # Add result to csv:
                      
            file = open(path, 'a+') 
            write = csv.writer(file)
            write.writerow(result)
            file.close()
        
    elif g_type == "er-graph":
        
        for n in n_range:

            result = [n, g_type]
                    
            # Generate Graph:
            
            p = lam * log(n)/n # See MIT slides
            g = nx.gnp_random_graph(n, p, seed = 42)
            
            # Check Connectivity by BFS:
            
            bfs_result = bfs_check(g, n)
            result.append(bfs_result[0])
            
            # Check connectivity by Adjacency matrix:
            
            adj_r = nx.to_numpy_matrix(g)
            adj_result = adjacency_check(adj_r, n)
            result.append(adj_result[0])
            
            # Check connectivity by Laplacian:
            
            lap_result = laplacian_check(g)
            result.append(lap_result[0])
            
            # Check connectivity:
            
            if bfs_result[1] and adj_result[1] and lap_result[1]:
                
                result.insert(2, "Connected")
                
            elif not bfs_result[1] and not adj_result[1] and not lap_result[1]:
                
                result.insert(2, "Disconnected")
                
            else:
                
                raise ValueError('The three results are not the same')
                
            
            # Add result to csv:
            
            file = open(path, 'a+') 
            write = csv.writer(file)
            write.writerow(result)
            file.close()
            
            
            
            

   

    
