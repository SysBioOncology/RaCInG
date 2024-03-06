"""
File with utility functions that are useful when analysing the output that
appears from RaCInG. There are methods to create the adjacency
of an input graph. Moreover, there is a method to count the amout of objects
of with given cell types given a raw object count
 (and the vertex type vector).
"""

import numpy as np
from scipy import sparse as sp

def EdgetoAdj(E, N):
    '''
    Converts a graph defined through an edge list into a graph defined through
    a sparse (compressed storage row) adjacancy matrix.

    Parameters
    ----------
    E : np.array with two columns and int entries;
        Edge list of the graph.
    N : int;
        Number of vertices of the graph.

    Returns
    -------
    Adj : sparse csr matrix;
        Adjacency matrix corresponding to E (and N).

    '''
    Adj = sp.coo_matrix((np.ones_like(E[:,0]),(E[:,0], E[:,1])), \
                        shape = (N, N)).tocsr()
    return Adj

def EdgetoAdj_No_loop(E, N):
    '''
    Converts a graph defined through an edge list into a graph defined through
    a sparse (compressed storage row) adjacancy matrix. In construction it will
    remove all loops. This is useful e.g. for triangle calculations.

    Parameters
    ----------
    E : np.array with two columns and int entries;
        Edge list of the graph.
    N : int;
        Number of vertices of the graph.

    Returns
    -------
    Adj : sparse csr matrix;
        Adjacency matrix corresponding to E (and N) without loops.

    '''
    Adj = sp.coo_matrix((np.ones_like(E[:,0][E[:,0]!=E[:,1]]),\
                         (E[:,0][E[:,0]!=E[:,1]], E[:,1][E[:,0]!=E[:,1]])), \
                        shape = (N, N)).tocsr()
    
    return Adj

def Count_Types(oblist, V, maxTypes = 0):
    """
    Counts unique occurances of elements in oblist with given types according to V.

    Parameters
    ----------
    oblist : list() with int entries;
        List with objects to be sorted (e.g. triangles)
    V : List() that maps each vertex to a type;
        List with rules to map the graph vertices (cells) to types.
    maxTypes : int; (OPTIONAL)
        Number of unique vertex types. If not specified, RaCInG tries to
        estimate it from the data provided.
        
        KNOWN ISSUE: If the input consits e.g. of 9 cell-types, but the ninth
        cell type is never generated, then the automatic function will think
        there are 8 cell-types. This might cause issues later in the pipeline.

    Returns
    -------
    counttensor : np.array() with int entries;
        The tensor in which for each possible type of obj its count is presented.
        (e.g. in the case of triangles a 3 at index [3,5,6] means that 3 triangles
         have been found that to from type 3 to 5 to 6.)
    """
    if maxTypes:
        dim = maxTypes
    else:
        dim = np.max(V) + 1
    counttensor = np.zeros(tuple([dim] * oblist.shape[1]))
    typelist = V[np.array(oblist)]
    for t in typelist:
        counttensor[tuple(t)] += 1
    return counttensor

def createSlurm(cancer, weight, feature, N, itNo, noPat, av, norm):
    if norm:
        filename = cancer + feature + "norm.sh"
    else:
        filename = cancer + feature + ".sh"
    
    with open(filename, "x", newline = '\n') as f:
        f.writelines(["#!/usr/bin/bash\n", "#SBATCH --nodes=1\n", "#SBATCH --ntasks=1\n","#SBATCH --partition=mcs.default.q\n"])
        if feature == "D":
            if norm:
                f.write("#SBATCH --output={}_{}_norm.out\n".format(cancer, feature))
            else:
                f.write("#SBATCH --output={}_{}.out\n".format(cancer, feature))
        else:
            if norm:
                f.write("#SBATCH --output={}_{}_{}_norm.out\n".format(cancer, feature, av))
            else:
                f.write("#SBATCH --output={}_{}_{}.out\n".format(cancer, feature, av))
        f.write("#SBATCH --error=panic-%j.err\n")
        f.write("#SBATCH --time=5-0\n")
        f.writelines(["WEIGHTTYPE=\"{}\"\n".format(weight),
                     "CANCER=\"{}\"\n".format(cancer),
                     "FEATURE=\"{}\"\n".format(feature),
                     "N={}\n".format(N),
                     "ITNO={}\n".format(itNo),
                     "NOPAT={}\n".format(noPat),
                     "AV={}\n".format(av),
                     "NORM={}\n".format(norm),
                     "echo \"$FEATURE\"\n",
                     "echo \"$WEIGHTTYPE,$NOPAT,$N,$ITNO,$AV\"\n",
                     "module load anaconda\n",
                     "source activate CCI-env\n",
                     "parallel python3 HPC_CLI.py $WEIGHTTYPE $CANCER $FEATURE {{}} $N $ITNO $AV $NORM ::: {{0..{}}}\n".format(noPat - 1),
                     "conda deactivate\n", "echo["])
    return


def poiBPFunc(x, M, sens):
    """
    Utility funciton that seeks to find the survival probability of a Poisson
    Branching process by searching for its root. This funciton outputs 
    the value of the survival function at the current estimate. Used in
    analytical computation of the GSCC.

    Parameters
    ----------
    x : np.array() with float entries;
        vector with current GSCC estimate.
    M : np.array() with float entries;
        Matrix with average sizes of each group of the branching process.
    sens : int;
        Size of x.

    Returns
    -------
    np.array() vector of current estimate

    """
    inter = np.tile(x, (sens, 1))
    result = inter * M
    return np.ones(sens) - x - np.exp(-np.sum(result, axis = 1))

if __name__ == "__main__":
    #Test Adj matrix construciton
    N = 5
    E = np.array([[1, 2], [4, 1], [4, 1], [0, 3], [4, 2], [3, 1], [2, 2]])
    print("With loops:")
    print(EdgetoAdj(E, N).todense())
    print("Without loops:")
    print(EdgetoAdj_No_loop(E, N).todense())
    
    #Test object count function
    oblist = np.array([[0, 3, 8, 1], [1, 4, 2,2], [1, 3, 2, 1], [0, 1, 2, 3]])
    V = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
    print("Object count no issue:")
    print(Count_Types(oblist, V))
    
    #Showing the issue in Count_Types()
    oblist = np.array([[0, 3, 8], [1, 4, 2], [1, 3, 2], [0, 1, 2]])
    V = np.array([0, 0, 0, 0, 2, 2, 2, 2, 0]) #Assume there are three cell types (type 2 was never generated)
    print("Object count issue with automatic type count:")
    print(Count_Types(oblist, V))
    print("Object count issue with corrected type count:")
    print(Count_Types(oblist, V, 3))   