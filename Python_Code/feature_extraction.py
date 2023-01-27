"""
Methods used to count the number of wedges and loops. This will be one of the
main files that extracts the featrues from the random graph outputs.
"""

import numpy as np
from Utilities import EdgetoAdj_No_loop


def Find_Number_Trust_Triangles_Unique(Adj):
    """
    Counts the number of trust triangles in a graph.
    
    NOTE: Disregards multi-edges, and does *not* remove triangles with a
    undirected edge (<-->).

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix with self loops removed

    Returns
    -------
    NoTriangles : int;
        Number of trust triangles.

    """
    n = Adj.get_shape()[0]
    A = Adj.sign()
    Wedge_matrix = (A ** 2)
    locfail = np.ravel_multi_index((A != Wedge_matrix.sign()).nonzero(), (n,n))
    alllock = np.ravel_multi_index((A + Wedge_matrix).nonzero(), (n,n))
    triangleloc = np.setdiff1d(alllock, locfail)
    NoTriangles = Wedge_matrix[np.unravel_index(triangleloc, (n,n))].sum()
    return int(NoTriangles)

def Find_Number_Triangles(Adj):
    """
    Finds number of triangles in a graph. Triangles over multi-edges are 
    counted multiple times.

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix of the graph.

    Returns
    -------
    No_triangles : int;
        Number of triangles.

    """
    A = Adj.copy()
    #n = A.get_shape()[0]
    #A[np.arange(n), np.arange(n)] = 0 #Use EtoAdj_no_loops(E, N) in Utilities
    triangle_matrix = A ** 3
    No_triangles = 1/3 * triangle_matrix.diagonal(0).sum()
    return int(No_triangles)

def Find_Number_Triangles_Unique(Adj):
    """
    Finds number of triangles in a graph. Triangles over multi-edges are not
    counted multiple times. These are loop triangles

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix of graph.

    Returns
    -------
    No_triangles : int
        Number of triangles

    """
    A = Adj.copy()
    n = A.get_shape()[0]
    #A[np.arange(n), np.arange(n)] = 0 #Use EtoAdj_no_loops(E, N) in Utilities
    trimmed_matrix = A.sign()
    triangle_matrix = trimmed_matrix **3
    No_triangles = 1/3 * triangle_matrix[np.arange(n),np.arange(n)].sum()
    return int(No_triangles)

def Find_Number_2Loops(Adj):
    """
    Finds number of 2-loops in a graph. Loops over multi-edges are counted
    multiple times.

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix of graph.

    Returns
    -------
    No_loops : int;
        Number of 2-loops.

    """
    A = Adj.copy()
    #n = A.get_shape()[0]
    #A[np.arange(n), np.arange(n)] = 0 #Use EtoAdj_no_loops(E, N) in Utilities
    loop_matrix = A ** 2
    No_loops = 1/2 * loop_matrix.diagonal(0).sum()
    return int(No_loops)

def Find_Number_2Loops_Unique(Adj):
    """
    Finds number of 2-loops in a graph. Loops over multi-edges are not counted
    multiple times.

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix of graph.

    Returns
    -------
    No_loops : int;
        Number of 2-loops.
    """
    A = Adj.copy()
    #n = A.get_shape()[0]
    #A[np.arange(n), np.arange(n)] = 0 #Use EtoAdj_no_loops(E, N) in Utilities
    trim = A.sign()
    loop_matrix = trim ** 2
    No_loops = 1/2 * loop_matrix.diagonal(0).sum()
    return int(No_loops)

def Find_Number_Wedges(Adj):
    """
    Finds number of wedges in a graph. Wedges over multi-edges are counted
    multiple times.

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix of graph.

    Returns
    -------
    No_wedges : int;
        Number of wedges.

    """
    A = Adj.copy()
    #n = A.get_shape()[0]
    #A[np.arange(n), np.arange(n)] = 0 #Use EtoAdj_no_loops(E, N) in Utilities
    wedge_matrix = A ** 2
    No_wedges = wedge_matrix.sum() - wedge_matrix.diagonal(0).sum()
    return int(No_wedges)

def Find_Number_Wedges_Unique(Adj):
    """
    Finds number of wedges in a graph. Wedges over multi-edges are not counted
    multiple times.
    
    NOTE: We do count wedges of the form [a, b, b] and [a, a, b].

    Parameters
    ----------
    Adj : sparse csr matrix;
        Adjacency matrix of graph.

    Returns
    -------
    No_wedges : int;
        Number of wedges.

    """
    A = Adj.copy()
    #n = A.get_shape()[0]
    #A[np.arange(n), np.arange(n)] = 0 #Use EtoAdj_no_loops(E, N) in Utilities
    trim = A.sign()
    wedge_matrix = trim ** 2
    No_wedges = wedge_matrix.sum() - wedge_matrix.diagonal(0).sum()
    return int(No_wedges)


def Trust_Triangles(Adj):
    """
    Counts the number of outward trust triangles
    a -> b
    |     \
    -> c <-
    
    and provides a list of the compositino of each triangle.

    NOTE: Discards multi-edges. Also, does *not* discard triangles with a
    looped edge (<-->).

    Parameters
    ----------
    Adj : Sparse csr matrix;
        Adjacency matrix of the graph

    Returns
    -------
    NoTriangles : int;
        Number of outward trust triangles
    Triangle_list : list() with int entries;
        List of each trust triangle in order [c, a, b]

    """
    NoTriangles = 0
    Triangle_list = []
    n = Adj.get_shape()[0]
    
    for v in range(n):
        neigh_v = Adj.getrow(v).nonzero()[1]
        for w in neigh_v:
            neigh_w = Adj.getrow(w).nonzero()[1]
            intersec = np.intersect1d(neigh_v, neigh_w)
            NoTriangles += len(intersec)
            for u in intersec:
                Triangle_list.append([u, v, w])
    
    return NoTriangles, np.array(Triangle_list)

def Cycle_Triangles(Adj):
    """
    Counts the number of cycle triangles, and produces a list of these
    triangles. 
    
    NOTE: Discards multi-edges. Also, does *not* discards triangles with a
    looped edge (<-->).

    Parameters
    ----------
    Adj : Sparse csr matrix;
        Adjacency matrix of the graph (loops removed).

    Returns
    -------
    NoTriangles : int;
        Number of triangles in the graph (symmetries removed).
    Triangle_list : list() with entries;
        List of each cycle triangle (symmetries present).

    """
    NoTriangles = 0
    Triangle_list = []
    AdjT = Adj.transpose().tocsr()
    n = Adj.get_shape()[0]
    
    for v in range(n):
        neigh_v = Adj.getrow(v).nonzero()[1]
        backneigh_v = AdjT.getrow(v).nonzero()[1]
        for w in neigh_v:
            neigh_w = Adj.getrow(w).nonzero()[1]
            intersec = np.intersect1d(backneigh_v, neigh_w)
            NoTriangles += len(intersec)
            for u in intersec:
                Triangle_list.append([u, v, w])
    
    return round(NoTriangles/3), np.array(Triangle_list)

def Wedges(Adj):
    """
    Counts number of wedges, and produces a list of these.
    
    NOTE: Disregards multi-edges, but dus not disregard looped edges (<-->).

    Parameters
    ----------
    Adj : Sparse csr-matrix;
        Adjacency matrix of graph.

    Returns
    -------
    NoWedges : int;
        Amount of wedges present.
    Wedge_list : list with int entries;
        List of wedges found in the graph.

    """
    NoWedges = 0
    Wedge_list = []
    n = Adj.get_shape()[0]
    
    for v in range(n):
        neigh_v = Adj.getrow(v).nonzero()[1]
        
        for w in neigh_v:
            neigh_w = Adj.getrow(w).nonzero()[1]
            neigh_w = neigh_w[neigh_w != v]
            
            for u in neigh_w:
                NoWedges += 1
                Wedge_list.append([v, w, u])
    
    return NoWedges, np.array(Wedge_list)

if __name__ == "__main__":
    #Small example graph
    IN = np.array([2,2,3,3,3,4,4,4])
    OUT = np.array([1,4,0,1,2,2,3,4])
    V = np.array([0,1,1,0,0])
    E = np.column_stack((IN, OUT))
    N = len(V)
    Adj = EdgetoAdj_No_loop(E, N) #Turn edge list into Adj matrix
    
    tt = Find_Number_Trust_Triangles_Unique(Adj)
    lt = Find_Number_Triangles_Unique(Adj)
    w = Find_Number_Wedges_Unique(Adj)
    nw, lw = Wedges(Adj)
    l = Find_Number_2Loops_Unique(Adj)
    ntt, ltt = Trust_Triangles(Adj)
    nlt, llt = Cycle_Triangles(Adj)
    
    #Due to the introduction of EdgetoAdj_No_loop the different counting
    #algorithms should do the same.
    print("Trust Triangles: {} or {}".format(tt, ntt))
    print("Loop Triangles: {} or {}".format(lt, nlt))
    print("Wedges: {} or {}".format(w, nw))
    print("Loops: {}".format(l))
    print("Trust triangles list")
    print(ltt)
    print("Loop triangles list")
    print(llt)   
    print("Wedge list")
    print(lw)

        