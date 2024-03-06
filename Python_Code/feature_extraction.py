"""
Methods used to count the number of wedges and loops. This will be one of the
main files that extracts the featrues from the random graph outputs.
"""

import numpy as np
from scipy import sparse as sp
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

def StrongConnect(v, Adj, Ind, Stack, counter, SCC):
    '''
    The main function of Tarjan's algorithm. It takes in the adjancency matrix
    of a graph, and recursively explores this graph to find out which vertices
    are strongly connected to one another.
    
    Parameters
    ----------
    v : int;
        A vertex number of the input graph.
    Adj : sparse matrix in csr format;
        The adjacency list of the input graph.
    Ind : 2 dimensional np.array with int entries;
        The index list of the exploration process of the input graph.
    Stack : List with int entries;
        The stack of recently explored vertices of the input graph. This needs
        to be a list because of an np.append bug during resursive functions. This
        throws out appended values again when move up in your recursion.
    counter : int;
        The (incrementing) counter of the number of recently explored vertices.
    SCC : List of lists with int entries;
        The strongly connected components of the input graph given in each list.

    Returns
    -------
    None. The function only updates all its input values recursively.

    '''
    counter += 1
    Ind[v, :] = np.array([counter, counter])
    Stack.append(v)
    
    for w in Adj.getrow(v).nonzero()[1]: # [1] because we want the column val
        if Ind[w, 0] == 0:
            StrongConnect(w, Adj, Ind, Stack, counter, SCC)
            Ind[v, 1] = np.minimum(Ind[v, 1], Ind[w, 1])
        
        elif Ind[w, 0] < Ind[v, 0] and np.any(Stack == w):
                Ind[v, 1] = np.minimum(Ind[v, 1], Ind[w, 0])
                
    if Ind[v, 0] == Ind[v, 1]:
        where = Stack.index(v)
        SCC.append(Stack[where:])
        del Stack[where:]
    return
        
def Tarjan(Adj):
    '''
    An implementation of Tarjan's algorithm it takes in the adjacency matrix of
    a graph, and outputs this graph's strongly connected components.

    Parameters
    ----------
    Adj : Sparse crs matrix;
        The adjacency matrix of the input graph.

    Returns
    -------
    SCC : List of lists with int entries;
        A list containing the vertices of strongly connected components in 
        sub-lists.

    '''
    counter = 0
    Ind = np.zeros((Adj.shape[0], 2))
    Stack = []
    SCC = []

    for v in np.arange(Adj.shape[0]):
        if Ind[v, 0] == 0:
            StrongConnect(v, Adj, Ind, Stack, counter, SCC)
    return SCC

def BFS(Adj, root):
    """
    Explores a component of a graph with the BFS algorithm. It outputs the
    vertices it has found.

    Parameters
    ----------
    Adj : Sparse csr-matrix;
        Adjacency matrix of the graph.
    root : Int;
        Initial vertex to start the exploration from.

    Returns
    -------
    Explored : List with int entries
        List of vertices in the explored component.

    """
    N = Adj.shape[0]
    seen = [False] * N
    Active = [root]
    Explored = []
    seen[root] = True
    while len(Active) > 0:
        curr = Active[0]
        for neigh in Adj.getrow(curr).nonzero()[1]:
            if not seen[neigh]:
                Active.append(neigh)
                seen[neigh] = True
                
        Active.remove(curr)
        Explored.append(curr)
    return Explored

def OUT(Adj):
    """
    Finds the OUT-component of a graph. It uses Tarjan's algorithm to find the
    GSCC and then applies a BFS on a vertex of the giant to find the giant +
    OUT component. Finally it takes the set difference between giant and 
    giant + Out to find the OUT-component.

    Parameters
    ----------
    Adj : Sparse csr matrix;
        The adjacency matrix of the graph.

    Returns
    -------
    Out_component : np.array() with int entries
        The vertices of the OUT-component.

    """
    giant = GSCC(Adj)
    giant_out = BFS(Adj, giant[0])
    Out_component = np.setdiff1d(giant_out, giant)
    return Out_component

def GSCC(Adj):
    """
    Finds the giant strongly connected component of a graph. It uses Tarjan's
    algorithm together with a single list search to find the largest SCC.

    Parameters
    ----------
    Adj : Sparse csr matrix
        Adjacency matrix of the graph.

    Returns
    -------
    giant : List with int entries;
        The vertices belonging to the giant SCC of the graph.
    """
    SCC = TarjanIterative(Adj)
    sizes = [len(l) for l in SCC]
    GSCC_index = sizes.index(max(sizes))
    giant = SCC[GSCC_index]
    return giant

def IN(Adj):
    """
    Finds the IN-component of the graph. It first uses Tarjan's algorithm to
    find that GSCC and then applies BFS on the arc reversed graph started at
    a GSCC-vertex to find the giant + IN component. Finally, it computes the
    difference between the giant and the giant + IN component to find the IN
    component.

    Parameters
    ----------
    Adj : Sparse csr matrix;
        Adjacency matrix of the graph.

    Returns
    -------
    In_component : np.array() with int entries;
        The vertices belonging to the IN-component.
    """
    trans = Adj.transpose()
    giant = GSCC(trans)
    giant_in = BFS(trans, giant[0])
    In_component = np.setdiff1d(giant_in, giant)
    return In_component

#----------------------------------------------------------------------------
#   TARJAN'S ALGORITHM, ITERATIVE   
#   Implementation below by Low-Level Bits (Jesper Öqvist); see https://llbit.se/?p=3379
#----------------------------------------------------------------------------

def TarjanIterative(Adj):
    """
    Finds the strongly connected components of a (directed) graph instance.
    The main function is sconnect; this function purely initialises all 
    lists/constants needed for that function.
    

    Parameters
    ----------
    Adj : np.array() with int entries;
        The adjacency matrix of the graph instance.

    Returns
    -------
    groups : list of lists with int entries;
        Every list in this list contains the indices of vertices that appear 
        together in a strongly connected component.

    """
    N = Adj.shape[0]
    new = 0 #New index
    index = [None] * N
    lowlink = [None] * N
    onstack = [False] * N
    stack = []
    nextgroup = 0 # Next SCC ID
    groups = [] # SCC: list of vertices
    groupid = {} # Map from vertex to SCC ID.
    
    for v in range(N):
        if index[v] == None:
            sconnect(v, Adj, new, index, lowlink, onstack, stack,\
                     nextgroup, groups, groupid)
    
    return groups
    

# Tarjan's algorithm, iterative version.
def sconnect(v, Adj, new, index, lowlink, onstack, stack, nextgroup,\
             groups, groupid):
    """
    Executes main function of Tarjan's algorithm. The function executes a
    depth first search, and identifies when it 'sees' vertices for the second
    time to determine which SCC the vertex belongs to.

    Parameters
    ----------
    v : int;
        The vertex ID.
    Adj : np.array with int entries;
        The adjacency matrix of the graph.
    new : int;
        New index.
    index : list of int or None entries;
        The moment when a certain vertex was found in the DFS.
    lowlink : list of int or None entries;
        The earlierst found vertex the current vertex connects to.
    onstack : list of Bool entries;
        Indicates whether a vertex is already on the explored stack or not.
    stack : list of int entries;
        The explored vertices.
    nextgroup : int;
        The index of the next SCC.
    groups : list of lists with int entries;
        The list of strongly connected components.
    groupid : dict with {int : int} entries;
        For each vertex (number) this dictionary indicates which SCC the vertex
        belongs to.

    Returns
    -------
    None. Only updates all its parameters.
    """
    work = [(v, 0)] # NEW: Recursion stack.
    while work:
        v, i = work[-1] # i is next successor to process.
        del work[-1]
        if i == 0: # When first visiting a vertex:
            index[v] = new
            lowlink[v] = new
            new += 1
            stack.append(v)
            onstack[v] = True
        recurse = False
        for j in range(i, Adj.getrow(v).nnz):
            w = Adj.getrow(v).nonzero()[1][j] #First [1] is for column val call
            if index[w] == None:
                work.append((v, j+1))
                work.append((w, 0))
                recurse = True
                break
            elif onstack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if recurse: continue
        if index[v] == lowlink[v]:
            com = []
            while True:
                w = stack[-1]
                del stack[-1]
                onstack[w] = False
                com.append(w)
                groupid[w] = nextgroup
                if w == v: break
            groups.append(com)
            nextgroup += 1
        if work: # NEW: v was recursively visited.
            w = v
            v, _ = work[-1]
            lowlink[v] = min(lowlink[v], lowlink[w])
    return

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
    print("Vertices of GSCC")
    print(GSCC(Adj))
        