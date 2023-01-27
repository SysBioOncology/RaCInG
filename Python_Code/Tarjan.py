import numpy as np
from scipy import sparse as sp

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

if __name__ == '__main__':
    #Simple test of the algorithms. Input data is provided in the same way as
    #it would be provided in the pipeline of RaCInG.
    rows = np.array([0,0,1,2,2,3,3,3,4,5])
    cols = np.array([1,2,3,3,4,0,4,5,6,6])
    data = np.ones_like(cols)
    Adj = sp.coo_matrix((data, (rows, cols)), shape = (9,9)).tocsr()
    
    print(GSCC(Adj))
    print(IN(Adj))
    print(OUT(Adj))

 
                