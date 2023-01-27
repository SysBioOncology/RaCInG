import numpy as np
from scipy import sparse as sp

def saveGraphUnique(E, V, types, lig, rec, N, filename):
    """
    Method that saves a realisation of our graph model as an npz file. In this
    file the first N x N matrix block contains the adjacency matrix of the graph.
    The entries of this block contains the (linear) index of the connection type.
    The final row first contains all the vertex types, and ends with the number
    of ligand and receptor types, respectively.
    
    NOTE: In the stored adjacency matrix only *unique* connections are stored.

    Parameters
    ----------
    E : np.array with int entries;
        Edge list of the graph.
    V : np.array with int entries;
        Vetex type list of the graph.
    types : np.array with int entries;
        Lig-rec connection type list.
    lig : int;
        Number of ligands.
    rec : int;
        Number of receptors.
    N : int;
        Number of cells.
    filename: string;
        Name of the file where the graph will be stored

    Returns
    -------
    "Graph successfully saved :)", and adds an npz-file of the input graph
    in the current folder.

    """
    print("Saving...")
    #Extract unique edges
    Eunique, indices = np.unique(E, axis = 0, return_index = True)
    #Save edge types in linear format
    lin_index = np.ravel_multi_index(np.transpose(types[indices, :]), (lig, rec))
    #Add the number of ligands and receptors at the end of the vertex type list
    Vaug = np.append(V, np.array([lig, rec]))
    #The data of the matrix will be the linear indices of the edge types, and
    #the vertex types of each vertex together with the ligand and receptor number
    data = np.append(lin_index, Vaug)
    #Places where the data will be added in the sparse matrix
    rows = np.append(Eunique[:, 0], N * np.ones(N + 2))
    cols = np.append(Eunique[:, 1], np.arange(N + 2))
    #Construction matrix in COO-format and immedeately change to CSR-format.
    Adj_ex = sp.coo_matrix((data,(rows,cols)), shape = (N + 1, N + 2)).tocsr()
    sp.save_npz(filename, Adj_ex)
    return print("Graph successfully saved :D")


def loadGraphUnique(filename):
    """
    Load and extract the graph data from an npz-file that was saved through 
    the saveGraphUnique(...) method. 
    
    NOTE: Information about multi-edges is lost through this procedure.

    Parameters
    ----------
    filename : The name of the file you want to load.

    Returns
    -------
    The vertex type list, edge type list, edge lsit, lig No, rec No and vertex
    number from the file.

    """ 
    print("Loading...")
    info = sp.load_npz(filename)
    size = info.get_shape()
    N = size[0] - 1 #Extract N from size of the matrix
    #Extract number of ligands and receptors from the right down entries
    lig, rec = info[N, N], info[N, N + 1]
    #Extract the vertex types from the remaining bottom row
    V = info[N, :N].toarray()
    #Extract edges from the nonzero indices of the N x N block.
    E = np.transpose(np.asarray(info[:N, :N].nonzero()))
    #Extract types from the data stored in the nonzero entries.
    values = info[:N, :N].data
    types = np.transpose(np.asarray(np.unravel_index(values, (lig, rec))))   
    return E, V, types, lig, rec, N

def saveGraphFull(E, V, types, lig, rec, N, filename):
    """
    Method to save a realisation of the random graph model as a (compressed)
    npz-file. In this file the first N x N block consists of the adjaceny matrix
    of the input graph. The next row contains the vertex types and the amount
    of ligand and receptor types. The final row contains information about
    the edge types of the connections (in a row major ordering).
    
    NOTE: Requires slightly more save space than the saveGraphUnique method.

    Parameters
    ----------
    E : np.array with int entries;
        The edge list of the graph to be saved.
    V : np.array with int entries;
        Vertex type list of the graph to be saved.
    types : np.array with int entries;
        Edge types of the edges of our graph (stored as tuples).
    lig : int;
        Number of ligand types.
    rec : int;
        Number of receptor types.
    N : int;
        Number of simulated cells.
    filename : string;
        Name of the file to store the graph in.

    Returns
    -------
    "Graph successfully saved :D", and adds an npz-file with the stored graph
    in the current folder.

    """
    print("Saving...")
    #Sort the edge list by row index. This is needed for row-major type storage.
    arg = np.argsort(E[:,0])
    Esort = E[arg, :]
    #Sort types like rows and convert tuple to linear index.
    typesort = types[arg, :]
    lin_index = np.ravel_multi_index(np.transpose(typesort), (lig, rec))
    #Generate data, rows and cols for creation of sparse COO-matrix
    Vaug = np.append(V, np.array([lig, rec]))
    ones = np.ones(lin_index.size)
    data = np.append(np.append(ones, Vaug), lin_index)
    rows = np.append(np.append(Esort[:,0], N * np.ones(N+2)), (N + 1) * ones)
    cols = np.append(np.append(Esort[:,1], np.arange(N + 2)), np.arange(ones.size))
    #Create desired sparse COO-matrix and turn it into a CSR-matrix
    Adj_ex = sp.coo_matrix((data, (rows, cols)), shape = (N + 2, max(N + 2, ones.size))).tocsr()
    sp.save_npz(filename, Adj_ex)
    return print("Graph successfully saved :D")

def loadGraphFull(filename):
    """
    Load and extract the data from a npz-file that was generate through the
    method saveGraphFull(...).

    Parameters
    ----------
    filename : string;
        Name of the file to be loaded. NOTE: Do not forget the .npz estention.

    Returns
    -------
    E : np.array with int entries;
        Edge list of the graph
    V : np.array with int entries;
        Type of each vertex.
    types : np.array with int entries;
        Ligand and receptor type of each connection given in E.
    lig : int;
        Number of ligand types.
    rec : int;
        Numer of receptor types.
    N : int;
        Number of cells in the simultation.

    """
    print("Loading...")
    info = sp.load_npz(filename).astype(int)
    size = info.get_shape()
    N = size[0] - 2 #Extract N from size of the matrix
    #Extract number of ligands and receptors from the right down entries
    lig, rec = info[N, N], info[N, N + 1]
    #Extract the vertex types from the remaining bottom row
    V = info[N, :N].toarray()
    #Extract types from final row
    types = np.transpose(np.asarray(np.unravel_index(info[N + 1, :].data, (lig, rec))))
    #Construct edge list from the nonzero values of the first N x N block,
    #and duplicate each edge according to the values in this block.
    duplicates = np.asarray(info[:N, :N].data)
    Eunique = np.transpose(np.asarray(info[:N, :N].nonzero()))
    E = np.repeat(Eunique, duplicates, axis = 0)
    return E, V, types, lig, rec, N
    