"""
This file generates random distribution for cell types, ligand/receptors and
cell structure. It does not assume any input data. The functions in this file
will be used for testing the random graph generation procedure of RaCInG.
"""
import numpy as np

def genRandomCellTypeDistr(cellTypeNo):
    """
    Generates a random cell type distribution.

    Parameters
    ----------
    cellTypeNo : int;
        The number of cell types you want.

    Returns
    -------
    Dcelltype : np.array() with float entries;
        Vector containing the probabilitiies of the cell types.
    """
    a = np.random.rand(cellTypeNo);
    Dcelltype = a / sum(a);
    return Dcelltype

def genRandomLigRecDistr(ligNo, recNo):
    """
    Generates a random ligand-receptor distribution.

    Parameters
    ----------
    ligNo : int;
        Number of ligands in the model.
    recNo : int;
        Number of receptors in the model.

    Returns
    -------
    Dligrec : 2D np.array() with float entries;
        Matrix containing the probabilities of each ligand-receptor pair.
    """
    a = np.random.rand(ligNo, recNo);
    Dligrec = a / np.sum(a);
    return Dligrec

def genRandomCellLigands(cellTypeNo, ligNo):
    """
    Generates a random cell-to-ligand compatability matrix.
    
    Parameters
    ----------
    cellTypeNo : integer; 
        Number of cell types.
    ligNo : integer; 
        Number of ligands.

    Returns
    -------
    CellStruclig : 2D np.array() with {0, 1} entries.
        A list linking ligand types to the cells they can appear in.
        The code ensures that every ligand at least belongs to one cell type.

    """
    CellStruclig = np.array([[0,0]]);
    while np.any(np.sum(CellStruclig, axis = 1) == 0):
        a = np.random.rand();
        CellStruclig = np.random.choice([0,1], (ligNo, cellTypeNo), p = [a, 1-a]);
    return CellStruclig

def genRandomCellReceptors(cellTypeNo, recNo):
    """
    Generates a random cell-to-receptor compatability matrix.
    
    Parameters
    ----------
    cellTypeNo : integer; 
        Number of cell types.
    recNo : integer; 
        Number of receptors.

    Returns
    -------
    CellStruclig : 2D np.array() with {0, 1} entries.
        A list linking receptor types to the cells they can appear in.
        The code ensures that every receptor at least belongs to one cell type.

    """
    CellStrucrec = np.array([[0,0]]);
    while np.any(np.sum(CellStrucrec, axis = 1) == 0):
        a = np.random.rand();
        CellStrucrec = np.random.choice([0,1], (recNo, cellTypeNo), p = [a, 1-a]);
    return CellStrucrec

def genRandomLigRecCompatibility(ligNo, recNo):
    """
    Generates a random matrix encoding how ligands and receptors can connect.
    
    NOTE: Not useable in RaCInG.
    
    Parameters
    ----------
    ligNo : integer; 
        Number of ligand types
    recNo : integer; 
        Number of receptor types.

    Returns
    -------
    LigRecCompat : 2D np.array() with {0, 1} entries;
        A random ligand-receptor compatibility matrix. When entry
        (i, j) contains a one, this means that ligand i is able to connect with
        receptor j.
    """
    check = True;
    a = np.random.rand();
    
    while check:
        LigRecCompat = np.random.choice([0,1], size = (ligNo, recNo), p = [a, 1 - a]);
        
        if not np.any(np.sum(LigRecCompat, axis = 0) == 0) and not \
            np.any(np.sum(LigRecCompat, axis = 1) == 0):
                check = False;
                
    return LigRecCompat

def genRandomDegreeDistrMeans(ligNo, recNo):
    """
    Generates random local degree for ligands and receptors on a given cell-type.
    
    NOTE: Not usable in RaCInG.

    Parameters
    ----------
    ligNo : int;
        Number of ligand types.
    recNo : int;
        Number of receptor types.

    Returns
    -------
    ligmeans : np.array() with int entries.
        Vector containing the average number of times that ligand i will appear
        on a (given) cell type.
    recmeans : np.array() with int entries.
        Vector containing the average number of times that receptor i will appear
        on a (given) cell type.

    """
    ligmeans = np.random.randint(0, 3, size = ligNo)
    recmeans = np.random.randint(0, 3, size = recNo)   
    return ligmeans, recmeans

def manual_experiment():
    """
    Provides a function that one can call in the other files with a manual
    experiment for RaCInG. 

    Returns
    -------
    Dcelltype : np.array() with float entries summing to 1
        Cell type distribution.
    LigRecDistr : 2D np.array() with float entries
        Matrix with probabilities of certain lig-rec connections appearing.

    """
    Dcelltype = np.array([0.4, 0.3, 0.2, 0.1])
    LigRecDistr = np.array([[0.1, 0.0, 0.0],[0.0, 0.1, 0.1],[0.0, 0.2, 0.0],[0.1, 0.0, 0.2],[0.2, 0.0, 0.0]])
    CellLigConnect = np.array([[1, 0, 0, 0, 1],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[1, 0, 0, 1, 1]])
    CellRecConnect = np.array([[1, 0, 0],[1, 1, 1],[0, 1, 0],[0, 1, 0]])
    return Dcelltype, LigRecDistr, CellLigConnect, CellRecConnect




