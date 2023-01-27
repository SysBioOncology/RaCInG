"""
The function in this file will calculate theoretical graph properties based on
input data. Hence, these functions will extract properties without the need of
generating graphs. The properties from this file are also exact. This is one of
the main features being extracted in the RaCInG pipeline.
"""

import numpy as np
import pandas as pd
from RaCInG_input_generation import generateInput
from RaCInG_input_generation import get_patient_names

def Calculate_expected_communication_probabilities(liglist, reclist, Dcell, Dconn, normalize = False):
    """
    Computes the exact direct communication probabilities between
    cell types given mRNA input data of model 1 for all patients. Precicely,
    if N(a,b) is the number of arcs between cell type a and b, then this method
    computes:
        
        \lim_{n \to \infty} N(a, b)/n.
        
    This quantity is equal to \kappa(a, b) q_a q_b, where q_a and q_b are the
    cell type quantifications and \kappa(a, b) is the kernel of the associated
    inhomogeneous random digraph.

    Parameters
    ----------
    liglist : np.array() 2D with {0, 1} entries;
        The matrix indicating whether cells can secrete ligands.
    reclist : 2D np.array() with {0, 1} entries;
        The matrix indicating whether cells can secrete receptors.
    Dcell : np.array() with non-negative float entries summing to 1.
        The relative cell-type quantifications.
    Dconn : 2D np.array() with non-negative float entries summing to 1.
        The probabilities that a ligend of type i connecting to a receptor
        of type j appears in the network.
    normalize : bool; (Optional)
        Indicator whether as input one wants to consider a uniform Dconn 
        matrix (with the same support as the input Dconn).
        
    Returns
    -------
    out : np.array() with float entries;
        Per patient (axis 2) it is given in what proportion cell type i
        communicates with cell type j (axis 0 & 1).
    """
    try:
        dim1 = Dcell.shape[1]
        dim2 = Dcell.shape[0]

    except:
        dim1 = len(Dcell)
        dim2 = 1
        Dcell = Dcell[None, :]
        Dconn = Dconn[:,:,None]
        
    commexpec = np.zeros((dim1, dim1, dim2))
    
    #Changing liglist and reclist to an all 1 list to make sure everything can connect
    if normalize:
        normvec = 1 / np.count_nonzero(Dconn, axis = (0,1))
        for i in range(Dconn.shape[2]):
            copy = Dconn[:,:,i]
            copy[copy > 0] = normvec[i]
            Dconn[:,:,i] = copy

    
    for patient in range(dim2):
        #print(patient)
        for i, ligrow in enumerate(Dconn[:,:,patient]):
            for j, ligrec in enumerate(ligrow):
                for k in range(dim1):
                    for l in range(dim1):
                        weightlig = np.sum(Dcell[patient,:] * liglist[:,i])
                        weightrec = np.sum(Dcell[patient, :] * reclist[:,j])
                        
                        #Add proportion of connections which will connect
                        #cell type i to j according to limiting value
                        if weightlig != 0 and weightrec != 0:
                            commexpec[k, l, patient] += ligrec * (Dcell[patient, k] * liglist[k,i] / weightlig) \
                                * (Dcell[patient, l] * reclist[l,j] / weightrec)
    
    #Normalize to take care of "missing mass" (lig/rec that could not connect)
    #and were dropped.
    out = commexpec / np.sum(commexpec, axis = (0,1))
    return out

def Save_Direct_Communication(out, cellnames, filename):
    """
    Saves the output the expected communication calculation.

    Parameters
    ----------
    out : 3D np.array() with float entries (adding to 1);
        Output of the expected direct communication calculation.
    cellnames : np.array() with str entries;
        Name of the cells in the same order as they appear along the rows of "out".
    filename : str;
        Name of the file to save the results to.

    Returns
    -------
    None. Creates a .npz file with the inputted results.

    """
    np.savez_compressed(filename, names = cellnames, prob = out)
    return


def Calculate_expected_communication_varying_cell(weighttype, cell, valrange, patient):
    """
    Computes direct communication probabilities over a range of cell fraction
    values of one cell type to gain insight into the influence of cell abundance
    on expression value.
    
    NOTE: Not used in RaCInG.

    Parameters
    ----------
    weighttype : str;
        Weight type of the input files
    cell : str;
        The cell type of which one wants to vary the relative abundance.
    valrange : np.array() with float entries between 0 and 1;
        The range of fractions you want the chosen cell to be present.
    patient : int;
        The number of the patient you want to consider.

    Returns
    -------
    cellnames : list of str;
        The names of cell considered in computations
    out : 3D np.array with float entries;
        The communication probabilities between cell types.

    """
    
    liglist, reclist, Dcell, Dconn, cellnames, \
        ligands, receptors, _ = generateInput(weighttype)
    
    commexpec = np.zeros((len(cellnames),len(cellnames),len(valrange)))
    
    for index, val in enumerate(valrange):
        print(val)
        #Changing the number of NK cells and ensuring the other fractions remain the same
        Dcell[patient, cellnames == cell] = val
        Dcell[patient, cellnames != cell] = Dcell[patient, cellnames != cell]/np.sum(Dcell[patient, cellnames != cell]) * (1 - val)
        for i, ligrow in enumerate(Dconn[:,:,patient]):
            for j, ligrec in enumerate(ligrow):
                for k in range(len(cellnames)):
                    for l in range(len(cellnames)):
                        weightlig = np.sum(Dcell[patient,:] * liglist[:,i])
                        weightrec = np.sum(Dcell[patient, :] * reclist[:,j])
                        
                        #Add proportion of connections which will connect
                        #cell type i to j according to limiting value
                        if weightlig != 0 and weightrec != 0:
                            commexpec[k, l, index] += ligrec * (Dcell[patient, k] * liglist[k,i] / weightlig) \
                                * (Dcell[patient, l] * reclist[l,j] / weightrec)
    
    #Normalize to take care of "missing mass" (lig/rec that could not connect)
    #and were dropped.
    out = commexpec / np.sum(commexpec, axis = (0,1))
                                
    np.savez_compressed("Limiting_Direct_Communication_Var_{}_cells_2.npz".format(cell),\
                        names = cellnames, prob = out)
    
    return cellnames, out

def Calculate_kernel(liglist, reclist, Dcell, Dconn, normalize = False):
    """
    Computes the kernel of the random graph model based on mRNA input 
    data for all patients. This kernel forms the backbone of all computations
    done in the random grpah model.

    Parameters
    ----------
    liglist : np.array() 2D with {0, 1} entries;
        The matrix indicating whether cells can secrete ligands.
    reclist : 2D np.array() with {0, 1} entries;
        The matrix indicating whether cells can secrete receptors.
    Dcell : np.array() with non-negative float entries summing to 1.
        The relative cell-type quantifications.
    Dconn : 2D np.array() with non-negative float entries summing to 1.
        The probabilities that a ligend of type i connecting to a receptor
        of type j appears in the network.
    normalize : bool; (Optional)
        Indicator whether as input one wants to consider a uniform Dconn 
        matrix (with the same support as the input Dconn).
        
    Returns
    -------
    out : np.array() with float entries;
        The kernel for each pair of cell interactions in each patient.
    """
    try:
        dim1 = Dcell.shape[1]
        dim2 = Dcell.shape[0]

    except:
        dim1 = len(Dcell)
        dim2 = 1
        Dcell = Dcell[None, :]
        Dconn = Dconn[:,:,None]
        
    commexpec = np.zeros((dim1, dim1, dim2))
    
    #Changing liglist and reclist to an all 1 list to make sure everything can connect
    if normalize:
        normvec = 1 / np.count_nonzero(Dconn, axis = (0,1))
        for i in range(Dconn.shape[2]):
            copy = Dconn[:,:,i]
            copy[copy > 0] = normvec[i]
            Dconn[:,:,i] = copy

    
    for patient in range(dim2):
        print(patient)
        for i, ligrow in enumerate(Dconn[:,:,patient]):
            for j, ligrec in enumerate(ligrow):
                for k in range(dim1):
                    for l in range(dim1):
                        weightlig = np.sum(Dcell[patient,:] * liglist[:,i])
                        weightrec = np.sum(Dcell[patient, :] * reclist[:,j])
                        
                        #Add proportion of connections which will connect
                        #cell type i to j according to limiting value
                        if weightlig != 0 and weightrec != 0:
                            commexpec[k, l, patient] += ligrec * (1 * liglist[k,i] / weightlig) \
                                * (1 * reclist[l,j] / weightrec)
    
    #Normalize to take care of "missing mass" (lig/rec that could not connect)
    #and were dropped.
    out = commexpec
    return out

def calculateWedges(cancer, bundle = True):
    load = np.load("kernel_{}.npz".format(cancer))
    out = load["prob"]
    out[out == 0] = 1
    cells = load["names"]
    cells[cells == "CD8+ T"] = "CD8"
    load2 = np.load("kernel_norm_{}.npz".format(cancer))
    outN = load2["prob"]
    outN[outN == 0] = 1
    names = get_patient_names(cancer)
    
    unifData = {}
    columnNames = []
    
    if bundle:
        for i, cell2 in enumerate(cells):
            for j, cell1 in enumerate(cells):
                for k in np.arange(start = j, stop = len(cells)):
                    cell3 = cells[k]
                    unifData["{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :] + out[k, j, :] * out[j, i, :]) / (outN[i,j, :] * outN[j, k, :] + outN[k,j, :] * outN[j,i, :])
                    columnNames.append("{}_{}_{}".format(cell1, cell2, cell3))
    else:
        for i, cell1 in enumerate(cells):
            for j, cell2 in  enumerate(cells):
                for k, cell3 in  enumerate(cells):
                    unifData["{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :] + out[k, j, :] * out[j, i, :]) / (outN[i,j, :] * outN[j, k, :] + outN[k,j, :] * outN[j,i, :])
                    columnNames.append("{}_{}_{}".format(cell1, cell2, cell3))
                
    df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
    df.to_csv("{}_min_weight_W_bundle.csv".format(cancer))
    return df

if __name__ == "__main__":
    from RaCInG_input_generation import generateInput as gi
    
    a, b, c, d, e, _, _, _ = gi("min", "SKCM", folder = "Example Input")
    out = Calculate_kernel(a,b,c[0,:],d[:,:,0], normalize=False)
    #np.savez_compressed("kernel_STAD.npz", names = e, prob = out)
    
    '''
    #Small example to test the results
    Dcelltype = np.array([0.4, 0.3, 0.2, 0.1])
    LigRecDistr = np.array([[0.1, 0.0, 0.0],[0.0, 0.1, 0.1],[0.0, 0.2, 0.0],[0.1, 0.0, 0.2],[0.2, 0.0, 0.0]])
    CellLigConnect = np.array([[1, 0, 0, 0, 1],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[1, 0, 0, 1, 1]])
    CellRecConnect = np.array([[1, 0, 0],[1, 1, 1],[0, 1, 0],[0, 1, 0]])
    out = Calculate_expected_communication_probabilities(CellLigConnect, CellRecConnect, Dcelltype, LigRecDistr)
    '''
