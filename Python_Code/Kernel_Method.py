"""
In this file all functionalities of the Kernel method are stored. There is a
function to compute the kernel as well as multiple functions to calculate 
graph properties from the kernel.
"""

import numpy as np
import pandas as pd
import scipy as sp
import Utilities as util
from RaCInG_input_generation import generateInput
from RaCInG_input_generation import get_patient_names


def Calculate_kernel(liglist, reclist, Dcell, Dconn, normalize = False):
    """
    Computes the kernel of the random graph model based on RaCInG input 
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
        The kernel is saved as numpy .npz file
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
    

    out = commexpec
    return out

def saveKernel(weight, cancer, folder, test = False):
    """
    Calculates and saves the kernel of a given dataset.

    Parameters
    ----------
    weight : str;
        Weighting used to calculate the LR-matrix.
    cancer : str;
        Cancer type indicator.
    folder : str;
        Location of input data.
    test : bool;
        Determines whether a test run is executed (that reduces the input data
        to only two patients). The default is False

    Returns
    -------
    None. Calculates and saves the kernel (normalized and unnormalized).

    """
    a, b, c, d, _, _, _, _ = generateInput(weight, cancer, folder = folder)
    if test:
        c = c[:2, :]
        d = d[:,:,:2]
    kernel = Calculate_kernel(a, b, c, d, normalize = False)
    unifKernel = Calculate_kernel(a, b, c, d, normalize = True)
    np.savez("kernel_{}.npz".format(cancer), kernel = kernel, unifKernel = unifKernel)
    return

def calculateDirect(cancer, bundle = True, norm = True, test = False):
    """
    Calculates the direct communication values of each patient using the kernel method.

    Parameters
    ----------
    cancer : str;
        Name of cancer type.
    bundle : bool, optional
        Determines whether directionality should be removed. The default is True.
    norm : bool, optional
        Determines whether normalized values should be computed or not
    test : bool;
        Determines whether a test run is executed (that reduces the input data
        to only two patients). The default is False

    Returns
    -------
    df : pandas dataframe
        Dataframe with direct communicatoin (Dir) values per patient.
        
    NOTE: It is tacitly assumed that the input parameter lambda equals 1. For
    normalized values the choice of lambda does not matter. For unnormalized
    values one needs to multiply all outputs by lambda.

    """
    load = np.load("kernel_{}.npz".format(cancer))
    out = load["kernel"]
    _, _, Dcell, _, cells, _, _, _ = generateInput("min", cancer)
    names = get_patient_names(cancer)
    cells[cells == "CD8+ T"] = "CD8"
    
    if test:
        names = names[:2]
        Dcell = Dcell[:2]
    
    
    if norm:
        outN = load["unifKernel"]
        unifData = {}
        columnNames = []
        
        if bundle:
            for i, cell1 in enumerate(cells):
                for i2, cell2 in enumerate(cells[i:]):
                    j=i+i2
                    unifData["Dir_{}_{}".format(cell1, cell2)] = (out[i, j, :]  + out[j, i, :]  ) / (outN[i, j, :] + outN[j, i, :])
                    columnNames.append("Dir_{}_{}".format(cell1, cell2))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_Dir_bundle_norm.csv".format(cancer))
        else:
            for i, cell1 in enumerate(cells):
                for j, cell2 in  enumerate(cells):
                    unifData["Dir_{}_{}".format(cell1, cell2)] = (out[i, j, :]) / (outN[i,j, :])
                    columnNames.append("Dir_{}_{}".format(cell1, cell2))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_Dir_norm.csv".format(cancer)) 
    else:
        unifData = {}
        columnNames = []
        
        if bundle:
            for i, cell1 in enumerate(cells):
                for i2, cell2 in enumerate(cells[i:]):
                    j=i+i2
                    unifData["Dir_{}_{}".format(cell1, cell2)] = Dcell[:, i] * Dcell[:, j] * (out[i, j, :]  + out[j, i, :]  )
                    columnNames.append("Dir_{}_{}".format(cell1, cell2))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_Dir_bundle.csv".format(cancer))
        else:
            for i, cell1 in enumerate(cells):
                for j, cell2 in  enumerate(cells):
                    unifData["Dir_{}_{}".format(cell1, cell2)] = Dcell[:, i] * Dcell[:, j] *(out[i, j, :])
                    columnNames.append("Dir_{}_{}".format(cell1, cell2))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_Dir.csv".format(cancer)) 
    return df

def calculateWedges(cancer, bundle = True, norm = True, test = False):
    """
    Calculates the wedge values of each patient using the kernel method.

    Parameters
    ----------
    cancer : str;
        Name of cancer type.
    bundle : bool, optional
        Determines whether directionality should be removed. The default is True.
    norm : bool, optional
        Determines whether normalized values should be computed or not. The default is True.
    test : bool;
        Determines whether a test run is executed (that reduces the input data
        to only two patients). The default is False

    Returns
    -------
    df : pandas dataframe
        Dataframe with wedge (W) values per patient.
        
    NOTE: It is tacitly assumed that the input parameter lambda equals 1. For
    normalized values the choice of lambda does not matter. For unnormalized
    values one needs to multiply all outputs by lambda^2.


    """
    load = np.load("kernel_{}.npz".format(cancer))
    out = load["kernel"]
    _, _, Dcell, _, cells, _, _, _ = generateInput("min", cancer)
    cells[cells == "CD8+ T"] = "CD8"
    names = get_patient_names(cancer)
    
    if test:
        names = names[:2]
        Dcell = Dcell[:2]
    
    if norm:
        outN = load["unifKernel"]
        unifData = {}
        columnNames = []
        
        if bundle:
            for i, cell2 in enumerate(cells):
                for j, cell1 in enumerate(cells):
                    for k in np.arange(start = j, stop = len(cells)):
                        cell3 = cells[k]
                        unifData["W_{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :]  + out[k, j, :] * out[j, i, :] ) / (outN[i,j, :] * outN[j, k, :] + outN[k,j, :] * outN[j,i, :])
                        columnNames.append("W_{}_{}_{}".format(cell1, cell2, cell3))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_W_bundle_norm.csv".format(cancer))
        else:
            for i, cell1 in enumerate(cells):
                for j, cell2 in  enumerate(cells):
                    for k, cell3 in  enumerate(cells):
                        unifData["W_{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :]) / (outN[i,j, :] * outN[j, k, :])
                        columnNames.append("W_{}_{}_{}".format(cell1, cell2, cell3))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_W_norm.csv".format(cancer))
    else:
        unifData = {}
        columnNames = []
        
        if bundle:
            for i, cell2 in enumerate(cells):
                for j, cell1 in enumerate(cells):
                    for k in np.arange(start = j, stop = len(cells)):
                        cell3 = cells[k]
                        unifData["W_{}_{}_{}".format(cell1, cell2, cell3)] = Dcell[:, i] * Dcell[:, j] * Dcell[:, k] * (out[i, j, :] * out[j, k, :]  + out[k, j, :] * out[j, i, :] )
                        columnNames.append("W_{}_{}_{}".format(cell1, cell2, cell3))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_W_bundle.csv".format(cancer))
        else:
            for i, cell1 in enumerate(cells):
                for j, cell2 in  enumerate(cells):
                    for k, cell3 in  enumerate(cells):
                        unifData["W_{}_{}_{}".format(cell1, cell2, cell3)] = Dcell[:, i] * Dcell[:, j] * Dcell[:, k] * (out[i, j, :] * out[j, k, :])
                        columnNames.append("W_{}_{}_{}".format(cell1, cell2, cell3))
            df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
            df.to_csv("{}_min_weight_W.csv".format(cancer))

    return df

def calculateTriangles(cancer, bundle = True, norm = True, test = False):
    """
    Calculates the triangle values of each patient using the kernel method.

    Parameters
    ----------
    cancer : str;
        Name of cancer type.
    bundle : bool, optional
        Determines whether directionality should be removed. The default is True.
     norm : bool, optional
         Determines whether normalized values should be computed or not. The default is True.
    test : bool;
        Determines whether a test run is executed (that reduces the input data
        to only two patients). The default is False

    Returns
    -------
    df : pandas dataframe
        Dataframe with triangle (Tr) values per patient. If bundle is FALSE, then
        triangles are split up in trust triangles (TT) and cycle triangles (CT)
        
    NOTE: It is tacitly assumed that the input parameter lambda equals 1. For
    normalized values the choice of lambda does not matter. For unnormalized
    values one needs to multiply all outputs by lambda^3.


    """
    load = np.load("kernel_{}.npz".format(cancer))
    out = load["kernel"]
    _, _, Dcell, _, cells, _, _, _ = generateInput("min", cancer)
    cells[cells == "CD8+ T"] = "CD8"
    
    names = get_patient_names(cancer)
    
    if test:
        names = names[:2]
        Dcell = Dcell[:2, :]
    
    
    if norm:
        outN = load["unifKernel"]
        unifData = {}
        columnNames = []
        
        if bundle:
            for i, cell1 in enumerate(cells):
                for i2, cell2 in enumerate(cells[i:]):
                    for i3, cell3 in enumerate(cells[(i + i2):]):
                        j = i + i2
                        k = j + i3
                        unifData["Tr_{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :] * out[k, i, :] + out[i, j, :] * out[j, k, :] * out[i, k, :] + \
                                                                            out[i, j, :] * out[k, j, :] * out[i, k, :] + out[i, j, :] * out[k, j, :] * out[k, i, :] + \
                                                                            out[j, i, :] * out[k, j, :] * out[k, i, :] + out[j, i, :] * out[j, k, :] * out[k, i, :] + \
                                                                            out[j, i, :] * out[j, k, :] * out[i, k, :] + out[j, i, :] * out[k, j, :] * out[i, k, :]) / \
                                                                            (outN[i, j, :] * outN[j, k, :] * outN[k, i, :] + outN[i, j, :] * outN[j, k, :] * outN[i, k, :] + \
                                                                            outN[i, j, :] * outN[k, j, :] * outN[i, k, :] + outN[i, j, :] * outN[k, j, :] * outN[k, i, :] + \
                                                                            outN[j, i, :] * outN[k, j, :] * outN[k, i, :] + outN[j, i, :] * outN[j, k, :] * outN[k, i, :] + \
                                                                                outN[j, i, :] * outN[j, k, :] * outN[i, k, :] + outN[j, i, :] * outN[k, j, :] * outN[i, k, :])
                        columnNames.append("Tr_{}_{}_{}".format(cell1, cell2, cell3))                    
                        df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
                        df.to_csv("{}_min_weight_Tr_bundle_norm.csv".format(cancer))
        else:
            for i, cell1 in enumerate(cells):
                for j, cell2 in enumerate(cells):
                    for k, cell3 in enumerate(cells):
                        unifData["TT_{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :] * out[i, k, :]) / (outN[i, j, :] * outN[j, k, :] * outN[i, k, :])
                        columnNames.append("TT_{}_{}_{}".format(cell1, cell2, cell3))
                        
                        if (i <= j)and(j <= k):
                            unifData["CT_{}_{}_{}".format(cell1, cell2, cell3)] = (out[i, j, :] * out[j, k, :] * out[k, i, :]) / (outN[i, j, :] * outN[j, k, :] * outN[k, i, :])
                            columnNames.append("CT_{}_{}_{}".format(cell1, cell2, cell3))
                       
                        df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
                        df.to_csv("{}_min_weight_Tr_norm.csv".format(cancer))
    else:
        unifData = {}
        columnNames = []
        
        if bundle:
            for i, cell1 in enumerate(cells):
                for i2, cell2 in enumerate(cells[i:]):
                    for i3, cell3 in enumerate(cells[(i + i2):]):
                        j = i + i2
                        k = j + i3
                        unifData["Tr_{}_{}_{}".format(cell1, cell2, cell3)] = Dcell[:, i] * Dcell[:, j] * Dcell[:, k] *(out[i, j, :] * out[j, k, :] * out[k, i, :] + out[i, j, :] * out[j, k, :] * out[i, k, :] + \
                                                                            out[i, j, :] * out[k, j, :] * out[i, k, :] + out[i, j, :] * out[k, j, :] * out[k, i, :] + \
                                                                            out[j, i, :] * out[k, j, :] * out[k, i, :] + out[j, i, :] * out[j, k, :] * out[k, i, :] + \
                                                                            out[j, i, :] * out[j, k, :] * out[i, k, :] + out[j, i, :] * out[k, j, :] * out[i, k, :])
                        columnNames.append("Tr_{}_{}_{}".format(cell1, cell2, cell3))                    
                        df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
                        df.to_csv("{}_min_weight_Tr_bundle.csv".format(cancer))
        else:
            for i, cell1 in enumerate(cells):
                for j, cell2 in enumerate(cells):
                    for k, cell3 in enumerate(cells):
                        unifData["TT_{}_{}_{}".format(cell1, cell2, cell3)] = Dcell[:, i] * Dcell[:, j] * Dcell[:, k] *(out[i, j, :] * out[j, k, :] * out[i, k, :]) 
                        columnNames.append("TT_{}_{}_{}".format(cell1, cell2, cell3))
                        
                        if (i <= j)and(j <= k):
                            unifData["CT_{}_{}_{}".format(cell1, cell2, cell3)] = Dcell[:, i] * Dcell[:, j] * Dcell[:, k] *(out[i, j, :] * out[j, k, :] * out[k, i, :]) 
                            columnNames.append("CT_{}_{}_{}".format(cell1, cell2, cell3))
                       
                        df = pd.DataFrame(data = unifData, columns = columnNames, index = names)
                        df.to_csv("{}_min_weight_Tr.csv".format(cancer))
                        
    return df



def getGSCCAnalytically(cancer, lab = 1, norm = True, test = False):
    """
    Calculates GSCC size based on kernel

    Parameters
    ----------
    cancer : str;
        Name of the cancer type
    lab : float;
        Value of the average degree in RaCInG
    norm : bool;
        Determines whether normalisation takes place or not
    test : bool;
        Determines whether a test run is executed (that reduces the input data
        to only two patients). The default is False

    Returns
    -------
    df : pandas dataframe
        Dataframe with GSCC values per patient.
    """
    load = np.load("kernel_{}.npz".format(cancer))
    out = load["kernel"]
    _, _, Dcell, _, cells, _, _, _ = generateInput("min", cancer)
    cells[cells == "CD8+ T"] = "CD8"
    outN = load["unifKernel"]

    names = get_patient_names(cancer)
    
    if test:
        names = names[:2]
        Dcell = Dcell[:2,:]
    
    sens = len(cells)
    GSCCsizes = np.zeros((len(names), len(cells) + 1))
    for k, name in enumerate(names):

        q = Dcell[k, :]
            
    
        muP = np.zeros((sens, sens))
        muM = muP.copy()
        
    
            
        for i in range(sens):
            for j in range(sens):
                muP[i, j] = lab * out[j, i, k] * q[j]
                muM[i, j] = lab * out[i, j, k] * q[j]
              
        sol = sp.optimize.root(util.poiBPFunc, np.ones(sens), args=(muP, sens))
        x = sol["x"]
        sol = sp.optimize.root(util.poiBPFunc, np.ones(sens), args=(muM, sens))
        y = sol["x"]
        
        GSCCsizes[k,:-1] = x * y * q
        GSCCsizes[k, -1] = np.sum(x * y * q)
    

    
    GSCCsizesN = np.zeros((len(names), len(cells) + 1))
    
    for k, name in enumerate(names):
        q = Dcell[k, :]
            
    
        muP = np.zeros((sens, sens))
        muM = muP.copy()
        
    
            
        for i in range(sens):
            for j in range(sens):
                muP[i, j] = lab *outN[j, i, k] * q[j]
                muM[i, j] = lab * outN[i, j, k] * q[j]
              
        sol = sp.optimize.root(util.poiBPFunc, np.ones(sens), args=(muP, sens))
        x = sol["x"]
        sol = sp.optimize.root(util.poiBPFunc, np.ones(sens), args=(muM, sens))
        y = sol["x"]
        GSCCsizesN[k, :-1] = x * y * q
        GSCCsizesN[k, -1] = np.sum(x * y * q)
        
    if norm:

        normvals = GSCCsizes / GSCCsizesN

        normvals[np.isnan(normvals)] = 1
    
        df = pd.DataFrame(data = normvals, columns = np.append(["GSCC_" + cell for cell in cells], "GSCC"), index = names)
        df.to_csv("{}_min_weight_GSCC_norm.csv".format(cancer))
    else:
        df = pd.DataFrame(data = GSCCsizes, columns = np.append(["GSCC_" + cell for cell in cells], "GSCC"), index = names)
        df.to_csv("{}_min_weight_GSCC.csv".format(cancer))

    return df

if __name__ == "__main__":
    cancer = "SKCM" #SKCM dataset with only first two patients
    weight = "min"
    folder = "Example input"
    test = True
    
    # Test kernel computation
    saveKernel(weight, cancer, folder, test)
    
    # Test wedge computation
    calculateWedges(cancer, norm = False, test = test)
    
    # Test GSCC computation
    getGSCCAnalytically(cancer, test = test)