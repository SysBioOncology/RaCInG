"""
File with examples on how to combine all files to generate graphs and 
extract features from them that get saved in raw .txt files.

@author: Mike van Santvoort
"""


import numpy as np
from RaCInG_input_generation import generateInput
from network_generation import model1
import Utilities
import feature_extraction as Loops

def countWedges(Dcell, Dconn, lig, rec, cellnames, N, av, itNo):
    """
    Generates the random graphs and extracts wedge counts for one patient.

    Parameters
    ----------
    Dcell : np.array() with float entries.
        Cell type quantification of one patient.
    Dconn : np.array() with float entries.
        Ligand-receptor interaction distribution.
    lig : np.array() with {0,1} entries.
        cell type compatibility with ligands.
    rec : np.array() with {0,1} entries.
        cell type compatibility with receptors.
    cellnames : np.array() with str entries;
        Array with the names of the different cell types.
    N : int;
        Number of cells.
    av : float;
        Averge degree per cell.
    itNo : int;
        Number of graphs to be generated.

    Returns
    -------
    av_triag : float;
        Average total number of triangles.
    std_triag : float;
        Standard deviation of total number of triangles.
    av_count : np.array() with float entries;
        Average number of triangles subdivided per triangle type.
    std_count : np.array() with float entries;
        Average number of triangles subdivided per triangle type.
    """
    triangletensor = np.zeros((len(cellnames), len(cellnames), len(cellnames), itNo))
    trianglecount = np.zeros(itNo)

    for i in range(itNo):
        V, E, _ = model1(N, av, lig, rec, Dcell, Dconn, [], False)
        Adj = Utilities.EdgetoAdj_No_loop(E, N)
        no, t_list = Loops.Wedges(Adj)
        
        trianglecount[i] = no
        
        triangletensor[:,:,:,i] = Utilities.Count_Types(t_list, V)
        
    av_triag = np.average(triangletensor, axis = 3)
    std_triag = np.std(triangletensor, axis = 3)
    av_count = np.average(trianglecount)
    std_count = np.std(trianglecount)
    return av_triag, std_triag, av_count, std_count

def countTrustTriangles(Dcell, Dconn, lig, rec, cellnames, N, av, itNo):
    """
    Generates the random graphs and extracts trust triangle counts for one patient.

    Parameters
    ----------
    Dcell : np.array() with float entries.
        Cell type quantification of one patient.
    Dconn : np.array() with float entries.
        Ligand-receptor interaction distribution.
    lig : np.array() with {0,1} entries.
        cell type compatibility with ligands.
    rec : np.array() with {0,1} entries.
        cell type compatibility with receptors.
    cellnames : np.array() with str entries;
        Array with the names of the different cell types.
    N : int;
        Number of cells.
    av : float;
        Averge degree per cell.
    itNo : int;
        Number of graphs to be generated.

    Returns
    -------
    av_triag : float;
        Average total number of triangles.
    std_triag : float;
        Standard deviation of total number of triangles.
    av_count : np.array() with float entries;
        Average number of triangles subdivided per triangle type.
    std_count : np.array() with float entries;
        Average number of triangles subdivided per triangle type.
    """
    triangletensor = np.zeros((len(cellnames), len(cellnames), len(cellnames), itNo))
    trianglecount = np.zeros(itNo)

    for i in range(itNo):
        V, E, _ = model1(N, av, lig, rec, Dcell, Dconn, [], False)
        Adj = Utilities.EdgetoAdj_No_loop(E, N)
        no, t_list = Loops.Trust_Triangles(Adj)
        
        trianglecount[i] = no
        
        triangletensor[:,:,:,i] = Utilities.Count_Types(t_list, V)
        
    av_triag = np.average(triangletensor, axis = 3)
    std_triag = np.std(triangletensor, axis = 3)
    av_count = np.average(trianglecount)
    std_count = np.std(trianglecount)
    return av_triag, std_triag, av_count, std_count


def countCycleTriangles(Dcell, Dconn, lig, rec, cellnames, N, av, itNo):
    """
    Generates the random graphs and extracts cycle triangle counts for one patient.

    Parameters
    ----------
    Dcell : np.array() with float entries.
        Cell type quantification of one patient.
    Dconn : np.array() with float entries.
        Ligand-receptor interaction distribution.
    lig : np.array() with {0,1} entries.
        cell type compatibility with ligands.
    rec : np.array() with {0,1} entries.
        cell type compatibility with receptors.
    cellnames : np.array() with str entries;
        Array with the names of the different cell types.
    N : int;
        Number of cells.
    av : float;
        Averge degree per cell.
    itNo : int;
        Number of graphs to be generated.

    Returns
    -------
    av_triag : float;
        Average total number of triangles.
    std_triag : float;
        Standard deviation of total number of triangles.
    av_count : np.array() with float entries;
        Average number of triangles subdivided per triangle type.
    std_count : np.array() with float entries;
        Average number of triangles subdivided per triangle type.
    """
    triangletensor = np.zeros((len(cellnames), len(cellnames), len(cellnames), itNo))
    trianglecount = np.zeros(itNo)

    for i in range(itNo):
        V, E, _ = model1(N, av, lig, rec, Dcell, Dconn, [], False)
        Adj = Utilities.EdgetoAdj_No_loop(E, N)
        no, t_list = Loops.Cycle_Triangles(Adj)
        
        trianglecount[i] = no
        
        triangletensor[:,:,:,i] = Utilities.Count_Types(t_list, V)
        
    av_triag = np.average(triangletensor, axis = 3)
    std_triag = np.std(triangletensor, axis = 3)
    av_count = np.average(trianglecount)
    std_count = np.std(trianglecount)
    return av_triag, std_triag, av_count, std_count

def countDirect(Dcell, Dconn, liglist, reclist, cellnames):
    """
    Generates exact direct communication information for one patient.

    Parameters
    ----------
    Dcell : np.array() with float entries.
        Cell type quantification of one patient.
    Dconn : np.array() with float entries.
        Ligand-receptor interaction distribution.
    lig : np.array() with {0,1} entries.
        cell type compatibility with ligands.
    rec : np.array() with {0,1} entries.
        cell type compatibility with receptors.
    cellnames : np.array() with str entries;
        Array with the names of the different cell types.

    Returns
    -------
    out : np.array() with float entries;
        Faction of direct communictaion covered by given cell types.
    """

    dim1 = len(Dcell)
        
    commexpec = np.zeros((dim1, dim1))
 
    for i, ligrow in enumerate(Dconn):
        for j, ligrec in enumerate(ligrow):
            for k in range(dim1):
                for l in range(dim1):
                    weightlig = np.sum(Dcell * liglist[:,i])
                    weightrec = np.sum(Dcell * reclist[:,j])
                    
                    #Add proportion of connections which will connect
                    #cell type i to j according to limiting value
                    if weightlig != 0 and weightrec != 0:
                        commexpec[k, l] += ligrec * (Dcell[k] * liglist[k,i] / weightlig) \
                            * (Dcell[l] * reclist[l,j] / weightrec)
    
    #Normalize to take care of "missing mass" (lig/rec that could not connect)
    #and were dropped.
    out = commexpec / np.sum(commexpec, axis = (0,1))
    return out

def runSimOne(cancer_type, weight_type, communication_type, pat, N = 10000, itNo = 100, av = 20, norm = False, folder = "Input_data_RaCInG"):
    """
    Provides an example funciton that reads input information, generates graphs,
    extracts their properties, and prints the output in console. This happens
    for one patient only. This implementation is usefull for paralallizing the
    patients for HPC computing.

    Parameters
    ----------
    cancer_type : str;
        Identifier for cancer type.
    weight_type : str;
        Indentifier for weight type used in interaction distribution.
    communication_type : str;
        Type of communication used. Only direct (D), wedge (W), trust triangle (TT),
        and cycle triangle (CT) implemented.
    pat : int;
        Number of the patient.
    N : int; (Optional)
        Number of cells per graph. The default is 10000.
    itNo : int; (Optional)
        Number of different graphs generated per patient. The default is 100.
    av : float; (Optional)
        Average degree per cell. The default is 15.
    norm : bool; (Optional)
        Run simulation with uniform interaction distribution. The default is False.

    Returns
    -------
    None. It prints the results in console.

    """
    #Read input information
    lig, rec, Dcell, Dconn, cells, _, _, _ = generateInput(weight_type, cancer_type, folder = folder)

    cellstring = np.array2string(cells)
    #If normalisation is true, the model is executed the ligand receptor distribution
    #set as a uniform distribution
    if norm:
        normvec = 1 / np.count_nonzero(Dconn, axis = (0,1))
        for i in range(Dconn.shape[2]):
            copy = Dconn[:,:,i]
            copy[copy > 0] = normvec[i]
            Dconn[:,:,i] = copy

        
    CellD = Dcell[pat,:]
    IntD = Dconn[:,:,pat]
    
    if communication_type == "D":
        toWrite = countDirect(CellD, IntD, lig, rec, cells)
        print("{}".format(pat))
        print(cellstring )
        for i, a in enumerate(toWrite):
            for j, b in enumerate(a):
                print("{},{},{}".format(i, j, b))
    elif communication_type == "W":
        av_triag, std_triag, av_count, std_count = countWedges(CellD, IntD, lig, rec, cells, N, av, itNo)
    elif communication_type == "TT":
        av_triag, std_triag, av_count, std_count = countTrustTriangles(CellD, IntD, lig, rec, cells, N, av, itNo)
    else:
        av_triag, std_triag, av_count, std_count = countCycleTriangles(CellD, IntD, lig, rec, cells, N, av, itNo)
        
    if communication_type != "D":
        print("{},{},{}".format(pat, N, av))
        print(cellstring )
        print("Count,{},{}".format(av_count, std_count))

        print("Composition - Average:")
        for i, a in enumerate(av_triag):
            for j, b in enumerate(a):
                for k, c in enumerate(b):
                    print("{},{},{},{}".format(i,j,k,c))
                    
        print("Composition - Std:")
        for i, a in enumerate(std_triag):
            for j, b in enumerate(a):
                for k, c in enumerate(b):
                    print("{},{},{},{}".format(i,j,k,c))
    return

def runSim(cancer_type, weight_type, communication_type, pats = "all", N = 10000, itNo = 100, av = 15, norm = False, folder = "Input_data_RaCInG"):
    """
    Provides an example funciton that reads input information, generates graphs,
    extracts their properties and outputs it in a raw .txt file.

    Parameters
    ----------
    cancer_type : str;
        Identifier for cancer type.
    weight_type : str;
        Indentifier for weight type used in interaction distribution.
    communication_type : str;
        Type of communication used. Only direct (D), wedge (W), trust triangle (TT),
        and cycle triangle (CT) implemented.
    pats : "all" or int; (Optional)
        The number of patients to include in the analysis. If int, it only
        analyses the first int patients. 
    N : int; (Optional)
        Number of cells per graph. The default is 10000.
    itNo : int; (Optional)
        Number of different graphs generated per patient. The default is 100.
    av : float; (Optional)
        Average degree per cell. The default is 15.
    norm : bool; (Optional)
        Run simulation with uniform interaction distribution. The default is False.

    Returns
    -------
    None. It creates a .txt file with all information after simulation.

    """
    #Read input information
    lig, rec, Dcell, Dconn, cells, _, _, _ = generateInput(weight_type, cancer_type, folder = folder)

    cellstring = np.array2string(cells)
    #If normalisation is true, the model is executed the ligand receptor distribution
    #set as a uniform distribution
    if norm:
        normvec = 1 / np.count_nonzero(Dconn, axis = (0,1))
        for i in range(Dconn.shape[2]):
            copy = Dconn[:,:,i]
            copy[copy > 0] = normvec[i]
            Dconn[:,:,i] = copy
            
        if communication_type == "D":
            filename = "{}_D_norm.out".format(cancer_type)
        else:
            filename = "{}_{}_{}_norm.out".format(cancer_type, communication_type, av)
    else:
        if communication_type == "D":
            filename = "{}_D.out".format(cancer_type)
        else:
            filename = "{}_{}_{}.out".format(cancer_type, communication_type, av)
    
    with open(filename, 'w') as f:
        if communication_type == "D":
            f.write("{},{},{}\n".format(weight_type,cancer_type,int(Dcell.shape[0])))
        else:
            f.write("{}\n".format(communication_type))
            f.write("{},{},{},{},{},{}\n".format(cancer_type,weight_type,int(Dcell.shape[0]),N,itNo,av))
        
        if pats == "all":
            limit = Dcell.shape[0]
        else:
            limit = pats
        
        for pat in range(limit):
            if communication_type == "D":
                f.write("{}\n".format(pat))
            else:
                f.write("{},{},{}\n".format(pat, N, av))
                
            f.write(cellstring + "\n")
                
            CellD = Dcell[pat,:]
            IntD = Dconn[:,:,pat]
            
            if communication_type == "D":
                toWrite = countDirect(CellD, IntD, lig, rec, cells)
                for i, a in enumerate(toWrite):
                    for j, b in enumerate(a):
                        f.write("{},{},{}\n".format(i, j, b))
            elif communication_type == "W":
                av_triag, std_triag, av_count, std_count = countWedges(CellD, IntD, lig, rec, cells, N, av, itNo)
            elif communication_type == "TT":
                av_triag, std_triag, av_count, std_count = countTrustTriangles(CellD, IntD, lig, rec, cells, N, av, itNo)
            else:
                av_triag, std_triag, av_count, std_count = countCycleTriangles(CellD, IntD, lig, rec, cells, N, av, itNo)
                
            if communication_type != "D":
                f.write("Count,{},{}\n".format(av_count, std_count))
    
                f.write("Composition - Average:\n")
                for i, a in enumerate(av_triag):
                    for j, b in enumerate(a):
                        for k, c in enumerate(b):
                            f.write("{},{},{},{}\n".format(i,j,k,c))
                            
                f.write("Composition - Std:\n")
                for i, a in enumerate(std_triag):
                    for j, b in enumerate(a):
                        for k, c in enumerate(b):
                            f.write("{},{},{},{}\n".format(i,j,k,c))
        f.write("\n")
    return

if __name__ == "__main__":
    runSim("SKCM", "min", "D", 2, 500, 10, 20, False, "Example Input")