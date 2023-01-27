"""
Use these functions to read in the input data for RaCInG's graph generation
procedure.
"""

import csv
import numpy as np
import sys
import os
import pathlib

def sortPermute(stringlist):
    '''
    Sort an array of strings (or numbers) alphabetically, and return
    an array with the permutation that will sort the input list. This
    is an auxiliary method that will help alleviate the issue when different
    input csv files do not have the same headers.

    Parameters
    ----------
    stringlist : list with string entries;
        The list to be sorted.

    Returns
    -------
    sortlist : np.array() with string entries;
        The sorted list.
    perm : np.array() with int entries;
        The permutation that will sort the input list.
    '''
    tupledList = [(stringlist[i], i) for i in range(len(stringlist))]
    tupledList.sort()
    sortlist, perm = zip(*tupledList)
    perm = np.asarray(perm)
    sortlist = np.asarray(sortlist)
    return sortlist, perm

def createCellLigList(filename):
    """
    Reads the .csv file with the cell to ligand connection information.
    To make this method compatible with other methods, we sort the cell-types
    by their name alphabetically. Of course, the connection matrix is then
    permuted in the same way as the sorting permutation to make rows still 
    correspond to the same cell type.

    Parameters
    ----------
    filename : string;
        Name of the .csv file where the cell-ligand connection information is
        stored.

    Returns
    -------
    CellLigList : np.array() with int entries;
        List that depicts whechter cell type i can connect to ligand type j
    ligands : np.array() with string entries;
        Names of ligands of a certain type.
    celltypes : np.array() with string entries;
        Names of cells of a certain type

    """
    #Try to open the file
    try:
        file = open(filename, 'r')
    except:
        sys.exit("ERROR: The file with cell-ligand interactions does not exist...")
     
    #Read the contents of the file
    csvreader = csv.reader(file, delimiter = ',' )
    ligands = []
    ligands = next(csvreader)
    celltypes = []
    CellLigList = []
    for row in csvreader:
        celltypes.append(row[0])
        CellLigList.append(list(map(int, row[1:])))
    file.close()
    
    #Turning the connection list into an np.array() remember the current config
    CellLigList = np.array(CellLigList)
    memory = CellLigList.copy()
    
    #Sort the cell-types by name, and swap the rows of the connection list
    #in the same way (so that each row still corresponds to the same cell type).
    celltypes, perm = sortPermute(celltypes)
    CellLigList = memory[perm, :]
    
    if not len(ligands) == len(CellLigList[0]):
        try:
            ligands.remove('')
        except:
            print("WARNING: List of ligands does not correspond to the amount of observed ligands.")
    
    ligands = np.array(ligands)
        
    return CellLigList, ligands, celltypes


def createCellRecList(filename):
    """
    Reads the .csv file with the cell to receptor connection information.
    To make this method compatible with other methods, we sort the cell-types
    by their name alphabetically. Of course, the connection matrix is then
    permuted in the same way as the sorting permutation to make rows still 
    correspond to the same cell types.

    Parameters
    ----------
    filename : string
        Name of the .csv file where the cell-receptor connection information is
        stored.

    Returns
    -------
    CellRecList : np.array() with int entries;
        List that depicts whechter cell type i can connect to receptor type j
    ligands : np.array() with string entries;
        Names of ligands of a certain type.
    celltypes : np.array() with string entries;
        Names of cells of a certain type
        
    """
    #Try to open file
    try:
        file = open(filename)
    except:
        sys.exit("ERROR: The file with cell-receptor interactions does not exist...")
      
    #Read the information from the file
    csvreader = csv.reader(file, delimiter = ',')
    receptors = []
    receptors = next(csvreader)
    celltypes = []
    CellRecList = []
    
    for row in csvreader:
        celltypes.append(row[0])
        CellRecList.append(list(map(int,row[1:])))

    file.close()
    
    #Turn connection information into np.array() and remember it
    CellRecList = np.array(CellRecList)
    memory = CellRecList.copy()
    
    #Sort cell types and sort rows with the same permultation
    celltypes, perm = sortPermute(celltypes)
    CellRecList = memory[perm, :]
    
    if not len(receptors) == len(CellRecList[0]):
        try:
            receptors.remove('')
        except:
            print("WARNING: List of receptors does not correspond to the amount of observed receptors.")
    
    receptors = np.array(receptors)
    
    return CellRecList, receptors, celltypes

def createCellTypeDistr(cells, filename):
    """
    Read the cell type distribution file. 
    
    NOTE: In case the length of the cell names array does not match with the
    length of the cell names in the cell type distribution file, then it will
    be assumed that the current file contains both M1 and M2 macrophages.
    Their relative quantities will be accumulated into one macrophage (M) class.

    Parameters
    ----------
    filename : string;
        Name of the file where the cell type distribution is in.

    Returns
    -------
    Dtypes : np.array() with float entries;
        The relative distribution of cell types per data-set (rows sum to 1)
    celltypes : np.array() with string entries;
        Names of the cell types

    """
    #Try opening the file
    try:
        file = open(filename)
    except:
        sys.exit("Cell type distribution file does not exist...")
    
    #Read the file
    csvreader = csv.reader(file, delimiter = ',')
    celltypes = []
    celltypes = next(csvreader)
    datasets = []
    Dtypes = []
    
    for row in csvreader:
        datasets.append(row[0])
        Dtypes.append(list(map(float, row[1:])))
        
    if not len(celltypes) == len(Dtypes[0]):
        try:
            celltypes.remove('')
        except:
            print("WARNING: Number of observed cell types does not match given list of cell types")
    
    #Normalize the rows
    nonNorm = np.array(Dtypes)
    normal = np.sum(nonNorm, axis = 1)
    normalized = nonNorm / normal[:,None]
    
    celltypes, perm = sortPermute(celltypes)
    Dtypes = normalized[:, perm]
    

    #These lines of code accumulate M1 and M2 macrophages into one M class.
    #ASSUMPTION: If there are more cell types in one of the two files, then
    #one of the two files contains M1 and M2 macrophages while the other does not.    
    if len(cells) != len(celltypes):
        #print("Warning: accumulating M1 and M2 macrophages into one M class")
        index_M1 = np.where(celltypes == "M1")[0]
        index_M2 = np.where(celltypes == "M2")[0]
        M1_Distr = Dtypes[:, index_M1]
        M2_Distr = Dtypes[:, index_M2]
        Dtypes[:, index_M1] = M1_Distr + M2_Distr
        Dtypes = np.delete(Dtypes, index_M2, axis=1)
        celltypes[index_M1] = "M"
        celltypes = np.delete(celltypes, index_M2)
    
    return Dtypes, celltypes, datasets


def createInteractionDistr(filename, ligands, receptors):
    """
    Input the ligand-receptor appearance data from a .csv input file.
    In this file for each sample the relative ligand receptor expression is
    given in the form [LIG]_[REC]. Hence, this information gets split and
    turned into a matrix when reading in the file.
    
    NOTE: Currently all "NA" values in the input data is replaced with a 0
    expression value.

    Parameters
    ----------
    filename : string;
        Name of the file with the ligand_receptor expression information.
    ligands : np.array() with string entries;
        Names of the ligand types in our model.
    receptors : np.array() with string entries;
        Names of the receptor types in our model.

    Returns
    -------
    The list of probabilities of each ligand-receptor interaction as a matrix.

    """
    #Import ligand and receptor type lists
    ligandtypes = list(ligands)
    receptortypes = list(receptors)
    
    #Try opening files
    try:
        file = open(filename)
    except:
        sys.exit("Interaction distribution file does not exist...")
    
    #Read out files
    csvreader = csv.reader(file, delimiter = ',')
    interactions = []
    interactions = next(csvreader)
    distr = []
    datasets = []
    
    for row in csvreader:
        datasets.append(row[0])
        no_NA_row = [0 if x == "NA" else x for x in row]
        distr.append(list(map(float, no_NA_row[1:])))
    
    if not len(interactions) == len(distr[0]):
        try:
            interactions.remove('')
        except:
            print("WARNING: Number of observed interactions does not match given list of interactions")
     
    #Turns lig-connection indices (in words) into indices.
    indices = [a.split('_') for a in interactions]
    indices = np.array(indices)
    rowWordIndex = list(indices[:, 0])
    columnWordIndex = list(indices[:, 1])
    row = np.array([ligandtypes.index(a) for a in rowWordIndex])
    col = np.array([receptortypes.index(a) for a in columnWordIndex])
    #Now we basically have informatino about all interaction matrices in COO-format
    
    #Store all patient specific data in a big tensor
    distr = np.array(distr)
    DconnectionTensor = np.zeros((len(ligands), len(receptors), len(datasets)))
    for i in range(len(datasets)):
        DconnectionTensor[row, col, i] = distr[i, :]    
    memory = DconnectionTensor.copy()
       
    #Normalise each matrix inside the tensor.
    normsum = np.sum(DconnectionTensor, axis = (0,1))
    DconnectionTensor = memory / normsum[None, None, :]

    return DconnectionTensor

def Read_Lig_Rec_Interaction(filename):
    """
    Reads the sign of each intereaction from the input csv file.
    
    NOTE: Not used in RaCInG.

    Parameters
    ----------
    filename : str;
        Name of the file with the signs.

    Returns
    -------
    sing_matrix : np.array() matrix with {-1, 0, 1} entries.
        Sign of the interaction for ligand i with receptor j.
        +1 : simulating.
        -1 : inhibiting.
        0 : unknown.
    ligand_names : np.array() with str entries;
        List with ligand names.
    receptor_names : np.array() with str entries;
        List with receptor names.
    """
    try:
        file = open(filename)
    except:
        sys.exit("Interaction sign file does not exist...")
    
    csvreader = csv.reader(file, delimiter = ',')
    
    receptor_names = next(csvreader)[1:]
    ligand_names = []
    sign_matrix = []
    
    for row in csvreader:
        ligand_names.append(row[0])
        sign_matrix.append(row[1:])
    
    return np.array(sign_matrix, dtype = int), ligand_names, receptor_names

def generateInput(weight_type, cancer_name, read_signs = False, folder = r"Input_data_RaCInG"):
    """
    Read in all input data for model 1 from the provided .csv files.
    
    NOTE: Use the following file names.
    
    cell -> lig compatibility : celltype_ligand.csv
    cell -> rec compatibility : celltype_receptor.csv
    cell type quantification : [cancer_name]_TMEmod_cell_fractions.csv
    ligand-receptor interaction weights : [cancer_name]_LRpairs_weights_[weight_type].csv
    
    Parameters
    ----------
    weight_type : str;
        The methodology used to compute the ligand/receptor pair weights (min,
        prod, etc.).
    cancer_name : str;
        Name (or abbreviation of) the cancer type used.
    read_signs : bool; (OPTIONAL)
        Indication of a sign file needs to be read.
    folder : str; (OPTIONAL)
        Name of the folder with input data.
    Returns
    -------
    CellLigList : np.array() with {0, 1} entries;
        The matrix that determines whether a ligand can connect to a specific
        cell type.
    CellRecList : np.array() with {0, 1} entries;
        The matrix that determines whether a receptor can connect to a specific
        cell type.
    Dtypes : np.array() with float entries;
        The vector that provides the probability of each cell types appearing.
    DconnectionTensor : np.array() with float entries;
        The matrix that provides the probability of each ligand-receptor
        connection appearing in the generated network.
    celltypes : np.array() with str entries;
        Names of cell types (ordered with the same indices as Dtypes).
    ligands : np.array() with str entries;
        Names of ligands (ordered with the same indices as rows in DconnectionTensor)
    receptors : np.array() with str entries;
        Names of receptors (ordered with the same indices as columns in DconnectionTensor)
    Sign_matrix : np.array() matrix with {0, -1, 1} entries.
        Matrix with the interaction sign of ligand i with receptor j.
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    
    CellLigList, ligands, celltypes = createCellLigList(os.path.join(current_path, \
                                                                     folder, r"celltype_ligand.csv"))
    
    CellRecList, receptors, celltypes2 = createCellRecList(os.path.join(current_path, \
                                                                     folder, r"celltype_receptor.csv"))

    
    Dtypes, celltypes3, bla = createCellTypeDistr(celltypes, os.path.join(current_path, \
                                                                          folder, r"{}_TMEmod_cell_fractions.csv".format(cancer_name)))


    DconnectionTensor = createInteractionDistr(os.path.join(current_path, \
                                               folder, r"{}_LRpairs_weights_{}.csv".format(cancer_name,weight_type)),\
                                               ligands, receptors)
    
    if read_signs:
        Sign_matrix, a, b = Read_Lig_Rec_Interaction(os.path.join(current_path, \
                                                                     folder, r"{}_LRpairs_sign_interaction.csv".formal(cancer_name)))
    else:
        Sign_matrix = np.zeros_like(DconnectionTensor[:,:,0])

    
    return CellLigList, CellRecList, Dtypes, DconnectionTensor, celltypes, ligands, receptors, Sign_matrix

def get_patient_names(cancer_type, folder = r"input_data_RaCInG"):
    """
    Gets the name tags of the patient in the input data.
    
    Parameters
    ----------
    cancer_type : str;
        The name (or abbreviation) of the cancer type being considered.

    Returns
    -------
    names : list of str type;
        Name tags of the patients.

    """
    
    current_path = pathlib.Path(__file__).parent.resolve()
    
    _, _, celltypes = createCellLigList(os.path.join(current_path, \
                                                                     folder, r"celltype_ligand.csv"))
    _, _, names = createCellTypeDistr(celltypes, os.path.join(current_path, \
                                                                          folder, r"{}_TMEmod_cell_fractions.csv".format(cancer_type)))
    return names
        

if __name__ == "__main__":
    #Little script to test reading input
    a, b, c, d, e, lig1, rec1, _ = generateInput("min", "NSCLC")
    #names = get_patient_names("STAD")
    
    
    
    