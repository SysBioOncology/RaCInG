"""
Calculate ligand-recptor contribution to certain communication types exactly.
"""

import numpy as np
from RaCInG_input_generation  import generateInput
from scipy import sparse
import pandas as pd
import csv
from RaCInG_input_generation  import createCellTypeDistr
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os

def Find_Patient_Subtype_Bagaev(data):
    """
    Finds the "MFP" classification of patients from the Bagaev
    repository (stored in file "Bagaev_mmc6.xlsx"), and outputs this
    classification as a list.

    Parameters
    ----------
    datasets : pandas dataframe
        The TCGA names and random graph statistics of the patients we want to consider.

    Returns
    -------
    patient_subtypes : list of str;
        Strings that describe the "MFP" classification of each patient.
        The string "NA" is added if the classification is not known.

    """
    #Read in excel file; skip first row; set second row as header
    #and first column as indices
    repository_data = pd.read_excel("Bagaev_mmc6.xlsx", sheet_name = 1, header = 1, index_col = 0)
    #Transform patient names, because "-01" does not appear in our dataset
    b = [a.replace(".", "-") for a in data]
    patient_names = [a[:12] for a in b]
    
    #Store attribute data in patient_subtypes. Try to locate the data and just
    #write in "NA" if this data does not exist.
    
    patient_subtypes = []
    
    for patient in patient_names:
        try:
            patient_subtypes.append(repository_data.loc[patient, "MFP"])
        except:
            patient_subtypes.append("NA")
    
    return patient_subtypes

def Metadata_csv_read_in(metadata_name, feature, data = "NA"):
    """
    Extracts a given feature from a given csv file.

    Parameters
    ----------
    metadata_name : str;
        Name of the csv file with the metadata.
    feature : str;
        Name of the feature you seek to extract (e.g. MFP or R).
    data : pd dataframe;
           The dataframe with the output from RaCInG. (optional) 
    Returns
    -------
    metadata.values : np.array();
        The array with the metadata of each patient.
    """
   
    #Reading in data of target file
    file = open(metadata_name)
    reader = csv.reader(file)
    header = next(reader)
    rows = []
    for row in reader:
        rows.append(row)
    file.close()
    
    #Transforming raw data into Data Frame
    metadata = pd.DataFrame(data = rows, columns = header)
    try:
        #Transform patient names, because "-01" does not appear in our dataset
        b = [a.replace(".", "-") for a in data.iloc[:,0]]
        patient_names = b
        metas = metadata[feature][np.where(np.isin(metadata["Patient"], np.array(patient_names)))[0]]
    except:
        metas = metadata[feature]
    return metas.values

def calculateBaysianInteractionScoreDirected(liglist, reclist, Dcell, Dconn, cellnames, c1, c2):
    """
    Calculates the Baysian ligand receptor pair scores for directed cell information.
    It computes how much each ligand-receptor pair contributes to each connection
    from c1 to c2.

    Parameters
    ----------
    liglist : np.array() with {0, 1} entries;
        Cell type with ligand compatibility matrix.
    reclist : np.array() with {0, 1} entries;
        Cell type with receptor compatibility matrix.
    Dcell : np.array() with float entries;
        Cell type quantification.
    Dconn : np.array() with float entries;
        Interaction distribution for the ligand-receptor pairs.
    cellnames : np.array() with str entries;
        Array with all cell names.
    c1 : str;
        Name of the sending cell type.
    c2 : str;
        Name of the receiving cell type.

    Returns
    -------
    scoring : np.array() with float entries;
        The baysian scoring of all ligand-receptor pairs given the cell type.
    """
    
    #Make sure that Dcell is always a 2D array and that Dconn is always a 3D array.
    try:
        cellNo = Dcell.shape[1]
        patNo = Dcell.shape[0]

    except:
        cellNo = len(Dcell)
        patNo = 1
        Dcell = Dcell[None, :]
        Dconn = Dconn[:,:,None]
        
    c1 = np.where(cellnames == c1)[0][0]
    c2 = np.where(cellnames == c2)[0][0]
    
    scoring = np.zeros((Dconn.shape[0],Dconn.shape[1],patNo))
    
    #Compute the normalisation constant for the Baysian method
    normlist = np.ones(patNo)
    for patient in range(patNo):
        print(patient)
        factors = np.zeros((Dconn.shape[0],Dconn.shape[1]))
        for i in range(Dconn.shape[0]):
            for j in range(Dconn.shape[1]):
                weightlig = np.sum(Dcell[patient,:] * liglist[:,i])
                weightrec = np.sum(Dcell[patient, :] * reclist[:,j])
                if weightlig != 0 and weightrec != 0:
                    scoring[i, j, patient] =  Dconn[i, j, patient] * liglist[c1, i] * reclist[c2, j] / (weightlig * weightrec)
                    factors[i,j] = Dconn[i, j, patient] * liglist[c1, i] * reclist[c2, j] / (weightlig * weightrec) 
        normlist[patient] = np.sum(factors)
    
    scoring = scoring / normlist[None, None, :]
    
    return scoring

def calculateBaysianInteractionScoreUndirected(liglist, reclist, Dcell, Dconn, cellnames, c1, c2):
    """
    Calculates the Baysian ligand receptor pair scores for undirected cell information.
    It computes how much each ligand-receptor pair contributes to each connection
    between c1 and c2.
    
    This is achieved by averaging over the directed cell type scoring from c1
    to c2 and vice versa.

    Parameters
    ----------
    liglist : np.array() with {0, 1} entries;
        Cell type with ligand compatibility matrix.
    reclist : np.array() with {0, 1} entries;
        Cell type with receptor compatibility matrix.
    Dcell : np.array() with float entries;
        Cell type quantification.
    Dconn : np.array() with float entries;
        Interaction distribution for the ligand-receptor pairs.
    cellnames : np.array() with str entries;
        Array with all cell names.
    c1 : str;
        Name of the sending cell type.
    c2 : str;
        Name of the receiving cell type.

    Returns
    -------
    scoring : np.array() with float entries;
        The baysian scoring of all ligand-receptor pairs given the cell type.
    """
    
    #Make sure that Dcell is always a 2D array and that Dconn is always a 3D array.
    try:
        cellNo = Dcell.shape[1]
        patNo = Dcell.shape[0]
    
    except:
        cellNo = len(Dcell)
        patNo = 1
        Dcell = Dcell[None, :]
        Dconn = Dconn[:,:,None]
        
    c1 = np.where(cellnames == c1)[0][0]
    c2 = np.where(cellnames == c2)[0][0]
    
    scoring = np.zeros((Dconn.shape[0],Dconn.shape[1],patNo))
    scoring2 = np.zeros((Dconn.shape[0],Dconn.shape[1],patNo))
    #Compute the normalisation constant for the Baysian method
    normlist = np.ones(patNo)
    normlist2 = np.ones(patNo)
    for patient in range(patNo):
        print(patient)
        factors = np.zeros((Dconn.shape[0],Dconn.shape[1]))
        factors2 = np.zeros((Dconn.shape[0],Dconn.shape[1]))
        for i in range(Dconn.shape[0]):
            for j in range(Dconn.shape[1]):
                weightlig = np.sum(Dcell[patient,:] * liglist[:,i])
                weightrec = np.sum(Dcell[patient, :] * reclist[:,j])
                if weightlig != 0 and weightrec != 0:
                    scoring[i, j, patient] =  Dconn[i, j, patient] * liglist[c1, i] * reclist[c2, j] / (weightlig * weightrec)
                    scoring2[i, j, patient] =  Dconn[i, j, patient] * liglist[c2, i] * reclist[c1, j] / (weightlig * weightrec)
                    factors[i,j] = Dconn[i, j, patient] * liglist[c1, i] * reclist[c2, j] / (weightlig * weightrec)
                    factors2[i,j] = Dconn[i, j, patient] * liglist[c2, i] * reclist[c1, j] / (weightlig * weightrec) 
        normlist[patient] = np.sum(factors)
        normlist2[patient] = np.sum(factors2)
    
    scoring = scoring / normlist[None, None, :]
    scoring2 = scoring2 / normlist2[None, None, :]
    
    scoring = (scoring + scoring2)/2
    
    return scoring

def findLigRecScoring(liglist, reclist, Dcell, Dconn, cellnames, c1, c2, c3):
    """
    Calculates the Baysian ligand receptor pair scores for triangles.
    It computes how much each ligand-receptor pair contributes to each triangle
    with cell types c1, c2 and c3.

    Parameters
    ----------
    liglist : np.array() with {0, 1} entries;
        Cell type with ligand compatibility matrix.
    reclist : np.array() with {0, 1} entries;
        Cell type with receptor compatibility matrix.
    Dcell : np.array() with float entries;
        Cell type quantification.
    Dconn : np.array() with float entries;
        Interaction distribution for the ligand-receptor pairs.
    cellnames : np.array() with str entries;
        Array with all cell names.
    c1 : str;
        Name of the first cell type in the triangle.
    c2 : str;
        Name of the second cell type in the triangle.
    c3 : str;
        Name of the third cell type in the triangle.

    Returns
    -------
    weights : np.array() with float entries;
        The baysian scoring of all ligand-receptor pairs given the cell type.
    """
    a = calculateBaysianInteractionScoreDirected(liglist, reclist, Dcell, Dconn, cellnames, c1, c2)
    b = calculateBaysianInteractionScoreDirected(liglist, reclist, Dcell, Dconn, cellnames, c2, c3)
    c = calculateBaysianInteractionScoreDirected(liglist, reclist, Dcell, Dconn, cellnames, c1, c3)
    

    weights= np.zeros(a.shape)
    for pat in range(a.shape[2]):
        A = sparse.csr_matrix(a[:,:,pat])
        B = sparse.csr_matrix(b[:,:,pat])
        C = sparse.csr_matrix(c[:,:,pat])
        AB = sparse.kron(A,B).sum()
        BC = sparse.kron(B,C).sum()
        AC = sparse.kron(A,C).sum()
        weights[:,:,pat]= a[:,:,pat] * BC + b[:,:,pat] * AC + c[:,:,pat] * AB
        
    weights = weights / 3
    return weights

def createPicture(weighttype, cancer, c1, c2, c3, g1, g2, feature, thresh, folder, triangle = True):
    """
    Make a bar chart with the most important ligand-receptor interactions for a
    given type of interaction.
    
    NOTE: Currently only direct communication and triangles implemented.

    Parameters
    ----------
    weighttype : str;
        Type of weights (e.g. min or prod).
    cancer : str;
        Abbreviation of cancer type (e.g. SKCM).
    c1 : str;
        Abbreviation of cell type (e.g. CD8).
    c2 : str;
        Abbreviation of cell type (e.g. CD8).
    c3 : str;
        Abbreviation of cell type (e.g. CD8).
    g1 : str;
        Name of group type (e.g. IE).
    g2 : str;
        Name of group type (e.g. IE).
    feature : str;
        Name of the type of feature (e.g. MFP)
    thresh : int;
        Only consider top thresh ligand-receptor pairs.
    folder :
        Name of the folder with the input data
    triangle : bool; (optional)
        Triangles or direct communication? The default is True.

    Returns
    -------
    None. It creates a picture

    """
    #Collect input data and subtypes
    current_path = pathlib.Path(__file__).parent.resolve()
    liglist, reclist, Dcell, Dconn, cellnames, ligands, receptors, _ = generateInput(weighttype, cancer, folder = folder)
    _, _, patients = createCellTypeDistr(cellnames, os.path.join(current_path, folder, r"{}_TMEmod_cell_fractions.csv".format(cancer)))
    
    NoPat = Dcell.shape[0]
    
    try:
        subtypes = Find_Patient_Subtype_Bagaev(patients)
    except:
        subtypes = Metadata_csv_read_in("metadata_{}.csv".format(cancer), feature)
    
    if np.all(np.array(subtypes) == "NA"):
        subtypes = Metadata_csv_read_in("metadata_{}.csv".format(cancer), feature)
    
    
    #Find probabilities
    if triangle:
        weights = findLigRecScoring(liglist, reclist, Dcell, Dconn, cellnames, c1, c2, c3)
    else:
        weights = calculateBaysianInteractionScoreUndirected(liglist, reclist, Dcell, Dconn, cellnames, c1, c2)

    
    #Compute average scores over group
    scoresA = np.average(weights[:,:,np.array(subtypes)[:NoPat] == g1], axis = 2)
    scoresB = np.average(weights[:, :, np.array(subtypes)[:NoPat] == g2], axis = 2)
    
    #Find max scores
    maxIndA = np.unravel_index(np.argsort(scoresA, axis=None), scoresA.shape)
    maxIndB =  np.unravel_index(np.argsort(scoresB, axis=None), scoresB.shape)
    
    #Highlight top tresh interactions and exctract their value and labels    
    bestLigA = ligands[maxIndA[0][-thresh:]]
    bestLigB = ligands[maxIndB[0][-thresh:]]
    bestRecA = receptors[maxIndA[1][-thresh:]]
    bestRecB = receptors[maxIndB[1][-thresh:]]
    maxA1 = scoresA[maxIndA[0][-thresh:], maxIndA[1][-thresh:]]
    maxA2 = scoresA[maxIndB[0][-thresh:], maxIndB[1][-thresh:]]
    maxB1 = scoresB[maxIndA[0][-thresh:], maxIndA[1][-thresh:]]
    maxB2 = scoresB[maxIndB[0][-thresh:], maxIndB[1][-thresh:]]
    maxValA = np.concatenate((maxA1,maxA2))
    maxValB = np.concatenate((maxB1,maxB2))
    plotlabels1 = ["{}_{}".format(bestLigA[i], bestRecA[i]) for i in range(len(bestLigA))]
    plotlabels2 = ["{}_{}".format(bestLigB[i], bestRecB[i]) for i in range(len(bestLigB))]
    plotlabels, indices = np.unique(plotlabels1 + plotlabels2, return_index= True)
    maxValA = maxValA[indices]
    maxValB = maxValB[indices]
    
    dictn = {"Interaction": np.tile(plotlabels, 2),\
             "Group": np.concatenate((np.repeat(g1, len(plotlabels)),np.repeat(g2, len(plotlabels)))),\
                 "Probability": np.concatenate((maxValA, maxValB))}
    df = pd.DataFrame(dictn)
    
    plt.figure(figsize = (10, 10))
    if triangle:
        plt.title("Interaction probabilities for Tr_{}_{}_{} feature in {}".format(c1, c2, c3, cancer))
    else:
        plt.title("Interaction probabilities for {} communication with {} in {}".format(c1, c2, cancer))
    sns.barplot(data = df, x = "Interaction", y = "Probability", hue = "Group")
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig("{}_ligrec.svg".format(cancer))
    return

if __name__ == "__main__":
    #Small test with actual input data
    createPicture("min", "kim", "B", "CD8+ T", "Grr", "R", "NR", "Response", 20, "Example input", triangle = False)
    