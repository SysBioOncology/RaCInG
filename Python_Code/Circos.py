# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 08:00:34 2022

@author: s159636
"""
import numpy as np
from RaCInG_input_generation import get_patient_names
import pandas as pd
import csv


def circosTxt(avGroup, cellnames, filename):
    """
    Generates .txt file with the (inflated) Kernel of RaCInG. This can then
    be used to generate a circos plot.

    Parameters
    ----------
    avGroup : np.array 2D;
        Average expression values of kernel per group.
    cellnames : list of str;
        Names of the cells.
    filename : str;
        Name of the output file.

    Returns
    -------
    None. Creates a .txt file with "filename" that can be used to create a
    circos plot.

    """
    with open("{}.txt".format(filename), 'w') as f:
        title = "labels \t "
        for cell in cellnames:
            if cell == "CD8+ T":
                cell = "CD8"
            title = title + cell + " \t "
        f.write(title + "\n")
        
        for i, cell in enumerate(cellnames):
            if cell == "CD8+ T":
                cell = "CD8"
            toWrite = cell + " \t "
            for j in avGroup[i, :]:
                toWrite = toWrite + str(j) + " \t "
            f.write(toWrite + "\n")
    return

def Find_Patient_Subtype_Bagaev(names):
    """
    Finds the "MFP" classification of patients from the Bagaev
    repository (stored in file "Bagaev_mmc6.xlsx"), and outputs this
    classification as a list.

    Parameters
    ----------
    names : list of str;
        The TCGA names and of the patients we want to consider.

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
    b = [a.replace(".", "-") for a in names]
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

def createCircosTxt(cancer_type, group, lam, filename, folder = "RaCInG_Input_Data", feature = 0):
    """
    Creates Circos .txt file from cancer type

    Parameters
    ----------
    cancer_type : str;
        Name of the cancer type.
    group : str;
        Group to analyse.
    lambda : float;
        Inflation factor of the model    
    filename : str;
        Name of the file to save the average kernel to.
    folder : str;
        Name of the folder with metadata
    feature : str;
        Name of meta-feature

    Returns
    -------
    None. Creates a .txt file with the average kernel values of a group.

    """
    load = np.load("kernel_{}.npz".format(cancer_type))
    out = load["prob"]
    sizeOut = out.shape[2]
    out[out == 0] = 0
    cells = load["names"]
    #load2 = np.load("kernel_norm_{}.npz".format(cancer_type))
    #outN = load2["prob"]
    #outN[outN == 0] = 1
    fold = (out) * lam
    if not feature:
        names = get_patient_names(cancer_type, folder)
        subtypes = Find_Patient_Subtype_Bagaev(names)
    else:
        subtypes = Metadata_csv_read_in("metadata_{}.csv".format(cancer_type), feature)
    data = np.average(fold[:,:,np.array(subtypes)[:sizeOut] == group], axis = 2)
    circosTxt(data, cells, filename)
    return

if __name__ == "__main__":
    df = pd.read_csv("SKCM_min_weight_W_bundle.csv")
    dataExact = pd.read_csv("SKCM_min_weight_W_bundle.csv").values[:,1:].astype(float)
    dataApprox = pd.read_csv("SKCM_W_10000_cells_15_deg_data_bundle.csv").values[:,1:].astype(float)
    
    error = (dataExact - dataApprox) / dataExact
    error = error[dataApprox != 1]
    print(len(error))
    error = error[:]
    
    import seaborn as sns
    
    sns.histplot(error[np.abs(error) < 1])
    print(sum(np.abs(error) < 1))