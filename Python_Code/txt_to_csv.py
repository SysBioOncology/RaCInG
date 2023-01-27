"""
Functions for creating csv file needed for statistical analysis

@author: Mike van Santvoort
"""
import numpy as np
import csv
from RaCInG_input_generation import generateInput as gi
from RaCInG_input_generation import get_patient_names as pn
import pandas as pd

def Triangle_Prop_Read(filename):
    """
    Reads the raw data (about triangles) from .txt files.

    Parameters
    ----------
    filename : str;
        Name of the file.

    Returns
    -------
    triangle_type : str;
        Triangle type read (e.g. W)
    triangle_data_a : np.array() with float entries;
        Average value of each subfeature.
    triangle_data_s : np.array() with float entries;
        Std of all subfeatures.
    triangle_raw_count : np.array() with float entries;
        For each patient the full count of the triangle_type. The rows are
        patients, the first column is the average count, and the second column
        is the std.
    weight_type : str;
        Type of weight used.
    cancer_type : str;
        Identifier for the type of dataset used.
    N : int;
        Number of cells used per graph.
    itNo : int;
        Number of graphs generated before extracting the features
    av : float;
        Average degree used for each graph.
    """
    
    with open(filename) as f:
        reader = csv.reader(f)
        triangle_type = next(reader) #"NA" if the line is missing
        cancer_type, weight_type, NoPat, N, itNo, av = next(reader)
        
        triangle_data_a = np.zeros((9, 9, 9, int(NoPat)))
        triangle_data_s = np.zeros((9, 9, 9, int(NoPat)))
        triangle_raw_count = np.zeros((int(NoPat), 2))
        
        p = next(reader)
        
        while p:
            p = p[0]
            next(reader)
            _, a_raw, s_raw = next(reader)
            
            triangle_raw_count[int(p), 0] = float(a_raw)
            triangle_raw_count[int(p), 1] = float(s_raw)
            
            next(reader)
            for _ in range(9):
                for _ in range(9):
                    for _ in range(9):
                        i, j, k, a = next(reader)
                        triangle_data_a[int(i), int(j), int(k), int(p)] = float(a)
                        
            next(reader)
            for _ in range(9):
                for _ in range(9):
                    for _ in range(9):
                        i, j, k, a = next(reader)
                        triangle_data_s[int(i), int(j), int(k), int(p)] = float(a)
                        
            p = next(reader)
    
    return triangle_type[0], triangle_data_a, triangle_data_s, triangle_raw_count, weight_type, cancer_type, int(N), int(itNo), float(av)

def Direct_Comm_Limit_Read(filename):
    """
    Reads in theoretical limits from a raw .txt file.

    Parameters
    ----------
    filename : str;
        Name of the file.

    Returns
    -------
    weight : str;
        Weight type used to compute the interaction distribution.
    cancer : str;
        (Abbreviation of) cancer type.
    NoPat : int;
        Number of patients in the dataset.
    direct_comm : np.array() with float entries;
        3D array where each matrix along the second axis is the direct communication
        between cell types in one patient.
    """
    
    with open(filename) as f:
        reader = csv.reader(f)
        comm_type = next(reader)
        weight, cancer, NoPat = next(reader)
        
        direct_comm = np.zeros((9,9, int(NoPat)))
        
        p = next(reader)
        
        while p:
            pat = p[0]
            next(reader)
            
            for _ in range(9):
                for _ in range(9):
                    i, j, a = next(reader)
                    direct_comm[int(i), int(j), int(pat)] = float(a)
                    
            p = next(reader)
        
    return weight, cancer, int(NoPat), direct_comm

def Generate_normalised_count_csv(cancer_type, weight_type, triangle_types, average = 15, noCells = 10000, folder = "Input_data_RaCInG", remove_direction = True):
    """
    Generates a .csv file with normalised entries based on a raw .txt file.
    Only to be used for graphlets consisting of three vertices.

    Parameters
    ----------
    cancer_type : str;
        Identifier of cancer type.
    weight_type : str;
        Identifier of weight type used in generating the interaction distribution.
    triangle_types : List of str;
        Triangle types to accumulate in one output csv.
    average : float; (Optional)
        The average degree used when generating the graphs. The default is 15.
    noCells : int; (Optional)
        The number of cells used per graph. The default is 10000.
    remove_direction : Bool; (Optional)
        Indicates whether directionality should be removed in the features. The default is True.

    Returns
    -------
    df : pandas dataframe
        The dataframe that also has been saved as .csv file.

    """
    
    #Generate input-data
    CellLig,CellRec,Dtypes,Dconn,celltypes,lig,rec,signs = gi(weight_type, cancer_type, folder = folder)
    celltypes[celltypes == "CD8+ T"] = "CD8"

    if remove_direction:
        for index, triangle_type in enumerate(triangle_types): #It is possible to accumulate multiple types by letting triangle types be a list
            #Reading in normalised and non-normalised data
            threetype,av,std,summary,weight,_,cellNo,itNo,avdeg = Triangle_Prop_Read("{}_{}_{}.out".format(cancer_type, triangle_type, average))
            threetype,avN,stdN,summaryN,weight,_,cellNo,itNo,avdeg = Triangle_Prop_Read("{}_{}_{}_norm.out".format(cancer_type, triangle_type, average))
            non_unif_data = {}
            column_names = []
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        non_unif_data["{}_{}_{}".format(celltypes[i], celltypes[j], celltypes[k])] = av[i,j,k,:]
                        column_names.append("{}_{}_{}".format(celltypes[i], celltypes[j], celltypes[k]))
            df = pd.DataFrame(data = non_unif_data)
            
            unif_data = {}
            column_names = []
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        unif_data["{}_{}_{}".format(celltypes[i], celltypes[j], celltypes[k])] = avN[i,j,k,:]
                        column_names.append("{}_{}_{}".format(celltypes[i], celltypes[j], celltypes[k]))
            dfN = pd.DataFrame(data = unif_data)
            
            if index == 0:
                new = pd.DataFrame(index = df.index)
                newN = pd.DataFrame(index = dfN.index)
            
            if triangle_type == "W":
                for col in df.columns:
                    label_elements = col.split('_')
                    if len(label_elements) > 1:
                        to_sort = [label_elements[0], label_elements[2]]
                        label_elements[0] = np.sort(to_sort)[0]
                        label_elements[2] = np.sort(to_sort)[1]
                        new_label = '_'.join(label_elements)
                    else:
                        new_label = col
                    
                    try:
                        new[new_label] += df[col]
                        newN[new_label] += dfN[col]
                    except:
                        new[new_label] = df[col]
                        newN[new_label] = dfN[col]
            else:
                for col in df.columns:
                    new_label = '_'.join(np.sort(col.split('_')))
                    
                    try:
                        new[new_label] += df[col]
                        newN[new_label] += dfN[col]
                    except:
                        new[new_label] = df[col]
                        newN[new_label] = dfN[col]
        
        if len(triangle_types) > 1:
            triangle_types = "Tr"

        
        #Normalising
        av = new.values
        avN = newN.values
        #Find the places where both normalised and non-normalised data has a zero
        zero = np.ravel_multi_index(np.nonzero(av == 0), np.shape(av))
        zeroN = np.ravel_multi_index(np.nonzero(avN == 0), np.shape(avN))
        intersec = np.intersect1d(zero, zeroN)
        indices_intersec = np.unravel_index(intersec, np.shape(av))

        
        #Set both arrays at these places to 1
        av[indices_intersec] = 1
        avN[indices_intersec] = 1
        #If there was no connection in avN, there now is one
        #Assumption: each connection will appear once at least per network
        avN[avN == 0] = 1
        #Normalising
        Norm = av/avN
        #Delete patients where either norm or normal sim decided to stop
        delete_indices = np.unique(np.concatenate((np.nonzero(summary[:,0] == 0)[0], np.nonzero(summaryN[:,0] == 0)[0])))
        print("These patients went wrong for non-normalised: {}".format(np.nonzero(summary[:,0] == 0)))
        print("These patients went wrong for normalised: {}".format(np.nonzero(summaryN[:,0] == 0)))
        Norm = np.delete(Norm, delete_indices, axis = 0)
        patients = np.delete(pn(cancer_type, folder = folder), delete_indices)
        
        df = pd.DataFrame(data = Norm, columns = new.columns, index = patients)
        df.to_csv("{}_{}_{}_cells_{}_deg_data_bundle.csv".format(cancer_type, triangle_types, noCells, average))
        return df
    
    
            
    #Reading in normalised and non-normalised data
    threetype,av,std,summary,weight,cellNo,itNo,avdeg = Triangle_Prop_Read("{}_{}_{}.out".format(cancer_type, triangle_types, average))
    threetype,avN,stdN,summaryN,weight,cellNo,itNo,avdeg = Triangle_Prop_Read("{}_{}_{}_norm.out".format(cancer_type, triangle_types, average))

    #Normalising
    #Find the places where both normalised and non-normalised data has a zero
    zero = np.ravel_multi_index(np.nonzero(av == 0), np.shape(av))
    zeroN = np.ravel_multi_index(np.nonzero(avN == 0), np.shape(avN))
    intersec = np.intersect1d(zero, zeroN)
    indices_intersec = np.unravel_index(intersec, np.shape(av))

    
    #Set both arrays at these places to 1
    av[indices_intersec] = 1
    avN[indices_intersec] = 1
    #If there was no connection in avN, there now is one
    #Assumption: each connection will appear once at least per network
    avN[avN == 0] = 1
    #Normalising
    Norm = av/avN
    
    delete_indices = np.unique(np.concatenate((np.nonzero(summary[:,0] == 0)[0], np.nonzero(summaryN[:,0] == 0)[0])))
    print("These patients went wrong for non-normalised: {}".format(np.nonzero(summary[:,0] == 0)))
    print("These patients went wrong for normalised: {}".format(np.nonzero(summaryN[:,0] == 0)))
    Norm = np.delete(Norm, delete_indices, axis = 3)
    patients = np.delete(pn(cancer_type, folder = folder), delete_indices)
    

    
    #Putting Norm data in dataframe
    Normdata = {}
    column_names = []
    for i in range(9):
        for j in range(9):
            for k in range(9):
                Normdata["{}_{}_{}".format(celltypes[i], celltypes[j], celltypes[k])] = Norm[i,j,k,:]
                column_names.append("{}_{}_{}".format(celltypes[i], celltypes[j], celltypes[k]))
    df = pd.DataFrame(data = Normdata)
    

    df.index = patients
    
    df.to_csv("{}_{}_{}_cells_{}_deg_data.csv".format(cancer_type, triangle_type, noCells, average))
    
    return df

def Generate_direct_communication_csv(cancer_type, weight_type, folder = "Input_data_RaCInG", remove_direction = True):
    """
    Generates the .csv file corresponding to the raw data of (exact) direct
    communication.

    Parameters
    ----------
    cancer_type : str;
        Name of cancer type.
    weight_type : str;
        Weights used when constructing interaction distribution.
    remove_direction : bool; (Optional)
        Do you want to remove directionality in the features. The default is True.

    Returns
    -------
    df : Pandas data frame
        Data frame that has also been saves as .csv file.
    """
    CellLig,CellRec,Dtypes,Dconn,celltypes,lig,rec,signs = gi(weight_type, cancer_type, folder = folder)
    celltypes[celltypes == "CD8+ T"] = "CD8"
    
    _, _, _, comm = Direct_Comm_Limit_Read(r"{}_D.out".format(cancer_type))
    _, _, Pat, commN = Direct_Comm_Limit_Read(r"{}_D_norm.out".format(cancer_type))
    
    
    if np.any(np.sum(comm, axis = (0,1)) < 0.99):
        print("Something is wrong")
        print(np.sum(comm, axis = (0,1)))
    
    
    if remove_direction:
        non_unif_data = {}
        column_names = []
        for i in range(9):
            for j in range(9):
                    non_unif_data["{}_{}".format(celltypes[i], celltypes[j])] = comm[i,j,:]
                    column_names.append("{}_{}".format(celltypes[i], celltypes[j]))
        df = pd.DataFrame(data = non_unif_data)
        
        unif_data = {}
        column_names = []
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    unif_data["{}_{}".format(celltypes[i], celltypes[j])] = commN[i,j,:]
                    column_names.append("{}_{}".format(celltypes[i], celltypes[j]))
        dfN = pd.DataFrame(data = unif_data)
        
        new = pd.DataFrame(index = df.index)
        newN = pd.DataFrame(index = dfN.index)
        
        for col in df.columns:
            new_label = '_'.join(np.sort(col.split('_')))
            
            try:
                new[new_label] += df[col]
                newN[new_label] += dfN[col]
            except:
                new[new_label] = df[col]
                newN[new_label] = dfN[col]
                
        comm = new.values
        commN = newN.values
        
        #Since we are working with limiting values, a zero at one is a zero at the
        #other. Their ratio should be one.
        comm[comm == 0] = 1
        commN[commN == 0] = 1
        Norm = comm/commN
        patients = pn(cancer_type, folder = folder)
        
        df = pd.DataFrame(data = Norm, index = patients, columns = new.columns)
        df.to_csv("{}_{}_weight_direct_communication_bundle.csv".format(cancer_type, weight_type))
        return df
        
    
    #Since we are working with limiting values, a zero at one is a zero at the
    #other. Their ratio should be one.
    comm[comm == 0] = 1
    commN[commN == 0] = 1
    Norm = comm/commN
    patients = pn(cancer_type, folder = folder)
    

    Normdata = {}
    column_names = []
    for i in range(9):
        for j in range(9):
            Normdata["{}_{}".format(celltypes[i], celltypes[j])] = Norm[i, j, :]
            column_names.append("{}_{}".format(celltypes[i], celltypes[j]))
            
    df = pd.DataFrame(data = Normdata)
    df.index = patients
    
    df.to_csv("{}_{}_weight_direct_communication.csv".format(cancer_type, weight_type))
    
    return df

