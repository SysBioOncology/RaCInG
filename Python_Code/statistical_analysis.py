# -*- coding: utf-8 -*-
"""
Functions needed for the statistical analysis of the output of RaCInG

@author: Mike van Santvoort
"""

import pandas as pd
import numpy as np
import scipy.stats as stat
import csv
from RaCInG_input_generation import get_patient_names as gp
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from adjustText import adjust_text
import math
from matplotlib import colors
from Kernel_Method import getGSCCAnalytically
#-------------------------
#Preprocessing functions
#-------------------------
def remove_celltype(data, celltype):
    """
    Removes the data of a certain cell type from the data-frame.
    
    Parameters
    ----------
    data : dict of pd dataframes;
        The fold changes of the cell type communications between normalised and non-normalised data. (Input data analysis)

    celltype : string;
        The celltype you want to remove
    
    Returns
    -------
    data : dict of pd dataframes;
        The new data frame with the communications including [celltype] removed.
    """
    
    for cancer in data.keys():
        for col in data[cancer].columns:
            cells = col.split('_')
            if celltype in cells:
                data[cancer].drop(col, inplace = True, axis = 1)
                
    return data


def readAllDataExact(cancer_types, investigation_types, datatype, remove_celltypes = [], weight = "min", bundle = True):
    """
    Reads in data from .csv files generated through the Kernel Method.
    
    Parameters
    ----------
    cancer_types : list with str entries;
        Identifiers of the base group of cancer types.
    investigation_types : list with str entries;
        Identifiers of the cancer types that will be compared with the cancer_type list.
    datatype : list with str entries;
        The names of the datatypes (e.g., wedge; W).
    remove_celltypes : list with str entries; (optional)
        The celltypes to remove from the analysis. The default is [].
    weight : str; (optional)
        Indicator of the weight type used when generating input data. The default is "min".
    bundle : bool; (optional)
        Indicates if directionality was removed or not. The default is True.

    Raises
    ------
    NameError
        If a certain cancer type does not exist, a NameError for the file is
        raised.

    Returns
    -------
    data : dict with pd dataframes;
        A dictionary with a pandas dataframe per entry.
    communication_types : list with str entries;
        The list of features we consider in our analysis (e.g. Dir_CD8_M).
    Nofeatures_global : int;
        Number of features we consider in our analysis. Usefull later for
        correction.
    """
    data = {}
    #Read in all cancer data for pan cancer analysis 
    for cancer in cancer_types:
        try:
            data[cancer] = dataReadExact(datatype, cancer, weight, bundle)
            b = [a.replace(".", "-") for a in data[cancer].iloc[:,0]]
            data[cancer].iloc[:,0] = b
    
        except:
            raise NameError("Data from cancer type {} does not exist...".format(cancer))
            
    #Add data for the investigation cancer types
    for cancer in investigation_types:
        if cancer not in cancer_types:
            try:
                data[cancer] = dataReadExact(datatype, cancer,weight, bundle)
                b = [a.replace(".", "-") for a in data[cancer].iloc[:,0]]
                data[cancer].iloc[:,0] = b
            except:
                raise NameError("Data from cancer type {} does not exist...".format(cancer))
         
        #Remove cell types that we do not want to consider in the investigation
    for cell_type in remove_celltypes:
        data = remove_celltype(data, cell_type)
        
    communication_types = data[cancer].columns[1:] #Array used later to reference the specific communication features
       
    #Print the number of features in each cancer type
    Nofeatures = {}
    for cancer in data.keys():
        print("We have {} unique features in {}.".format(len(data[cancer].columns)-1, cancer))
        Nofeatures[cancer] = len(data[cancer].columns)-1
    Nofeatures_global = Nofeatures[cancer] #Will later be referenced if we need to know the number of features in the analysis
    return data, communication_types, Nofeatures_global

def dataReadExact(datatypes, cancer, weight = "min", bundle = True):
    """
    Read in data from one cancer type out a .csv file
    generated from the Kernel method.

    Parameters
    ----------
    datatypes : list of str;
        Different types of data considered in the analysis (e.g. wedges; W).
    cancer : str;
        Inicator of cancer type to read in.
    weight : str; (optional)
        Indicator of the weight type used when generating input data.
    bundle : bool; (optional)
        Indicates if directionality was removed or not.

    Returns
    -------
    frame : Pandas data frame;
        The data from the csv file in a pandas dataframe.
    """
    
    if len(datatypes) == 1:
        #Get name of data file
        if datatypes[0] == "Dir":
            if bundle:
                data_name = "{}_{}_weight_{}_bundle_norm.csv".format(cancer, weight, datatypes[0])
            else:
                data_name = "{}_{}_weight_{}_norm.csv".format(cancer, weight, datatypes[0])
        else:
            if bundle:
                data_name = "{}_{}_weight_{}_bundle_norm.csv".format(cancer, weight, datatypes[0]) #Name of file with network properties per patient
            else:
                data_name = "{}_{}_weight_{}_norm.csv".format(cancer, weight, datatypes[0])
        return pd.read_csv(data_name)
    else:
        for index, t in enumerate(datatypes):
            #Get name of data file
            if t == "Dir":
                if bundle:
                    data_name = "{}_{}_weight_{}_bundle_norm.csv".format(cancer, weight, t)
                else:
                    data_name = "{}_{}_weight_{}_norm.csv".format(cancer, weight, t)
            else:
                if bundle:
                    data_name = "{}_{}_weight_{}_bundle_norm.csv".format(cancer, weight, t) #Name of file with network properties per patient
                else:
                    data_name = "{}_{}_weight_{}_norm.csv".format(cancer, weight, t)
            if index == 0:
                data = pd.read_csv(data_name)
                vals = data.values
                cols = list(data.columns)
                columns = cols #[t + "_" + col for col in cols]
            else:
                data = pd.read_csv(data_name)
                val = data.values[:,1:]
                vals = np.hstack((vals, val))
                cols = list(data.columns[1:])
                columns.extend(cols)#[t + "_" + col for col in cols])
      
        frame = pd.DataFrame(data = vals, columns = columns)
        return frame

def data_read(datatypes, cancer, noCells = 10000, average = 15, weight = "min", bundle = True):
    """
    Read in data from .csv files (with fold changes).

    Parameters
    ----------
    datatypes : list of str;
        Different types of data considered in the analysis (e.g. wedges; W).
    cancer : str;
        Inicator of cancer type to read in.
    noCells : int; (optional)
        Amount of cells used per simulation.
    average : float; (optional)
        Average degree used per cell.
    weight : str; (optional)
        Indicator of the weight type used when generating input data.
    bundle : bool; (optional)
        Indicates if directionality was removed or not.

    Returns
    -------
    frame : Pandas data frame;
        The data from the csv file in a pandas dataframe.
    """
    
    if len(datatypes) == 1:
        #Get name of data file
        if datatypes[0] == "Dir":
            if bundle:
                data_name = "{}_{}_weight_direct_communication_bundle.csv".format(cancer, weight)
            else:
                data_name = "{}_{}_weight_direct_communication.csv".format(cancer, weight)
        else:
            if bundle:
                data_name = "{}_{}_{}_cells_{}_deg_data_bundle.csv".format(cancer, datatypes[0], noCells, average) #Name of file with network properties per patient
            else:
                data_name = "{}_{}_{}_cells_{}_deg_data.csv".format(cancer, datatypes[0], noCells, average)
        return pd.read_csv(data_name)
    else:
        for index, t in enumerate(datatypes):
            #Get name of data file
            if t == "Dir":
                if bundle:
                    data_name = "{}_{}_weight_direct_communication_bundle.csv".format(cancer, weight)
                else:
                    data_name = "{}_{}_weight_direct_communication.csv".format(cancer, weight)
            else:
                if bundle:
                    data_name = "{}_{}_{}_cells_{}_deg_data_bundle.csv".format(cancer, t, noCells, average) #Name of file with network properties per patient
                else:
                    data_name = "{}_{}_{}_cells_{}_deg_data.csv".format(cancer, t, noCells, average)
            if index == 0:
                data = pd.read_csv(data_name)
                vals = data.values
                cols = list(data.columns)
                columns = [t + "_" + col for col in cols]
            else:
                data = pd.read_csv(data_name)
                val = data.values[:,1:]
                vals = np.hstack((vals, val))
                cols = list(data.columns[1:])
                columns.extend([t + "_" + col for col in cols])
      
        frame = pd.DataFrame(data = vals, columns = columns)
        return frame
    
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
    try:
        b = [a.replace(".", "-") for a in data.iloc[:,0]]
        patient_names = [a[:12] for a in b]
    except:

        b = gp(data) 
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

def Metadata_csv_read_in(metadata_name, feature, data = [], names = 0):
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
    names : list of str;
           List with patient names. (optional)
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
    if len(data):
        #Transform patient names, because "-01" does not appear in our dataset
        b = [a.replace(".", "-") for a in data.iloc[:,0]]
        patient_names = b
        metas = metadata[feature][np.where(np.isin(metadata["Patient"], np.array(patient_names)))[0]]
    elif names:
        metas = metadata[feature][np.where(np.isin(metadata["Patient"], np.array(names)))[0]]
    else:
        metas = metadata[feature]
    return metas.values

def readAllData(cancer_types, investigation_types, datatype, remove_celltypes = [], noCells = 10000, average = 15, weight = "min", bundle = True):
    """
    Parameters
    ----------
    cancer_types : list with str entries;
        Identifiers of the base group of cancer types.
    investigation_types : list with str entries;
        Identifiers of the cancer types that will be compared with the cancer_type list.
    datatype : list with str entries;
        The names of the datatypes (e.g., wedge; W).
    remove_celltypes : list with str entries; (optional)
        The celltypes to remove from the analysis. The default is [].
    noCells : int; (optional)
        Amount of cells used per simulation. The default is 10000.
    average : float; (optional)
        Average degree used per cell. The default is 15.
    weight : str; (optional)
        Indicator of the weight type used when generating input data. The default is "min".
    bundle : bool; (optional)
        Indicates if directionality was removed or not. The default is True.

    Raises
    ------
    NameError
        If a certain cancer type does not exist, a NameError for the file is
        raised.

    Returns
    -------
    data : dict with pd dataframes;
        A dictionary with a pandas dataframe per entry.
    communication_types : list with str entries;
        The list of features we consider in our analysis (e.g. Dir_CD8_M).
    Nofeatures_global : int;
        Number of features we consider in our analysis. Usefull later for
        correction.
    """
    data = {}
    #Read in all cancer data for pan cancer analysis 
    for cancer in cancer_types:
        try:
            data[cancer] = data_read(datatype, cancer, noCells, average, weight, bundle)
            b = [a.replace(".", "-") for a in data[cancer].iloc[:,0]]
            data[cancer].iloc[:,0] = b
    
        except:
            raise NameError("Data from cancer type {} does not exist...".format(cancer))
            
    #Add data for the investigation cancer types
    for cancer in investigation_types:
        if cancer not in cancer_types:
            try:
                data[cancer] = data_read(datatype, cancer, noCells, average, weight, bundle)
                b = [a.replace(".", "-") for a in data[cancer].iloc[:,0]]
                data[cancer].iloc[:,0] = b
            except:
                raise NameError("Data from cancer type {} does not exist...".format(cancer))
         
        #Remove cell types that we do not want to consider in the investigation
    for cell_type in remove_celltypes:
        data = remove_celltype(data, cell_type)
        
    communication_types = data[cancer].columns[1:] #Array used later to reference the specific communication features
       
    #Print the number of features in each cancer type
    Nofeatures = {}
    for cancer in data.keys():
        print("We have {} unique features in {}.".format(len(data[cancer].columns)-1, cancer))
        Nofeatures[cancer] = len(data[cancer].columns)-1
    Nofeatures_global = Nofeatures[cancer] #Will later be referenced if we need to know the number of features in the analysis
    return data, communication_types, Nofeatures_global

def addMetadata(data, cancer_types, investigation_types, feature_pan, feature_inv):
    """
    Adds desired metadata at the end of an existing dataframe.

    Parameters
    ----------
    data : dict with Pandas dataframes;
        A dictionary with RaCInG output data per entry.
    cancer_types : list with str entries;
        Identifiers of the base group of cancer types.
    investigation_types : list with str entries;
        Identifiers of the cancer types that will be compared with the cancer_type list.
    feature_pan : str;
        Identifier for the type of metadata one wants to consider for the
        large group of cancer types (e.g., MFP).
    feature_inv : str;
        Identifier for the type of metadata one wants to consider for the
        smaller group of cancer types (e.g., MFP).

    Returns
    -------
    data : dict with Pandas dataframes;
        A dictionary with RaCInG output data per entry. Metadata is placed in
        the final column of the dataframes.

    """
    for cancer in cancer_types:
        try:
            meta_data = Find_Patient_Subtype_Bagaev(data[cancer]) #Getting meta-data (MFP) from Bagaev xslx-file.
            data[cancer][feature_pan] = meta_data #Adding meta-data at the end of the data-frame
        except:
            print("Bagaev did not work for {}".format(cancer))
            pass
    
    
        try:
            meta_data = Metadata_csv_read_in("metadata_{}.csv".format(cancer), feature_pan, data = data[cancer])
            data[cancer][feature_pan] = meta_data
        except:
            print("Metadatafile not found for {}".format(cancer))
            pass
        
    for cancer in investigation_types:
        try:
            meta_data = Find_Patient_Subtype_Bagaev(data[cancer]) #Getting meta-data (MFP) from Bagaev xslx-file.
            data[cancer][feature_inv] = meta_data #Adding meta-data at the end of the data-frame
        except:
            print("Bagaev did not work for {}".format(cancer))
            pass
    
    
        try:
            meta_data = Metadata_csv_read_in("metadata_{}.csv".format(cancer), feature_inv, data = data[cancer])
            data[cancer][feature_inv] = meta_data
        except:
            print("Metadatafile not found for {}".format(cancer))
            pass
    return data

def removeOutliers(data, threshold = 50):
    """
    Removes patients from the cvs that show extreme output values. This function
    was needed, since the HPC cluster has a tendacy to fail randomly, resulting
    in some patients with extreme output results.

    Parameters
    ----------
 data : dict with Pandas dataframes;
     A dictionary with RaCInG output data per entry. Metadata is placed in
     the final column of the dataframes.
    threshold : float; (optional)
        The treshold value for the fold change in the cvs file. The default
        is 50.

    Returns
    -------
    data : dict with Pandas dataframes;
        The data for the statistical analysis with for each cancer types the
        extreme patients removed.
    """
    for cancer in data.keys():
        wrong = []
        for column in data[cancer].columns:
            try:
                wrong.extend(data[cancer][data[cancer][column] > threshold].index.values.tolist())
                #wrong.extend(data[cancer][data[cancer][column] < 1/wrongthresh].index.values.tolist())
            except:
                pass
        
        data[cancer].drop(np.unique(wrong), axis = 0, inplace=True)
        print("I dropped {} bad patients in {}".format(len(wrong), cancer))
        print("The maximum in {} currently is {}".format(cancer, np.amax(data[cancer].iloc[:,1:].values)))
        print("The minimum in {} currently is {}".format(cancer, np.amin(data[cancer].iloc[:,1:].values)))
    return data

#--------------------
#Correlation analysis
#--------------------
def computeCorrelation(data, thresh = 20, save = False):
    """
    Computes the correlation between immune response score and the RaCInG
    feature values.

    Parameters
    ----------
    data : dict with pandas dataframe entries;
        The data to be used in the analysis.
    thresh : int; (optional)
        The amount of correlations to plot. The default is 20.
    save : bool; (optional)
        Save the figure or not. The default is false.

    Returns
    -------
    None. Creates a figures and saves it.

    """
    IR = {}

    #Add immune response to each cancer type in the dataframe
    for cancer in data.keys():
    
        immune_response = pd.read_csv("immuneresponse_score_{}.csv".format(cancer))
    
        new_column = ["NA"]*data[cancer].shape[0]
        for place, label in enumerate(immune_response.iloc[:,0]):
            try:
                row = np.nonzero(data[cancer].iloc[:,0].values == label)[0][0]
                new_column[row] = immune_response.iloc[place, 1]
            except:
                pass
    
        IR[cancer] = np.array(new_column)
    
    #Compute spearman rho correlation between immune response score and 
    #feature values for each feature per cancer type
    correlations = {}
    
    for cancer in data.keys():
        correlations[cancer] = [] #Records spear rho corr (col 0) and p-value (col 1)
        ir_score = IR[cancer]
        refCancer = cancer
    
        for col in data[cancer].columns[1:-1]:
                feat_val = data[cancer].loc[:, col].values.astype(float)
                rho, _ = stat.spearmanr(feat_val, ir_score)
                correlations[cancer].append(rho)
    
    correlations["pan"] = []
    for col in data[refCancer].columns[1:-1]:

        pandata = [] #Sample all data from all cancer types
        irScores = [] #corresponding IR scores
        
        for cancer in data.keys():
            pandata.extend(data[cancer].loc[:, col].values.astype(float))
            irScores.extend(IR[cancer])
            
        rho, _ = stat.spearmanr(pandata, irScores)
        correlations["pan"].append(rho)
    
    keys = list(data.keys())
    keys.append("pan")
    #Create heatmatps with largest correlations
    for cancer in keys:
        try:
            labels = data[cancer].columns[1:-1]
        except:
            labels = data[keys[0]].columns[1:-1]
        highlightsBig = np.argsort(correlations[cancer])[-(round(thresh/2)):]
        highlightsSmall = np.argsort(-np.array(correlations[cancer]))[-(round(thresh/2)):]
        highlights = np.append(highlightsBig, highlightsSmall)
        
        fig, ax = plt.subplots(1,1)
        
        x = np.arange(thresh + 1)
        
        best_corr = np.array(correlations[cancer])[highlights]
        types = labels[highlights]
        
        sorted_data = np.sort(best_corr)
        sorted_types = types[np.argsort(best_corr)]
        
        try:
            empty_index = np.nonzero(sorted_data < 0)[0][-1]
        except:
            empty_index = -1
            
        colors_rainfall = np.array(['r'] * (thresh + 1))
        colors_rainfall[empty_index + 1 :] = 'g'
        
        sorted_data = np.insert(sorted_data, empty_index + 1, 0)
        sorted_types = np.insert(sorted_types, empty_index + 1, ' ')
        ax.bar(x, sorted_data, width = 0.9, tick_label = sorted_types, color = colors_rainfall)
        ax.set_title("{}".format(cancer))
        ax.set_ylabel("Spearman rho correlation")
        ax.set_ylim([-1, 1])
        plt.xticks(rotation=90)
        
        if save:
            plt.savefig("Correlation_IR_features_rainfall_{}.svg".format(cancer), bbox_inches='tight')
    return

#------------------------
# Wilcoxon rank sum test
#------------------------
def groupPatients(data, cancer_types, investigation_types, Group_Names_pan, Group_Names_inv, feature_pan, feature_inv):
    """
    Groups patients in groups corresponding to metadata.

    Parameters
    ----------
    data : dict with pandas dataframe entries;
          The data to be used in the analysis.
    cancer_types : list with str entries;
         Identifiers of the base group of cancer types.
    investigation_types : list with str entries;
         Identifiers of the cancer types that will be compared with the cancer_type list.
   Group_Names_pan : list of str;
         Names of the groups in the pan cancer set (the large set).
   Group_Names_inv : list of str;
        Names of the group in the cancer set to be compared with the large set.
   feature_pan : str;
       Identifier for the type of metadata one wants to consider for the
       large group of cancer types (e.g., MFP).
   feature_inv : str;
       Identifier for the type of metadata one wants to consider for the
       smaller group of cancer types (e.g., MFP).

    Raises
    ------
    NameError
        A name error is raised when certain groups do not exist.

    Returns
    -------
    Groups : dict of dict with pandas dataframe entries;
        A dictionary where per cancer type and per group only the data is
        given with the patients from said group.

    """
    #For each considered cancer-type, split up the patients in their respective groups
    Groups={}
    for cancer in cancer_types:
        Groups[cancer] = {}
        for group in Group_Names_pan:
            try:
                Groups[cancer][group] = data[cancer][data[cancer][feature_pan] == group]
                print("In cancer {} group {} has size {}".format(cancer, group, len(Groups[cancer][group])))
            except:
                raise NameError("Patients from group {} do not exist...".format(group))
                
    for cancer in investigation_types:
        Groups[cancer] = {}
        for group in Group_Names_inv:
            try:
                Groups[cancer][group] = data[cancer][data[cancer][feature_inv] == group]
                print("In cancer {} group {} has size {}".format(cancer, group, len(Groups[cancer][group])))
            except:
                raise NameError("Patients from group {} do not exist...".format(group))
    return Groups

def createPanCancerData(Groups, cancer_types, Group_Names_pan, Nofeatures_global):
    """
    Bundles the data together of the all cancer types in the "pan cancer group".

    Parameters
    ----------
    Groups : dict of dict with pandas dataframe entries;
        A dictionary where per cancer type and per group only the data is
        given with the patients from said group.
    cancer_types : list with str entries;
        Identifiers of the base group of cancer types.
     Group_Names_pan : list of str;
        Names of the groups in the pan cancer set (the large set).
    Nofeatures_global : int;
        The number of features we are investigating in total.

    Returns
    -------
    pan_cancer_data : dict with pandas dataframes;
        The dat of all cancers in the pan cancer analysis grouped together.
    pan_cancer_labels : dict with np.arrays;
        The labels that say for each comparison which patients belongs in
        which group.
    """
    pan_cancer_data = {}
    pan_cancer_labels = {}
    for number, group1 in enumerate(Group_Names_pan):
        for group2 in Group_Names_pan[(number+1):]:
            #try:
                #For each comparison, group the data of all cancers together with labels of the immune phenotypes
                combined_data = np.zeros((1, Nofeatures_global))
                labels_data = np.array([2])
                for cancer in cancer_types:
                    for index, group in enumerate([group1, group2]):
                        group_data = Groups[cancer][group].to_numpy()[:,1:-1].astype(float)
                        combined_data = np.vstack((combined_data, group_data))
                        labels_data = np.hstack((labels_data, np.repeat(index, group_data.shape[0])))
                
                #Delete the dummy row (to set dimensions)
                pan_cancer_data["{}_{}".format(group1, group2)] = np.delete(combined_data, 0, axis = 0)
                pan_cancer_labels["{}_{}".format(group1, group2)] = np.delete(labels_data, 0, axis = 0)
            #except:
            #    print("{} vs. {} comparison failed".format(group1, group2))
            #    pass
    return pan_cancer_data, pan_cancer_labels

def wilcoxon(data, Groups, investigation_types, Group_Names_inv, pan_cancer_data, pan_cancer_labels, Group_Names_pan, communication_types, Nofeatures_global, Cross_feature_analysis = False, alpha = 0.05):
    """
    Executes the Wilcoxon rank sum test to compare groups in the dataset. One
    either compare differences between group in the same cancer type, or the
    same groups across cancer types.

    Parameters
    ----------
    data : dict with pandas dataframes;
        The data for each cancer type.
    Groups : dict of dict with pandas dataframe;
        The data of "data" split up into their respective groups.
    investigation_types : list with str entries;
        The names of the cancer types in the smaller set (that is compared with
        the pan cancer set).
    Group_Names_inv : list with str entries;
        The list of the group names in the smaller set.
    pan_cancer_data : dict with pandas dataframes;
        Bundeled pan cancer data split into different groups.
    pan_cancer_labels : dict with np.arrays;
        The labels that indicate which patient in pan_cancer_data belongs to
        which group.
     Group_Names_pan : list of str;
        Names of the groups in the pan cancer set (the large set).
    communication_types : list of str;
        Names of the different features we considered.
   Nofeatures_global : int;
       The number of features we are investigating in total.
    alpha : float; (optional)
        The significance level to test at. The default is 0.05.
    Cross_feature_analysis : bool; (optional)
        If True, compares differences of the same feature across cancer types. 
        The default is False.
        
        NOTE: Only works if there are only two features (e.g. R vs. NR).

    Returns
    -------
    p-values, fold changes, and significant features of the tests.

    """


    if Cross_feature_analysis:
    #For each group comparison in each cancer type we compute the p-value of the Mann-Whitney U test
    
        pvals = {} #Values from Mann-Whitney-U test (corrected for multiple hypothesis testing)
        fold_change = {} #How many times are values from group A higher than B (on average)
        signif = {} #Which are the significant features
    
        for cancer in investigation_types:
            pvals[cancer] = {}
            fold_change[cancer] = {}
            signif[cancer] = {}
            for i, G1 in enumerate(Group_Names_inv):
                #Per MFP group, execture the Mann-Whitney-U test for each feature
                    G2 = Group_Names_inv[1 - i]
                    p = []
                    f = []
                    cols = []
                    for index, att in enumerate(data[cancer].columns[1:-1]):
                        try:
                            t1 = Groups[cancer][G1].loc[:, att]
                            t2 = pan_cancer_data["{}_{}".format(G1, G2)][pan_cancer_labels["{}_{}".format(G1, G2)] == i, index]
                            f.append(np.mean(t1) / np.mean(t2))
                            _, val = stat.ranksums(t1,t2) #Maybe Willcoxon Rank sum test
                            p.append(val)
                            cols.append(att) #The names of the stored pathways in order
                        except:
                            t1 = Groups[cancer][G1].loc[:, att]
                            t2 = pan_cancer_data["{}_{}".format(G2, G1)][pan_cancer_labels["{}_{}".format(G2, G1)] == i, index]
                            f.append(np.mean(t1) / np.mean(t2))
                            _, val = stat.ranksums(t1,t2) #Maybe Willcoxon Rank sum test
                            p.append(val)
                            cols.append(att) #The names of the stored pathways in order
    
                    pvals[cancer]["{}".format(G1)] = p
                    fold_change[cancer]["{}".format(G1)] = f
    
                    #Find singificant values using Bonferroni correction
                    signif[cancer]["{}".format(G1)] = np.nonzero(np.array(p) <= alpha / Nofeatures_global)[0]
    
        return pvals, fold_change, signif
    else:
        #For each group comparison in each cancer type we compute the p-value of the Mann-Whitney U test
    
        pvals = {} #Values from Mann-Whitney-U test (corrected for multiple hypothesis testing)
        fold_change = {} #How many times are values from group A higher than B (on average)
        signif = {} #Which are the significant features
    
        for cancer in investigation_types:
            pvals[cancer] = {}
            fold_change[cancer] = {}
            signif[cancer] = {}
            for i, G1 in enumerate(Group_Names_inv):
                for G2 in Group_Names_inv[(i+1):]:
                #Per MFP group, execture the Mann-Whitney-U test for each feature
                    p = []
                    f = []
                    cols = []
                    for att in data[cancer].columns:
                        t1 = Groups[cancer][G1].loc[:, att]
                        t2 = Groups[cancer][G2].loc[:, att]
    
                        try:
                            f.append(np.mean(t1) / np.mean(t2))
                            _, val = stat.ranksums(t1,t2) #Maybe Willcoxon Rank sum test
                            p.append(val)
                            cols.append(att) #The names of the stored pathways in order
                        except:
                            pass
    
                    pvals[cancer]["{}_{}".format(G1, G2)] = p
                    fold_change[cancer]["{}_{}".format(G1, G2)] = f
    
                    #Find singificant values using Bonferroni correction
                    signif[cancer]["{}_{}".format(G1, G2)] = np.nonzero(np.array(p) <= alpha / Nofeatures_global)[0]
        
        #Computing pan cancer p-values and fold change
        pan_cancer_p = {}
        pan_cancer_fold = {}
        pan_cancer_signif = {}
        for i, G1 in enumerate(Group_Names_pan):
            for G2 in Group_Names_pan[(i+1):]:
    
                #Per MFP group, execture the Mann-Whitney-U test for each feature
                p = []
                f = []
                cols = []
                for index, att in enumerate(communication_types):
                    t1 = pan_cancer_data["{}_{}".format(G1, G2)][pan_cancer_labels["{}_{}".format(G1, G2)] == 0, index]
                    t2 = pan_cancer_data["{}_{}".format(G1, G2)][pan_cancer_labels["{}_{}".format(G1, G2)] == 1, index]
    
                    f.append(np.mean(t1) / np.mean(t2))
                    _, val = stat.ranksums(t1,t2) #Maybe Willcoxon Rank sum test
                    p.append(val)
    
                pan_cancer_p["{}_{}".format(G1, G2)] = p
                pan_cancer_fold["{}_{}".format(G1, G2)] = f
    
                #Find significant values using Bonferroni correction
                pan_cancer_signif["{}_{}".format(G1, G2)] = np.nonzero(np.array(p) <= alpha / Nofeatures_global)[0]        
        return pvals, fold_change, signif, pan_cancer_p, pan_cancer_fold, pan_cancer_signif
    
#-------------------------
# Volcano plot creation
#-------------------------
def volcanoPan(pan_cancer_p, pan_cancer_fold, pan_cancer_signif, Group_Names_pan, communication_types, datatype, feature_pan, Pan_cancer_name,  Nofeatures_global, alpha = 0.05, thresh = 20, save = False):
    """
    Generates volcato plots comparing the different groups within the pan cancer
    bundle of cancer types.

    Parameters
    ----------
    pan_cancer_p : dict with np.arrays;
        p-values of all the comparisions between groups.
    pan_cancer_fold : dict with np.array;
        Fold changes of average expression levels in each group.
    pan_cancer_signif : dict with lists;
        List of siginificantly different features for each comparison.
    Group_Names_pan : list of str;
        Names of all groups in the pan cancer analysis.
    communication_types : list of str;
        Names of all the features considered in the comparison.
    datatype : list of str;
        Names of the datatypes considered (e.g. wedge; W).
    feature_pan : str;
        Name of the metadata feature used in the pan cancer comparison.
    Pan_cancer_name : str;
        Indicator of how to call the pan cancer bundle.
    Nofeatures_global : int;
        Amount of feature we consider at the same time.
    alpha : float, optional
        Significance level of the tests. The default is 0.05.
    thresh : int, optional
        Amount of top features to label. The default is 20.
    save : bool, optional
        Saving the figure or not. The default is False.

    Returns
    -------
    None. It may save the generated volcano plot.

    """
    
    counter = 0
    row_length = max(1, int(np.ceil(math.comb(len(Group_Names_pan), 2))/2))
    
    
    if len(Group_Names_pan) == 2:
        fig, ax = plt.subplots(1, row_length)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for i, G1 in enumerate(Group_Names_pan):
            for j, G2 in enumerate(Group_Names_pan):
                try:
                    ydata = -np.log(pan_cancer_p["{}_{}".format(G1,G2)])/np.log(10)
                    xdata = np.log(pan_cancer_fold["{}_{}".format(G1,G2)])/np.log(2)
    
                    #Get most important features in dataset
                    important_features = np.intersect1d(np.argsort(ydata)[-(thresh):], pan_cancer_signif["{}_{}".format(G1,G2)])
    
                except:
                    continue
    
                ax.scatter(xdata, ydata, marker='x', color = "gainsboro")
                ax.axline((-2, -np.log(alpha/Nofeatures_global)/np.log(10)), slope = 0, color = 'r')
                ax.set_xlim([-1.5,1.5])
                [ax.scatter(xdata[col], ydata[col], color = "blue") for col in important_features]
                texts = [ax.text(xdata[col], ydata[col], communication_types[col]) for col in important_features]
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle = '-', color='black'))
                counter += 1
    
        fig.suptitle("{} comparison {}".format(feature_pan, Pan_cancer_name))
        fig.supxlabel("2-log of fold change")
        fig.supylabel("Negative 10-log of p-value")
        fig.tight_layout()
        if save:
            plt.savefig("Volcano_plots_pan_cancer_seperate_comparisons_{}_{}_bundle.png".format(feature_pan, datatype))
        
        
        
        
    else:    
        fig, ax = plt.subplots(2, row_length)
        ax = ax.ravel()
        fig.set_figheight(15)
        fig.set_figwidth(20)
        for i, G1 in enumerate(Group_Names_pan):
            for j, G2 in enumerate(Group_Names_pan):
                try:
                    ydata = -np.log(pan_cancer_p["{}_{}".format(G1,G2)])/np.log(10)
                    xdata = np.log(pan_cancer_fold["{}_{}".format(G1,G2)])/np.log(2)
    
                    #Get most important features in dataset
                    important_features = np.intersect1d(np.argsort(ydata)[-(thresh):], pan_cancer_signif["{}_{}".format(G1,G2)])
    
                except:
                    continue
    
                ax[counter].scatter(xdata, ydata, marker='x', color = "gainsboro")
                ax[counter].axline((-2, -np.log(alpha/Nofeatures_global)/np.log(10)), slope = 0, color = 'r')
                ax[counter].set_xlim([-1.5,1.5])
                [ax[counter].scatter(xdata[col], ydata[col], color = "blue") for col in important_features]
                texts = [ax[counter].text(xdata[col], ydata[col], communication_types[col]) for col in important_features]
                adjust_text(texts, ax=ax[counter], arrowprops=dict(arrowstyle = '-', color='black'))
                
                #Change final labels axes
                labels = [item.get_text() for item in ax[counter].get_xticklabels()]
                labels[0] = labels[0]
                labels[-1] = labels[-1]
                ticks = ax[counter].get_xticks()
                ax[counter].set_xticks(ticks)
                ax[counter].set_xticklabels(labels, fontsize = "large")
                labels = [item.get_text() for item in ax[counter].get_yticklabels()]
                ticks = ax[counter].get_yticks()
                ax[counter].set_yticks(ticks)
                ax[counter].set_yticklabels(labels, fontsize = "large")
                ax[counter].text(0.05, 0.05, "More expressed in {}".format(G2), transform=ax[counter].transAxes, ha="left", va="center", rotation=0, size=12, bbox=dict(boxstyle="larrow,pad=0.3", fc = "w", lw=2))
                ax[counter].text(0.95, 0.05, "More expressed in {}".format(G1), transform=ax[counter].transAxes, ha="right", va="center", rotation=0, size=12, bbox=dict(boxstyle="rarrow,pad=0.3", fc = "w", lw=2))
                counter += 1
    
        fig.suptitle("Volcano plots of features distinguishing {} groups in {} comparison".format(feature_pan, Pan_cancer_name))
        fig.supxlabel("2-log of fold change")
        fig.supylabel("Negative 10-log of p-value")
        fig.tight_layout()
        if save:
            plt.savefig("Volcano_plots_pan_cancer_seperate_comparisons_{}_{}_bundle.png".format(feature_pan, datatype))
    return

def volcanoInd(pvals, fold_change, signif, Group_Names_inv, communication_types, datatype, feature_inv, investigation_types,  Nofeatures_global, alpha = 0.05, thresh = 20, save = False):
    """
    Creates a volcano plot for each cancer type comparing the different
    subgroups.

    Parameters
    ----------
    pvals : dict with np.arrays;
        p-values of all the comparisions between groups.
    fold_change : dict with np.array;
        Fold changes of average expression levels in each group.
    signif : dict with lists;
        List of siginificantly different features for each comparison..
    Group_Names_inv : list of str;
        Names of all groups in the pan cancer analysis.
    communication_types : list of str;
        Names of all the features considered in the comparison.
    datatype : list of str;
        Names of the datatypes considered (e.g. wedge; W).
    feature_inv : str;
        Name of the metadata feature.
    investigation_types : list of str;
        Names of the cancer types we seek to investigate.
    Nofeatures_global : int;
        Amount of feature we consider at the same time.
    alpha : float, optional
        Significance level of the tests. The default is 0.05.
    thresh : int, optional
        Amount of top features to label. The default is 20.
    save : bool, optional
        Saving the figure or not. The default is False.

    Returns
    -------
    None. Might create a saved figure of all volcano plots.

    """
    for cancer in investigation_types:   
        counter = 0
        row_length = max(1, int(np.ceil(math.comb(len(Group_Names_inv), 2))/2))
        if len(Group_Names_inv) == 2:
            fig, ax = plt.subplots(1, row_length)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            for i, G1 in enumerate(Group_Names_inv):
                for j, G2 in enumerate(Group_Names_inv):
                    try:
                        ydata = -np.log(pvals[cancer]["{}_{}".format(G1,G2)])/np.log(10)
                        xdata = np.log(fold_change[cancer]["{}_{}".format(G1,G2)])/np.log(2)
    
                        #Get most important features in dataset
                        important_features = np.intersect1d(np.argsort(ydata)[-(thresh):], signif[cancer]["{}_{}".format(G1,G2)])
                    except:
                        continue
    
                    ax.scatter(xdata, ydata, marker='x', color = "gainsboro")
                    ax.axline((-2, -np.log(alpha/Nofeatures_global)/np.log(10)), slope = 0, color = 'r')
                    ax.set_xlim([-1.5,1.5])
                    [ax.scatter(xdata[col], ydata[col], color = "blue") for col in important_features]
                    texts = [ax.text(xdata[col], ydata[col], communication_types[col]) for col in important_features]
                    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle = '-', color='black'))

                    
                    #Change final labels axes
                    labels = [item.get_text() for item in ax.get_xticklabels()]
                    labels[0] = labels[0]
                    labels[-1] = labels[-1]
                    ticks = ax.get_xticks()
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels, fontsize = "large")
                    labels = [item.get_text() for item in ax.get_yticklabels()]
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(labels, fontsize = "large")
                    ax.text(0.05, 0.05, "More expressed in {}".format(G2), transform=ax.transAxes, ha="left", va="center", rotation=0, size=12, bbox=dict(boxstyle="larrow,pad=0.3", fc = "w", lw=2))
                    ax.text(0.95, 0.05, "More expressed in {}".format(G1), transform=ax.transAxes, ha="right", va="center", rotation=0, size=12, bbox=dict(boxstyle="rarrow,pad=0.3", fc = "w", lw=2))
                    counter += 1
        
        else:
            fig, ax = plt.subplots(2, row_length)
            ax = ax.ravel()
            fig.set_figheight(15)
            fig.set_figwidth(20)
            for i, G1 in enumerate(Group_Names_inv):
                for j, G2 in enumerate(Group_Names_inv):
                    try:
                        ydata = -np.log(pvals[cancer]["{}_{}".format(G1,G2)])/np.log(10)
                        xdata = np.log(fold_change[cancer]["{}_{}".format(G1,G2)])/np.log(2)
    
                        #Get most important features in dataset
                        important_features = np.intersect1d(np.argsort(ydata)[-(thresh):], signif[cancer]["{}_{}".format(G1,G2)])
                    except:
                        continue
    
                    ax[counter].scatter(xdata, ydata, marker='x', color = "gainsboro")
                    ax[counter].axline((-2, -np.log(alpha/Nofeatures_global)/np.log(10)), slope = 0, color = 'r')
                    ax[counter].set_xlim([-1.5,1.5])
                    [ax[counter].scatter(xdata[col], ydata[col], color = "blue") for col in important_features]
                    texts = [ax[counter].text(xdata[col], ydata[col], communication_types[col]) for col in important_features]
                    adjust_text(texts, ax=ax[counter], arrowprops=dict(arrowstyle = '-', color='black'))

    
                    
                    #Change final labels axes
                    labels = [item.get_text() for item in ax[counter].get_xticklabels()]
                    labels[0] = labels[0]
                    labels[-1] = labels[-1]
                    ticks = ax[counter].get_xticks()
                    ax[counter].set_xticks(ticks)
                    ax[counter].set_xticklabels(labels, fontsize = "large")
                    labels = [item.get_text() for item in ax[counter].get_yticklabels()]
                    ticks = ax[counter].get_yticks()
                    ax[counter].set_yticks(ticks)
                    ax[counter].set_yticklabels(labels, fontsize = "large")
                    ax[counter].text(0.05, 0.05, "More expressed in {}".format(G2), transform=ax[counter].transAxes, ha="left", va="center", rotation=0, size=12, bbox=dict(boxstyle="larrow,pad=0.3", fc = "w", lw=2))
                    ax[counter].text(0.95, 0.05, "More expressed in {}".format(G1), transform=ax[counter].transAxes, ha="right", va="center", rotation=0, size=12, bbox=dict(boxstyle="rarrow,pad=0.3", fc = "w", lw=2))
                    counter += 1
    
        fig.suptitle("Volcano plots of features distinguishing {} groups in {}".format(feature_inv, cancer))
        fig.supxlabel("2-log of fold change")
        fig.supylabel("Negative 10-log of p-value")
        fig.tight_layout()
        if save:
            plt.savefig("Volcano_plots_{}_seperate_comparisons_{}_{}_bundle.png".format(cancer, feature_inv, datatype))
    return

def volcanoCross(pvals, fold_change, signif, cancer_types, investigation_types, Group_Names_inv, communication_types, datatype, Nofeatures_global, alpha = 0.05, thresh = 20, save = False):
    """
    Creates volcano plots for the cross feature analyses.

    Parameters
    ----------
    pvals : dict with np.arrays;
        p-values of all the comparisions between groups.
    fold_change : dict with np.array;
        Fold changes of average expression levels in each group.
    signif : dict with lists;
        List of siginificantly different features for each comparison..
    cancer_types : list with str entries;
        Identifiers of the base group of cancer types.
    investigation_types : list of str;
        Names of the cancer types we seek to investigate.
    Group_Names_inv : list of str;
        Names of all groups in the pan cancer analysis.
    communication_types : list of str;
        Names of all the features considered in the comparison
    datatype : list of str;
        Names of the datatypes considered (e.g. wedge; W).
    Nofeatures_global : int;
        Amount of feature we consider at the same time.
    alpha : float, optional
        Significance level of the tests. The default is 0.05.
    thresh : int, optional
        Amount of top features to label. The default is 20.
    save : bool, optional
        Saving the figure or not. The default is False.

    Returns
    -------
    None. Creates volcano plots

    """
    row_length = max(1, int(np.ceil(math.comb(len(Group_Names_inv), 2))/2))
    cancer = investigation_types[0]
    for group in Group_Names_inv:
        fig, ax = plt.subplots(1, row_length)
        fig.set_figheight(10)
        fig.set_figwidth(15)
   
        ydata = -np.log(pvals[cancer]["{}".format(group)])/np.log(10)
        xdata = np.log(fold_change[cancer]["{}".format(group)])/np.log(2)
   
        #Get most important features in dataset
        important_features = np.intersect1d(np.argsort(ydata)[-(thresh):], signif[cancer]["{}".format(group)])
   
   
        ax.scatter(xdata, ydata, marker='x', color = "gainsboro")
        ax.axline((-2, -np.log(alpha/Nofeatures_global)/np.log(10)), slope = 0, color = 'r')
        ax.set_xlim([-1.5,1.5])
        [ax.scatter(xdata[col], ydata[col], color = "blue") for col in important_features]
        texts = [ax.text(xdata[col], ydata[col], communication_types[col]) for col in important_features]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle = '-', color='black'))
        at = AnchoredText("{}".format(group), prop=dict(size=15), frameon=True, loc='lower left')
        ax.add_artist(at)
        
        fig.suptitle("{} over {}".format(cancer, cancer_types[0]))
        fig.supxlabel("2-log of fold change")
        fig.supylabel("Negative 10-log of p-value")
        fig.tight_layout()
        if save:
            plt.savefig("Volcano_plots_{}_in_{}_seperate_comparisons_{}_{}_bundle.png".format(datatype, group, cancer, cancer_types[0])) 
    return


#-------------------
# Heatmaps
#-------------------
def findLargeFold(pan_cancer_p, p_vals, pan_cancer_fold, fold_change, pan_cancer_signif, signif, Group_Names_pan, investigation_types, Nofeatures_global, thresh = 20):
    """
    Finds the biggest differences between the pan cancer feature values and
    the cancer specific feature values.

    Parameters
    ----------
 pan_cancer_p : dict with np.arrays;
     p-values of all the comparisions between groups.
   pvals : dict with np.arrays;
       p-values of all the comparisions between groups.
    pan_cancer_fold : dict with np.array;
        Fold changes of average expression levels in each group.
   fold_change : dict with np.array;
       Fold changes of average expression levels in each group.
    pan_cancer_signif : dict with lists;
        List of siginificantly different features for each comparison.   
    signif : dict with lists;
        List of siginificantly different features for each comparison
    Group_Names_pan : list of str;
        Names of all groups in the pan cancer analysis.
    investigation_types : list of str;
        Names of the cancer types we seek to investigate.
    Nofeatures_global : int;
        Amount of feature we consider at the same time.
    thresh : int, optional
        Amount of top features to label. The default is 20.

    Returns
    -------
    big_diff_path : dict with np.array entries;
        Labels of the features that differ most and indicators that show 
        whether the feature went down or up.
    warning : bool;
        Indicates whether pan cancer groups did not correspond to individual cancer groups.
    """
    big_diff_path = {}
    for cancer in investigation_types:
        big_diff_path[cancer] = {}
        for i, G1 in enumerate(Group_Names_pan):
            for G2 in Group_Names_pan[(i+1):]:
                try: #Pan cancer groups coincide with cancer specific groups
                    warning = False
                    most_diff = []
                    #Compute fold change difference between pan cancer and cancer specific group
                    difference = np.array(fold_change[cancer]["{}_{}".format(G1, G2)]) - np.array(pan_cancer_fold["{}_{}".format(G1, G2)])
                    #Make sure to only consider feature that are significant in at least one cancer type
                    signif_features = np.union1d(pan_cancer_signif["{}_{}".format(G1, G2)], signif[cancer]["{}_{}".format(G1,G2)])
                    #Sort the features in the order of smallest fold-change difference to highest fold-change difference
                    places = np.argsort(np.abs(difference))
                    #Record whether fold change is higher or lower than before 
                    sign_diff = np.sign(difference)
                    #Get the features with the biggest fold-change difference that are also significant for one comparison and record sign
                    counter = 0
                    while len(most_diff) < thresh and counter < Nofeatures_global:
                        if places[- counter] in signif_features:
                            most_diff.append([places[-counter],sign_diff[places[-counter]]])
                        counter += 1
                    big_diff_path[cancer]["{}_{}".format(G1, G2)] = np.array(most_diff).astype(int) #Change the list into an int np.array
                
                except: #Pan cancer groups do not coincide with cancer specific groups
                    print("Warning: Pan-cancer groups do not coincide with cancer-specific groups.")
                    print("Assumption: Cancer-specific groups are R and NR. We will now compare these groups with MFP pan-cancer groups.")
                    print("Make sure to order the immune subtypes [IE, D, IE/F, F]. Then, they are order according to response to therapy.")
                    warning = True
                    most_diff = []
                    
                    #Take most explaining features pan cancer and investigate how R vs. NR compares
                    places = np.argsort(np.array(pan_cancer_p["{}_{}".format(G1, G2)]))[:thresh]
                    
                    for place in places:
                        if place in signif_features:
                            most_diff.append([place, np.sign(np.array(fold_change[cancer]["R_NR"]) - np.array(pan_cancer_fold["{}_{}".format(G1, G2)]))[place]])
    
                    big_diff_path[cancer]["{}_{}".format(G1, G2)] = np.array(most_diff).astype(int)
    return big_diff_path, warning

def plot_heatmap(ax, key, heatmap_data, heatmap_shapes, col_labels, row_labels, first = False, last = False):
    """
    Plots heatmaps of the largest fold change differences between the pan cancer
    setting and the cancer specific setting.

    Parameters
    ----------
    ax : axis object.
    key : dict key object.
    heatmap_data : data to be plotted.
    heatmap_shapes : shapes of the scatter in the heatmap.
    col_labels : labels of the columns in the heatmap
    row_labels : labels of the rows.
    first : indicator whether the current heatmap to be plotted is the first.
    last : indicator whether the current heatmap will be the last.

    Returns
    -------
    ax : returns ax object with figure.

    """
    # Set cmap
    cmap = cm.get_cmap('coolwarm')    

    # Iterate over types of plastic
    for i in range(heatmap_data[key].shape[1]):
        # Select data for the given type of cancer/comparison
        d = heatmap_data[key][:,i]
        markers = heatmap_shapes[key][:,i].astype(int)

        # Get values for the x and y axes
        y = np.arange(len(row_labels[key]))
        #y = np.array(row_labels[key])
        x = np.array([i] * len(y))

        # Generate colors. Numbers already normalised
        color = cmap(d) 
        # Plot the markers for the selected pathway
        ax.scatter(x[markers == 1], y[markers == 1], color=color[markers == 1], s=120, marker = '^')
        ax.scatter(x[markers == -1], y[markers == -1], color=color[markers == -1], s=120, marker = 'v')
        ax.scatter(x[markers == 0], y[markers == 0], color=color[markers == 0], s=120, marker = 'o')
        ax.set_yticks(y)
        ax.set_yticklabels(row_labels[key])


    # Remove all spines
    ax.set_frame_on(False)

    # Set grid lines with some transparency
    ax.grid(alpha=0.4)

    # Make sure grid lines are behind other objects
    ax.set_axisbelow(True)

    if last:
            # Set position for x ticks
        ax.set_xticks(np.arange(len(col_labels)))
            # Remove tick marks by setting their size to 0. Set text color to "0.3" (a type of grey)
        ax.tick_params(size=0, colors="0.3")
    # Set labels for the x ticks (the names of the types of cancer/comparison type)
        ax.set_xticklabels(col_labels)
        # Set label for horizontal axis.
        ax.set_xlabel("Cancer Types", loc="right")
    else:
        # Set position for x ticks
        ax.set_xticks(np.arange(len(col_labels)))
         # Remove tick marks by setting their size to 0. Set text color to "0.3" (a type of grey)
        ax.tick_params(size=0, colors="0.3")
        ax.set_xticklabels([])


    # Remove tick marks by setting their size to 0. Set text color to "0.3" (a type of grey)
    ax.tick_params(size=0, colors="0.3")
    ax.set_ylabel("{}".format(key), loc="top")
    
    if first:
        ax.set_title("Comparison TCGA with Gideauslanderpd1")
    return ax

def heatmap(big_diff_path, warning, pan_cancer_p, p_vals, pan_cancer_fold, fold_change, pan_cancer_signif, signif, Group_Names_pan, Pan_cancer_name, investigation_types, communication_types, feature_inv, datatype, Nofeatures_global, thresh = 20, save = False):
    """
    Creates heatmap with the biggest fold change difference between specific 
    cancer and pan cancer comparison.

    Parameters
    ----------
    big_diff_path : dict with np.array entries;
        Labels of the features that differ most and indicators that show 
        whether the feature went down or up.
    warning : bool;
        Indicates whether pan cancer groups did not correspond to individual cancer groups.
    pan_cancer_p : dict with np.arrays;
        p-values of all the comparisions between groups.
    pan_cancer_fold : dict with np.array;
        Fold changes of average expression levels in each group.
    pan_cancer_signif : dict with lists;
        List of siginificantly different features for each comparison.
    pvals : dict with np.arrays;
        p-values of all the comparisions between groups.
    fold_change : dict with np.array;
        Fold changes of average expression levels in each group.
    signif : dict with lists;
        List of siginificantly different features for each comparison..
    Group_Names_pan : list of str;
        Names of all groups in the pan cancer analysis.
    Pan_cancer_name : str;
        Name of the pan cancer cancer types as a bundle.
    investigation_types : list of str;
        Names of the cancer types we seek to investigate.
    communication_types : list of str;
        The labels of the features included in the analysis.
    feature_inv : str;
        Name of the metadata feature in the individual cancer type.
    datatype : list of str;
        Types of features used in the analysis (e.g. wedge; W).
    Nofeatures_global : int;
        Amount of feature we consider at the same time.
    thresh : int, optional
        Amount of top features to label. The default is 20.
    save : bool, optional
        If true the funciton saves the heatmap. The default is False.
    Returns
    -------
    None.

    """
    try: #In reference and specific cancer type the groups overlap
        communication_types = np.array(communication_types)
        #Creating data for heatmap
        heatmap_cols = {}
        for key in pan_cancer_fold.keys():
            cols_comparison = []
            #Find the feature indices of the features that had the biggest difference with the pan-cancer analysis
            for cancer in investigation_types:
                cols_comparison = np.union1d(cols_comparison, big_diff_path[cancer][key][:,0])
            heatmap_cols[key] = cols_comparison
    
        #Now, for all the "interesting" features we will extract the data for the investigation cancer types and pan cancer.
        heatmap_data = {}
        row_labels = {}
        col_labels = investigation_types.copy()
    
    
        for key in pan_cancer_fold.keys():
            heatmap_data[key] = []
    
    
            for cancer in investigation_types:
                #Extract the fold change values of the top differences
                heatmap_data[key].append(np.log(np.array(fold_change[cancer][key])[heatmap_cols[key].astype(int)])/np.log(2))
    
            #Pan cancer data of the features where the investigation types differ most
            heatmap_data[key].append(np.log(np.array(pan_cancer_fold[key])[heatmap_cols[key].astype(int)])/np.log(2))
            row_labels[key] = communication_types[heatmap_cols[key].astype(int)]
            heatmap_data[key] = np.array(heatmap_data[key]).transpose()
        col_labels.append("Rest TCGA") #Call this column in the heatmap "Pan"
    
        #Allocate the signs to the correct features
        heatmap_shapes = {}
        for key in pan_cancer_fold.keys():
            heatmap_shapes[key] = np.zeros_like(heatmap_data[key])
            for place, cancer in enumerate(investigation_types):
                #Find the locations in heatmap_shapes[key] where the signs belong
                places = np.nonzero(np.isin(row_labels[key], communication_types[big_diff_path[cancer][key][:,0].astype(int)]))[0]
    
                #Place the signs at the correct place
                for i in places:
                    #Find the index in the feature list where the current row_label belongs
                    value = list(communication_types).index(row_labels[key][i])
                    #Find the sign location of this feature in the big_diff_path array and add this sign to the shapes array
                    heatmap_shapes[key][i, place] = big_diff_path[cancer][key][big_diff_path[cancer][key][:,0].astype(int) == value ,1]
    
        #Sort the data such that on average redder values appear on top
        for key in pan_cancer_fold.keys():
             # Normalise such that values are between 0 and 1.
            pivot = max(np.amax(heatmap_data[key]), -np.amin(heatmap_data[key]))
            for i in range(heatmap_data[key].shape[1]):
                # Select data for the given type cancer type and comparison
                d = heatmap_data[key][:,i]
                heatmap_data[key][:, i] = (d + pivot) / (2 * pivot)
                #heatmap_data[key][:, i] = (d) / (pivot)
    
            av = np.argsort(np.average(heatmap_data[key], axis = 1))
            heatmap_data[key] = heatmap_data[key][av, :]
            heatmap_shapes[key] =  heatmap_shapes[key][av,:]
            row_labels[key] = row_labels[key][av]
            
    except: #Groups do not overlap
        print("Assumption: R and NR are being compared with something else.")
        communication_types = np.array(communication_types)
            #Creating data for heatmap
        heatmap_cols = {}
        for key in pan_cancer_fold.keys():
            cols_comparison = []
            #Find the feature indices of the features that had the biggest difference with the pan-cancer analysis
            for cancer in investigation_types:
                cols_comparison = np.union1d(cols_comparison, big_diff_path[cancer][key][:,0])
            heatmap_cols[key] = cols_comparison
    
        #Now, for all the "interesting" features we will extract the data for the investigation cancer types and pan cancer.
        heatmap_data = {}
        row_labels = {}
        col_labels = investigation_types.copy()
    
    
        for key in pan_cancer_fold.keys():
            heatmap_data[key] = []
    
    
            for cancer in investigation_types:
                #Extract the fold change values of the top differences
                heatmap_data[key].append(np.log(np.array(fold_change[cancer]["R_NR"])[heatmap_cols[key].astype(int)])/np.log(2))
    
            #Pan cancer data of the features where the investigation types differ most
            heatmap_data[key].append(np.log(np.array(pan_cancer_fold[key])[heatmap_cols[key].astype(int)])/np.log(2))
            row_labels[key] = communication_types[heatmap_cols[key].astype(int)]
            heatmap_data[key] = np.array(heatmap_data[key]).transpose()
        col_labels.append(Pan_cancer_name) #Call this column in the heatmap "Pan"
    
        #Allocate the signs to the correct features
        heatmap_shapes = {}
        for key in pan_cancer_fold.keys():
            heatmap_shapes[key] = np.zeros_like(heatmap_data[key])
            for place, cancer in enumerate(investigation_types):
                #Find the locations in heatmap_shapes[key] where the signs belong
                places = np.nonzero(np.isin(row_labels[key], communication_types[big_diff_path[cancer][key][:,0].astype(int)]))[0]
    
                #Place the signs at the correct place
                for i in places:
                    #Find the index in the feature list where the current row_label belongs
                    value = list(communication_types).index(row_labels[key][i])
                    #Find the sign location of this feature in the big_diff_path array and add this sign to the shapes array
                    heatmap_shapes[key][i, place] = big_diff_path[cancer][key][big_diff_path[cancer][key][:,0].astype(int) == value ,1]
    
        #Sort the data such that on average redder values appear on top
        for key in pan_cancer_fold.keys():
             # Normalise such that values are between 0 and 1.
            pivot = max(np.amax(heatmap_data[key]), -np.amin(heatmap_data[key]))
            for i in range(heatmap_data[key].shape[1]):
                # Select data for the given type cancer type and comparison
                d = heatmap_data[key][:,i]
                heatmap_data[key][:, i] = (d + pivot) / (2 * pivot)
                #heatmap_data[key][:, i] = (d) / (pivot)
    
            av = np.argsort(np.average(heatmap_data[key], axis = 1))
            heatmap_data[key] = heatmap_data[key][av, :]
            heatmap_shapes[key] =  heatmap_shapes[key][av,:]
            row_labels[key] = row_labels[key][av]
            
    PlotNo = len(row_labels.keys())
    fig, ax = plt.subplots(PlotNo ,figsize=(5, 20))
    
    try:
        for i, key in enumerate(row_labels.keys()):
            if i == 0:
                plot_heatmap(ax[i], key, heatmap_data, heatmap_shapes, col_labels, row_labels, True, False)
            elif i == PlotNo - 1:
                plot_heatmap(ax[i], key, heatmap_data, heatmap_shapes, col_labels, row_labels, False, True)
            else:
                plot_heatmap(ax[i], key, heatmap_data, heatmap_shapes, col_labels, row_labels)
            
            if warning:            
                cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="coolwarm"), ax=ax[i], orientation = "vertical", shrink = 0.5,\
                     anchor = (0.0, 1.0), ticks = [0, 1])
                cbar.set_ticklabels(["NR", "R"])
            else:
                cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="coolwarm"), ax=ax[i], orientation = "vertical", shrink = 0.5,\
                     anchor = (0.0, 1.0), ticks = [0, 1])
                cbar.set_ticklabels(["{}".format(key.split('_')[1]), "{}".format(key.split('_')[0])])  
    except:
        plot_heatmap(ax, key, heatmap_data, heatmap_shapes, col_labels, row_labels, True, True)
      
    
        if warning:            
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="coolwarm"), ax=ax, orientation = "vertical", shrink = 0.5,\
                 anchor = (0.0, 1.0), ticks = [0, 1])
            cbar.set_ticklabels(["NR or F", "R or IE"])
        else:
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap="coolwarm"), ax=ax, orientation = "vertical", shrink = 0.5,\
                 anchor = (0.0, 1.0), ticks = [0, 1])
            cbar.set_ticklabels(["{}".format(key.split('_')[1]), "{}".format(key.split('_')[0])]) 
    
    fig.tight_layout()
    if save:
        plt.savefig("{}_comparison_heatmap_{}_{}_bundle.png".format(investigation_types[0], feature_inv, datatype))
    return

#-----------------------
#   GSCC
#-----------------------

def contributionAnalysisGSCC(lmin, lmax, datasets, celltype, feat, norm, save):
    """
    Analyses contribution of celltype in GSCC for different patient groups over 
    differet datasets for varying average degree in the graph using the kernel.
    

    Parameters
    ----------
    lmin : float;
        Minimal average degree inputted.
    lmax : float;
        Maximal average degree inputted.
    datasets : list of str;
        List of datasets investigated.
    celltype : str;
        Name of cell type to be investigated.
    feat : str;
        Feature from meta-data to be compared.
    norm : bool;
        Normalize the output or not?
    save : bool;
        Save the output or not.

    Returns
    -------
    None. A figure with the evolution of the GSCC for each group. Indicating the
    average of the group, 1st quartile and 3rd quartile.

    """
    # Do the simulations
    lamlist = np.arange(lmin, lmax, (lmax - lmin)/100)
    GSCCtumor = {}
    for dat in datasets:
        GSCCdat = []
        for la in lamlist:
            df = getGSCCAnalytically(dat, lab = la, norm = norm)
            GSCCdat.append(list(df[celltype].values))
        GSCCtumor[dat] = GSCCdat
    
    # Add the patient subgroups
    GSCCresponses = {}
    for dat in datasets:
        try:
            names = gp(dat)
            df = Metadata_csv_read_in("metadata_{}.csv".format(dat), "MFP", names = names)
            GSCCresponses[dat] = list(df)
        except:
            subtypes = Find_Patient_Subtype_Bagaev(dat)
            GSCCresponses[dat] = subtypes
    
    pan = True
    if pan:
        for i, dat in enumerate(datasets):
            if i == 0:
                numValues = GSCCtumor[dat]
                typeValues = GSCCresponses[dat]
            else:
                numValues = np.append(numValues, GSCCtumor[dat], axis = 1)
                typeValues = np.append(typeValues, GSCCresponses[dat])
                
        GSCCresponses = {}
        GSCCtumor = {}
        GSCCresponses["pan"] = typeValues
        GSCCtumor["pan"] = numValues
        datasets = ["pan"]
        
    
    # Compare the differences
    pVals = {}
    signif = {}
    groupvals = {}
    for dat in datasets:
        keylist = list(np.unique(GSCCresponses[dat]))
        try:
            keylist.remove("")
        except:
            pass

        for key in keylist:
            locs = np.where(np.array(GSCCresponses[dat]) == key)
            vals = np.array(GSCCtumor[dat])[:, locs[0]]
            #groupvals[key + " in " + dat] = vals
            groupvals[key] = vals
    keylist2 = groupvals.keys()
    

    
    for key in keylist2:
        for key2 in keylist2:
            _, pVals[key + " < " + key2] = stat.mannwhitneyu(groupvals[key], groupvals[key2], alternative = "less", axis = 1)
            signif[key + " < " + key2] = (pVals[key + " < " + key2]  < 0.05/len(keylist2)**2)
        
    # Plotting the significant values

    fig, ax = plt.subplots()
    toTest = list(signif.keys())
    
    for key in keylist2:
        try:
            toTest.remove(key + " < " + key)
        except:
            pass
    
    
    for key in keylist:
        try:
            toTest.remove(key + " < " + key)
        except:
            pass
    for i, key in enumerate(toTest):
        ax.scatter(lamlist, (i + 2) * signif[key], color = "b", marker = "o")
        ax.scatter(lamlist, (i + 2) * ~signif[key], color = "r", marker = "o")
    ticks = np.arange(2, len(toTest) + 2)
    labels = toTest
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(1, len(toTest) + 2)
    ax.set_xlabel("Average degree inputted (λ)")
    ax.set_ylabel("Comparison (in {})".format(dat))
    ax.set_title("Signif. of one sided Mann-Whitney U")
    if save:
        plt.tight_layout()
        plt.savefig("MannWhitneyU_comparison_GSCC_{}.png".format(dat))
        plt.savefig("MannWhitneyU_comparison_GSCC_{}.svg".format(dat))
        plt.show()
    else:
        plt.show()
        
    
    colors = ['#9467bd','#d62728', '#ff7f0e', '#2ca02c', '#8c564b']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']
    for i, key in enumerate(keylist2):
            try:
                vals = groupvals[key]
                av = np.median(vals, axis = 1)
                q1 = np.percentile(vals, 25, axis = 1)
                q3 = np.percentile(vals, 75, axis = 1)
                plt.plot(lamlist, av, color = colors[i], linestyle = linestyles[i], label = key)
                plt.fill_between(lamlist, q3, q1, color = colors[i], alpha = 0.2)
            except:
                pass
    plt.legend()
    plt.ylim([1, 1.5])
    plt.xlabel("Average degree inputted (λ)")
    if norm:
        plt.ylabel("{} contribution to GSCC (normalized)".format(celltype))
    else:
        plt.ylabel("{} contribution to GSCC".format(celltype))
    plt.title("Evolution of {} cells in GSCC for different groups".format(celltype))
    if save:
        plt.tight_layout()
        plt.savefig("GSCC_evol_{}_{}_groups_norm.png".format(dat, feat))
        plt.savefig("GSCC_evol_{}_{}_groups_norm.svg".format(dat, feat))
        plt.show()
    else:
        plt.show()
    return pVals, signif







if __name__ == "__main__":
    #--------------------------------------------
    #   Test GSCC function Monte-Carlo
    #--------------------------------------------
    pvals, signif = contributionAnalysisGSCC(1, 15, ["STAD", "SKCM"], "GSCC_Tumor", "MFP", True, False)
    
    
    
    
    #--------------------------------------------------
    # Test heatmap, volcano and correlation kernel
    #--------------------------------------------------
    cancer_types = ["STAD", "SKCM"] #Names of the cancer types for pan cancer analysis
    investigation_types = ["STAD"] #Names of cancer types to investigate
    datatype = ["W"] #Type of communication data
    
    
    Group_Names_pan = ["IE", "IE/F", "F", "D"] #Names of groups in pan cancer analysis
    Group_Names_inv = ["IE", "IE/F", "F", "D"] #Names of groups in cancer specific analysis
    feature_pan = "MFP" #Metadata feature for pan cancer group
    feature_inv = "MFP" #Metadata feature for investigation group
    Pan_cancer_name = "Rest TCGA" #How to name the reference set (cancer_types)
    
    remove_celltypes = [] #List of celltypes to remove.
    Remove_low_var = False #Remove low variance features
    Cross_feature_analysis = False #Instead of creating volcanoplots of comparing features between cancer types, you create heatmaps that compare the same features across cancer types
    thresh = 20 #Only take top thresh pathways in analysis
    data, communication_types, Nofeatures_global = readAllDataExact(cancer_types, investigation_types, datatype, remove_celltypes = [], weight = "min", bundle = True)
    data = addMetadata(data, cancer_types, investigation_types, feature_pan, feature_inv)
    computeCorrelation(data, thresh)
    Groups = groupPatients(data, cancer_types, investigation_types, Group_Names_pan, Group_Names_inv, feature_pan, feature_inv)
    pan_cancer_data, pan_cancer_labels = createPanCancerData(Groups, cancer_types, Group_Names_pan, Nofeatures_global)
    pvals, fold_change, signif, pan_cancer_p, pan_cancer_fold, pan_cancer_signif = wilcoxon(data, Groups, investigation_types, Group_Names_inv, pan_cancer_data, pan_cancer_labels, Group_Names_pan, communication_types, Nofeatures_global, Cross_feature_analysis = False, alpha = 0.05)
    volcanoPan(pan_cancer_p, pan_cancer_fold, pan_cancer_signif, Group_Names_pan, communication_types, datatype, feature_pan, Pan_cancer_name,  Nofeatures_global)
    volcanoInd(pvals, fold_change, signif, Group_Names_inv, communication_types, datatype, feature_inv, investigation_types,  Nofeatures_global)
    big_diff_path, warning = findLargeFold(pan_cancer_p, pvals, pan_cancer_fold, fold_change, pan_cancer_signif, signif, Group_Names_pan, investigation_types, Nofeatures_global, thresh = 20)
    heatmap(big_diff_path, warning, pan_cancer_p, pvals, pan_cancer_fold, fold_change, pan_cancer_signif, signif, Group_Names_pan, Pan_cancer_name, investigation_types, communication_types, feature_inv, datatype, Nofeatures_global, thresh = 20, save = False)
    

    
    
    #--------------------------------------------------
    # Test heatmap, volcano and correlation monte-carlo
    #--------------------------------------------------
    cancer_types = ["SKCM"] #Names of the cancer types for pan cancer analysis
    investigation_types = ["SKCM"] #Names of cancer types to investigate
    datatype = ["W"] #Type of communication data
    
    
    Group_Names_pan = ["IE", "IE/F", "F", "D"] #Names of groups in pan cancer analysis
    Group_Names_inv = ["IE", "IE/F", "F", "D"] #Names of groups in cancer specific analysis
    feature_pan = "MFP" #Metadata feature for pan cancer group
    feature_inv = "MFP" #Metadata feature for investigation group
    Pan_cancer_name = "Rest TCGA" #How to name the reference set (cancer_types)
    
    remove_celltypes = [] #List of celltypes to remove.
    Remove_low_var = False #Remove low variance features
    Cross_feature_analysis = False #Instead of creating volcanoplots of comparing features between cancer types, you create heatmaps that compare the same features across cancer types
    thresh = 20 #Only take top thresh pathways in analysis
    data, communication_types, Nofeatures_global = readAllData(cancer_types, investigation_types, datatype, remove_celltypes, noCells = 500, average = 20, weight = "min", bundle = True)
    data = addMetadata(data, cancer_types, investigation_types, feature_pan, feature_inv)
    computeCorrelation(data, thresh)
    Groups = groupPatients(data, cancer_types, investigation_types, Group_Names_pan, Group_Names_inv, feature_pan, feature_inv)
    pan_cancer_data, pan_cancer_labels = createPanCancerData(Groups, cancer_types, Group_Names_pan, Nofeatures_global)
    pvals, fold_change, signif, pan_cancer_p, pan_cancer_fold, pan_cancer_signif = wilcoxon(data, Groups, investigation_types, Group_Names_inv, pan_cancer_data, pan_cancer_labels, Group_Names_pan, communication_types, Nofeatures_global, Cross_feature_analysis = False, alpha = 0.05)
    volcanoPan(pan_cancer_p, pan_cancer_fold, pan_cancer_signif, Group_Names_pan, communication_types, datatype, feature_pan, Pan_cancer_name,  Nofeatures_global)
    volcanoInd(pvals, fold_change, signif, Group_Names_inv, communication_types, datatype, feature_inv, investigation_types,  Nofeatures_global)
    big_diff_path, warning = findLargeFold(pan_cancer_p, pvals, pan_cancer_fold, fold_change, pan_cancer_signif, signif, Group_Names_pan, investigation_types, Nofeatures_global, thresh = 20)
   
    
    
