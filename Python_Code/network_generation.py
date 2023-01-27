"""
Author : Mike van Santvoort (m.v.santvoort@tue.nl)

Generates one graph relasation of RaCInG based on input data. These are:
    1. The probability list that a certain ligand-receptor list is expressed
    2. The relative frequency of different cell types
    3. The list of how certain cells can secrete ligands/receptors.
"""

import numpy as np
import distribution_generation as dg




class informationMicroEnv:
    """
    Provides information about the micro-environment. When this class is being
    set up, it generates a distribution of cell type, a distribution of ligand
    receptor connections, and the connection structure of ligands/receptors to
    cells. It can generate a list of cell type connections based on the distri-
    butions; this requres the amount of desired edges.
    """
    
    def __init__(self, cellTypeNo, ligNo, recNo, genRandom = True ,\
                 Dcelltype = [], Dligrec = [], structurelig = [], \
                     structurerec = []):
        if genRandom:
            self.Dcelltype = dg.genRandomCellTypeDistr(cellTypeNo)
            self.Dligrec = dg.genRandomLigRecDistr(ligNo, recNo)
            self.structurelig = dg.genRandomCellLigands(cellTypeNo, ligNo)
            self.structurerec = dg.genRandomCellReceptors(cellTypeNo, recNo)
        else:
            self.Dcelltype = Dcelltype
            self.Dligrec = Dligrec
            self.structurelig = structurelig
            self.structurerec = structurerec
            
        self.ligNo = ligNo
        self.recNo = recNo
        self.vertextypelist = []
        self.edgetypelist = []
        self.edgelist = []
        return
    
    def genRandomEdgeList(self, edgeNo):
        """
        Generates a random edge list corresponding to a graph realisation of
        RaCInG. 

        Parameters
        ----------
        edgeNo : int;
            The number of edges to be generated.

        Returns
        -------
        edgelist : 2D np.array() with int entries;
            The edge list of the graph realisation. If e.g. [3, 5] appears in
            the list this means there is a connection from vertex 3 to vertex 5.
        edgetypelist : 2D np.array() with int entries;
            The list with the ligand-receptor pairs used for each arc. If the
            first entrie of the edge list is [1 , 2] and the first entry of this
            list is [3, 4], then this means that there is an arc from vertex 1
            to 2 where the ligand used was of type 3 and the receptor of type 4.
        """
        
        #First generate a list of ligand-receptor connections present
        distr = self.Dligrec.flatten();
        linearEdgeList = np.random.choice(distr.size, edgeNo, p = distr)
        self.edgetypelist = np.unravel_index(linearEdgeList, self.Dligrec.shape)
        self.edgelist = np.unravel_index(linearEdgeList, self.Dligrec.shape)
        
        #Assign each connection to uniformly chosen cells that can accept them
        for i in np.arange(self.ligNo):
            count = np.sum(self.edgetypelist[0] == i);
            acceptingCellTypes = np.nonzero(self.structurelig[:,i]==1)
            ConnectionChoice = np.nonzero(np.in1d(self.vertextypelist,\
                                          acceptingCellTypes))
            try:
                temp = np.random.choice(ConnectionChoice[0], count)
            except:
                #print("WARNING: Ligand {} cannot bind {} times...".format(i, count))
                temp = - np.ones(count)
            self.edgelist[0][self.edgetypelist[0] == i] = temp;
            
        for i in np.arange(self.recNo):
            count = np.sum(self.edgetypelist[1] == i)
            acceptingCellTypes = np.nonzero(self.structurerec[:,i]==1);
            ConnectionChoice = np.nonzero(np.in1d(self.vertextypelist,\
                                          acceptingCellTypes))
            
            try:
                temp = np.random.choice(ConnectionChoice[0], count)
            except:
                #print("WARNING: Receptor {} cannot bind {} times...".format(i, count))
                temp = -np.ones(count)
                
            self.edgelist[1][self.edgetypelist[1] == i] = temp
        
        #Transform edge list in the correct data structure
        self.edgelist = np.stack(self.edgelist, axis = -1)
        self.edgetypelist = np.stack(self.edgetypelist, axis = -1)
        
        #Fetching indices of non matched ligands/receptors
        deleteindex1 = np.nonzero(self.edgelist[:, 0] == -1)[0]
        deleteindex2 = np.nonzero(self.edgelist[:, 1] == -1)[0]
        deleteindex = np.concatenate((deleteindex1, deleteindex2))
        
        self.edgelist = np.delete(self.edgelist, deleteindex, axis = 0)   
        self.edgetypelist = np.delete(self.edgetypelist, deleteindex, axis = 0)
        
        return self.edgelist, self.edgetypelist
            
    
    def genRandomCellTypeList(self, vertexNo):
        """
        Generates the vertices (and their types) in the graph generation
        procedure.

        Parameters
        ----------
        vertexNo : int;
            Number of vertices.

        Returns
        -------
        vertextypelist : np.array() with int entries;
            A list with the type of each vertex. For example, if entry 5 is given
            by 3, then cell 5 has cell-type 3.
        """
        self.vertextypelist = np.random.choice(self.Dcelltype.size, vertexNo, \
                                               p = self.Dcelltype)
        return self.vertextypelist
 
    
def model1(N, avdeg,  cellLigList = [], cellRecList = [], \
           Dcelltype = [], Dligrec = [], Signmatrix = [], genRandom = True):
    """
    Generates a graph instance according to model 1. Hence, it generates
    a fixed number of edges according to Dligrec, and assigns them to cells
    generated through Dcelltype uniformly at random according to the connection
    rules given in cellLigList and cellRecList.

    Parameters
    ----------
    N : int;
        Number of cells.
    avdeg : float;
        Average number of in-/out-connections per cell.
    cellLigList : 2D np.array() with {0,1}-entries, optional
        The rules indicating which cell types (rows) can possibly connect to which
        ligand types (columns). The default is [].
    cellRecList : 2D np.array() with {0,1}-entries, optional
        The rules indicating which cell types (rows) can possibly connect to which
        receptor types (columns). The default is [].
    Dcelltype : 1D np.array() with float entries, optional
        The distribution of cell types. The default is [].
    Dligrec : 2D np.array() with float entries, optional
        The distribution of ligand/receptor-pairs. The default is [].
    Signmatrix : 2D np.array() with int entries, optional
        The signs of the lig_rec interactions.
    genRandom : bool, optional
        Indicates whether to read input data from a file or to generate
        synthetic input data from the distribution_generation file. 
        The default is True (read in from actual mRNA input file).

    Returns
    -------
    V : 1D np.array() with int entries;
        The cell-type of all vertices (as a number).
    E : 2D np.array() with int entries;
        The edge list of the generated graph.
    types : 2D np.array() with int entries;
        The ligand and receptor type for each edge. Optional (depending on the
        sign matrix input): the sign of each ligand-receptor interaction behind 
        the interaction.
    """
    
    if genRandom:
        cellTypeNo = 11 #Number of cell types
        ligNo = 10 #Number of ligand types
        recNo = 10 #Number of receptor types
        M = int(round(avdeg*N)) #Number of edges
    else:
        cellTypeNo = Dcelltype.size
        ligNo = Dligrec.shape[0]
        recNo = Dligrec.shape[1]
        M = int(round(avdeg*N))
    
    #Generate info about the micro-env, meaning
    #1. How cells can connect
    #2. With what probability connections appear.
    #3. With what probability cell types appear.
    if genRandom:
        info = informationMicroEnv(cellTypeNo,ligNo,recNo, genRandom)
    else:
        info = informationMicroEnv(cellTypeNo,ligNo,recNo, genRandom,\
                                   Dcelltype, Dligrec, cellLigList, cellRecList)
    
    #Generate specific cell type list and edge list
    V = info.genRandomCellTypeList(N)
    E, types = info.genRandomEdgeList(M)
    
    if len(Signmatrix) > 0:
        #Add signs to all ligand-receptor interaction if needed.
        interactions = Signmatrix[types[:,0],types[:,1]]
        types = np.column_stack((types, interactions))
    
    return V, E, types


if __name__ == "__main__":
    #Small example based on manual input data.  
    Dcelltype, LigRecDistr, CellLigConnect, CellRecConnect = dg.manual_experiment()
    V, E, types = model1(1000000, 20, CellLigConnect, CellRecConnect, Dcelltype, LigRecDistr, [], False)




   

