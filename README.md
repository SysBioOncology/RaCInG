# RaCInG
The random graph model to infer cell-cell communication networks based on bulk RNA-seq data. The ropo contains the code used for the paper "Mathematically mapping the network of cells in the tumor microenvironment" by van Santvoort et al.

## General description
In this study we used patient specific bulk RNA-seq input together with non-patient specific prior knowledge on possible ligand-receptor interactions to reconstruct cell-cell interaction networks in an indivudal patient's tumour. The model operates in four main steps:
1. It transforms mRNA-seq input data into four matrices used by the graph generation procedure. These four input matrices contain information about cellular deconvolution (C-matrix), ligand-receptor interaction pair quantification (LR-matrix), ligand compatibility with cell-types (L-matrix) and receptor compatibility with cell types (R-matrix).
2. It uses the matrices to construct an ensemble of possible cell-cell interaction networks for individual patients. It does this by generating a fixed number of individual cells from the C-matrix, and a fixed number of ligand-receptor pair from the LR-matrix. Then, it binds the LR-pairs to indivudual cells unfiromly at random, respecting the compatibility of the ligand/receptor with the cell-type until everything is paired.
3. From the ensemble of networks certain fingerprints are extracted (e.g. the count of triangles). By averaging over the the counts for different graphs in the ensemble a feature value for the individual patient is created. Features are only extracted if they are expected to remain statistically consistent over the networks ensemble.
4. The features are used as biomarkers for the individual patient's tumor micro-environment, and through statistical testing meta-features of a patient (like resonse to immunotherapy) can be analysed. 

To validate the model, we have applied it to extract 444 network features related to the tumor microenvironment in 3213 solid cancer patients, and unveiled associations with immune response and subtypes, and identified cancer-specific differences in inter-cellular signaling. Additionally, we have used RaCInG in 118 patients with known response to immunotherapy to explain how immune phenotypes regulated by context-specific intercellular communication affect their response. RaCInG is a modular pipeline, and we envision its application for cell-cell interaction reconstruction in different contexts.

## Uses of the model
The code can be used to transform RNA-seq data into input matrices for RaCInG, and execute it to construct cell-cell interaction networks for individual patients. Moreover, it can be used to extract some predefined features for these networks, and do some statistical tests on these features. In order to run this model, the folder *R_code* contains all functionalities to turn RNA-seq data into input matrices for the model. The folder *Python_Code* contains the functions to execute the additional functionalities:
- The function *execute_sim.py* allows the user to construct networks based on the input matrices and extract certain features from them.
- The function *txt_to_csv.py* combines the output of a uniform and non-uniform run of the random graph model to create normalized feature values for each patient.
- The function *statistical_analysis.py* allows the user to do some statistical analysis on the output of the random graph model.
- The function *retrieveLigRecInfo.py* allows the user to compute the ligand-receptor interaction probabilities in given cell-types.
- The function *Circos.py* allows the user to compute direct communication values using the theoretical kernel of the random graph model to be plotted in a Circos plot.

In execution of the Python code make sure that all Python functions are located together. They each have their own functionalities, but work together to execute the model. Moreover, make sure that all input matrices for the model are located in the same folder (in our case the folder *Example input*) and that the metadata is located in the same folder as the *statistical_analysis.py* function.

:information_source: **A demo of the Python functionalities is available as Jupyter notebook in the Python_Code folder (called *Demo.ipynb*).**

:information_source: **A demo of R functionalities is available in the R_Code folder as Rmarkdown files.**

## Software
### R-code

We used R version 4.2.0 and the following R packages were used:

- liana 0.1.10
- OmnipathR 3.7.0
- immunedeconv 2.1.0
- easier 1.4.0
- EPIC 1.1.5
- MCPcounter 1.2.0
- quantiseqr 1.6.0
- xCell 1.1.0
- ConsensusTME 0.0.1.9000
- corrplot 0.92
- dplyr 1.0.10

### Python-code
The code was developed and run in Anaconda on Python 3.8.11. The following Python packages are used:
- Numpy 1.16.6
- Scipy 1.7.1
- Matplotlib 3.4.3
- Seaborn 0.11.2
- Pandas 1.2.4
- Adjusttext 0.7.3

The Adjusttext package does not have a standard build in Anaconda. We recommend installing it with pip from the Powershell terminal.
```
pip install Adjusttext==0.7.3
```

Finally, in the *Envs* folder we have uploaded the file *RaCInG_Environment.yaml* which contains all dependencies in the environmemt on which we ran the model. You can import this file into Anaconda (from the Environments tab) to recreate the environment on which we ran the Python code.

## Files
### R files
- **RaCInG_ccc_prior_knowledge.Rmd**: RMarkdown notebook to extract knowledge of cell-type compatibility of ligands and receptors. We rendered this .Rmd file into a HTML file which reports the main results of each step of the analysis (**RaCInG_ccc_prior_knowledge.html**).
- **RaCInG_input_tcga.Rmd**: RMarkdown notebook used to quantify cell-type abundance, ligand-receptor pair activation and immune response for patients from The Cancer Genome Atlas (TCGA).
- **RaCInG_input_published_cohorts.Rmd**: RMarkdown notebook used to to quantify cell-type abundance and ligand-receptor pair activation for the validation cohorts (datasets of patients treated with immunotherapy).
- **utils/run_TMEmod_deconvolution**: Runs in silico deconvolution from bulk-tumor RNA-seq data through `immunedeconv` R package.

### Python files
We explain the basic functionalities of all Python files. Please consult the demo for the usage. Additionally, all functions in the files have been given docstrings that should explain their funcitonality in more detail.
- **RaCInG_input_generation.py**: Reads the input matrices for the graph generation procedure from .csv files.
- **distribution_generation.py**: Generates random input matrices for the graph generation procedure (for testing purposes; not used in paper).
- **network_generation.py**: Creates single network instances based on the input matrices.
- **Theoretical_Direct_Communication.py**: Computes direct communication values with the kernel method.
- **Utilities.py**: Contains utitlity functions to create adjacency matrices from edge lists and quickly count the number of certain objects from an object list.
- **feature_extraction.py**: Contains functions to extract feature from generated graphs.
- **Tarjan.py**: Computes the size of the giant strongly connected component in generated networks using Tarjan's algorithm (not used in paper).
- **Graph_saving.py**: Contains functionalities to save generated graphs (for reproduction and testing purposes; not used in paper).
- **execute_sim.py**: Combines the functionalities of all the above files into one pipeline to create graphs and analyse them from input matrices.
- **HPC_CLI.py**: Creates a command line interface setup using **execute_sim.py** that allows its execution in a HPC environment[^GNU] paralellized over the patients.
- **txt_to_csv.py**: Combines non-uniform and uniform output from **execute_sim.py** into normalized feature values.
- **statistical_analysis.py**: Executes statistical analysis on normalized feature values based on metadata from patients.
- **retrieveLigRecInfo.py**: Computes probabilities of ligand-receptor interactions conditioned on the appearance of a ligand-receptor interaction between two given cell types (and plots the pairs with the largerst conditional probability).
- **Circos.py**: Computes direct communcation values for all patients based on the network kernel, and outputs the average for a group of patients in a format that can be read by the online tool Circos[^circosNote] to create a plot.

## Data

- **expressed_ccc_table_omnipath_ramilowski_quantiseq_filtered_LR_10tpm.csv**: .csv file containg the extracted prior knowledge of cell-cell communication (returned as output by **RaCInG_ccc_prior_knowledge.Rmd**).

## Additional Folders
The folder *Envs* contains the environments used when executing RaCInG to obtain the results in the paper. The folder *Manuals* contains a more detailed description of the input matrices used in RaCInG, and the outputs that the graph generation pipeline creates. 

## Notes
[^circosNote]: You can create Circos plots based on the output of **Circos.py** [here](http://circos.ca/).
[^GNU]: We have used [GNU Parallel](https://www.gnu.org/software/parallel/) to use the command line interface from **HPC.py** in a HPC environment.
