# -*- coding: utf-8 -*-
"""
Command line interface to run the code in an HPC environment
"""

import argparse
from Monte_Carlo_Method import runSimOne as run

#-------------------------------
#     Command Line Interface    
#-------------------------------

#Creating parser
my_parser = argparse.ArgumentParser(prog = "RaCInG", 
                                    description = "Computes graph features of given patient")

my_parser.add_argument("weight_type", type = str, help = "Type of edge weights used")
my_parser.add_argument("cancer_type", type = str, help = "Type of cancer considered")
my_parser.add_argument("feature", type = str, help = "Feature indicator")
my_parser.add_argument("patient", type = int, help = "The number of the patient to be analyzed")
my_parser.add_argument("N", type = int, help = "The number of vertices/cells per graph")
my_parser.add_argument("itNo", type = int, help = "The number of graphs used for calculations")
my_parser.add_argument("av", type = float, help = "The average in-/out- degree of each vertex")
my_parser.add_argument("norm", type = int, help = "Determines whether objects get counted on the normalised version of the graph or not")

args = my_parser.parse_args()


run(args.cancer_type, args.weight_type, args.feature, args.patient, args.N, args.itNo, args.av, args.norm)