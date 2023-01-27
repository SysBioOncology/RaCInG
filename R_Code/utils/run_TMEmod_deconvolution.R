# --------------------------------------------------- #
# Script to run deconvolution computational methods   #
# Francesca Finotello wrote this code.                #
# --------------------------------------------------- #

run_TMEmod_deconvolution <- function(RNA_tpm, cancer_type){
  
  # packages
  # install.packages("remotes")
  # remotes::install_github("icbi-lab/immunedeconv")
  # install.packages("devtools")
  # devtools::install_github("cansysbio/ConsensusTME")
  
  library('immunedeconv')
  library('ConsensusTME')

  cellfrac.consensusTME <- ConsensusTME::consensusTMEAnalysis(as.matrix(RNA_tpm), 
                                                              cancer = cancer_type, 
                                                              statMethod = "ssgsea")
  
  cellfrac.quanTIseq <- deconvolute_quantiseq(RNA_tpm,
                                              tumor = TRUE, 
                                              arrays = FALSE,
                                              scale_mrna = TRUE)
  
  cellfrac.EPIC <- deconvolute_epic(RNA_tpm, 
                                    tumor = TRUE, 
                                    scale_mrna = TRUE)
  
  cellfrac.mcpcounter <- deconvolute_mcp_counter(RNA_tpm,
                                                 feature_types = "HUGO_symbols")
  
  cellfrac.xCell <- deconvolute_xcell(RNA_tpm,
                                      arrays = FALSE)
  
  cellfrac.TIMER <- deconvolute_timer(RNA_tpm, 
                                      indications = rep(cancer_type, ncol(RNA_tpm)))
  
  
  return(list(quanTIseq=cellfrac.quanTIseq, 
       EPIC=cellfrac.EPIC, 
       mcpcounter=cellfrac.mcpcounter, 
       xCell=cellfrac.xCell, 
       TIMER=cellfrac.TIMER, 
       consensusTME=cellfrac.consensusTME))

}
