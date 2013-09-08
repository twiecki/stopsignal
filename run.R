args <- commandArgs(trailingOnly=TRUE)

samples	= as.integer(args[1])
burn.in 	= as.integer(args[2])
number.of.chains = as.integer(args[3])
thinning = as.integer(args[4])
estimates.for.all = (args[5] == "All")
summary.statistics = args[6] == "1"
posterior.distributions = args[7] == "1"
mcmc.chains = args[8] == "1"
deviance = args[9] == "1"
posterior.predictors = args[10] == "1"
posterior.predictors.samples = as.integer(args[11]) 
num.cores = as.integer(args[12])
data.file = args[13]

name.data.file = gsub(".csv", "", data.file)

if (samples < 1){
        stop("The total number of MCMC samples must be greater than zero.")
}

if (samples <= burn.in){
        stop("The total number of MCMC samples must be greater than the number of burn-in samples.")
}

if (thinning < 1){
       stop("The thinning factor must be greater than 0.")
}

if  (((samples-burn.in)/thinning) < 1){
        stop("No MCMC samples will be retained. Increse the number of retained samples or decrese the thinning factor.")
}

if (posterior.predictors == T){
  if ((((samples-burn.in)/thinning)*number.of.chains) < posterior.predictors.samples){
        stop("The number of posterior predictive samples cannot be greater than the number of retained MCMC samples.")
  }
}
            
source("Aux_BEEST.R")  # <- the wd for these will be the location of this run.R file        
file.location <- dirname(data.file)
setwd(file.location)  # set the wd to the file location

#print(getwd())
#print(data.file)

if (summary.statistics == T|posterior.distributions==T|mcmc.chains==T|posterior.predictors==T){
  mcmc.samples = read_prep(n_chains = number.of.chains,name_data.file = name.data.file)
  
  if (summary.statistics==T){
    summary_stats(pars = mcmc.samples,all_pars=estimates.for.all,name_data.file = name.data.file)
    print("Summary statistics are saved to file.")
  }
  
  if (posterior.distributions==T){
    plot_posteriors(pars = mcmc.samples,all_pars=estimates.for.all,name_data.file = name.data.file) 
    print("Posterior distributions are saved to file.")
  }
  
  if (mcmc.chains==T){
    plot_chains(pars = mcmc.samples,all_pars=estimates.for.all,name_data.file = name.data.file)  
    print("MCMC chains are saved to file.")
  }
  
  if (posterior.predictors==T){
    dat = read.csv(file=data.file,head=TRUE,sep=",")
    print("Running posterior predictive model checks. This might take a while...")
    suppressPackageStartupMessages(library("gamlss.dist"))
    suppressPackageStartupMessages(library("vioplot"))
    #library("gamlss.dist")
    #library("vioplot")
    posterior_predictions(pars = mcmc.samples,n_post_samples = posterior.predictors.samples,data=dat,name_data.file = name.data.file) 
    print("Results of posterior predictive model checks are saved to file.")
  }
}

#save(samples,file="Traces.RData")
#    print(samples)
#    print(burn.in)
#    print(number.of.chains)
#    print(thinning)
#    print(estimates.for.all)
#    print(summary.statistics)
#    print(posterior.distributions)
#    print(mcmc.chains)
#    print(deviance)
#    print(posterior.predictors)
#    print(posterior.predictors.samples)
#    print(num.cores)
#    print(data.file)