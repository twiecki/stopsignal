#Load python traces in R and make array for further use
read_prep = function(n_chains,name_data.file) {

	for(i in 1:n_chains){
		chain = read.csv(file=paste(paste(paste(name_data.file,"_parameters",i,sep=""),sep=""),".csv",sep=""),head=TRUE,sep=";")
		my_chain = as.matrix(chain[,order(names(chain))])
		n_subject = ifelse(ncol(my_chain)==6,1,(ncol(my_chain)-12)/6)
		
		if (i == 1) {
			n_dim = dim(chain)
			my_names = dimnames(my_chain)[[2]]
      n_samples = n_dim[1]
			traces = array(NA,dim=c(n_dim,n_chains),dimnames=list(1:n_dim[1],my_names,paste("chain", 1:n_chains)))
		} 
		traces[,,i] = my_chain

	}
	return(list(traces = traces, n_subject = n_subject, my_names = my_names, n_samples = n_samples, n_chains = n_chains))
}

#Computes summary statistics for each parameter per participant (collapsed over chains) and save output to csv.
summary_stats_individual = function(pars, params, subject_idx = NULL,name_data.file){

	n_pars = 6
	
	summary = matrix(NA,n_pars+4,7)
	colnames(summary) = c("Mean","Sd","2.5%","25%","50%","75%","97.5%")
	rownames(summary) = c(params,"mean go","sd go","mean SSRT","sd SSRT")
  #unnecesary...
	mu_go_subj = grep(glob2rx("mu_go*"), params, value=TRUE)
	sigma_go_subj = grep(glob2rx("sigma_go*"), params, value=TRUE)
	tau_go_subj = grep(glob2rx("tau_go*"), params, value=TRUE)
	
	mu_stop_subj = grep(glob2rx("mu_stop*"), params, value=TRUE)
	sigma_stop_subj = grep(glob2rx("sigma_stop*"), params, value=TRUE)
	tau_stop_subj = grep(glob2rx("tau_stop*"), params, value=TRUE)

	for(par in params){
		summary[par,1] = round(mean(as.vector(pars$traces[,par,])),4)
		summary[par,2] = round(sd(as.vector(pars$traces[,par,])),4)
		summary[par,3:7] = round(quantile(as.vector(pars$traces[,par,]),prob = c(0.025,0.25,0.5,0.75,0.975)),4)
	}

	meanGo = as.vector(pars$traces[,mu_go_subj,])+as.vector(pars$traces[,tau_go_subj,])
	sdGo = sqrt(as.vector(pars$traces[,sigma_go_subj,])^2+as.vector(pars$traces[,tau_go_subj,])^2)
	meanSRRT = as.vector(pars$traces[,mu_stop_subj ,])+as.vector(pars$traces[,tau_stop_subj ,])
	sdSRRT = sqrt(as.vector(pars$traces[,sigma_stop_subj,])^2+as.vector(pars$traces[,tau_stop_subj ,])^2)

	summary["mean go",1] = round(mean(meanGo),4)
	summary["mean go",2] = round(sd(meanGo),4)
	summary["mean go",3:7] = round(quantile(meanGo),4)

	summary["sd go",1] = round(mean(sdGo),4)
	summary["sd go",2] = round(sd(sdGo),4)
	summary["sd go",3:7] = round(quantile(sdGo),4)

	summary["mean SSRT",1] = round(mean(meanSRRT),4)
	summary["mean SSRT",2] = round(sd(meanSRRT),4)
	summary["mean SSRT",3:7] = round(quantile(meanSRRT),4)

	summary["sd SSRT",1] = round(mean(sdSRRT),4)
	summary["sd SSRT",2] = round(sd(sdSRRT),4)
	summary["sd SSRT",3:7] = round(quantile(sdSRRT),4)

	write.csv(summary,file=paste(paste(paste(name_data.file,"_individual_summary",sep=""),subject_idx,sep=""),".csv",sep=""))
}

#Computes summary statistics for each group parameter (collapsed over chains) and save output to csv.
summary_stats_group = function(pars,name_data.file){

	#only group nodes
	group_pars = c("mu_go","mu_go_var","mu_stop","mu_stop_var","sigma_go","sigma_go_var","sigma_stop","sigma_stop_var","tau_go","tau_go_var","tau_stop","tau_stop_var")
	summary = matrix(NA,12,7)
	colnames(summary) = c("Mean","Sd","2.5%","25%","50%","75%","97.5%")
	rownames(summary) = group_pars

	for(par in group_pars){
		summary[par,1] = round(mean(as.vector(pars$traces[,par,])),4)
		summary[par,2] = round(sd(as.vector(pars$traces[,par,])),4)
		summary[par,3:7] = round(quantile(as.vector(pars$traces[,par,]),prob = c(0.025,0.25,0.5,0.75,0.975)),4)
	}
	write.csv(summary,file = paste(name_data.file,"_group_parameter_summary.csv",sep=""))
}

#Summary stats using summary_stats_individual() and summary_stats_group() functions
summary_stats = function(pars,all_pars=T,name_data.file){
	if (all_pars){
		if (pars$n_subject==1) ###Individual model
			summary_stats_individual(pars, params = c("mu_go","mu_stop","sigma_go","sigma_stop","tau_go","tau_stop"),name_data.file = name_data.file)
		else {
			summary_stats_group(pars,name_data.file = name_data.file)
		
			for (n in 1:pars$n_subject){ #Seperate table for each participant
				param_name = c(paste("mu_go_subj.",n,sep=""),paste("mu_stop_subj.",n,sep=""),
                        paste("sigma_go_subj.",n,sep=""),paste("sigma_stop_subj.",n,sep=""),
                        paste("tau_go_subj.",n,sep=""),paste("tau_stop_subj.",n,sep=""))
			
			summary_stats_individual(pars, params = param_name,subject_idx = n,name_data.file = name_data.file)
			}
		} 
	} else summary_stats_group(pars,name_data.file = name_data.file)
}

#Plots posteriors for individual participants for the individual model 
plot_individual_posteriors = function(pars,params,name_data.file){

  prior_den = list(mu_go = dunif(1,0.001,1000), mu_stop = dunif(1,0.001,600),
			sigma_go = dunif(1,1,500),sigma_stop = dunif(1,1,350),
		  tau_go = dunif(1,1,500),tau_stop = dunif(1,1,350))
	
	pdf(paste(paste(name_data.file,"_individual_posteriors",sep=""),".pdf",sep=""),paper = "special",width=8, height=7.5)
	layout(matrix(1:6,3,byrow=T))
	par(cex.main=1.4)

	for(par in params){
    lim = c(min(pars$traces[,par,])-30,max(pars$traces[,par,])+30)
    plot(density(pars$traces[,par,]),xlim = lim, main = par,xlab="RT (ms)",ylab = "Density")
    lines(c(0,lim[2]),c(prior_den[names(prior_den)==par],prior_den[names(prior_den)==par]),lty=2)
  }
  
	dev.off()
}

#Plots posteriors for individual participants for the individual model 
plot_individual_posteriors_for_hier = function(pars,params,subject_idx,name_data.file){

	pdf(paste(paste(paste(name_data.file,"_individual_posteriors",sep=""),subject_idx,sep=""),".pdf",sep=""),paper = "special",width=8, height=7.5)
	layout(matrix(1:6,3,byrow=T))
	par(cex.main=1.4)

	for(par in params){
    lim = c(min(pars$traces[,par,])-30,max(pars$traces[,par,])+30)
    plot(density(pars$traces[,par,]),xlim = lim, main = par,xlab="RT (ms)",ylab = "Density")
  }
  
	dev.off()
}

#Plots posterior of group level parameters for hierarchical model
plot_group_posteriors = function(pars,name_data.file){

  prior_den_group = list(mu_go = dunif(1,0.001,1000), mu_go_var = dunif(1,0.01,300),
                   mu_stop = dunif(1,0.001,600), mu_stop_var = dunif(1,0.01,300),
                   sigma_go = dunif(1,1,500), sigma_go_var = dunif(1,0.01,200),
                   sigma_stop = dunif(1,1,350), sigma_stop_var = dunif(1,0.01,200),
                   tau_go = dunif(1,1,500), tau_go_var = dunif(1,0.01,200),
                   tau_stop = dunif(1,1,350), tau_stop_var = dunif(1,0.01,200))
		  
  pdf(paste(name_data.file,"_group_parameter_posteriors.pdf",sep=""),paper = "special" ,width=14, height=6.5)
	layout(matrix(1:12,2))
	par(cex.main=1.4)
	
	group_pars = c("mu_go","mu_go_var","mu_stop","mu_stop_var","sigma_go","sigma_go_var","sigma_stop", "sigma_stop_var","tau_go","tau_go_var","tau_stop","tau_stop_var")
	
	for(par in group_pars){
    		lim = c(min(pars$traces[,par,])-30,max(pars$traces[,par,])+30)
    		plot(density(pars$traces[,par,]),xlim = lim, main = par,xlab="RT (ms)",ylab = "Density")
    		lines(c(0,lim[2]),c(prior_den_group[names(prior_den_group)==par],prior_den_group[names(prior_den_group)==par]),lty=2)
  	}

	dev.off()
}

#Plots posteriors using plot_individual_posteriors() and plot_group_posteriors() functions
plot_posteriors = function(pars,all_pars=T,name_data.file){
	if (all_pars){
		if (pars$n_subject==1) ###Individual model
			plot_individual_posteriors(pars, params = c("mu_go","mu_stop","sigma_go","sigma_stop","tau_go","tau_stop"),name_data.file = name_data.file) 
                           
		else {
			plot_group_posteriors(pars,name_data.file = name_data.file)
		
			for (n in 1:pars$n_subject){ #Seperate figure for each participant
				param_name = c(paste("mu_go_subj.",n,sep=""),paste("mu_stop_subj.",n,sep=""),
                        paste("sigma_go_subj.",n,sep=""),paste("sigma_stop_subj.",n,sep=""),
                        paste("tau_go_subj.",n,sep=""),paste("tau_stop_subj.",n,sep=""))

				plot_individual_posteriors_for_hier(pars, params = param_name, subject_idx = n,name_data.file = name_data.file)
			}
		} 
	} else plot_group_posteriors(pars,name_data.file = name_data.file)
}

#Plots chains for individual participants for the individual model as well as the hierarchical model
plot_individual_chains = function(pars,params,subject_idx = NULL,name_data.file){

	pdf(paste(paste(paste(name_data.file,"_individual_chains",sep=""),subject_idx,sep=""),".pdf",sep=""),paper = "special",width=8, height=7.5)
	layout(matrix(1:6,3,byrow=T))
	par(cex.main=1.4)

	for(par in params){
    lim = c(min(pars$traces[,par,])-10,max(pars$traces[,par,])+10)
    plot(1:pars$n_samples,pars$traces[,par,1],ylim = lim, xlim = c(1,pars$n_samples), main = par,xlab="Iteration",ylab = par,type="l")
		
		if(pars$n_chains>1){  #if multiple chains, draw lines
			for(j in 2:pars$n_chains){
				lines(1:pars$n_samples,pars$traces[,par,j],col=j)
			}
		}
	}
	dev.off()
}

#Plots chains of group level parameters for hierarchical model
plot_group_chains = function(pars,name_data.file){
  pdf(paste(name_data.file,"_group_parameter_chains.pdf",sep=""),paper = "special" ,width=16, height=5.5)
	layout(matrix(1:12,2))
	par(cex.main=1.4)
	
	group_pars = c("mu_go","mu_go_var","mu_stop","mu_stop_var",
                "sigma_go","sigma_go_var", "sigma_stop","sigma_stop_var",
                 "tau_go","tau_go_var","tau_stop","tau_stop_var")

	for(par in group_pars){
    		lim = c(min(pars$traces[,par,])-10,max(pars$traces[,par,])+10)
    		plot(1:pars$n_samples,pars$traces[,par,1],ylim = lim, xlim = c(1,pars$n_samples), main = par,xlab="Iteration",ylab = par,type="l")
		
		if(pars$n_chains>1){ #if multiple chains, draw lines
			for(j in 2:pars$n_chains){
				lines(1:pars$n_samples,pars$traces[,par,j],col=j)
			}
		}
  }

	dev.off()
}

#Plots chains using plot_individual_chains() and plot_group_chains() functions
plot_chains = function(pars,all_pars=T,name_data.file){
	if (all_pars){
		if (pars$n_subject==1) ###Individual model
			plot_individual_chains(pars, params = c("mu_go","mu_stop","sigma_go","sigma_stop","tau_go","tau_stop"),name_data.file = name_data.file)
		else {
			plot_group_chains(pars,name_data.file = name_data.file)
		
			for (n in 1:pars$n_subject){ #Seperate figure for each participant
				param_name = c(paste("mu_go_subj.",n,sep=""),paste("mu_stop_subj.",n,sep=""),
                        paste("sigma_go_subj.",n,sep=""),paste("sigma_stop_subj.",n,sep=""),
                        paste("tau_go_subj.",n,sep=""),paste("tau_stop_subj.",n,sep=""))
				
				plot_individual_chains(pars, params = param_name, subject_idx = n,name_data.file = name_data.file)
			}
		} 
	} else plot_group_chains(pars,name_data.file = name_data.file)
}

load_prep_observed_data= function(pars,subject_idx,data){

	if(pars$n_subject == 1){
		data = cbind(subj_idx = rep(1,nrow(data)),data)
	}

	#select delays with at least one srrt
	delays = unique(data$ssd[data$subj_idx==subject_idx&data$"ss_presented"==1&data$inhibited==0&data$rt!=-999])
	n_delays = length(delays)

	#observed SRRTs per delay
	median_observed_srrt = rep(NA,length(delays))
	n_observed_srrt = rep(NA,length(delays))
	n_observed_inhibit = rep(NA,length(delays))

	for (d in 1:n_delays){
		srrt_temp = data$rt[data$subj_idx==subject_idx&data$ss_presented==1&data$inhibited==0&data$ssd==delays[d]]
		median_observed_srrt[d] = median(srrt_temp)
		n_observed_srrt[d] = length(srrt_temp)
		n_observed_inhibit[d] = nrow(data[data$subj_idx==subject_idx&data$ss_presented==1&data$inhibited==1&data$rt==-999&data$ssd==delays[d],])
	}

	#number of stopsignal trials per delay
	n_observed = n_observed_srrt + n_observed_inhibit

	return(list(delays = delays, n_delays = n_delays, median_observed_srrt = median_observed_srrt, n_observed_srrt = n_observed_srrt, n_observed = n_observed))
}

# Sample from the joint distribution of the parameters
sample_joint_posterior = function(pars,n_post_samples,subject_idx = NULL){
	
	#n_post_samples must to be smaller than pars$n_samples*pars$n_chains!!!
	if(pars$n_subject==1){
		params = c("mu_go","mu_stop","sigma_go","sigma_stop","tau_go","tau_stop")
	} else {
		params = c(paste("mu_go_subj.",subject_idx,sep=""),paste("mu_stop_subj.",subject_idx,sep=""),
				paste("sigma_go_subj.",subject_idx,sep=""),paste("sigma_stop_subj.",subject_idx,sep=""),
				paste("tau_go_subj.",subject_idx,sep=""),paste("tau_stop_subj.",subject_idx,sep=""))
	}

	par_vectors = array(NA,dim = c(n_post_samples,6),list(NULL,params))
	it_ind = sample(1:(pars$n_samples*pars$n_chains),n_post_samples,replace=F)
	
	for(p in params){
		par_vectors[,p] = as.vector(pars$traces[,p,])[it_ind]
	}

	return(list(par_vectors = par_vectors, params = params))
}

# Generates posterior predictions and corresponding p values
generate_posterior_predicitions = function(delays,n_delays,n_observed,median_observed_srrt,par_vectors,n_post_samples,params){
	
	median_post_pred = matrix(NA,n_delays,n_post_samples)
	pvalue_one = rep(NA,n_delays)
	pvalue_two = rep(NA,n_delays)
	post_pred = sapply(as.character(delays),function(x) NULL)
	
	for(d in 1:n_delays){
		for(j in 1:n_post_samples){
			go_rt = rexGAUS(n_observed[d],mu = par_vectors[j,params[1]],sigma = par_vectors[j,params[3]],nu = par_vectors[j,params[5]])
			stop_rt = delays[d] + rexGAUS(n_observed[d],mu=par_vectors[j,params[2]],sigma = par_vectors[j,params[4]],nu = par_vectors[j,params[6]])
			srrt = go_rt[go_rt<=stop_rt]
			post_pred[[d]][[j]] = list(srrt)
			median_post_pred[d,j] = median(srrt)	
		}
		pvalue_one[d] = mean(median_post_pred[d,]> median_observed_srrt[d],na.rm=T)
	}
	ipvalue_one = 1-pvalue_one
	
	for(d in 1:n_delays){
			pvalue_two[d] = 2*min(pvalue_one[d],ipvalue_one[d])
	}		
	average_post_pred = apply(median_post_pred,1,mean,na.rm=T)
	return(list(median_post_pred = median_post_pred, average_post_pred = average_post_pred, pvalue_one = pvalue_one, pvalue_two = pvalue_two))
}

# Save results of posterior predictive model checks, incl. p values to csv
save_post_pred = function(delays,n_delays,median_observed_srrt,n_observed_srrt,average_post_pred,pvalue_one,pvalue_two,subject_idx=NULL,name_data.file){

	post_pred_summary = rbind(round(rbind(n_observed_srrt,median_observed_srrt,average_post_pred),2),
                            round(rbind(pvalue_one,pvalue_two),3))
	colnames(post_pred_summary) = paste(rep("SSD=",each=n_delays),delays,sep="")
	rownames(post_pred_summary) = c("Number of observed SRRT", "Observed median SRRT","Average posterior prediction","One-sided p value","Two-sided p value")	
	
	write.csv(post_pred_summary,file = paste(paste(paste(name_data.file,"_summary_posterior_predictions",sep=""),subject_idx,sep=""),".csv",sep=""))
}

# Violin plots for posterior predictive model checks
plot_post_pred = function(delays,n_delays, median_observed_srrt,median_post_pred,subject_idx=NULL,name_data.file){

	my_min = min(c(min(median_post_pred,na.rm=T),min(median_observed_srrt)))-80
	my_max = max(c(max(median_post_pred,na.rm=T),max(median_observed_srrt)))+80
	ran = round(c(my_min,my_max))
	pdf(paste(paste(paste(name_data.file,"_plot_posterior_predictions",sep=""),subject_idx,sep=""),".pdf",sep=""),paper = "special" ,width=9, height=9)
	par(cex=1.2,cex.lab = 1.2,mar=c(5,8,2,2),cex.main=1.4)
	plot(1:length(delays),median_observed_srrt,pch=17,xlab="SSD (ms)",axes=F,
		ylim=ran,xlim=c(0,n_delays+1),ylab ="",main="Posterior predictive model checks",cex=1.3)
	axis(1,at=0:(n_delays+1),labels=c(NA,delays,NA))
	axis(2,at=ran,las=2)
	mtext("Median SRRT (ms)",2, cex=1.4,line=1)
	for(i in 1:n_delays){
    preds_med = median_post_pred[i,][!is.na(median_post_pred[i,])]
    if (length(preds_med) >0) {
      vioplot(preds_med,add=T,at=i,col="gray",cex=2)
    }
	}

	points(1:n_delays,median_observed_srrt,cex=1.5,pch=17)
	lines(1:n_delays,median_observed_srrt,lwd=2,lty=2)
	dev.off()
}

posterior_predictions = function(pars,n_post_samples,data,name_data.file){

	if(pars$n_subject==1){
		obs = load_prep_observed_data(pars = pars,subject_idx = 1,data = data)
		post_pars = sample_joint_posterior(pars = pars,n_post_samples = n_post_samples)
		post_predictions = generate_posterior_predicitions(delays = obs$delays,n_delays = obs$n_delays, n_observed = obs$n_observed, 
			median_observed_srrt = obs$median_observed_srrt, par_vectors = post_pars$par_vectors, n_post_samples = n_post_samples, params = post_pars$params)
		save_post_pred(delays = obs$delays,median_observed_srrt = obs$median_observed_srrt,n_observed_srrt = obs$n_observed_srrt,
				average_post_pred = post_predictions$average_post_pred,pvalue_one = post_predictions$pvalue_one,pvalue_two = post_predictions$pvalue_two, name_data.file = name_data.file)
		plot_post_pred(delays = obs$delays,n_delays = obs$n_delays, median_observed_srrt = obs$median_observed_srrt,
				median_post_pred = post_predictions$median_post_pred, name_data.file = name_data.file)
				
		} else{
		
      	for(n in 1:pars$n_subject){
      		obs = load_prep_observed_data(pars = pars,subject_idx = n,data = data)
        		post_pars = sample_joint_posterior(pars = pars,n_post_samples = n_post_samples,subject_idx = n)
        		post_predictions = generate_posterior_predicitions(delays = obs$delays,n_delays = obs$n_delays, n_observed = obs$n_observed, 
         			median_observed_srrt = obs$median_observed_srrt, par_vectors = post_pars$par_vectors, n_post_samples = n_post_samples, params = post_pars$params)
        		save_post_pred(delays = obs$delays,median_observed_srrt = obs$median_observed_srrt,n_observed_srrt = obs$n_observed_srrt,
          			average_post_pred = post_predictions$average_post_pred,pvalue_one = post_predictions$pvalue_one,pvalue_two = post_predictions$pvalue_two,subject_idx = n, name_data.file = name_data.file)
        		plot_post_pred(delays = obs$delays,n_delays = obs$n_delays, median_observed_srrt = obs$median_observed_srrt,
        			median_post_pred = post_predictions$median_post_pred,subject_idx = n,name_data.file = name_data.file)
      	}
    	}
}