from __future__ import division
#print "Loading packages...",
import os
import sys
import multiprocessing
import cStringIO
import stopsignal
from kabuki.analyze import gelman_rubin
from kabuki.utils import get_traces
from kabuki.utils import save_csv
from kabuki.utils import load_csv
import numpy
import math

#print "Packages loaded."

if (len(sys.argv) != 14):
	print("no good!")
	exit(1)

#print sys.argv

samples	= int(sys.argv[1])
burnIn 	= int(sys.argv[2])
numberOfChains = int(sys.argv[3])
thinning = int(sys.argv[4])
estimatesForAll = (sys.argv[5]=="All")
summaryStatistics = (int(sys.argv[6]) != 0)
posteriorDistributions = (int(sys.argv[7]) != 0)
mcmcChains = (int(sys.argv[8]) != 0)
deviance = (int(sys.argv[9]) != 0)
posteriorPredictors = (int(sys.argv[10]) != 0)
posteriorPredictorsSamples = int(sys.argv[11])  
numCores = int(sys.argv[12])
dataFile = sys.argv[13]

actual_cores = multiprocessing.cpu_count()

if numCores < 1:
        print('Warning: Inputting 0 CPU core is silly. BEESTs will use the deault number of ' + str(actual_cores-1) + ' cores.')
        numCores = actual_cores -1
        
if samples < 1:
        #print "The total number of MCMC samples must be greater than zero."
        sys.exit() 

if samples <= burnIn:
        #print "The total number of MCMC samples must be higher than the number of burn-in samples."
        sys.exit() 

if thinning < 1:
        #print "The thinning factor must be higher than 0."
        sys.exit() 

if  ((samples-burnIn)/thinning) < 1:
        #print "No MCMC samples will be retained. Increse the number of retained samples or decrese the thinning factor."
        sys.exit() 

if posteriorPredictors == True:
        if (((samples-burnIn)/thinning)*numberOfChains) < posteriorPredictorsSamples:
        #print "The number of posterior predictive samples cannot be higher than the number of retained MCMC samples."
                sys.exit()

abspath = os.path.abspath(dataFile)
dname = os.path.dirname(abspath)
os.chdir(dname)
data=load_csv(dataFile)

#Check if rt data is in msec vs. sec
rts = data["rt"]
rts = rts[rts!=-999]
#print(min(rts))
#print(max(rts))
if ((min(rts)<80) | (max(rts)>2500)):
        print('The maximum and/or minimum RT in your dataset is unlikley. Are you sure that your data is supplied in milliseconds?')

models =[]
local_models = []

def run_model(i):
    ss = stopsignal.StopSignal(data)
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    ss.find_starting_values()
    ss.sample(samples,burn=burnIn,thin=thinning,tune_throughout=False, db='pickle', dbname='remote_traces' + str(i+1) + '.db')
    sys.stdout = save_stdout
    return ss

if __name__ == "__main__":
        if actual_cores < numCores:
                if actual_cores==1:
                        print('Your system doesn\'t have ' + str(numCores) + ' cores. BEESTs will use 1 core.')
                        numCores = actual_cores
                else:
                      print('Your system doesn\'t have ' + str(numCores) + ' cores. BEESTs will use the default number of ' + str(actual_cores-1) + ' core(s).')
                      numCores = actual_cores-1
        else:
                print('Your system has ' + str(actual_cores) + ' cores. BEESTs will use ' + str(numCores) + ' core(s)')
        
        n_runs = math.ceil((numberOfChains/numCores))
        n_runs = numpy.array(n_runs).astype('int')
        num_remote = numberOfChains-n_runs
        num_remote = numpy.array(num_remote).astype('int')
        n_pool = numCores-1
        #print n_runs
        #print num_remote
        #print n_pool
        
        if num_remote>0:
                remote_model = []
                pool = multiprocessing.Pool(processes=n_pool)
                results = pool.map_async(run_model, range(num_remote))

        for i in range(n_runs):
                run_id = i+1
                beg = run_id + (i * n_pool)
                end = beg + n_pool
                if i == (n_runs-1):
                        end = numberOfChains
                ran_chains = range(beg,end+1)
                print('Running chain(s) ' + str(ran_chains))

                ss = stopsignal.StopSignal(data)
                print('\n Computing start values. It may take a few minutes.')
                save_stdout = sys.stdout
                sys.stdout = cStringIO.StringIO()
                ss.find_starting_values()
                sys.stdout = save_stdout
                ss.sample(samples,burn=burnIn,thin=thinning,tune_throughout=False, db='pickle', dbname='local_traces' + str(i+1) + '.db')
                local_models.append(ss)
                if i == (n_runs-1):
                        print('Waiting for the other chains to finish...')

        if num_remote>0:
                models = results.get()

        for i in range(n_runs):
                models.append(local_models[i])

        #print(len(models))
        print "Finished sampling!"

        if numberOfChains > 1:
                Rhat = gelman_rubin(models)
                print('\n Gelman-Rubin Rhat diagnostic:')
                for key in Rhat.keys():
                        print((key, Rhat[key]))

        #TOEGEVOEGD
        name_dataFile = dataFile.replace(".csv","")

        for i in range(numberOfChains):
                save_csv(get_traces(models[i]), name_dataFile + '_parameters'+str(i+1)+'.csv', sep = ';')

        print "Posterior samples are saved to file."    

        if deviance == True:
                for i in range(numberOfChains):
                        dev = models[i].mc.db.trace('deviance')()
                        numpy.savetxt(name_dataFile + '_deviance'+str(i+1)+'.csv', dev, delimiter=";") 
                print "Deviance values are saved to file"

##    print samples
##    print burnIn
##    print numberOfChains
##    print thinning
##    print estimatesForAll
##    print summaryStatistics
##    print posteriorDistributions
##    print mcmcChains
##    print deviance
##    print posteriorPredictors
##    print posteriorPredictorsSamples
##    print numCores
##    print dataFile

