#####Tests go loglikelihood for one dataset with a given mu, sigma, and tau parameters
##import os
##import stop_likelihoods
##import numpy
##import csv
##os.chdir('C:/Dropbox/My Documents/StopSignalPython/test')
##
##reader=csv.reader(open("Test_Go.csv","rb"),delimiter=',')
##x=list(reader)
##data=numpy.array(x).astype('double')
##rt = data[4:data.shape[0],0]
##Mu = data[0,0]
##Sigma = data[1,0]
##Tau= data[2,0]
##
##print(Mu)
##print(Sigma)
##print(Tau)
##
##print('Python:', stop_likelihoods.Go(value=rt, imu_go=Mu, isigma_go=Sigma, itau_go=Tau))
##print('R:', data[3,0])

###Tests go loglikelihood for multiple dataset with varying mu, sigma, and tau parameters
import os
import stop_likelihoods
import numpy
import csv
os.chdir('C:/Dropbox/My Documents/StopSignalPython/test')

reader=csv.reader(open("Test_Go_Rep.csv","rb"),delimiter=',')
x=list(reader)
data=numpy.array(x).astype('double')

for i in range(0,data.shape[1]):
    rt = data[4:data.shape[0],i]
    LL_py = stop_likelihoods.Go(value=rt, imu_go=data[0,i], isigma_go=data[1,i], itau_go=data[2,i])
    LL_R = data[3,i]
    print('******Rep*******', i)
    print('Tau>Sigma*0.05:', data[2,i]> data[1,i]*0.05)
    print(round(LL_py,10) == round(LL_R,10))
    print('Python:', LL_py)
    print('R:',LL_R )
