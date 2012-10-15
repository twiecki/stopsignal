###Tests go loglikelihood for multiple dataset with varying mu, sigma, and tau parameters
import os
import stop_likelihoods
import numpy
import csv
os.chdir('C:/Dropbox/My Documents/StopSignalPython/test')

reader1=csv.reader(open("Test_SRRT_Rep.csv","rb"),delimiter=',')
x=list(reader1)
reader2=csv.reader(open("Test_SRRT_SSD_Rep.csv","rb"),delimiter=',')
y=list(reader2)

data=numpy.array(x).astype('double')
SSD=numpy.array(y).astype('int')

for i in range(0,data.shape[1]):
    rt = data[7:data.shape[0],i]
    ssd = SSD[:,i]
    LL_py = stop_likelihoods.SRRT(value=rt, issd=ssd,imu_go=data[0,i], isigma_go=data[1,i], itau_go=data[2,i],
                                  imu_stop=data[3,i], isigma_stop=data[4,i], itau_stop=data[5,i])
    LL_R = data[6,i]
    print('******Rep*******', i)
    print('Tau_Go>Sigma_Go*0.05:', data[2,i]> data[1,i]*0.05,'Tau_Stop>Sigma_Stop*0.05:', data[5,i]> data[4,i]*0.05)
    print(round(LL_py,10) == round(LL_R,10))
    print('Python:', LL_py)
    print('R:',LL_R )
