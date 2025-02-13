import pandas as pd

from pyTVDN import TVDNDetect
from pathlib import Path
import rpy2.robjects as robj
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import seaborn as sns


#load data for subject 1
#data=pd.read_csv("E:\petra\TVDNsub1-main\ef.csv")
#np.save("efdata.npy",data)
fMRI = np.load("efdata.npy")

#Construct the TVDN Detection Class
kappa = 1.591
lamb = 1e-6
r = 6
Lmin = 10
fct = 2
nsim = 50
fMRIdet = TVDNDetect(Ymat=fMRI, saveDir=None, dataType="fMRI", fName="fMRIsample_", r = 16, kappa=1.3956, fct=2, lamb=1e-6, downRate=4, MaxM=10, Lmin=10, freq=0.5)

#Run the Main Function
fMRIdet()

#Check the current results
print(fMRIdet)

#Tuning the kappa parameters: (0.5,10)->1.6515; (1.5,1.7)->1.591,
kappas = np.linspace(1, 1.7, 100)
#kappas = [1.45, 1.55, 1.65, 1.75, 1.85, 1.95]
fMRIdet.TuningKappa(kappas)
print("The optimal kappas are:", fMRIdet.optKappa)
print("The optimal number of change point under the range of kappa we speicified is:", fMRIdet.optKappaOptNumChg)
print("The optimal number of change point is:", fMRIdet.optNumChg)
#fMRIdet.UpdateEcpts(numChg=2)
fMRIdet.UpdateEcpts()
print(fMRIdet)
dir(fMRIdet)
print(fMRIdet.ecpts)

#Plot the detection results
fMRIdet.PlotEcpts(GT=[30, 110, 150], saveFigPath="fMRIecpt.jpg")
#Plot the reconstructed curve
fMRIdet.PlotRecCurve(bestK=3, saveFigPath=None)
fMRIdet.PlotRecCurve(idxs=[13, 7, 89], saveFigPath=None)
fMRIdet.PlotRecCurve(idxs=[13, 7, 89], saveFigPath=None, is_smoothCurve=True)
#Plot the eigenvalue curve
#fMRIdet.PlotEigenCurve()

#Extract the Eigvals and EigVectors
fMRIdet.GetFeatures()
Ur = fMRIdet.curEigVecs
lamMs = fMRIdet.curEigVals

plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.title("Real part")
sns.heatmap(np.array(lamMs).real.T)
plt.show()

plt.subplot(122)
plt.title("Imaginary part")
sns.heatmap(np.array(lamMs).imag.T)
plt.show()


#Change the number of change point
fMRIdet.UpdateEcpts(3) # Reconstruct the Xmat and estiamte the eigen values, again
fMRIdet.GetFeatures()
lamMs = fMRIdet.curEigVals
plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.title("Real part")
sns.heatmap(np.array(lamMs).real.T)
plt.show()
plt.subplot(122)
plt.title("Imaginary part")
sns.heatmap(np.array(lamMs).imag.T)
plt.show()

#save ecpts value
ecptsValue=np.array(fMRIdet.ecpts)
np.save("fMRIdet.cpts",ecptsValue)
ecp=np.load("E:\petra\TVDNsub1-main/fMRIdet.cpts.npy")
print(ecp)

#save r value
rValue=np.array(fMRIdet.paras)
np.save("fMRIdet.paras['r']",rValue)
mat=np.load("E:\petra\TVDNsub1-main/fMRIdet.paras['r'].npy",allow_pickle=True).item()
print(mat)
print(mat['r'])

# fMRIdet.curEigVals,
EigValue=np.array(fMRIdet.curEigVals)
np.save("fMRIdet.curEigVals",EigValue)
CEV=np.load("E:\petra\TVDNsub1-main/fMRIdet.curEigVals.npy")
print(CEV)

# fMRIdet.curEigVectors
EigVec=np.array(fMRIdet.curEigVecs)
np.save("fMRIdet.curEigVectors",EigVec)
CEVe=np.load("E:\petra\TVDNsub1-main/fMRIdet.curEigVectors.npy")
print(CEVe)
