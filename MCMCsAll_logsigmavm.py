
# coding: utf-8

# In[ ]:


# %load MCMCsAll_logsigmavm_r1alpha.py


# In[ ]:


# %load MCMCsAll_logsigmavm_r1alpha.py


# In[ ]:


# %load MCMCsAll_logsigmavm.py
import os,sys
import numpy as np
import pandas as pd #for loading csv Excel files
import string #for loading files with apostrophe in file name
from scipy.interpolate import interp1d, interp2d
from scipy import optimize as opt #for numerical root finding
from scipy.optimize import fsolve #for numerical root finding of a set of equations
from scipy.integrate import odeint #to solve differntial equations
import scipy.integrate as integrate #for (numerical) integrating
from scipy import special as sp #for gamma function, modified bessel function of first kind
import mpmath as mp #for incomplete beta function
import random #to generate integer random numbers
import itertools #to merge lists
import time #for printing out timing
import emcee #for running MCMCs
import argparse
from multiprocessing import Pool

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Group data or simulation data",type=str,required=True,choices=['groupsdata','simsdata'])
parser.add_argument("--parameter-space", help="Parameters to scan over, either [M200,c,sigmavm] or [rho0,sigma0,sigmavm]",type=str,nargs="+",required=True,choices=['M200csigmavm','rho0sigma0sigmavm'])
parser.add_argument("--groups", help="Groups included in run",nargs="+",type=int,required=True,choices=range(16))
parser.add_argument("--profiles", help="Profiles included in run",nargs="+",type=str,required=True,choices=['NFW','Bl','Gn'])
parser.add_argument("--coregrowingcollapse", help="Select core growing or core collapse solution",type=str,required=True,choices=['CoreGrowing','CoreCollapse'])
parser.add_argument("--burn-ins", help="set number of burn-in runs. Default is 5",default=5,type=int)
parser.add_argument("--nwalkers", help="set number of walkers. Default is 224.",default=224,type=int)
parser.add_argument("--burn-in-samples", help="set number of samples for burn-in runs. Default is 50.",default=50,type=int)
parser.add_argument("--full-run-samples", help="set number of samples for full runs. Default is 500.",default=500,type=int)

args=parser.parse_args()

data=args.data #args.data
parameterspaceList=args.parameter_space #args.parameter_space
galnumvals=args.groups
DMprofileList=args.profiles
CoreGrowingCollapse=args.coregrowingcollapse
burnin_val=args.burn_ins
nwalker_val=args.nwalkers
burnin_samples=args.burn_in_samples
full_samples=args.full_run_samples

output_dir="./MCMC_results/"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

print("Running Galaxy Numbers: "+str(galnumvals)+", DM Profiles: "+str(DMprofileList))
print("Data is "+str(data)+", and parameter space is "+str(parameterspaceList))

# # Constants

H0=70.*10.**(-3.) #km/s/kpc;
h=H0/(100.*10.**(-3.)) #for H0 in km/s/kpc
#print(h)

Omegam=0.3
OmegaL=0.7
G=4302./10.**9. #4.302*10^-6 kpc MSun^-1 (km/s)^2 

#tage = (3.154*10.**17.)/2.
tage = 5.*10**9.*365.*24.*60.*60 #Assuming tage= 5 Gyr = 5*10^9*365*24*60*60 s
MSun_in_g = 1.98855*10.**30.*10.**3. #Msun in g
km_in_kpc = 1./(3.0857*10.**16.) #kpc=3.0857*10^16 km
cm_in_kpc = km_in_kpc/(10.**3.*10**2.)


# # Load data

# ## Observation data

if data == 'groupsdata':
    #_____Group observation data DelUps015_____
    ObservationData=pd.read_csv('ObservationData/Group observation data DelUps015.csv')
    [names,arsec_in_kpc,circularization,seeing_arcsec,slitwidth_arcsec,zvals,log10M200obsvals,log10M200errorvals,YSPSvals,kappabarobsvals,kappabarobserrorvals,thetaEinstein,log10Sigmacr,kappabaryons]=[ObservationData.values[:,j] for j in range(0,14)]
    thetaEinstein=thetaEinstein*arsec_in_kpc
elif data == 'simsdata':
    #_____Simulation data_____
    ObservationData=pd.read_csv('ObservationData/Sim observation data.csv')
    [names,arsec_in_kpc,circularization,seeing_arcsec,slitwidth_arcsec,zvals,log10M200obsvals,log10M200errorvals,YSPSvals,kappabarobsvals,kappabarobserrorvals,thetaEinstein_kpc,log10Sigmacr,kappabaryons,logrho0stars,a_stellar]=[ObservationData.values[:,j] for j in range(0,16)]
    #Convert to float numbers:
    [arsec_in_kpc,circularization,seeing_arcsec,slitwidth_arcsec,zvals,log10M200obsvals,log10M200errorvals,YSPSvals,kappabarobsvals,kappabarobserrorvals,thetaEinstein_kpc,log10Sigmacr,kappabaryons,logrho0stars,a_stellar]=[arsec_in_kpc.astype(float),circularization.astype(float),seeing_arcsec.astype(float),slitwidth_arcsec.astype(float),zvals.astype(float),log10M200obsvals.astype(float),log10M200errorvals.astype(float),YSPSvals.astype(float),kappabarobsvals.astype(float),kappabarobserrorvals.astype(float),thetaEinstein_kpc.astype(float),log10Sigmacr.astype(float),kappabaryons.astype(float),logrho0stars.astype(float),a_stellar.astype(float)]
    thetaEinstein=thetaEinstein_kpc

log10YSPSvals=np.array([np.log10(YSPSvals[i]) for i in range(0,len(names))])

REvals = thetaEinstein
Sigmacrvals = 10.**log10Sigmacr

seeing=seeing_arcsec*arsec_in_kpc
sigmaPSFvals=seeing/2.355
slitwidthvals=slitwidth_arcsec*arsec_in_kpc


# ## Stellar data (bins+obs. LOS velocity dispersion)

if data == 'groupsdata':
    #_____Group observation data DelUps015_____
    Loadstellardata=pd.read_csv('ObservationData/Group stellar kinematics data.csv')
    stellardata=np.array([[Loadstellardata.values[j] 
                       for j in list(itertools.chain.from_iterable(np.argwhere(Loadstellardata.values[:,0]==names[i])))]
                      for i in range(0,len(names))])
    #_____Truncate the bins to avoid double-computing the same bins_____
    #[bin min, bin max]
    binsminmax=[abs(np.array([[stellardata[i][j][2],stellardata[i][j][3]] for j in range(0,len(stellardata[i]))])*arsec_in_kpc[i]*circularization[i]) for i in range(0,len(stellardata))]
    #Sort absolute values of bins from min to max + add bin [0,...]
    binaround0=[list(itertools.chain.from_iterable(np.argwhere(binsminmax[i][:,0]==binsminmax[i][:,1])))[0] for i in range(0,len(stellardata))]
    fullbins=[np.array(list(itertools.chain.from_iterable(
        [[[min(binsminmax[i][k]),max(binsminmax[i][k])] for k in range(0,binaround0[i])],
         [[0.,min(binsminmax[i][binaround0[i]])]],
         [[min(binsminmax[i][k]),max(binsminmax[i][k])] for k in range(binaround0[i]+1,len(binsminmax[i]))]
        ]))) for i in range(0,len(stellardata))]
    binvals=[np.unique(fullbins[i],axis=0) for i in range(0,len(stellardata))] #Removes duplicates and sorts data
    binpositionvals=[list(itertools.chain.from_iterable([np.argwhere(binvals[i][:,0]==fullbins[i][:,0][k])[0] 
        for k in range(0,len(fullbins[i][:,0]))])) #Position of bins in list of full bins
     for i in range(0,len(stellardata))]
elif data == 'simsdata':
    #_____Simulation data_____
    Loadstellardata=pd.read_csv('ObservationData/Sim stellar kinematics data.csv')
    stellardata=np.array([[Loadstellardata.values[j] 
                       for j in list(itertools.chain.from_iterable(np.argwhere(Loadstellardata.values[:,0]==names[i])))]
                      for i in range(0,len(names))])
    #_____Bins already sorted_____
    #[bin min, bin max]
    binsminmax=[abs(np.array([[stellardata[i][j][2],stellardata[i][j][3]] for j in range(0,len(stellardata[i]))])*circularization[i]) for i in range(0,len(stellardata))]
    binvals=binsminmax
    binpositionvals=[list(itertools.chain.from_iterable([np.argwhere(binvals[i][:,0]==binvals[i][:,0][k])[0] 
        for k in range(0,len(binvals[i][:,0]))])) #Position of bins in list of full bins
     for i in range(0,len(stellardata))]

sigmaLOSobsvals=np.array([[stellardata[i][j][4] for j in range(0,len(stellardata[i]))] for i in range(0,len(stellardata))])
sigmaLOSerrorvals=np.array([[stellardata[i][j][5] for j in range(0,len(stellardata[i]))] for i in range(0,len(stellardata))])


# ## Baryon profiles

# ### Sersic profiles with and without M/L gradient

###Need to load Sersic profiles for groups aso for simulation data to generate r values
groupnames=pd.read_csv('ObservationData/Group observation data DelUps015.csv').values[:,0]
#print(groupnames)

#With M/L gradient:
rhoSersicGradTab=[np.loadtxt('ObservationData/SersicDensityProfile_DelUps015_'+groupnames[i]+'.dat') for i in range(0,len(groupnames))]
rhoSersicGradInt=[interp1d(rhoSersicGradTab[i][:,0],rhoSersicGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(groupnames))]
MSersicGradTab=[np.loadtxt('ObservationData/SersicEnclMass_DelUps015_'+groupnames[i]+'.dat') for i in range(0,len(groupnames))]
MSersicGradInt=[interp1d(MSersicGradTab[i][:,0],MSersicGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(groupnames))]

#Without M/L gradient (needed to compute Sigmastars_sigmaLOS2):
rhoSersicNoGradTab=[np.loadtxt('ObservationData/SersicDensityProfile_DelUps0_'+groupnames[i]+'.dat') for i in range(0,len(groupnames))]
rhoSersicNoGradInt=[interp1d(rhoSersicNoGradTab[i][:,0],rhoSersicNoGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(groupnames))]
MSersicNoGradTab=[np.loadtxt('ObservationData/SersicEnclMass_DelUps0_'+groupnames[i]+'.dat') for i in range(0,len(groupnames))]
MSersicNoGradInt=[interp1d(MSersicNoGradTab[i][:,0],MSersicNoGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(groupnames))]

rvals=np.array([MSersicGradTab[0][:,0][i] for i in range(1,len(MSersicGradTab[0][:,0])-1)]) #Range of r such that FindRoot[...] works
[rmin,rmax]=[rvals[0],rvals[-1]]  
#print(rvals)
print([rmin,rmax])

Rvals = np.array([rvals[l] for l in range(0,len(rvals))]) #range for the ACSIDM mass and density profiles
[Rmin,Rmax]=[Rvals[0],Rvals[-1]]  
print([Rmin,Rmax])


# ### Stellar density profiles for simulation data

simsnames=pd.read_csv('ObservationData/Sim observation data.csv').values[:,0]
logrho0stars=pd.read_csv('ObservationData/Sim observation data.csv').values[:,14]
rho0stars0=np.array([10.**logrho0stars[i] for i in range(0,len(simsnames))]) #in Msun/kpc^3
a_stellar=pd.read_csv('ObservationData/Sim observation data.csv').values[:,15]

def rhostars(galnum,r):
    rho0stars=rho0stars0[galnum]
    astars=a_stellar[galnum]
    rhostars=rho0stars/(1+r**2./astars**2.)**(3./2.)
    return rhostars

def Mstars(galnum,r):
    rho0stars=rho0stars0[galnum]
    astars=a_stellar[galnum]
    x=r/astars
    Mstars=4.*np.pi*rho0stars*astars**3.*(np.log(x+np.sqrt(1+x**2.))-x/np.sqrt(1+x**2.))
    #Mstarstest=4.*np.pi*integrate.quad(rhostars,0,10.)[0] #Ok.
    return Mstars

simsnames=pd.read_csv('ObservationData/Sim observation data.csv').values[:,0]
rhostarsTab=np.array([[[r,rhostars(i,r)] for r in rvals] for i in range(0,len(simsnames))])
rhostarsInt=[interp1d(rhostarsTab[i][:,0],rhostarsTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(simsnames))]

MstarsTab=np.array([[[r,Mstars(i,r)] for r in rvals] for i in range(0,len(simsnames))])
MstarsInt=[interp1d(MstarsTab[i][:,0],MstarsTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(simsnames))]

if data == 'groupsdata':
    #Sersic profiles with  M/L gradient_
    MbInt=MSersicGradInt
elif data == 'simsdata':
    MbInt=MstarsInt


# # ACNFW profile

# ## Functions

def rhocrit(z):
    H2 = H0**2.*(Omegam*(1.+z)**3.+OmegaL)
    rhocrit = (3.*H2)/(8.*np.pi*G)
    return rhocrit 
def rhos(z,c):
    gc=(np.log(1.+c)-c/(1.+c))**(-1.)
    return (200./3.)*c**3.*gc*rhocrit(z)
def r200(z,M200,c):
    return M200**(1./3.)*((4.*np.pi)/3.*200.*rhocrit(z))**(-1./3.)
def rs(z,M200,c):
    r200 = M200**(1./3.)*((4.*np.pi)/3.*200.*rhocrit(z))**(-1./3.)
    return r200/c

#[A,w]=[1.,0.] for NFW; [1.,1.] for Blumenthal (Bl), [0.8,0.85] for Gnedin (Gn);
def ACNFWProfile(DMprofile,galnum,Y,M200,c,r): #DMprofile = NFW, Bl, Gn
    #_____Group properties_____
    z = zvals[galnum]
    def Mb(R):
        #return Y*MSersicGradInt[galnum](R) 
        return Y*MbInt[galnum](R)
    #_____NFW profile_____
    def MNFW(R):
        return 4.*np.pi*rhos(z,c)*rs(z,M200,c)**3.*(np.log((R+rs(z,M200,c))/rs(z,M200,c))-R/(R+rs(z,M200,c)))
    #_____AC NFW profile_____

    Mb_r200=Mb(r200(z,M200,c))
    
    if DMprofile == 'NFW': [A,w]=[1.,0.]
    if DMprofile == 'Bl': [A,w]=[1.,1.]
    if DMprofile == 'Gn': [A,w]=[0.8,0.85]
    def rival(R):
        def f(ri):
            def bar(R):
                return r200(z,M200,c)*A*(R/r200(z,M200,c))**w
            return (ri/R*(1+Mb_r200/M200)-1.)*MNFW(bar(ri))-Mb(bar(R))
        #_____Find root_____
        ristart=r
        ri=opt.fsolve(f,ristart,xtol=10.**(-3.))[0] #default: xtol=1.49012e-08
        return ri
    def MACdm(R):
        return MNFW(rival(R)) 
    epsilon = 10.**(-3.)
    M1=MACdm(r-epsilon*r);
    M2=MACdm(r+epsilon*r);
    Mavg=(M1+M2)/2. 
    Mdmprime=(M2-M1)/(2.*epsilon*r)
    rhoACdm=Mdmprime/(4.*np.pi*r**2.)
    return [rhoACdm,Mavg]      

# # ACSIDM profile in terms of [M200,c,sigmavm]

def IsothermalProfileInt(galnum,Y,rho0,sigma0):
    #_____Group properties_____
    def Mb(R):
        #return Y*MSersicGradInt[galnum](R)
        return Y*MbInt[galnum](R)
    #_____Isothermal profile_____
    def rhoMiso(rhoM,R):
        [rhodm, Mdm] = rhoM
        drhodm_dR = -((G*rhodm)/(sigma0**2.*R**2.))*(Mdm+Mb(R))
        dMdm_dR = 4.*np.pi*R**2.*rhodm
        return [drhodm_dR,dMdm_dR]
    #_____Initial conditions_____
    rhoMini = [rho0,(4.*np.pi)/3.*rho0*rmin**3.]
    #_____Solve ODE_____
    sol = odeint(rhoMiso,rhoMini,rvals) #sol=[rhoiso(R),Miso(R) for R in rvals]
    rhoiso=interp1d(rvals,sol[:,0], kind='cubic',fill_value='extrapolate')
    Miso=interp1d(rvals,sol[:,1], kind='cubic',fill_value='extrapolate')
    return [rhoiso,Miso]


# In[ ]:


def rho0sigma0ini(alphastart,r1,ratioNFW,rhoACNFW_val):
    #_____Isothermal profile in terms of h(r)_____
    def h(alpha):
        def hr(y,r):
            [h,dhdr]=y
            d2hdr2=-(2.*dhdr)/r-alpha*np.exp(h)
            return [dhdr,d2hdr2]
        y0 = [0.0, -(alpha*rmin)/3.] #initial conditions at r=rmin
        y = odeint(hr,y0,rvals,full_output=0) #full_output=True
        hInt=interp1d(rvals,y[:,0], kind='cubic',fill_value='extrapolate')
        dhdr_exphr_Int=interp1d(rvals,y[:,1]/np.exp(y[:,0]), kind='cubic',fill_value='extrapolate')
        return [hInt,dhdr_exphr_Int]

    #_____Find root_____
    def diffratio(alpha):
        [hInt,dhdr_exphr_Int]=h(alpha)
        ratio=-(1./(alpha*r1))*dhdr_exphr_Int(r1)
        return ratio-ratioNFW
    #alphastart=0.01
    alpha=opt.fsolve(diffratio,alphastart)[0] #default: xtol=1.49012e-08 #xtol=10.**(-3.)
    #_____[rho0,sigma0]_____
    rho1=rhoACNFW_val
    rho0=rho1*np.exp(-h(alpha)[0](r1))
    sigma0=np.sqrt((4.*np.pi*G*rho0)/alpha)
    return [alpha,rho0,sigma0]


# In[ ]:


def ACSIDMProfileM200csigmavm(galnum,DMprofile,CoreGrowingCollapse,Y,M200,c,sigmavm,rho0,sigma0,alphastart,success):
    #_____Group properties_____
    z = zvals[galnum]
    def Mb(R):
        #return Y*MSersicGradInt[galnum](R)
        return Y*MbInt[galnum](R)
    
    #_____AC NFW profile_____
    rhoACNFWvals=np.array([ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[0] for R in Rvals])
    rhoACNFWInt=interp1d(Rvals,rhoACNFWvals,kind='cubic',fill_value='extrapolate')
    
    MACNFWvals=np.array([ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[1] for R in Rvals])
    MACNFWInt=interp1d(Rvals,MACNFWvals,kind='cubic',fill_value='extrapolate')

    # dummy default values
    rhoACSIDMInt=1.
    MtotInt=1.
    alpha=1.
    DeltaU=0.

    r200_val=r200(z,M200,c)

    #_____Find r1_____
    #Inverse function
    R_rhoACNFWInt=interp1d(rhoACNFWvals,Rvals,kind='cubic',fill_value='extrapolate')
    rhoACNFWr1=1./(MSun_in_g*sigmavm*km_in_kpc*cm_in_kpc**2.*tage)
    r1=R_rhoACNFWInt(rhoACNFWr1)
    
    #_____ACSIDM profile__________

    #if ratio > 0.5: #MatchingSuccess=ratio>0.5&&ratio<Log[$MaxNumber]
    if success:
        rhoACNFW_val=rhoACNFWInt(r1)
        MACNFW_val=MACNFWInt(r1)
        ratio = MACNFW_val/(4.*np.pi*rhoACNFW_val*r1**3.)
        def Findrho0sigma0(rho0sigma0):
            [rho0,sigma0] = rho0sigma0
            sol=IsothermalProfileInt(galnum,Y,rho0,sigma0)
            [rho1,M1]=[sol[0](r1),sol[1](r1)]
            equation1 = M1/(4.*np.pi*r1**3.) - ratio*rho1
            equation2 = rhoACNFW_val - rho1
            return [equation1,equation2]
        [alpha,rho0start,sigma0start]=rho0sigma0ini(alphastart,r1,ratio,rhoACNFW_val)
        #if alpha<10.**(-5.) or alpha>10.**5.:
        #    success=False
        [rho0,sigma0] = abs(opt.fsolve(Findrho0sigma0,[rho0start,sigma0start],xtol=10.**(-5.))) #default: xtol=1.49012e-08
        sol=IsothermalProfileInt(galnum,Y,rho0,sigma0)
        [rho1,M1]=[sol[0](r1),sol[1](r1)]
        #_____Matching success tests_____
        MatchingSuccessTestrho = abs((rho1-rhoACNFW_val)/rho1)
        MatchingSuccessTestM = abs((M1-MACNFW_val)/M1)
        #if MatchingSuccessTest <=0.01: Matching success test passed.
        if MatchingSuccessTestrho > 0.01 or MatchingSuccessTestM > 0.01:
            success=False
    else:
        success=False
    if success:
        try:
            def rhoACSIDM(R):
                if R > r1:
                    #rhodm = rhoACNFW(M200,c,R)
                    rhodm = rhoACNFWInt(R)
                else: 
                    #rhodm = rhoiso(rho0,sigma0,R)
                    rhodm = sol[0](R)
                return rhodm 
            def MACSIDM(R):
                if R > r1:
                    #Mdm = MACNFW(M200,c,R)
                    Mdm = MACNFWInt(R)
                else: 
                    #Mdm = Miso(rho0,sigma0,R)
                    Mdm = sol[1](R)
                return Mdm
            rhoACSIDMInt=interp1d(Rvals,[rhoACSIDM(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
            MtotInt=interp1d(Rvals,[MACSIDM(R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
            MACSIDMInt=interp1d(Rvals,[MACSIDM(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
            #_____CoreGrowingCollapse_____
            #Potential Phi(R)
            def Phiintegrand(R,func):
                return func(R)/R*2.
            PhiACNFW=integrate.quad(Phiintegrand,Rmax,Rmin,args=(MACNFWInt),limit=200)[0]
            PhiACSIDM=integrate.quad(Phiintegrand,Rmax,Rmin,args=(MACSIDMInt),limit=200)[0]
            #Potential energy U(R)=m1*Phi(R) for R=Rmin (maximum change), and for test mass m1=1.
            DeltaU=PhiACSIDM-PhiACNFW 
            #Core growing: DeltaU > 0
            if CoreGrowingCollapse == 'CoreGrowing' and DeltaU<0:
                success=False
            # Core collapse: DeltaU < 0
            if CoreGrowingCollapse == 'CoreCollapse' and DeltaU>0:
                success=False
        except:
            success=False

    vel=(4./np.sqrt(np.pi))*sigma0
    xsctn=sigmavm/vel
    return [MtotInt,rhoACSIDMInt,np.log10(M200),np.log10(c),np.log10(rho0),np.log10(sigma0),r1,sigmavm,xsctn,r200_val,vel,alphastart,alpha,DeltaU,success]

# In[ ]:


# # ACSIDM profile in terms of [rho0,sigma0,sigmavm]

#_____x1=r1/rs_____
x1vals=np.logspace(-3., 10., 1500, endpoint=True)
def Ratio(x1):
    return (1./x1**2.)*(1.+x1)**2*(np.log(1.+x1)-x1/(x1+1.))
Ratiovals = np.array(list(itertools.chain.from_iterable([[0.5],[Ratio(x1) for x1 in x1vals]])))
x1vals = np.array(list(itertools.chain.from_iterable([[0.],[x1 for x1 in x1vals]])))
x1Int=interp1d(Ratiovals,x1vals, kind='cubic', fill_value='extrapolate')
def X1(ratio):
    return x1Int(ratio)

def ACSIDMProfilerho0sigma0sigmavm(galnum,DMprofile,Y,rho0,sigma0,sigmavm,M200,c,xsctn,r1,success):
    #_____Group properties_____
    z = zvals[galnum]
    def Mb(R):
        #return Y*MSersicGradInt[galnum](R)
        return Y*MbInt[galnum](R)
    
    #_____AC NFW profile_____    
    def rhoACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[0]
    def MACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[1]

    # dummy default values
    rhoACSIDMInt=1.
    MtotInt=1.
    r200_val=1.
    
    #_____Isothermal profile_____
    sol=IsothermalProfileInt(galnum,Y,rho0,sigma0)
    def rhoiso(R):
        #return interp1d(rvals,sol[:,0], kind='cubic', fill_value='extrapolate')(R)
        return sol[0](R)
    def Miso(R):
        #return interp1d(rvals,sol[:,1], kind='cubic', fill_value='extrapolate')(R)
        return sol[1](R)
    
    def rhoACSIDM(M200,c,r1,R):
        if R > r1:
            rhodm = rhoACNFW(M200,c,R)
        else: 
            rhodm = rhoiso(R)
        return rhodm 
    def MACSIDM(M200,c,r1,R):
        if R > r1:
            Mdm = MACNFW(M200,c,R)
        else: 
            Mdm = Miso(R)
        return Mdm 
    
    xsctnmin=1./(MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*cm_in_kpc**2.*tage*rhoiso(Rmin))
    xsctnmax=1./(MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*cm_in_kpc**2.*tage*rhoiso(Rmax))
    if xsctnmin < xsctn < xsctnmax:
        #_____ACSIDM profile_____
        try:
            def Findr1(R):
                return rhoiso(R)-1./(MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*xsctn*cm_in_kpc**2.*tage)
            r1 = opt.brentq(Findr1,Rmin,Rmax,maxiter=150)
            #_____Matching success tests_____    
            ratio = Miso(r1)/(4.*np.pi*rhoiso(r1)*r1**3.)
            #####Solutions for [M200,c] do strongly depend on [M200start,cstart].
            if ratio <= np.log(1.+10.**10.)-1.:
                x1 = X1(ratio)
            else: 
                x1 = np.exp(1.+ratio)-1.
            rhosstart = rhoiso(r1)*x1*(1.+x1)**2.
            rsstart = r1/x1
            def Findcstart(c):
                return rhosstart - (200.*rhocrit(z)*c**3.)/(3.*(np.log(1.+c)-c/(1.+c)))
            cstart = opt.brentq(Findcstart,10.**-10.,10.**10.,maxiter=150)
            M200start = (4.*np.pi)/3.*200.*rhocrit(z)*cstart**3.*rsstart**3
            #If[M200start<10^10,M200start=10^13]; Start value for M200 in LogMIntrhoInttab is 10^10.
            def FindM200c(M200c):
                [M200,c] = M200c
                equation1 = MACNFW(M200,c,r1)/(4.*np.pi*r1**3.) - ratio*rhoACNFW(M200,c,r1)
                equation2 = rhoACNFW(M200,c,r1) - rho1
                return [equation1,equation2]
            rho1=rhoiso(r1)
            M1=Miso(r1)
            [M200,c] = opt.fsolve(FindM200c,[M200start,cstart])
            r200_val=r200(z,M200,c)
            if r1 >= r200_val:
                success=False
            #_____Matching success tests_____
            if success:
                MatchingSuccessTestrho = abs((rho1-rhoACNFW(M200,c,r1))/rho1)
                MatchingSuccessTestM = abs((M1-MACNFW(M200,c,r1))/M1)
                #if MatchingSuccessTest <=0.01: Matching success test passed.
                if MatchingSuccessTestrho >0.01 or MatchingSuccessTestM >0.01:
                    success=False
        except:
            success=False
    else:
        success=False
    if success:
        try:
            rhoACSIDMInt=interp1d(Rvals,[rhoACSIDM(M200,c,r1,R) for R in Rvals], kind='cubic', fill_value='extrapolate')
            MtotInt=interp1d(Rvals,[MACSIDM(M200,c,r1,R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        except:
            success=False
    return [MtotInt,rhoACSIDMInt,np.log10(M200),np.log10(c),np.log10(rho0),np.log10(sigma0),r1,sigmavm,r200_val,success]   


# # Mean convergence kappabar

def kappabartot(galnum,Y,rhoACSIDMInt):
    #_____Group properties_____ 
    RE=REvals[galnum] #Einstein radius in kpc
    Sigmacr=Sigmacrvals[galnum] #critical density
    kappab=kappabaryons[galnum] #convergence due to baryons only with M/L=1
    #_____Mean total convergence kappabar_____
    def integrand1(R):
        return 4.*np.pi*R**2.*rhoACSIDMInt(R)
    def integrand2(R):
        return integrand1(R)*(1.-np.sqrt(R**2.-RE**2.)/R) 
    Mdmencl = integrate.quad(integrand1,Rmin,RE,limit=200)[0]+integrate.quad(integrand2,RE,Rmax,limit=200)[0]
    kappadm=Mdmencl/(np.pi*RE**2.*Sigmacr)
    kappatot=kappadm+Y*kappab
    return kappatot


# # Binned LOS velocity dispersion

# ## Projected stellar surface density: Sigmastars(R)

R=np.array(list(itertools.chain.from_iterable([10.**np.arange(-1.,3.5,0.02),[3162.28]])))
r=np.array([0.5*(R[i]+R[i+1]) for i in range(0,len(R)-1)])
Delta=np.array([R[i+1]-R[i] for i in range(0,len(R)-1)])

def Sigmastars(galnum):
    if data == 'groupsdata':
        #rhostars=rhoSersicGradInt[galnum]
        rhostars=rhoSersicNoGradInt[galnum]
        integrand=np.array([rhostars(r[j]) for j in range(0,len(r))])
        def integral(i):
            integral1=2.*integrand[i]*np.sqrt(2.*Delta[i]*r[i])
            integral2=np.sum([(2.*Delta[j]*r[j]*integrand[j])/np.sqrt(((R[j+1]+R[j])/2.)**2.-R[i]**2.) for j in range(i+1,len(r))])
            return integral1+integral2
        Int=np.array([integral(i) for i in range(0,len(r))])
        Inttab=np.array([[R[i],Int[i]] for i in range(0,len(R)-1)])
    elif data == 'simsdata':
        def Sigmastarssims(R):
            rho0stars=rho0stars0[galnum]
            astars=a_stellar[galnum]
            Sigmastars=(rho0stars*astars)/(1+R**2./astars**2.)
            return Sigmastars
        Inttab=np.array([[R[i],Sigmastarssims(R[i])] for i in range(0,len(R)-1)])
    return Inttab


# ## Surface density times velocity dispersion squared: Sigmastars_sigmaLOS2(R)

#For (regularized) incomplete beta function Beta(z,a,b): 
#see sympy package mpmath Beta(z,a,b)=betainc(a, b, x1=0, x2=z, regularized=False) (http://docs.sympy.org/0.7.1/modules/mpmath/functions/gamma.html)
#scipy Beta(z,a,b)=sp.beta(a, b)*sp.betainc(a, b, z) not sufficient cannot compute incomplete beta function for z=0,1 
#(https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betainc.html#scipy.special.betainc)

#For gamma function Gamma(z): sympy package mpmath cannot evluate mp.gamma(z=0)=np.inf, 
#use scipy package sp.gamma(z=0)=np.inf

def fbeta(beta,w):
    #_____Regularized incomplete beta function Beta(z,a,b)_____
    def Beta(z,a,b):
        return mp.betainc(a, b, x1=0, x2=z, regularized=False)
    #_____Function f(beta,w) for nonzero beta, w=R**2./r**2._____
    f=(w**((1./2.)-beta)/2.)*(beta*Beta(w,beta+(1./2.),1./2.)-Beta(w,beta-(1./2.),1./2.)
            +(sp.gamma(beta-(1./2.))*np.sqrt(np.pi)*(3.-2.*beta))/(2.*sp.gamma(beta)))  #sp.gamma(z)=Gamma function \Gamma(z) 
    return f

alphavals=np.arange(-25.,25.,.1)
wvals=1./(np.exp(alphavals)+1)

def f_beta(beta):
    #w=Rpr**2./r**2.
    def fw(w):
        if beta==0.:
            f=np.sqrt(1-w)
        else: 
            f=fbeta(beta,w)
        return f
    f_betavals=list(np.float_([fw(w) for w in wvals]))
    f_betaInt=interp1d(wvals,f_betavals, kind='cubic')
    return f_betaInt

def Sigmastars_sigmaLOS2(galnum,MtotInt,beta):
    if data == 'groupsdata':
        #Sersic profiles WITHOUT  M/L gradient
        rhostars=rhoSersicNoGradInt[galnum]
    elif data == 'simsdata':
        rhostars=rhostarsInt[galnum]
    #Very small difference to Sean's code due to interpolation function for f(beta,w)
    fInt=f_beta(beta)
    def F(r,R):
        w=R**2./r**2.
        return fInt(w)*r
    integrand=np.array([G*(MtotInt(rj)/rj**2.)*rhostars(rj) for rj in r])
    def integral(i):
        integral1=np.sum([2.*Delta[j]*F(r[j],R[i])*integrand[j] for j in range(i,len(r))])
        return integral1
    Int=np.array(list(np.float_([integral(i) for i in range(0,len(r))])))
    Inttab=np.array([[R[i],Int[i]] for i in range(0,len(R)-1)])
    return Inttab

# ## Seeing and binning

def SeeingBinned(galnum,func):
    bins=binvals[galnum]
    width=slitwidthvals[galnum]
    sigmaPSF=sigmaPSFvals[galnum]
    funcInt=interp1d(func[:,0],func[:,1], kind='cubic')
    def integrand(r1,r2):
        return (r2/sigmaPSF**2.)*sp.iv(0,(r1*r2)/sigmaPSF**2.)*np.exp(-((r1**2.+r2**2.)/(2.*sigmaPSF**2.)))*funcInt(r2)
    def Seeing_bin(binnum):
        #_____Seeing:_____
        #_____Make R values for bin centers_____
        [Nx,Ny,NR]=[20.,20.,5.]
        [Rbinmin,Rbinmax]=bins[binnum]
        Deltax=(Rbinmax-Rbinmin)/Nx
        Deltay=width/Ny
        Abin=width*(Rbinmax-Rbinmin)
        xvals=np.arange(Rbinmin+Deltax/2.,Rbinmax,Deltax)
        yvals=np.arange(-width/2.+Deltay/2.,width/2.,Deltay)
        Rvals=np.array([[np.sqrt(x**2.+y**2.) for y in yvals] for x in xvals])
        #_____Calculate seeing-corrected function between Rmin and Rmax_____
        Rvalsflattened=np.array(list(itertools.chain.from_iterable(Rvals)))
        Rmin=min(Rvalsflattened)
        Rmax=max(Rvalsflattened)
        DeltaR=(Rmax-Rmin)/NR
        def integralRpr(r):
            #_____rvals_____
            rad=np.arange(r-3.*sigmaPSF,r+3.*sigmaPSF, min(0.1*sigmaPSF,0.1*r))
            rvalspos_min=np.where(max(func[:,0][0],0.1*sigmaPSF)<rad)
            rvals_min=np.array(list(itertools.chain.from_iterable([rad[pos] for pos in rvalspos_min])))
            rvalspos_max=np.where(rad<func[:,0][-1])
            rvals_max=np.array(list(itertools.chain.from_iterable([rad[pos] for pos in rvalspos_max])))
            rvals=np.intersect1d(rvals_min,rvals_max) #Find common values of two arrays
            #_____Deltarvals_____
            Deltarvals=np.array(list(itertools.chain.from_iterable(
                [[(rvals[1]-rvals[0])/2.],
                [(rvals[i+1]-rvals[i-1])/2. for i in range(1,len(rvals)-1)],
                [(rvals[-1]-rvals[-2])/2.]
                ])))
            #_____integral_____
            integral=np.sum([integrand(r, rvals[ip])*Deltarvals[ip] for ip in range(0,len(rvals))])
            return integral
        funcSeeingtab=np.array([[r,integralRpr(r)] for r in np.arange(Rmin,Rmax+DeltaR,DeltaR)])
        funcSeeing=interp1d(funcSeeingtab[:,0],funcSeeingtab[:,1], kind='cubic', fill_value='extrapolate')
        #fill_value='extrapolate': interpolate such that Rmin and Rmax are included in interpolation range
        #_____Binning:_____
        funcSeeing_bin=np.sum([[Deltax*Deltay*funcSeeing(Rvals[i][j]) for j in range(0,int(Ny))] for i in range(0,int(Nx))])/Abin
        return funcSeeing_bin
    
    output=np.array([Seeing_bin(binnum) for binnum in range(0,len(bins))])
    return output

Sigmastars_seeing_binned_vals=np.array([SeeingBinned(galnumber,Sigmastars(galnumber)) for galnumber in range(0,len(names))])
print(Sigmastars_seeing_binned_vals)


def sigmaLOS_seeing_binned(galnum,MtotInt,beta):
    binpositions=binpositionvals[galnum]
    Sigmastars_seeing_binned=Sigmastars_seeing_binned_vals[galnum]
    Sigmastars_sigmaLOS2_seeing_binned=SeeingBinned(galnum,Sigmastars_sigmaLOS2(galnum,MtotInt,beta))
    sigmaLOS=np.sqrt(Sigmastars_sigmaLOS2_seeing_binned/Sigmastars_seeing_binned)
    sigmaLOSfull=np.array([sigmaLOS[i] for i in binpositions])
    return sigmaLOSfull

###Very small differences to Sean's code come from differences when computing Mtot and interpolation function for f(beta,w).
###Difference to Sean's code less than 0.1% (1-2% is accuracy of Sean's code).


# In[ ]:


# # Set up MCMCs

# ## Logarithm of probability: lnprob

#Scan over log10 of free paramters: paramters always > 0, priors not necessary
#params has to be the first entry in lnprob to make emcee work
#def lnprob(params,galnum,DMprofile,CoreGrowingCollapse):
def lnprobM200csigmavm(params,galnum,DMprofile,CoreGrowingCollapse):
    [log10Y,beta,log10M200,log10c,log10sigmavm,log10alphastart]=params
    success=True
    #_____Free parameters_____ 
    Y=10.**log10Y
    M200=10.**log10M200
    c=10.**log10c
    sigmavm=10.**log10sigmavm
    alphastart=10.**log10alphastart
    #_____Group properties_____ 
    z = zvals[galnum]
    kappabarobs=kappabarobsvals[galnum]
    kappabarobserror=kappabarobserrorvals[galnum]
    log10M200obs=log10M200obsvals[galnum]
    log10M200error=log10M200errorvals[galnum]
    sigmaLOSobs=sigmaLOSobsvals[galnum]
    sigmaLOSerror=sigmaLOSerrorvals[galnum]
    log10YSPS=log10YSPSvals[galnum]

    # dummy definitions
    rho0=1
    sigma0=1.
    vel=1.
    sigmaLOS=[1 for x in range(len(sigmaLOSobs))]
    log10rho0=1.
    log10sigma0=np.log10(sigma0)
    xsctn=1.
    prob=0.
    ChiSqDisp=np.inf
    ChiSqLensing=np.inf
    ChiSqMass=np.inf
    kappabar=1.
    r1=1.
    r200_val=1.
    alpha=1.
    DeltaU=0.
    ChiSqTot=np.inf
    lnprob=-np.inf
    
    #_____Priors/physical values for free parameters_____     
    if abs(log10Y-log10YSPS) > 0.4:
        success=False
    elif abs(beta) > 0.3:
        success=False
    elif sigmavm < 0.: 
        success=False
    elif alphastart<10.**(-5.) or alphastart>10.**5.:
        success=False
       
    #_____ACSIDM profile_____
    if success:
        [MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1,sigmavm,xsctn,r200_val,vel,alphastart,alpha,DeltaU,success]=ACSIDMProfileM200csigmavm(galnum,DMprofile,CoreGrowingCollapse,Y,M200,c,sigmavm,rho0,sigma0,alphastart,success)
        if xsctn < 0. or xsctn > 10.:
            success=False
    if success:
        #_____\chi^2 lensing_____ 
        kappabar = kappabartot(galnum,Y,rhoACSIDMInt)
        if np.isnan(kappabar):
            success=False
    if success:
        ChiSqLensing = (kappabar-kappabarobs)**2./kappabarobserror**2.
        #_____\chi^2 mass_____ 
        #Mass-concentration relation with redshift dependence
        a=0.520+(0.905-0.520)*np.exp(-0.617*z**1.21)
        b=-0.101+0.026*z
        log10cNFW=a+b*(log10M200-np.log10(10.**12.*h**(-1.)))  #M200 in units of MSun
        ###Modified:
        log10cerror=0.15 #error estimate
        ChiSqMass=(log10M200-log10M200obs)**2./log10M200error**2.+(log10c-log10cNFW)**2/log10cerror**2
        #_____\chi^2 velocity dispersion_____
        sigmaLOS=sigmaLOS_seeing_binned(galnum,MtotInt,beta)
        ChiSqDisp=np.sum([(sigmaLOS[i]-sigmaLOSobs[i])**2./sigmaLOSerror[i]**2. for i in range(0,len(sigmaLOS))])
        #_____\chi^2 M/L ratio close to Salpeter_____ 
        log10YSPSerror=0.1 #error estimate
        ChiSqML=(log10Y-log10YSPS)**2./log10YSPSerror**2.
        #_____Total \chi^2_____
        ChiSqTot=ChiSqLensing+ChiSqMass+ChiSqDisp #+ChiSqML
        prob=np.exp(-ChiSqTot/2.)*sigmavm
        lnprob=np.log(prob)

    blobs=[[log10rho0,log10sigma0,np.log10(xsctn),log10Y,beta,prob,ChiSqDisp,ChiSqLensing,ChiSqMass],[sigmaLOS[i] for i in range(len(sigmaLOSobs))],[kappabar,r1,r200_val,log10M200,log10c,vel,sigmavm,log10sigmavm,log10alphastart,alphastart,alpha,DeltaU,xsctn,ChiSqTot,success]]
    flatblobs=np.array(list(itertools.chain.from_iterable(blobs)))
    return lnprob, flatblobs


def lnprobrho0sigma0sigmavm(params,galnum,DMprofile):
    [log10Y,beta,log10rho0,log10sigma0,log10sigmavm]=params
    success=True
    #_____Free parameters_____ 
    Y=10.**log10Y
    rho0=10.**log10rho0
    sigma0=10.**log10sigma0
    sigmavm=10.**log10sigmavm
    #_____Group properties_____ 
    z = zvals[galnum]
    kappabarobs=kappabarobsvals[galnum]
    kappabarobserror=kappabarobserrorvals[galnum]
    log10M200obs=log10M200obsvals[galnum]
    log10M200error=log10M200errorvals[galnum]
    sigmaLOSobs=sigmaLOSobsvals[galnum]
    sigmaLOSerror=sigmaLOSerrorvals[galnum]
    log10YSPS=log10YSPSvals[galnum]

    # determine cross-section
    vel=(4./np.sqrt(np.pi))*sigma0
    xsctn=sigmavm/vel

    # dummy values
    log10M200=0.
    log10c=0.
    M200=1.
    c=1.
    sigmaLOS=[1 for x in range(len(sigmaLOSobs))]
    prob=0.
    ChiSqDisp=np.inf
    ChiSqLensing=np.inf
    ChiSqMass=np.inf
    kappabar=1.
    r1=1.
    r200_val=1.
    ChiSqTot=np.inf
    lnprob=-np.inf

    
    #_____Priors/physical values for free parameters_____     
    if abs(log10Y-log10YSPS) > 0.4:
        success=False
    elif abs(beta) > 0.3:
        success=False
    elif sigmavm < 0.: 
        success=False
    elif xsctn < 0. or xsctn > 10.:
        success=False
    if success: 
        #_____ACSIDM profile_____
        [MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1,sigmavm,r200_val,success]=ACSIDMProfilerho0sigma0sigmavm(galnum,DMprofile,Y,rho0,sigma0,sigmavm,M200,c,xsctn,r1,success)
        #_____\chi^2 lensing_____
        if success:
            kappabar = kappabartot(galnum,Y,rhoACSIDMInt)
            if np.isnan(kappabar):
                success=False
        if success:
            ChiSqLensing = (kappabar-kappabarobs)**2./kappabarobserror**2.
            #_____\chi^2 mass_____ 
            #Mass-concentration relation with redshift dependence
            a=0.520+(0.905-0.520)*np.exp(-0.617*z**1.21)
            b=-0.101+0.026*z
            log10cNFW=a+b*(log10M200-np.log10(10.**12.*h**(-1.)))  #M200 in units of MSun
            ###Modified:
            log10cerror=0.15 #error estimate
            ChiSqMass=(log10M200-log10M200obs)**2./log10M200error**2.+(log10c-log10cNFW)**2/log10cerror**2
            #_____\chi^2 velocity dispersion_____
            sigmaLOS=sigmaLOS_seeing_binned(galnum,MtotInt,beta)
            ChiSqDisp=np.sum([(sigmaLOS[i]-sigmaLOSobs[i])**2./sigmaLOSerror[i]**2. for i in range(0,len(sigmaLOS))])
            #_____\chi^2 M/L ratio close to Salpeter_____ 
            log10YSPSerror=0.1 #error estimate
            ChiSqML=(log10Y-log10YSPS)**2./log10YSPSerror**2.
            #_____Total \chi^2_____ 
            ChiSqTot=ChiSqLensing+ChiSqMass+ChiSqDisp #+ChiSqML
            prob=np.exp(-ChiSqTot/2.)*sigmavm
            lnprob=np.log(prob)

    blobs=[[log10rho0,log10sigma0,np.log10(xsctn),log10Y,beta,prob,ChiSqDisp,ChiSqLensing,ChiSqMass],[sigmaLOS[i] for i in range(len(sigmaLOSobs))],[kappabar,r1,r200_val,log10M200,log10c,vel,sigmavm,log10sigmavm,xsctn,ChiSqTot,success]]
    flatblobs=np.array(list(itertools.chain.from_iterable(blobs)))
    return lnprob, flatblobs

# ## Find random initial points for walkers

# ## Find random initial points for walkers

def walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile,CoreGrowingCollapse,parspace):
    print('Determine starting points for walkers:')
    #_____Starting points for walkers_____
    params = initialparams
    initpoint=True
    chainsini=[]
    while len(chainsini) < nwalkers:
        if initpoint:
            initpoint=False
            if parspace=='M200csigmavm':
                lnprobval,blobs = lnprobM200csigmavm(params,galnum,DMprofile,CoreGrowingCollapse)
            elif parspace=='rho0sigma0sigmavm':
                lnprobval,blobs = lnprobrho0sigma0sigmavm(params,galnum,DMprofile,CoreGrowingCollapse)
            if lnprobval==-np.inf or np.isnan(lnprobval):
                continue
            else:
                chainsini.append(blobs)
                #print(blobs)
        else:
            params=initialparams+paramserrors*np.random.randn(len(initialparams))
            #print(params)
            if parspace=='M200csigmavm':
                lnprobval,blobs = lnprobM200csigmavm(params,galnum,DMprofile,CoreGrowingCollapse)
            elif parspace=='rho0sigma0sigmavm':
                lnprobval,blobs = lnprobrho0sigma0sigmavm(params,galnum,DMprofile,CoreGrowingCollapse)
            if lnprobval==-np.inf or np.isnan(lnprobval):
                continue
            else:
                chainsini.append(blobs)
                print(blobs)
    return np.array(chainsini)

# ## Module to run MCMCs

def MCMCM200csigmavm(galnum,DMprofile,CoreGrowingCollapse,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun,parspace): #nwalkers should be > 100.
    #_____MCMC properties_____
    header=[["log10rho0","log10sigma0","np.log10(xsctn)","log10Y","beta","prob","ChiSqDisp","ChiSqLensing","ChiSqMass"],["sigmaLOS"+str(i) for i in range(len(sigmaLOSobsvals[galnum]))] ,["kappabar","r1","r200","log10M200","log10c","vel","sigmavm","log10sigmavm","log10alphastart","alphastart","alpha","DeltaU","xsctn","ChiSqTot","success"]]
    #header=[["log10rho0","log10sigma0","np.log10(xsctn)","log10Y","beta","prob","ChiSqDisp","ChiSqLensing","ChiSqMass"],["sigmaLOS"+str(i) for i in range(len(sigmaLOSobsvals[galnum]))] ,["kappabar","r1","r200","log10M200","log10c","vel","sigmavm","xsctn","ChiSqTot","success"]]
    #header=['log10Y', 'beta', 'log10M200', 'log10c','log10rho0','log10sigma0','r1','sigmavm','xsctn', 'Chi2']
    header=np.array(list(itertools.chain.from_iterable(header)))
    chainlength = nwalkers*nsamples_finalrun
    filename=str(names[galnum])+'_SersicDelUps015_'+str(DMprofile)+'_'+str(CoreGrowingCollapse)+'_'+str(data)+'_chainlength'+str(chainlength)+'_nwalkers'+str(nwalkers)+'_nsamples'+str(nsamples_finalrun)
    #_____Set up the MCMC_____
    #Number of free parameters:
    ndim=6 #=len(params)
    initialparams=[0.3222192947339193, 0.0, 14.141639613890037, 0.95018288373578297, np.log10(2094.2717341292714),0.]
    paramserrors=np.array([0.1,0.3,2.,0.5,np.log10(50.),5.])
    #_____Starting points for walkers_____
    #Starting point p0 for each of the walkers:
    #Number of walkers must be the same for burn in and finalrun because of the set up of the initial conditions:
    #p0=np.array([initialparams+paramserrors*np.random.randn(ndim) for i in range(nwalkers)])
    starting_point_start = time.time()
    chainsini=walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile,CoreGrowingCollapse,parspace)
    print(chainsini)
    formatter=['%.18e' for x in range(len(header)-1)]
    formatter.append('%d')
    np.savetxt(output_dir+'Startingpointswalkers_'+str(parspace)+'_log10sigmavm_'+filename+'.dat',chainsini, header=str(header),fmt=formatter)
    print('Startingpointswalkers_'+str(parspace)+'_log10sigmavm_'+filename+'.dat exported.')
    p0=np.array([[chainsini[i][j] for j in [3,4,-12,-11,-8,-7]] for i in range(nwalkers)])
    print(p0)
    starting_point_end = time.time()
    spoint_time=starting_point_end-starting_point_start
    print('Time to calculate starting points for galnum '+str(galnum)+' and profile '+str(DMprofile)+'='+str(spoint_time))    
    #_____MCMCs_____
    parallelstart = time.time()
    for i in range(nburnins+1):
        if i==0:
            paramsini=p0    
        if i in range(nburnins): #Burn-in runs 
            nsamples=nsamples_burnin #chainlength = nwalkers*nsamples
            runname='run'+str(i+1)
        else: #MCMC production run
            nsamples=nsamples_finalrun #chainlength = nwalkers*nsamples
            runname='finalrun'
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobM200csigmavm, args=(galnum,DMprofile,CoreGrowingCollapse), pool=pool)
            paramsini_tmp,lnprobval,state,blobstmp=sampler.run_mcmc(paramsini,nsamples)
            paramsini=paramsini_tmp
            blobs=sampler.get_blobs(flat=True)
            accfrac=np.mean(sampler.acceptance_fraction)
            sampler.reset()
        print('MCMC '+runname+' completed. Acceptance fraction: '+str(accfrac)) 
        #Save results on computer:
        np.savetxt(output_dir+'Chains_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat',blobs, header=str(header),fmt=formatter)
        print('Chains_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat exported.')
    parallel_end = time.time()
    tparallel=parallel_end - parallelstart
    print('tparallel for galnum '+str(galnum)+' and profile '+str(DMprofile)+'='+str(tparallel))
    #_____Best fit_____
    Chi2vals=blobs[:,-2]
    imin=list(itertools.chain.from_iterable(np.argwhere(Chi2vals==min(Chi2vals))))
    bestfit=blobs[imin][0]
    print('Best fit: '+str(header)+'='+str(bestfit))
    #Save results:
    np.savetxt(output_dir+'Bestfitparams_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat',bestfit)
    print('Bestfitparams_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat exported.')
    return 'Done.'

def MCMCrho0sigma0sigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun,parspace): #nwalkers should be > 100.
    #_____MCMC properties_____
    header=[["log10rho0","log10sigma0","np.log10(xsctn)","log10Y","beta","prob","ChiSqDisp","ChiSqLensing","ChiSqMass"],["sigmaLOS"+str(i) for i in range(len(sigmaLOSobsvals[galnum]))] ,["kappabar","r1","r200","log10M200","log10c","vel","sigmavm","log10sigmavm","xsctn","ChiSqTot","success"]]
    #header=['log10Y', 'beta', 'log10M200', 'log10c', 'log10rho0', 'log10sigma0','r1', 'sigmavm','xsctn', 'Chi2']
    #header=[["log10rho0","log10sigma0","np.log10(xsctn)","log10Y","beta","prob","ChiSqDisp","ChiSqLensing","ChiSqMass"],["sigmaLOS"+str(i) for i in range(len(sigmaLOSobsvals[galnum]))] ,["kappabar","r1","r200","log10M200","log10c","vel","sigmavm","xsctn","ChiSqTot","success"]]
    header=np.array(list(itertools.chain.from_iterable(header)))
    chainlength = nwalkers*nsamples_finalrun   
    filename=str(names[galnum])+'_SersicDelUps015_'+str(DMprofile)+'_'+str(data)+'_chainlength'+str(chainlength)+'_nwalkers'+str(nwalkers)+'_nsamples'+str(nsamples_finalrun)
    #_____Set up the MCMC_____
    #Number of free parameters:
    ndim=5 #=len(params)
    initialparams=[np.log10(2.1),0.,np.log10(10**7.5),np.log10(580.),np.log10((4./np.sqrt(np.pi))*580.*1.5)]
    paramserrors=np.array([0.1,0.3,1.2,0.25,np.log10(50.)])
    #_____Starting points for walkers_____
    #Starting point p0 for each of the walkers:
    #Number of walkers must be the same for burn in and finalrun because of the set up of the initial conditions:
    #p0=np.array([initialparams+paramserrors*np.random.randn(ndim) for i in range(nwalkers)])
    starting_point_start = time.time()
    chainsini=walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile,parspace)
    print(chainsini)
    formatter=['%.18e' for x in range(len(header)-1)]
    formatter.append('%d')
    np.savetxt(output_dir+'Startingpointswalkers_'+str(parspace)+'_log10sigmavm_'+filename+'.dat',chainsini,header=str(header),fmt=formatter)
    print('Startingpointswalkers_'+str(parspace)+'_log10sigmavm_'+filename+'.dat exported.')
    p0=np.array([[chainsini[i][j] for j in [3,4,0,1,-4]] for i in range(nwalkers)])
    starting_point_end = time.time()
    spoint_time=starting_point_end-starting_point_start
    print('Time to calculate starting points for galnum '+str(galnum)+' and profile '+str(DMprofile)+'='+str(spoint_time))    
    #_____MCMCs_____
    parallelstart = time.time()
    for i in range(nburnins+1):
        if i==0:
            paramsini=p0    
        if i in range(nburnins): #Burn-in runs 
            nsamples=nsamples_burnin #chainlength = nwalkers*nsamples
            runname='run'+str(i+1)
        else: #MCMC production run
            nsamples=nsamples_finalrun #chainlength = nwalkers*nsamples
            runname='finalrun'
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobrho0sigma0sigmavm, args=(galnum,DMprofile), pool=pool)
            paramsini_tmp, lnprobvals, state, blobstmp = sampler.run_mcmc(paramsini,nsamples) 
            paramsini=paramsini_tmp
            blobs=sampler.get_blobs(flat=True)
            accfrac=np.mean(sampler.acceptance_fraction)
            sampler.reset()
        print('MCMC '+runname+' completed. Acceptance fraction: '+str(accfrac)) 
        #Save results on computer:
        np.savetxt(output_dir+'Chains_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat',blobs, header=str(header),fmt=formatter)
        print('Chains_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat exported.')
    parallel_end = time.time()
    tparallel=parallel_end - parallelstart
    print('tparallel for galnum '+str(galnum)+' and profile '+str(DMprofile)+'='+str(tparallel))
    #_____Best fit_____
    Chi2vals=blobs[:,-2]
    imin=list(itertools.chain.from_iterable(np.argwhere(Chi2vals==min(Chi2vals))))
    bestfit=blobs[imin][0]
    print('Best fit: '+str(header)+'='+str(bestfit))
    #Save results:
    np.savetxt(output_dir+'Bestfitparams_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat',bestfit)
    print('Bestfitparams_'+parameter_space+'_log10sigmavm_'+filename+'_'+runname+'.dat exported.')
    return 'Done.'

for parameter_space in parameterspaceList:
    for galnum in galnumvals:
        for DMprofile in DMprofileList:
            #_____Number of burn-in runs_____
            nburnins=burnin_val
            #_____Number of walkers (must be the same for burn in and finalrun because of the set up of the initial conditions)
            nwalkers=nwalker_val
            #_____Chain lengths_____
            nsamples_burnin = burnin_samples
            nsamples_finalrun = full_samples
            #_____Print properties of MCMCs_____
            burninlength = nwalkers*nsamples_burnin
            chainlength = nwalkers*nsamples_finalrun
            print(str(names[galnum])+': ')
            print('Burn in: nburnins='+str(nburnins)+', burninlength='+str(burninlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_burnin]))
            print('Final run: chainlength='+str(chainlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_finalrun]))
            #_____Run MCMCs____
            if parameter_space=='M200csigmavm':
                print(MCMCM200csigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun,parameter_space))
            elif parameter_space=='rho0sigma0sigmavm':
                print(MCMCrho0sigma0sigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun,parameter_space))
end = time.time()
ttot=end - start
print('ttot='+str(ttot))
print("Successfully finished running.")



