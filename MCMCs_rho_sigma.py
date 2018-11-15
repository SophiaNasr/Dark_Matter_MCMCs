# coding: utf-8
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

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--groups", help="Groups included in run",nargs="+",type=int,required=True,choices=range(10))
parser.add_argument("--profiles", help="Profiles included in run",nargs="+",type=str,required=True,choices=['NFW','Bl','Gn'])
parser.add_argument("--threads", help="set the number of threads. Default is 1 thread.",default=1,type=int)
parser.add_argument("--burn-ins", help="set number of burn-in runs. Default is 5",default=5,type=int)
parser.add_argument("--nwalkers", help="set number of walkers. Default is 224.",default=224,type=int)
parser.add_argument("--burn-in-samples", help="set number of samples for burn-in runs. Default is 50.",default=50,type=int)
parser.add_argument("--full-run-samples", help="set number of samples for full runs. Default is 500.",default=500,type=int)

args=parser.parse_args()

galnumvals=args.groups
DMprofileList=args.profiles
threads=args.threads
burnin_val=args.burn_ins
nwalker_val=args.nwalkers
burnin_samples=args.burn_in_samples
full_samples=args.full_run_samples

parameter_space="rhosigma0sigmavm"

output_dir="./MCMC_results/"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

print("Running Galaxy Numbers: "+str(galnumvals)+", DM Profiles: "+str(DMprofileList)+" with "+str(threads)+" threads.")

# ## Group observation data DelUps015

GroupObservationDataDelUps015 = pd.read_csv('ObservationData/Group observation data DelUps015.csv')
[names,arsec_in_kpc,circularization,seeing_arcsec,slitwidth_arcsec,zvals,log10M200obsvals,log10M200errorvals,YSPSvals,kappabarobsvals,kappabarobserrorvals,thetaEinstein,log10Sigmacr,kappabaryons] = [GroupObservationDataDelUps015.values[:,j] for j in range(0,14)]
print(names)
print(zvals)

log10YSPSvals=np.array([np.log10(YSPSvals[i]) for i in range(0,len(names))])
print(YSPSvals)
print(log10YSPSvals)

REvals = thetaEinstein*arsec_in_kpc
Sigmacrvals = 10.**log10Sigmacr
print(REvals)

seeing=seeing_arcsec*arsec_in_kpc
sigmaPSFvals=seeing/2.355
slitwidthvals=slitwidth_arcsec*arsec_in_kpc

# ## Stellar data (bins+obs. LOS velocity dispersion)

Loadstellardata=pd.read_csv('ObservationData/Group stellar kinematics data.csv')
stellardata=np.array([[Loadstellardata.values[j] 
                       for j in list(itertools.chain.from_iterable(np.argwhere(Loadstellardata.values[:,0]==names[i])))]
                      for i in range(0,len(names))])
sigmaLOSobsvals=np.array([[stellardata[i][j][4] for j in range(0,len(stellardata[i]))] for i in range(0,len(stellardata))])
sigmaLOSerrorvals=np.array([[stellardata[i][j][5] for j in range(0,len(stellardata[i]))] for i in range(0,len(stellardata))])
print(sigmaLOSobsvals)

#Truncate the bins to avoid double-computing the same bins
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
print(binvals[0])



# ### Load Sersic profiles with and without M/L gradient
rhoSersicNoGradTab=[np.loadtxt('ObservationData/SersicDensityProfile_DelUps0_'+names[i]+'.dat') for i in range(0,len(names))]
rhoSersicNoGradInt=[interp1d(rhoSersicNoGradTab[i][:,0],rhoSersicNoGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

MSersicNoGradTab=[np.loadtxt('ObservationData/SersicEnclMass_DelUps0_'+names[i]+'.dat') for i in range(0,len(names))]
MSersicNoGradInt=[interp1d(MSersicNoGradTab[i][:,0],MSersicNoGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

print(MSersicNoGradInt[0](10.))
print(rhoSersicNoGradInt[0](10.))


rhoSersicGradTab=[np.loadtxt('ObservationData/SersicDensityProfile_DelUps015_'+names[i]+'.dat') for i in range(0,len(names))]
rhoSersicGradInt=[interp1d(rhoSersicGradTab[i][:,0],rhoSersicGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

MSersicGradTab=[np.loadtxt('ObservationData/SersicEnclMass_DelUps015_'+names[i]+'.dat') for i in range(0,len(names))]
MSersicGradInt=[interp1d(MSersicGradTab[i][:,0],MSersicGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

print(MSersicGradInt[0](10.))
print(rhoSersicGradInt[0](10.))


# # ACNFW profile (MakeACNFWTableIntMod)

# ### Test parameters
[Ytest,betatest,rho0test,sigma0test,sigmamtest]=[5.,0.,5.*10**8.,300.,0.1]


# ### Constants
H0=70.*10.**(-3.) #km/s/kpc;
h=H0/(100.*10.**(-3.)) #for H0 in km/s/kpc
print(h)

Omegam=0.3
OmegaL=0.7
G=4302./10.**9. #4.302*10^-6 kpc MSun^-1 (km/s)^2 


tage = 5.*10**9.*365.*24.*60.*60 #Assuming tage= 5 Gyr = 5*10^9*365*24*60*60 s
MSun_in_g = 1.98855*10.**30.*10.**3. #Msun in g
km_in_kpc = 1./(3.0857*10.**16.) #kpc=3.0857*10^16 km
cm_in_kpc = km_in_kpc/(10.**3.*10**2.)

#rvals=MSersicGradTab[0][:,0]
rvals=np.array([MSersicGradTab[0][:,0][i] for i in range(1,len(MSersicGradTab[0][:,0])-1)]) #Range of r such that FindRoot[...] works
[rmin,rmax]=[rvals[0],rvals[-1]]  
print(len(rvals))
print([rmin,rmax])

Rvals = np.array([rvals[l] for l in range(0,len(rvals))]) #range for the ACSIDM mass and density profiles
[Rmin,Rmax]=[Rvals[0],Rvals[-1]]  
print([Rmin,Rmax])

# ### Functions

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

print(rhocrit(0.))

def ACNFWProfile(DMprofile,galnum,Y,M200,c,r): #DMprofile = NFW, Bl, Gn
    #_____Group properties_____
    z = zvals[galnum]
    def Mb(R):
        return Y*MSersicGradInt[galnum](R) 
    #_____NFW profile_____
    def MNFW(R):
        return 4.*np.pi*rhos(z,c)*rs(z,M200,c)**3.*(np.log((R+rs(z,M200,c))/rs(z,M200,c))-R/(R+rs(z,M200,c)))
    #_____AC NFW profile_____
    if DMprofile == 'NFW': [A,w]=[1.,0.]
    if DMprofile == 'Bl': [A,w]=[1.,1.]
    if DMprofile == 'Gn': [A,w]=[0.8,0.85]
    def rival(R):
        def f(ri):
            def bar(R):
                return r200(z,M200,c)*A*(R/r200(z,M200,c))**w
            return (ri/R*(1+Mb(r200(z,M200,c))/M200)-1.)*MNFW(bar(ri))-Mb(bar(R))
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
    
#print(ACNFWProfile(DMprofile,galnum,1.,10.**12.,1.,100.,))


# # ACSIDM profile in terms of [rho0,sigma0,xsctn]

def IsothermalProfile(galnum,Y,rho0,sigma0):
    #_____Group properties_____
    def Mb(R):
        return Y*MSersicGradInt[galnum](R)
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
    return sol

#_____x1=r1/rs_____
x1vals=np.logspace(-3., 10., 1500, endpoint=True)
def Ratio(x1):
    return (1./x1**2.)*(1.+x1)**2*(np.log(1.+x1)-x1/(x1+1.))
Ratiovals = np.array(list(itertools.chain.from_iterable([[0.5],[Ratio(x1) for x1 in x1vals]])))
x1vals = np.array(list(itertools.chain.from_iterable([[0.],[x1 for x1 in x1vals]])))
x1Int=interp1d(Ratiovals,x1vals, kind='cubic', fill_value='extrapolate')
def X1(ratio):
    return x1Int(ratio)

print(X1(1.))

def ACSIDMProfile(galnum,DMprofile,Y,rho0,sigma0,xsctn):
    #_____Group properties_____
    z = zvals[galnum]
    def Mb(R):
        return Y*MSersicGradInt[galnum](R)
    
    #_____AC NFW profile_____    
    def rhoACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[0]
    def MACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[1]
    
    #_____Isothermal profile_____
    sol = IsothermalProfile(galnum,Y,rho0,sigma0)
    def rhoiso(R):
        return interp1d(rvals,sol[:,0], kind='cubic', fill_value='extrapolate')(R) 
    def Miso(R):
        return interp1d(rvals,sol[:,1], kind='cubic', fill_value='extrapolate')(R)
    
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
            if ratio > 0.5:
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
                    equation1 = MACNFW(M200,c,r1)/(4.*np.pi*r1**3.*rhoACNFW(M200,c,r1)) - ratio
                    equation2 = rhoACNFW(M200,c,r1) - rhoiso(r1)
                    return [equation1,equation2]
                [M200val,cval] = opt.fsolve(FindM200c,[M200start,cstart])
                [rhosval,rsval] = [rhos(z,cval),rs(z,M200val,cval)]
                MatchingSuccessTest = abs((rhoiso(r1)-rhoACNFW(M200val,cval,r1))/rhoiso(r1))
                #if MatchingSuccessTest <=0.01: Matching success test passed.
                if MatchingSuccessTest <=0.01:
                    if r1 < r200(z,M200val,cval):
                        [r1,M200val,cval]=[r1,M200val,cval]
                    else:
                        [r1,M200val,cval]=[100.,1.,10.**12.]
                else:
                    [r1,M200val,cval]=[100.,1.,10.**12.]
            else:
                [r1,M200val,cval]=[100.,1.,10.**12.] 
        except:
            [r1,M200val,cval]=[100.,1.,10.**12.] #If an error occurs,e.g. in the numerical root finding, output are dummy variables.                           
    else:
        [r1,M200val,cval]=[100.,1.,10.**12.]     
    
    try:
        rhoACSIDMInt=interp1d(Rvals,[rhoACSIDM(M200val,cval,r1,R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        MtotInt=interp1d(Rvals,[MACSIDM(M200val,cval,r1,R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
    except:
        #ACSIDM solution fails, SIDM solution taken instead and dummy variables for [M200,c] split out
        [r1,M200val,cval]=[100.,1.,10.**12.]
        rhoACSIDMInt=interp1d(Rvals,[rhoiso(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        MtotInt=interp1d(Rvals,[Miso(R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        
    sigmavm=sigma0*(4./np.sqrt(np.pi))*xsctn
    return [MtotInt,rhoACSIDMInt,np.log10(M200val),np.log10(cval),np.log10(rho0),np.log10(sigma0),r1,sigmavm,xsctn]

# # ACSIDM profile in terms of [rho0,sigma0,sigmavm]

def ACSIDMProfilesigmavm(galnum,DMprofile,Y,rho0,sigma0,sigmavm):
    xsctn=sigmavm/((4./np.sqrt(np.pi))*sigma0)
    #_____Group properties_____
    z = zvals[galnum]
    def Mb(R):
        return Y*MSersicGradInt[galnum](R)
    
    #_____AC NFW profile_____    
    def rhoACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[0]
    def MACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[1]
    
    #_____Isothermal profile_____
    sol = IsothermalProfile(galnum,Y,rho0,sigma0)
    def rhoiso(R):
        return interp1d(rvals,sol[:,0], kind='cubic', fill_value='extrapolate')(R) 
    def Miso(R):
        return interp1d(rvals,sol[:,1], kind='cubic', fill_value='extrapolate')(R)
    
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
            #if ratio > 0.5: #MatchingSuccess=ratio>0.5&&ratio<Log[$MaxNumber]
            if ratio > 0.5:
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
                    equation2 = rhoACNFW(M200,c,r1) - rhoiso(r1)
                    return [equation1,equation2]
                [M200val,cval] = opt.fsolve(FindM200c,[M200start,cstart])
                [rhosval,rsval] = [rhos(z,cval),rs(z,M200val,cval)]
                #_____Matching success tests_____
                MatchingSuccessTestrho = abs((rhoiso(r1)-rhoACNFW(M200val,cval,r1))/rhoiso(r1))
                MatchingSuccessTestM = abs((Miso(r1)-MACNFW(M200val,cval,r1))/Miso(r1))
                #if MatchingSuccessTest <=0.01: Matching success test passed.
                if MatchingSuccessTestrho <=0.01 and MatchingSuccessTestM <=0.01:
                    if r1 < r200(z,M200val,cval):
                        [r1,M200val,cval]=[r1,M200val,cval]
                    else:
                        [r1,M200val,cval]=[100.,1.,10.**12.]
                else:
                    [r1,M200val,cval]=[100.,1.,10.**12.]
            else:
                [r1,M200val,cval]=[100.,1.,10.**12.] 
        except:
            [r1,M200val,cval]=[100.,1.,10.**12.] #If an error occurs,e.g. in the numerical root finding, output are dummy variables.                           
    else:
        [r1,M200val,cval]=[100.,1.,10.**12.]     
    
    try:
        rhoACSIDMInt=interp1d(Rvals,[rhoACSIDM(M200val,cval,r1,R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        MtotInt=interp1d(Rvals,[MACSIDM(M200val,cval,r1,R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
    except:
        #ACSIDM solution fails, SIDM solution taken instead and dummy variables for [M200,c] split out
        [r1,M200val,cval]=[100.,1.,10.**12.]
        rhoACSIDMInt=interp1d(Rvals,[rhoiso(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        MtotInt=interp1d(Rvals,[Miso(R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        
    return [MtotInt,rhoACSIDMInt,np.log10(M200val),np.log10(cval),np.log10(rho0),np.log10(sigma0),r1,sigmavm,xsctn]   

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
    Mdmencl = integrate.quad(integrand1,Rmin,RE)[0]+integrate.quad(integrand2,RE,Rmax)[0]
    kappadm=Mdmencl/(np.pi*RE**2.*Sigmacr)
    kappatot=kappadm+Y*kappab
    return kappatot


# # Binned LOS velocity dispersion

# ### Projected stellar surface density: Sigmastars(R)

R=np.array(list(itertools.chain.from_iterable([10.**np.arange(-1.,3.5,0.02),[3162.28]])))
r=np.array([0.5*(R[i]+R[i+1]) for i in range(0,len(R)-1)])
Delta=np.array([R[i+1]-R[i] for i in range(0,len(R)-1)])

def Sigmastars(galnum):
    rhostars=rhoSersicNoGradInt[galnum]
    integrand=np.array([rhostars(r[j]) for j in range(0,len(r))])
    def integral(i):
        integral1=2.*integrand[i]*np.sqrt(2.*Delta[i]*r[i])
        integral2=np.sum([(2.*Delta[j]*r[j]*integrand[j])/np.sqrt(((R[j+1]+R[j])/2.)**2.-R[i]**2.) for j in range(i+1,len(r))])
        return integral1+integral2
    Int=np.array([integral(i) for i in range(0,len(r))])
    Inttab=np.array([[R[i],Int[i]] for i in range(0,len(R)-1)])
    return Inttab


# ### Surface density times velocity dispersion squared: Sigmastars_sigmaLOS2(R)

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
print(len(wvals))
print([min(wvals),max(wvals)])

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
    rhostars=rhoSersicNoGradInt[galnum]
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

# ### Seeing and binning

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


# # ACSIDM group fit probability

def ACSIDMGroupFitProbability(galnum,DMprofile,log10Y,beta,log10rho0,log10sigma0,sigmavm):
    #_____Free parameters_____ 
    Y=10.**log10Y
    rho0=10.**log10rho0
    sigma0=10.**log10sigma0
    #_____Group properties_____ 
    z = zvals[galnum]
    kappabarobs=kappabarobsvals[galnum]
    kappabarobserror=kappabarobserrorvals[galnum]
    log10M200obs=log10M200obsvals[galnum]
    log10M200error=log10M200errorvals[galnum]
    sigmaLOSobs=sigmaLOSobsvals[galnum]
    sigmaLOSerror=sigmaLOSerrorvals[galnum]
    log10YSPS=log10YSPSvals[galnum]
    #_____ACSIDM profile_____ 
    [MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1,sigmavm,xsctn]=ACSIDMProfilesigmavm(galnum,DMprofile,Y,rho0,sigma0,sigmavm)
    #_____\chi^2 lensing_____ 
    kappabar = kappabartot(galnum,Y,rhoACSIDMInt)
    ChiSqLensing = (kappabar-kappabarobs)**2./kappabarobserror**2.
    #_____\chi^2 mass_____ 
    #Mass-concentration relation with redshift dependence
    a=0.520+(0.905-0.520)*np.exp(-0.617*z**1.21)
    b=-0.101+0.026*z
    log10cNFW=a+b*(log10M200-np.log10(10.**12.*h**(-1.)))  #M200 in units of MSun
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
    prob=np.exp(-ChiSqTot/2.)
    output=np.array([ChiSqLensing,ChiSqMass,ChiSqDisp,prob])
    return output


###Different values for prob in comparison to Sean's code come from differences when computing Mtot and interpolation function for f(beta,w).

# # Set up MCMCs

# ### Logarithm of probability: lnprob

#Scan over log10 of free paramters: paramters always > 0, priors not necessary
#params has to be the first entry in lnprob to make emcee work
#def lnprob(params,galnum,DMprofile,CoreGrowingCollapse):
def lnprob(params,galnum,DMprofile):
    [log10Y,beta,log10rho0,log10sigma0,sigmavm]=params 
    #_____Free parameters_____ 
    Y=10.**log10Y
    rho0=10.**log10rho0
    sigma0=10.**log10sigma0
    #_____Group properties_____ 
    z = zvals[galnum]
    kappabarobs=kappabarobsvals[galnum]
    kappabarobserror=kappabarobserrorvals[galnum]
    log10M200obs=log10M200obsvals[galnum]
    log10M200error=log10M200errorvals[galnum]
    sigmaLOSobs=sigmaLOSobsvals[galnum]
    sigmaLOSerror=sigmaLOSerrorvals[galnum]
    log10YSPS=log10YSPSvals[galnum]
    #_____Priors/physical values for free parameters_____     
    if abs(log10Y-log10YSPS) > 0.4:
        lnprob = -np.inf #prob=0  
    elif abs(beta) > 0.3:
        lnprob = -np.inf
    elif sigmavm < 0.: 
        lnprob = -np.inf
    else: 
        #_____ACSIDM profile_____       
        [MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1,sigmavm,xsctn]=ACSIDMProfilesigmavm(galnum,DMprofile,Y,rho0,sigma0,sigmavm)
        if log10M200 == 12. and log10c==0.:
            lnprob = -np.inf
        elif xsctn > 10.:
            lnprob = -np.inf
        elif log10c > np.log10(20.): #0 <= c <= 20. equivalemt to log10c > np.log10(20.)
            lnprob = -np.inf
        else:     
            #_____\chi^2 lensing_____ 
            kappabar = kappabartot(galnum,Y,rhoACSIDMInt)
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
            prob=np.exp(-ChiSqTot/2.)
            lnprob=-ChiSqTot/2.
    return lnprob


def walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile):
    print('Determine starting points for walkers:')
    #_____Starting points for walkers_____
    j=0
    jmax=20*nwalkers
    while j < jmax:
        if j==0:
            params = initialparams
            lnprobval = lnprob(params,galnum,DMprofile)
            chainsini = [list(itertools.chain.from_iterable([params,[(-2.)*lnprobval]]))]
            j+= 1
        else:
            params=initialparams+paramserrors*np.random.randn(len(initialparams))
            lnprobval = lnprob(params,galnum,DMprofile)
            if lnprobval==-np.inf:
                chainsini=chainsini
            else:
                output = np.array(list(itertools.chain.from_iterable([params,[(-2.)*lnprobval]])))
                chainsini.append(output)
            
            if len(chainsini)==nwalkers:
                j=jmax+1
            else:
                j+= 1
    
    return np.array(chainsini)


# ### Module to run MCMCs

def MCMCNewsigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun): #nwalkers should be > 100.
    #_____MCMC properties_____
    header=['log10Y', 'beta', 'log10rho0', 'log10sigma0', 'sigmavm', 'Chi2']
    chainlength = nwalkers*nsamples_finalrun   
    filename=str(names[galnum])+'_SersicDelUps015_'+str(DMprofile)+'_chainlength'+str(chainlength)+'_nwalkers'+str(nwalkers)+'_nsamples'+str(nsamples_finalrun)+'_threads'+str(threads)
    #_____Set up the MCMC_____
    #Number of free parameters:
    ndim=5 #=len(params)
    initialparams=[np.log10(2.1),0.,np.log10(10**7.5),np.log10(580.),(4./np.sqrt(np.pi))*580.*1.5]
    paramserrors=np.array([0.1,0.3,1.2,0.25,50.])
    #_____Starting points for walkers_____
    #Starting point p0 for each of the walkers:
    #Number of walkers must be the same for burn in and finalrun because of the set up of the initial conditions:
    #p0=np.array([initialparams+paramserrors*np.random.randn(ndim) for i in range(nwalkers)])
    starting_point_start = time.time()
    chainsini=walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile)
    print(chainsini)
    np.savetxt(output_dir+'Startingpointswalkers_'+parameter_space+'_'+filename+'.dat',chainsini, header=str(header))
    print('Startingpointswalkers_'+parameter_space+'_'+filename+'.dat exported.')
    p0=np.array([[chainsini[i][j] for j in range(0,len(initialparams))] for i in range(0,nwalkers)])
    starting_point_end = time.time()
    spoint_time=starting_point_end-starting_point_start
    print('Time to calculate starting points for galnum '+str(galnum)+' and profile '+str(DMprofile)+'='+str(spoint_time))    
    #_____MCMCs_____
    parallelstart = time.time()
    for i in range(0,nburnins+1):
        if i==0:
            paramsini=p0    
        if i in range(0,nburnins): #Burn-in runs 
            nsamples=nsamples_burnin #chainlength = nwalkers*nsamples
            runname='run'+str(i+1)
        else: #MCMC production run
            nsamples=nsamples_finalrun #chainlength = nwalkers*nsamples
            runname='finalrun'
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(galnum,DMprofile), threads=threads)
        paramsini, lnprobvals, state = sampler.run_mcmc(paramsini,nsamples)
    
        params = sampler.flatchain
        lnprobvals = sampler.flatlnprobability
        accfrac=np.mean(sampler.acceptance_fraction)
        sampler.reset()
        print('MCMC '+runname+' completed. Acceptance fraction: '+str(accfrac)) 
        Chi2vals =-2.*lnprobvals
        chains=np.array([[params[j][0],params[j][1],params[j][2],params[j][3],params[j][4],Chi2vals[j]] 
                         for j in range(0,len(params))])
        #Save results on computer:
        np.savetxt(output_dir+'Chains_'+parameter_space+'_'+filename+'_'+runname+'.dat',chains, header=str(header))
        print('Chains_'+parameter_space+'_'+filename+'_'+runname+'.dat exported.')
    parallel_end = time.time()
    tparallel=parallel_end - parallelstart
    print('tparallel for galnum '+str(galnum)+' and profile '+str(DMprofile)+'='+str(tparallel))
    #_____Best fit_____
    imin=list(itertools.chain.from_iterable(np.argwhere(Chi2vals==min(Chi2vals))))
    Chi2=Chi2vals[imin][0]
    bestfitparams=params[imin][0]
    print('Best fit: Chi2='+str(Chi2)+', [log10Y,beta,log10rho0,log10sigma0,sigmavm]='+str(bestfitparams))
    bestfit=np.array([bestfitparams[0],bestfitparams[1],bestfitparams[2],bestfitparams[3],bestfitparams[4],Chi2])
    #Save results:
    np.savetxt(output_dir+'Bestfitparams_'+parameter_space+'_'+filename+'_'+runname+'.dat',bestfit)
    return 'Done.'  

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
        print(MCMCNewsigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun))
end = time.time()
ttot=end - start
print('ttot='+str(ttot))
print("Successfully finished running.")
