
# coding: utf-8

# # Packages

# In[6]:


import numpy as np
import pandas as pd #for loading csv Excel files
import string #for loading files with apostrophe in file name
from scipy.interpolate import interp1d, interp2d
from scipy import optimize as opt #for numerical root finding
from scipy.optimize import fsolve #for numerical root finding of a set of equations
from scipy.integrate import odeint #to solve differntial equations
import scipy.integrate as integrate #for (numerical) integrating
#from scipy.integrate import quad, dblquad, nquad #for (numerical) integrating with infinite bounds
from scipy import special as sp #for gamma function, modified bessel function of first kind
import mpmath as mp #for incomplete beta function
import random #to generate integer random numbers
import itertools #to merge lists
#import matplotlib.pyplot as pl #for plots
#import matplotlib.mlab as mlab #for plotting Gaussian ditrsibutions
##from random import randint #to generate random numbers
import time #for printing out timing
import emcee #for running MCMCs
#import corner #for corner plots
#import sys #to split out minimum and maximum values of numbers allowd in Python
from sys import argv
##import autograd.numpy as np
##from autograd import grad #for computing numerical derivatives
#
##sp.init_printing()


# # MCMC settings:

# In[7]:


start = time.time()


# In[49]:


#pathchains = 'MCMC_results/'
#print (pathchains)

# galnum=4
# DMprofile='NFW'
# threads=32

galnum=int(argv[1])
DMprofile=argv[2]
threads=int(argv[3])
print([galnum,DMprofile,threads])


# # Group properties

# ## Group observation data DelUps015

# In[4]:


#GroupObservationDataDelUps015 = pd.read_csv(path+'Dropbox/SIDM on Intermediate Scales/Mathematica packages/Group observation data DelUps015.csv')
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
#print(Sigmacrvals)

seeing=seeing_arcsec*arsec_in_kpc
sigmaPSFvals=seeing/2.355
slitwidthvals=slitwidth_arcsec*arsec_in_kpc
#print(sigmaPSFvals)
#print(slitwidth)


# ## Stellar data (bins+obs. LOS velocity dispersion)

# In[5]:


#Loadstellardata=pd.read_csv(path+'Dropbox/SIDM on Intermediate Scales/Mathematica packages/Group stellar kinematics data.csv')
Loadstellardata=pd.read_csv('ObservationData/Group stellar kinematics data.csv')
stellardata=np.array([[Loadstellardata.values[j] 
                       for j in list(itertools.chain.from_iterable(np.argwhere(Loadstellardata.values[:,0]==names[i])))]
                      for i in range(0,len(names))])

#print(stellardata[0][0])
#print(stellardata[1][0])
#print(len(stellardata)) #len(stellardata)=len(names)

sigmaLOSobsvals=np.array([[stellardata[i][j][4] for j in range(0,len(stellardata[i]))] for i in range(0,len(stellardata))])
sigmaLOSerrorvals=np.array([[stellardata[i][j][5] for j in range(0,len(stellardata[i]))] for i in range(0,len(stellardata))])
print(sigmaLOSobsvals)
#print(sigmaLOSerrorvals)


# In[6]:


#Truncate the bins to avoid double-computing the same bins
#data=obsdata1[galnum]
#stellarkinematics=stellardata[galnum]
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
#print(fullbins[0])
binpositionvals=[list(itertools.chain.from_iterable([np.argwhere(binvals[i][:,0]==fullbins[i][:,0][k])[0] 
        for k in range(0,len(fullbins[i][:,0]))])) #Position of bins in list of full bins
     for i in range(0,len(stellardata))]
print(binvals[0])
#print(pos)


# ### Load Sersic profiles with and without M/L gradient

# In[7]:


#rhoSersicNoGradTab=[np.loadtxt(path+'/Dropbox/SIDM on Intermediate Scales/Working group 1 Groups and ellipticals/Loading Sersic Profiles/SersicDensityProfile_DelUps0_'+names[i]+'.dat') for i in range(0,len(names))]
rhoSersicNoGradTab=[np.loadtxt('ObservationData/SersicDensityProfile_DelUps0_'+names[i]+'.dat') for i in range(0,len(names))]
rhoSersicNoGradInt=[interp1d(rhoSersicNoGradTab[i][:,0],rhoSersicNoGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

#MSersicNoGradTab=[np.loadtxt(path+'/Dropbox/SIDM on Intermediate Scales/Working group 1 Groups and ellipticals/Loading Sersic Profiles/SersicEnclMass_DelUps0_'+names[i]+'.dat') for i in range(0,len(names))]
MSersicNoGradTab=[np.loadtxt('ObservationData/SersicEnclMass_DelUps0_'+names[i]+'.dat') for i in range(0,len(names))]
MSersicNoGradInt=[interp1d(MSersicNoGradTab[i][:,0],MSersicNoGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

print(MSersicNoGradInt[0](10.))
print(rhoSersicNoGradInt[0](10.))


# In[8]:


#rhoSersicGradTab=[np.loadtxt(path+'/Dropbox/SIDM on Intermediate Scales/Working group 1 Groups and ellipticals/Loading Sersic Profiles/SersicDensityProfile_DelUps015_'+names[i]+'.dat') for i in range(0,len(names))]
rhoSersicGradTab=[np.loadtxt('ObservationData/SersicDensityProfile_DelUps015_'+names[i]+'.dat') for i in range(0,len(names))]
rhoSersicGradInt=[interp1d(rhoSersicGradTab[i][:,0],rhoSersicGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

#MSersicGradTab=[np.loadtxt(path+'/Dropbox/SIDM on Intermediate Scales/Working group 1 Groups and ellipticals/Loading Sersic Profiles/SersicEnclMass_DelUps015_'+names[i]+'.dat') for i in range(0,len(names))]
MSersicGradTab=[np.loadtxt('ObservationData/SersicEnclMass_DelUps015_'+names[i]+'.dat') for i in range(0,len(names))]
MSersicGradInt=[interp1d(MSersicGradTab[i][:,0],MSersicGradTab[i][:,1], kind='cubic', fill_value='extrapolate') for i in range(0,len(names))]

print(MSersicGradInt[0](10.))
print(rhoSersicGradInt[0](10.))


# # ACNFW profile (MakeACNFWTableIntMod)

# ### Test parameters

# In[9]:


[Ytest,betatest,rho0test,sigma0test,sigmamtest]=[5.,0.,5.*10**8.,300.,0.1]


# ### Constants

# In[10]:


H0=70.*10.**(-3.) #km/s/kpc;
h=H0/(100.*10.**(-3.)) #for H0 in km/s/kpc
print(h)

Omegam=0.3
OmegaL=0.7
G=4302./10.**9. #4.302*10^-6 kpc MSun^-1 (km/s)^2 

#tage = (3.154*10.**17.)/2.
tage = 5.*10**9.*365.*24.*60.*60 #Assuming tage= 5 Gyr = 5*10^9*365*24*60*60 s
MSun_in_g = 1.98855*10.**30.*10.**3. #Msun in g
km_in_kpc = 1./(3.0857*10.**16.) #kpc=3.0857*10^16 km
cm_in_kpc = km_in_kpc/(10.**3.*10**2.)
#print(G)
#print tage
#print cm_in_kpc

#_____Range of interpolation functions for rhoSersic and MSersic (same for all groups)_____
#[rmin,rmax]=[MSersicGradTab[0][:,0][0],MSersicGradTab[0][:,0][-1]]  
#print([rmin,rmax])

#rvals=MSersicGradTab[0][:,0]
rvals=np.array([MSersicGradTab[0][:,0][i] for i in range(1,len(MSersicGradTab[0][:,0])-1)]) #Range of r such that FindRoot[...] works
[rmin,rmax]=[rvals[0],rvals[-1]]  
print(len(rvals))
print([rmin,rmax])

Rvals = np.array([rvals[l] for l in range(0,len(rvals))]) #range for the ACSIDM mass and density profiles
[Rmin,Rmax]=[Rvals[0],Rvals[-1]]  
print([Rmin,Rmax])

##Bounds for numerical root finding
#print(sys.float_info)
##[FindRootmin, FindRootmax]=[sys.float_info.min,sys.float_info.max]
##print(FindRootmin, FindRootmax)
##Maximum number of iteration for numerical root finding
##maxiternum = 1000


# ### Functions

# In[11]:


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


# In[12]:


#start = time.time()

#[A,w]=[1.,0.] for NFW; [1.,1.] for Blumenthal (Bl), [0.8,0.85] for Gnedin (Gn);
def ACNFWProfile(DMprofile,galnum,Y,M200,c,r): #DMprofile = NFW, Bl, Gn
    #_____Group properties_____
    z = zvals[galnum]
    #Y = NewYvals[galnum]
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
        #ri = opt.minimize_scalar(f, method='Brent', bounds=[rmin, rmax]) 
                #ri = opt.minimize_scalar(f, method='Brent', bounds=[rmin, rmax]) 
        #ri = opt.brentq(f,rmin,rmax,maxiter=150)
        #ri = opt.brentq(f,Rmin,Rmax,maxiter=150)
        #ri = opt.brentq(f,10.**-10.,10.**10.,maxiter=100) #maxiter=150
        ristart=r
        ri=opt.fsolve(f,ristart,xtol=10.**(-3.))[0] #default: xtol=1.49012e-08
        #sol=root(f,ristart) #fsolve faster
        #ri=sol.x
        return ri
    def MACdm(R):
        return MNFW(rival(R)) 
    epsilon = 10.**(-3.)
    M1=MACdm(r-epsilon*r);
    M2=MACdm(r+epsilon*r);
    Mavg=(M1+M2)/2. 
    Mdmprime=(M2-M1)/(2.*epsilon*r)
    rhoACdm=Mdmprime/(4.*np.pi*r**2.)
    #return [M200,c,r,Mavg,rhoACdm]
    return [rhoACdm,Mavg]  
    

#print(ACNFWProfile('NFW',0,1.,10.**12.,1.,100.,))
#print(ACNFWProfile('Bl',0,1.,10.**12.,1.,100.,))
#print(ACNFWProfile('Gn',0,1.,10.**12.,1.,100.,))
print(ACNFWProfile(DMprofile,galnum,1.,10.**12.,1.,100.,))

#end = time.time()
#print(end - start)


# # ACSIDM profile in terms of [rho0,sigma0,xsctn]

# In[13]:


def IsothermalProfile(galnum,Y,rho0,sigma0):
    #_____Group properties_____
    #Y = NewYvals[galnum]
    def Mb(R):
        return Y*MSersicGradInt[galnum](R)
    #_____Isothermal profile_____
    def rhoMiso(rhoM,R):
        [rhodm, Mdm] = rhoM
        drhodm_dR = -((G*rhodm)/(sigma0**2.*R**2.))*(Mdm+Mb(R))
        dMdm_dR = 4.*np.pi*R**2.*rhodm
        return [drhodm_dR,dMdm_dR]
    #_____Initial conditions_____
    #rhodm(rmin) = rho0, Mdm(rmin) = (4 Pi)/3 rho0*rmin^3
    #rhoMini = [rho0,(4.*np.pi)/3.*rho0*rmin**3.]
    rhoMini = [rho0,(4.*np.pi)/3.*rho0*rmin**3.]
    #_____Solve ODE_____
    #R = rvals
    sol = odeint(rhoMiso,rhoMini,rvals) #sol=[rhoiso(R),Miso(R) for R in rvals]
    #def rhoiso(R):
    #    return interp1d(rvals,sol[:,0], kind='cubic')(R) 
    #def Miso(R):
    #    return interp1d(rvals,sol[:,1], kind='cubic')(R) 
    #return [rhoiso(r),Miso(r)]
    return sol

#print(IsothermalProfile(1, 5.*10.**9.,600.))


# In[14]:


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


# In[15]:


#start = time.time()

#def ACSIDMProfile(galnum,DMprofile,CoreGrowingCollapse,Y,rho0,sigma0,xsctn):
def ACSIDMProfile(galnum,DMprofile,Y,rho0,sigma0,xsctn):
    #_____Group properties_____
    z = zvals[galnum]
    #Y = NewYvals[galnum]
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
            #r1=r/.FindRoot[rho0iso[r]*MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*xsctn*kpcovercm**2*tage==1,{r,1}];
            def Findr1(R):
                return rhoiso(R)-1./(MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*xsctn*cm_in_kpc**2.*tage)
            r1 = opt.brentq(Findr1,Rmin,Rmax,maxiter=150)
            #print(r1)
            #_____Matching success tests_____    
            ratio = Miso(r1)/(4.*np.pi*rhoiso(r1)*r1**3.)
            #if ratio > 0.5: #MatchingSuccess=ratio>0.5&&ratio<Log[$MaxNumber]
            #print(ratio)
            if ratio > 0.5:
                #####Solutions for [M200,c] do strongly depend on [M200start,cstart].
                if ratio <= np.log(1.+10.**10.)-1.:
                    x1 = X1(ratio)
                else: 
                    x1 = np.exp(1.+ratio)-1.
                rhosstart = rhoiso(r1)*x1*(1.+x1)**2.
                rsstart = r1/x1
                #cstart=cval/.FindRoot[rhosSIDM==(200rhocrit[z]cval^3)/(3(Log[1+cval]-cval/(1+cval))),{cval,10,0.1,1000}]
                def Findcstart(c):
                    return rhosstart - (200.*rhocrit(z)*c**3.)/(3.*(np.log(1.+c)-c/(1.+c)))
                #cstart = opt.brentq(Findcstart,10.**-15.,10.**15.,maxiter=150)
                cstart = opt.brentq(Findcstart,10.**-10.,10.**10.,maxiter=150)
                M200start = (4.*np.pi)/3.*200.*rhocrit(z)*cstart**3.*rsstart**3
                #If[M200start<10^10,M200start=10^13]; Start value for M200 in LogMIntrhoInttab is 10^10.
                def FindM200c(M200c):
                    [M200,c] = M200c
                    #Minterp[M200val,cval,r1]/(4 pi r1^3 rhointerp[M200val,cval,r1])==MSIDM[r1]/(4 pi r1^3 rhoSIDM[r1])
                    equation1 = MACNFW(M200,c,r1)/(4.*np.pi*r1**3.*rhoACNFW(M200,c,r1)) - ratio
                    equation2 = rhoACNFW(M200,c,r1) - rhoiso(r1)
                    return [equation1,equation2]
                [M200val,cval] = opt.fsolve(FindM200c,[M200start,cstart])
                [rhosval,rsval] = [rhos(z,cval),rs(z,M200val,cval)]
                #print([M200val,cval])
                MatchingSuccessTest = abs((rhoiso(r1)-rhoACNFW(M200val,cval,r1))/rhoiso(r1))
                #if MatchingSuccessTest <=0.01: Matching success test passed.
                #print([M200val,cval,r1])
                if MatchingSuccessTest <=0.01:
                    if r1 < r200(z,M200val,cval):
                        [r1,M200val,cval]=[r1,M200val,cval]
                        ##_____Core-growing vs. core-collapse solution_____  
                        ##drhoiso/dr:
                        #drhoiso1=-G*rhoiso(r1)*(Miso(r1)+Mb(r1))/(sigma0**2.*r1**2.)
                        ##drhoNFW/dr ###Also use this condition in case of ACNFW
                        #drhoNFW1=-(rhosval/rsval)*(1.+3.*(r1/rsval))*(rsval/r1)**2.*(1.+(r1/rsval))**(-3.)
                        #s1=drhoiso1/drhoNFW1 #s1 > 1 = core growing, s1 < 1 = core collapse
                        ##print([drhoiso1,drhoNFW1,s1])
                        #if CoreGrowingCollapse == 'CoreGrowing' and s1>1:
                        #    print('CoreGrowing.')
                        #    [r1,M200val,cval]=[r1,M200val,cval]
                        #elif CoreGrowingCollapse == 'CoreCollapse' and s1<1:
                        #    [r1,M200val,cval]=[r1,M200val,cval]
                        #    print('CoreCollapse.')
                        #else:
                        #    [r1,M200val,cval]=[100.,1.,10.**12.]
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
        #fill_value='extrapolate': interpolate such that Rmin and Rmax are included in interpolation range
        #MACSIDMInt=interp1d(Rvals,[MACSIDM(M200val,cval,R) for R in Rvals], kind='cubic')
        MtotInt=interp1d(Rvals,[MACSIDM(M200val,cval,r1,R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
    except:
        #ACSIDM solution fails, SIDM solution taken instead and dummy variables for [M200,c] split out
        [r1,M200val,cval]=[100.,1.,10.**12.]
        rhoACSIDMInt=interp1d(Rvals,[rhoiso(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        #fill_value='extrapolate': interpolate such that Rmin and Rmax are included in interpolation range
        #MACSIDMInt=interp1d(Rvals,[MACSIDM(M200val,cval,R) for R in Rvals], kind='cubic')
        MtotInt=interp1d(Rvals,[Miso(R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        
    #return [MtotInt,rhoACSIDMInt,np.log10(M200val),np.log10(cval),r1]
    #return [M200val,cval,r1]
    sigmavm=sigma0*(4./np.sqrt(np.pi))*xsctn
    return [M200val,cval,r1,sigmavm]

#print(ACSIDMProfile(0,'NFW','CoreGrowing',Ytest,rho0test,sigma0test,sigmamtest))
print(ACSIDMProfile(galnum,DMprofile,Ytest,rho0test,sigma0test,sigmamtest))

#end = time.time()
#print(end - start) #returning interpolation function is faster than returning tables 


# # ACSIDM profile in terms of [M200,c,sigmavm]

# In[16]:


def IsothermalProfileInt(galnum,Y,rho0,sigma0):
    #_____Group properties_____
    #Y = NewYvals[galnum]
    def Mb(R):
        return Y*MSersicGradInt[galnum](R)
    #_____Isothermal profile_____
    def rhoMiso(rhoM,R):
        [rhodm, Mdm] = rhoM
        drhodm_dR = -((G*rhodm)/(sigma0**2.*R**2.))*(Mdm+Mb(R))
        dMdm_dR = 4.*np.pi*R**2.*rhodm
        return [drhodm_dR,dMdm_dR]
    #_____Initial conditions_____
    #rhodm(rmin) = rho0, Mdm(rmin) = (4 Pi)/3 rho0*rmin^3
    #rhoMini = [rho0,(4.*np.pi)/3.*rho0*rmin**3.]
    rhoMini = [rho0,(4.*np.pi)/3.*rho0*rmin**3.]
    #_____Solve ODE_____
    #R = rvals
    sol = odeint(rhoMiso,rhoMini,rvals) #sol=[rhoiso(R),Miso(R) for R in rvals]
    rhoiso=interp1d(rvals,sol[:,0], kind='cubic',fill_value='extrapolate')
    Miso=interp1d(rvals,sol[:,1], kind='cubic',fill_value='extrapolate')
    #def rhoiso(R):
    #    return interp1d(rvals,sol[:,0], kind='cubic')(R) 
    #def Miso(R):
    #    return interp1d(rvals,sol[:,1], kind='cubic')(R) 
    return [rhoiso,Miso]
    #return sol

print(IsothermalProfileInt(galnum,5,5.*10.**9.,600.)[0](100.))


# In[17]:


#start = time.time()

#def ACSIDMProfileM200cxsctn(galnum,DMprofile,Y,M200,c,xsctn):
def ACSIDMProfileM200csigmavm(galnum,DMprofile,Y,M200,c,sigmavm):
    #_____Group properties_____
    z = zvals[galnum]
    #Y = NewYvals[galnum]
    def Mb(R):
        return Y*MSersicGradInt[galnum](R)
    
    #_____AC NFW profile_____
    def rhoACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[0]
    def MACNFW(M200,c,R):
        return ACNFWProfile(DMprofile,galnum,Y,M200,c,R)[1]
    [rhosval,rsval] = [rhos(z,c),rs(z,M200,c)]
    
    #_____Isothermal profile_____
    #sol = IsothermalProfile(galnum,Y,rho0,sigma0)
    def rhoiso(rho0,sigma0,R):
        return IsothermalProfileInt(galnum,Y,rho0,sigma0)[0](R)
    def Miso(rho0,sigma0,R):
        return IsothermalProfileInt(galnum,Y,rho0,sigma0)[1](R)
    
    def rhoACSIDM(M200,c,rho0,sigma0,r1,R):
        if R > r1:
            rhodm = rhoACNFW(M200,c,R)
        else: 
            rhodm = rhoiso(rho0,sigma0,R)
        return rhodm 
    def MACSIDM(M200,c,rho0,sigma0,r1,R):
        if R > r1:
            Mdm = MACNFW(M200,c,R)
        else: 
            Mdm = Miso(rho0,sigma0,R)
        return Mdm
    
    #_____Isothermal profile_____
    def Findr1(R):
        #return (rhoACNFW(M200,c,R)-1./(MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*xsctn*cm_in_kpc**2.*tage))**2.
        #sigmavm=sigma0*(4./np.sqrt(np.pi))*xsctn
        return (rhoACNFW(M200,c,R)-1./(MSun_in_g*sigmavm*km_in_kpc*cm_in_kpc**2.*tage))**2.
        #return (rhoACNFWInt(R)-1./(MSun_in_g*sigma0*km_in_kpc*(4./np.sqrt(np.pi))*xsctn*cm_in_kpc**2.*tage))**2.
    #r1 = opt.brentq(Findr1,Rmin,Rmax,maxiter=5) #maxiter=150
    r1start=10.
    r1=opt.fsolve(Findr1,r1start)[0] #default: ,xtol=10.**(-3.), xtol=1.49012e-08
    #print(r1)
    
    #_____ACSIDM profile__________
    #ratio = Miso(rho0,sigma0,r1)/(4.*np.pi*rhoiso(rho0,sigma0,r1)*r1**3.)
    ratio = MACNFW(M200,c,r1)/(4.*np.pi*rhoACNFW(M200,c,r1)*r1**3.)
    #if ratio > 0.5: #MatchingSuccess=ratio>0.5&&ratio<Log[$MaxNumber]
    if ratio > 0.5:
        def Findrho0sigma0(rho0sigma0):
            [rho0,sigma0] = rho0sigma0
            rho1=rhoiso(rho0,sigma0,r1)
            M1=Miso(rho0,sigma0,r1)
            equation1 = M1/(4.*np.pi*r1**3.) - ratio*rho1
            equation2 = rhoACNFW(M200,c,r1) - rho1
            return [equation1,equation2]
        [rho0start,sigma0start]=[10.**7.5,580.]
        #[rho0start,sigma0start]=[10.**8.,400.]
        #[rho0,sigma0]=opt.minimize(Findmin,[rho0start,sigma0start]).x[0] ### too slow
        [rho0,sigma0] = abs(opt.fsolve(Findrho0sigma0,[rho0start,sigma0start],xtol=10.**(-5.))) #default: xtol=1.49012e-08
        #sol=root(Findrho0sigma0,[rho0start,sigma0start],tol=0.1) #as fast as fsolve
        #[rho0,sigma0] =sol.x
        #_____Matching success tests_____
        #MatchingSuccessTest = abs((rhoiso(r1)-rhoACNFW(M200val,cval,r1))/rhoiso(r1))
        MatchingSuccessTestrho = abs((rhoiso(rho0,sigma0,r1)-rhoACNFW(M200,c,r1))/rhoiso(rho0,sigma0,r1))
        MatchingSuccessTestM = abs((Miso(rho0,sigma0,r1)-MACNFW(M200,c,r1))/Miso(rho0,sigma0,r1))
        #if MatchingSuccessTest <=0.01: Matching success test passed.
        if MatchingSuccessTestrho <=0.01 and MatchingSuccessTestM <=0.01:
            if r1 < r200(z,M200,c):
                #[r1,M200val,cval]=[r1,M200val,cval]
                [rho0,sigma0,sigmavm]=[rho0,sigma0,sigmavm]
                ##_____Core-growing vs. core-collapse solution_____  
                ##drhoiso/dr:
                #drhoiso1=-G*rhoiso(rho0,sigma0,r1)*(Miso(rho0,sigma0,r1)+Mb(r1))/(sigma0**2.*r1**2.)
                ##drhoNFW/dr ###Also use this condition in case of ACNFW
                #drhoNFW1=-(rhosval/rsval)*(1.+3.*(r1/rsval))*(rsval/r1)**2.*(1.+(r1/rsval))**(-3.)
                #s1=drhoiso1/drhoNFW1 #s1 > 1 = core growing, s1 < 1 = core collapse.
                #if CoreGrowingCollapse == 'CoreGrowing' and s1>1:
                #    #print('CoreGrowing.')
                #    #[r1,M200,c]=[r1,M200,c]
                #    [rho0,sigma0,r1]=[rho0,sigma0,r1]
                #elif CoreGrowingCollapse == 'CoreCollapse' and s1<1:
                #    #print('CoreCollapse.')
                #    [rho0,sigma0,r1]=[rho0,sigma0,r1]
                #else:
                #    #print('None of these.')
                #    [rho0,sigma0,r1]=[10**8.,600.,100.]
            else:
                [rho0,sigma0,sigmavm]=[10**8.,600.,50.]
        else:
            [rho0,sigma0,sigmavm]=[10**8.,600.,50.]
    else:
        [rho0,sigma0,sigmavm]=[10**8.,600.,50.]
    try:
        rhoACSIDMInt=interp1d(Rvals,[rhoACSIDM(M200,c,rho0,sigma0,r1,R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        #fill_value='extrapolate': interpolate such that Rmin and Rmax are included in interpolation range
        MtotInt=interp1d(Rvals,[MACSIDM(M200,c,rho0,sigma0,r1,R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
    except:
        ##ACSIDM solution fails, ACNFW solution taken instead and dummy variables for [rho0,sigma0,sigmavm] split out
        ##[rho0,sigma0,sigmavm]=[10**8.,600.,50.]
        [rho0,sigma0,sigmavm]=[10**8.,600.,50.]
        rhoACSIDMInt=interp1d(Rvals,[rhoACNFW(M200,c,R) for R in Rvals], kind='cubic', fill_value='extrapolate')
        #fill_value='extrapolate': interpolate such that Rmin and Rmax are included in interpolation range
        MtotInt=interp1d(Rvals,[MACNFW(M200,c,R)+Mb(R) for R in Rvals], kind='cubic', fill_value='extrapolate')
    
    #return [MtotInt,rhoACSIDMInt,np.log10(M200),np.log10(c)]
    xsctn=sigmavm/((4./np.sqrt(np.pi))*sigma0)
    return [MtotInt,rhoACSIDMInt,np.log10(M200),np.log10(c),np.log10(rho0),np.log10(sigma0),r1,sigmavm,xsctn]    

#end = time.time()
#print((end - start))


# In[18]:


#start = time.time()

[Ytest,rho0test,sigma0test,sigmamtest]=[5.,10**8.,600.,0.5]
#sigmavmtest2=sigma0test2*(4./np.sqrt(np.pi))*sigmamtest2
[M200test,ctest,r1test,sigmavmtest]=ACSIDMProfile(galnum,DMprofile,Ytest,rho0test,sigma0test,sigmamtest)
print([np.log10(rho0test),np.log10(sigma0test),r1test,sigmavmtest,sigmamtest])
#print(ACSIDMProfileM200csigmavm(0,'NFW',Ytest2,M200test2,ctest2,sigmavmtest2))
print([ACSIDMProfileM200csigmavm(galnum,DMprofile,Ytest,M200test,ctest,sigmavmtest)[i] for i in [-5,-4,-3,-2,-1]])

#end = time.time()
#print((end - start))


# # Test parameters

# In[19]:


#[Ytest,rho0test,sigma0test,sigmamtest]=[5.,10**8.,600.,0.5]
[Ytest,rho0test,sigma0test,sigmamtest]=[5.,10**8.5,500,0.5]
#[M200test,ctest,r1test]=ACSIDMProfile(0,'NFW','CoreGrowing',Ytest,rho0test,sigma0test,sigmamtest)
[M200test,ctest,r1test,sigmavmtest]=ACSIDMProfile(galnum,DMprofile,Ytest,rho0test,sigma0test,sigmamtest)
print([M200test,ctest,sigmavmtest])


# In[20]:


#start = time.time()

[Ytest,rho0test,sigma0test,sigmamtest]=[5.,10**8.5,500,0.5]
#[M200test,ctest,r1test]=ACSIDMProfile(0,'NFW','CoreGrowing',Ytest,rho0test,sigma0test,sigmamtest)
[M200test,ctest,r1test,sigmavmtest]=ACSIDMProfile(galnum,DMprofile,Ytest,rho0test,sigma0test,sigmamtest)
print([np.log10(rho0test),np.log10(sigma0test),r1test,sigmavmtest,sigmamtest])
#print([ACSIDMProfileM200cxsctn(0,'NFW','CoreGrowing',Ytest,M200test,ctest,sigmamtest)[i] for i in [-3,-2,-1]])
print([ACSIDMProfileM200csigmavm(galnum,DMprofile,Ytest,M200test,ctest,sigmavmtest)[i] for i in [-5,-4,-3,-2,-1]])

#end = time.time()
#print((end - start))


# # Mean convergence kappabar

# In[21]:


#start = time.time()

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
    #numpoints = 10000
    #RminREvals = np.linspace(Rmin,RE,numpoints)
    #RERmaxvals = np.linspace(RE,Rmax,numpoints)
    #Mdmencl = integrate.simps([integrand1(r) for r in RminREvals],RminREvals)+integrate.simps([integrand1(r) for r in RERmaxvals],RERmaxvals)
    Mdmencl = integrate.quad(integrand1,Rmin,RE)[0]+integrate.quad(integrand2,RE,Rmax)[0]
    kappadm=Mdmencl/(np.pi*RE**2.*Sigmacr)
    kappatot=kappadm+Y*kappab
    return kappatot

#print(kappabartot(0,Ytest,ACSIDMProfileM200cxsctn(0,'NFW',Ytest,M200test,ctest,sigmamtest)[1])) #rhoACSIDMInt=ACSIDMProfile(...)[1]
print(kappabartot(galnum,Ytest,ACSIDMProfileM200csigmavm(galnum,DMprofile,Ytest,M200test,ctest,sigmavmtest)[1])) #rhoACSIDMInt=ACSIDMProfile(...)[1]
#print(kappabartest)

#end = time.time()
#print(end - start) #returning interpolation function is faster than returning tables 


# # Binned LOS velocity dispersion

# ### Projected stellar surface density: Sigmastars(R)

# In[22]:


#R=np.logspace(np.log10(Rmin),np.log10(Rmax),250)
R=np.array(list(itertools.chain.from_iterable([10.**np.arange(-1.,3.5,0.02),[3162.28]])))
r=np.array([0.5*(R[i]+R[i+1]) for i in range(0,len(R)-1)])
Delta=np.array([R[i+1]-R[i] for i in range(0,len(R)-1)])
#print(Delta)


# In[23]:


#start = time.time()

def Sigmastars(galnum):
    #rhostars=rhoSersicGradInt[galnum]
    rhostars=rhoSersicNoGradInt[galnum]
    integrand=np.array([rhostars(r[j]) for j in range(0,len(r))])
    def integral(i):
        integral1=2.*integrand[i]*np.sqrt(2.*Delta[i]*r[i])
        integral2=np.sum([(2.*Delta[j]*r[j]*integrand[j])/np.sqrt(((R[j+1]+R[j])/2.)**2.-R[i]**2.) for j in range(i+1,len(r))])
        return integral1+integral2
    Int=np.array([integral(i) for i in range(0,len(r))])
    Inttab=np.array([[R[i],Int[i]] for i in range(0,len(R)-1)])
    return Inttab

#print(Sigmastars(0))

#end = time.time()
#print(end - start)  


# ### Surface density times velocity dispersion squared: Sigmastars_sigmaLOS2(R)

# In[24]:


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

print(fbeta(0.1,0.5))


# In[25]:


#alphavals=np.arange(-20.,21.,.1)
alphavals=np.arange(-25.,25.,.1)
wvals=1./(np.exp(alphavals)+1)
print(len(wvals))
print([min(wvals),max(wvals)])


# In[26]:


#start = time.time()

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
#list(np.float_([f(w) for w in wvals]))

print(f_beta(0.1)(0.5))

#end = time.time()
#print(end - start)  


# In[27]:


#start = time.time()

def Sigmastars_sigmaLOS2(galnum,MtotInt,beta):
    #rhostars=rhoSersicGradInt[galnum]
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

#Mtot=ACSIDMProfileM200cr1(0,'NFW',Ytest,M200test,ctest,r1test)[0]
#Mtot=ACSIDMProfileM200csigmavm(0,'NFW',Ytest,M200test,ctest,sigmavm)[0]
Mtot=MSersicGradInt[0]
SigmastarssigmaLOS2=Sigmastars_sigmaLOS2(0,Mtot,0.1)
#print(SigmastarssigmaLOS2)

#end = time.time()
#print(end - start)  


# ### Seeing and binning

# In[28]:


#start = time.time()

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

Sigmastars_seeing_binned0=SeeingBinned(0,Sigmastars(0))
#Sigmastars_sigmaLOS2_seeing_binned0=SeeingBinned(0,SigmastarssigmaLOS20)
print(Sigmastars_seeing_binned0)
#print(Sigmastars_sigmaLOS2_seeing_binned0)

#end = time.time()
#print(end - start) 


# In[51]:


Sigmastars_seeing_binned_vals=np.array([SeeingBinned(galnumber,Sigmastars(galnumber)) for galnumber in range(0,len(names))])
print(Sigmastars_seeing_binned_vals)


# In[30]:


#start = time.time()

def sigmaLOS_seeing_binned(galnum,MtotInt,beta):
    binpositions=binpositionvals[galnum]
    Sigmastars_seeing_binned=Sigmastars_seeing_binned_vals[galnum]
    Sigmastars_sigmaLOS2_seeing_binned=SeeingBinned(galnum,Sigmastars_sigmaLOS2(galnum,MtotInt,beta))
    sigmaLOS=np.sqrt(Sigmastars_sigmaLOS2_seeing_binned/Sigmastars_seeing_binned)
    sigmaLOSfull=np.array([sigmaLOS[i] for i in binpositions])
    return sigmaLOSfull

#Mtot=ACSIDMProfileM200cxsctn(0,'NFW',Ytest,M200test,ctest,sigmamtest)[0]
Mtot=ACSIDMProfileM200csigmavm(galnum,DMprofile,Ytest,M200test,ctest,sigmavmtest)[0]
print(sigmaLOS_seeing_binned(galnum,Mtot,betatest))
#print(sigmaLOStest)
###Very small differences to Sean's code come from differences when computing Mtot and interpolation function for f(beta,w).
###Difference to Sean's code less than 0.1% (1-2% is accuracy of Sean's code).

#end = time.time()
#print(end - start)   


# # ACSIDM group fit probability

# ### ACSIDM group fit probability

# In[31]:


#start = time.time()

#ACSIDMProfileM200cr1(0,'NFW','CoreGrowing',Y,M200,c,r1)
#def ACSIDMGroupFitProbability(galnum,DMprofile,log10Y,beta,log10M200,log10c,xsctn):
def ACSIDMGroupFitProbability(galnum,DMprofile,log10Y,beta,log10M200,log10c,sigmavm):
    #_____Free parameters_____ 
    Y=10.**log10Y
    #rho0=10.**log10rho0
    #sigma0=10.**log10sigma0
    #xsctn=10.**log10xsctn
    M200=10.**log10M200
    c=10.**log10c
    #r1=10.**log10r1
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
    #[MtotInt,rhoACSIDMInt,log10M200,log10c] = ACSIDMProfile(galnum,DMprofile,CoreGrowingCollapse,Y,rho0,sigma0,xsctn)
    #[MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1]=ACSIDMProfileM200cxsctn(galnum,DMprofile,CoreGrowingCollapse,Y,M200,c,xsctn)
    [MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1,sigmavm,xsctn]=ACSIDMProfileM200csigmavm(galnum,DMprofile,Y,M200,c,sigmavm)
    #_____\chi^2 lensing_____ 
    kappabar = kappabartot(galnum,Y,rhoACSIDMInt)
    ChiSqLensing = (kappabar-kappabarobs)**2./kappabarobserror**2.
    #_____\chi^2 mass_____ 
    #Mass-concentration relation with redshift dependence
    a=0.520+(0.905-0.520)*np.exp(-0.617*z**1.21)
    b=-0.101+0.026*z
    log10cNFW=a+b*(log10M200-np.log10(10.**12.*h**(-1.)))  #M200 in units of MSun
    log10cerror=0.1 #error estimate
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
    #output=np.array(list(itertools.chain.from_iterable([[prob,ChiSqDisp,ChiSqLensing,ChiSqMass],sigmaLOS,[kappabar]]))) #+ChiSqML
    output=np.array([ChiSqLensing,ChiSqMass,ChiSqDisp,prob])
    return output

#print(ACSIDMGroupFitProbability(0,'NFW',np.log10(Ytest),betatest,np.log10(M200test),np.log10(ctest),sigmamtest))
print(ACSIDMGroupFitProbability(galnum,DMprofile,np.log10(Ytest),betatest,np.log10(M200test),np.log10(ctest),sigmavmtest))
#print([ChiSqLensingtest,ChiSqMasstest,ChiSqVelDisptest,probtest])
##print(probtest)
###Different values for prob in comparison to Sean's code come from differences when computing Mtot and interpolation function for f(beta,w).

#end = time.time()
#print((end - start))


# # Set up MCMCs

# ### Logarithm of probability: lnprob

# In[32]:


#start = time.time()

#Scan over log10 of free paramters: paramters always > 0, priors not necessary
#params has to be the first entry in lnprob to make emcee work
#def lnprob(params,galnum,DMprofile,CoreGrowingCollapse):
def lnprob(params,galnum,DMprofile):
    #[log10Y,beta,log10rho0,log10sigma0,log10xsctn]=params 
    #[log10Y,beta,log10M200,log10c,log10r1]=params
    #[log10Y,beta,log10M200,log10c,xsctn]=params
    [log10Y,beta,log10M200,log10c,sigmavm]=params
    #[Y,beta,rho0,sigma0,xsctn]=params
    #[DMprofile,galnum]=externalparams
    #_____Free parameters_____ 
    Y=10.**log10Y
    #rho0=10.**log10rho0
    #sigma0=10.**log10sigma0
    #xsctn=10.**log10xsctn
    M200=10.**log10M200
    c=10.**log10c
    #r1=10.**log10r1
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
    #if Y > 10.:
    #    lnprob = -np.inf #Prior on Y, for too large values of Y FindRoot for r1 for 'Gn' doesn't work any more  
    elif abs(beta) > 0.3:
        lnprob = -np.inf
    elif log10c > np.log10(20.): #0 <= c <= 20. equivalemt to log10c > np.log10(20.)
        lnprob = -np.inf
    elif sigmavm < 0.: 
        lnprob = -np.inf
    #elif sigmavm > 10.: 
    #    lnprob = -np.inf
    #elif xsctn < 0.: 
    #    lnprob = -np.inf
    #elif r1 < 0.: 
    #    lnprob = -np.inf
    #elif sigma0 < 160:
    #    lnprob = -np.inf #Prior on sigma0, for too large values of Y FindRoot for r1 for 'NFW' doesn't work any more      
    else: 
        #_____ACSIDM profile_____ 
        #[MtotInt,rhoACSIDMInt,log10M200,log10c] = ACSIDMProfile(galnum,DMprofile,CoreGrowingCollapse,Y,rho0,sigma0,xsctn)
        #[MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1]=ACSIDMProfileM200cxsctn(galnum,DMprofile,CoreGrowingCollapse,Y,M200,c,xsctn)
        #[MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1]=ACSIDMProfileM200cxsctn(galnum,DMprofile,Y,M200,c,xsctn)
        [MtotInt,rhoACSIDMInt,log10M200,log10c,log10rho0,log10sigma0,r1,sigmavm,xsctn]=ACSIDMProfileM200csigmavm(galnum,DMprofile,Y,M200,c,sigmavm)
        #if M200 == 10.**12 and c==1.:
        #    lnprob = -np.inf
        #if log10M200 == 12. and log10c==0.:
        #    lnprob = -np.inf
        #[rho0,sigma0,r1]=[10.**7.5,580.,100.]
        if log10rho0 == 8. and log10sigma0==np.log10(600.) and sigmavm==50.:
            lnprob = -np.inf
        elif xsctn < 0.:
            lnprob = -np.inf
        elif xsctn > 10.:
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
            #log10cerror=0.1 #error estimate
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
            #output=np.array(list(itertools.chain.from_iterable([[prob,ChiSqDisp,ChiSqLensing,ChiSqMass],sigmaLOS,[kappabar]]))) #+ChiSqML
            #output=np.array([ChiSqLensing,ChiSqMass,ChiSqDisp,prob])
    #return prob
    return lnprob

#testparams=[np.log10(Ytest),betatest,np.log10(rho0test),np.log10(sigma0test),np.log10(sigmamtest)]
#print(lnprob(testparams,0,'NFW','CoreGrowing'))
testparams=[np.log10(Ytest),betatest,np.log10(M200test),np.log10(ctest),sigmavmtest]
#print(lnprob(testparams,0,'NFW','CoreGrowing'))
print(lnprob(testparams,galnum,DMprofile))

#end = time.time()
#print((end - start))


# In[33]:


testparams=[np.log10(Ytest),betatest,np.log10(10.),np.log10(ctest),sigmavmtest]
#print(lnprob(testparams,0,'NFW','CoreGrowing'))
print(lnprob(testparams,galnum,DMprofile))


# ### Find random initial points for walkers

# In[34]:


#[log10Y,beta,log10rho0,log10sigma0,log10xsctn]
#initialparams=[np.log10(2.1),0.,np.log10(10**7.5),np.log10(580.),np.log10(1.5)]
[Yini,betaini,rho0ini,sigma0ini,xsctnini]=[2.1,0.,10**7.3,580.,1.6]
sigmavmini=sigma0ini*(4./np.sqrt(np.pi))*xsctnini
#[M200ini,cini,r1ini]=ACSIDMProfile(0,'NFW','CoreGrowing',Yini,rho0ini,sigma0ini,xsctnini)
#[M200ini,cini,r1ini]=ACSIDMProfile(0,'NFW','CoreGrowing',Yini,rho0ini,sigma0ini,xsctnini)
[M200ini,cini,r1inim,sigmavmini]=ACSIDMProfile(galnum,DMprofile,Yini,rho0ini,sigma0ini,xsctnini)
print([M200ini,cini,r1inim,sigmavmini])


# In[35]:


#paramslnprob=[log10Y,beta,log10M200,log10c,r1]
initialparams=[np.log10(Yini),betaini,np.log10(M200ini),np.log10(cini),sigmavmini]
#for galnum in range(0,len(names)):
#    for DMprofile in ['NFW','Bl', 'Gn']:
#        #for CoreGrowingCollapse in ['CoreGrowing']:
#        #print(lnprob(initialparams,galnum,DMprofile,CoreGrowingCollapse))
#        print(lnprob(initialparams,galnum,DMprofile))
#Initial points for work all cases. Ok.
print(lnprob(initialparams,galnum,DMprofile))


# In[36]:


[Yini,betaini,rho0ini,sigma0ini,xsctnini]=[2.1,0.,10**7.3,580.,1.6]
sigmavmini=sigma0ini*(4./np.sqrt(np.pi))*xsctnini
[M200ini,cini,r1inim,sigmavmini]=ACSIDMProfile(galnum,DMprofile,Yini,rho0ini,sigma0ini,xsctnini)
initialparams=[np.log10(Yini),betaini,np.log10(M200ini),np.log10(cini),sigmavmini]
print(initialparams)


# In[37]:


#initialparams=[np.log10(Yini),betaini,np.log10(M200ini),np.log10(cini),xsctnini]
#initialparams=[0.3222192947339193, 0.0, 14.141639613890037, 0.95018288373578297, 1.6]
initialparams=[0.3222192947339193, 0.0, 14.141639613890037, 0.95018288373578297, 2094.2717341292714]


# In[38]:


#def walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile,CoreGrowingCollapse):
def walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile):
    print('Determine starting points for walkers:')
    #_____Starting points for walkers_____
    j=0
    jmax=20*nwalkers
    while j < jmax:
        if j==0:
            params = initialparams
            #lnprobval = lnprob(params,galnum,DMprofile,CoreGrowingCollapse)
            lnprobval = lnprob(params,galnum,DMprofile)
            #print(lnprobval)
            #(-2.)*np.log(prob)=Chi2
            chainsini = [list(itertools.chain.from_iterable([params,[(-2.)*lnprobval]]))]
            #print(chainsini)
            #print(j)
            j+= 1
        else:
            params=initialparams+paramserrors*np.random.randn(len(initialparams))
            #cov=np.array(paramserrors)**2.*np.identity(len(initialparams)) #* makes maric multiplcation automatically
            #params = np.random.multivariate_normal(params,cov)
            #lnprobval = lnprob(params,galnum,DMprofile,CoreGrowingCollapse)
            lnprobval = lnprob(params,galnum,DMprofile)
            if lnprobval==-np.inf:
                chainsini=chainsini
            else:
                output = np.array(list(itertools.chain.from_iterable([params,[(-2.)*lnprobval]])))
                chainsini.append(output)
                #print(chainsini)
                #print(len(chainsini))
                #print(j)
            
            if len(chainsini)==nwalkers:
                #print('Initial points for walkers determined:')
                #print(chainsini)
                #np.savetxt(pathchains+'Initialpoints_'+filename+'.dat',chainsini, header=str(header))
                #print('Initialpoints_'+filename+'.dat exported.')
                
                #p0vals=np.array([[chainsini[i][j] for j in range(0,len(initialparams))] for i in range(0,nwalkers)])
                #print(p0vals)
                j=jmax+1
            else:
                j+= 1
    
    #return p0vals
    return np.array(chainsini)


# In[39]:


initialparams=[np.log10(Yini),betaini,np.log10(M200ini),np.log10(cini),sigmavmini]
paramserrors=np.array([0.1,0.3,2.,0.5,50.])
#[galnum,DMprofile,CoreGrowingCollapse]=[0,'NFW','CoreGrowing']
#[galnum,DMprofile]=[0,'NFW']
nwalkers=10
#print(walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile,CoreGrowingCollapse))
print(walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile))


# ### Module to run MCMCs

# In[40]:


def MCMCNewM200csigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun): #nwalkers should be > 100.
    #_____MCMC properties_____
    header=['log10Y', 'beta', 'log10M200', 'log10c', 'sigmavm', 'Chi2']
    #burninlength = nwalkers*nsamples_burnin
    chainlength = nwalkers*nsamples_finalrun
    #print('Burn in: burninlength='+str(burninlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_burnin]))
    #print('Final run: chainlength='+str(chainlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_finalrun]))
    filename=str(names[galnum])+'_SersicDelUps015_'+str(DMprofile)+'_chainlength'+str(chainlength)+'_nwalkers'+str(nwalkers)+'_nsamples'+str(nsamples_finalrun)+'_threads'+str(threads)
    #_____Set up the MCMC_____
    #Number of free parameters:
    ndim=5 #=len(params)
    initialparams=[0.3222192947339193, 0.0, 14.141639613890037, 0.95018288373578297, 2094.2717341292714]
    paramserrors=np.array([0.1,0.3,2.,0.5,50.])
    #_____Starting points for walkers_____
    #Starting point p0 for each of the walkers:
    #Number of walkers must be the same for burn in and finalrun because of the set up of the initial conditions:
    #p0=np.array([initialparams+paramserrors*np.random.randn(ndim) for i in range(nwalkers)]) 
    chainsini=walkersini(initialparams,paramserrors,nwalkers,galnum,DMprofile)
    print(chainsini)
    np.savetxt('MCMC_results/Startingpointswalkers_M200csigmavm_'+filename+'.dat',chainsini, header=str(header))
    print('Startingpointswalkers_M200csigmavm_'+filename+'.dat exported.')
    p0=np.array([[chainsini[i][j] for j in range(0,len(initialparams))] for i in range(0,nwalkers)])
    #print(p0)
    #_____MCMCs_____
    for i in range(0,nburnins+1):
        if i==0:
            paramsini=p0    
        if i in range(0,nburnins): #Burn-in runs 
            #nsamples=burninlength/nwalkers 
            nsamples=nsamples_burnin #chainlength = nwalkers*nsamples
            runname='run'+str(i+1)
        else: #MCMC production run
            #nsamples=chainlength/nwalkers 
            nsamples=nsamples_finalrun #chainlength = nwalkers*nsamples
            runname='finalrun'
    
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(galnum,DMprofile,CoreGrowingCollapse), threads=threads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(galnum,DMprofile), threads=threads)
        #threads=number of cores
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
        np.savetxt('MCMC_results/Chains_M200csigmavm_'+filename+'_'+runname+'.dat',chains, header=str(header))
        print('Chains_M200csigmavm_'+filename+'_'+runname+'.dat exported.')
    
    #_____Best fit_____
    imin=list(itertools.chain.from_iterable(np.argwhere(Chi2vals==min(Chi2vals))))
    Chi2=Chi2vals[imin][0]
    bestfitparams=params[imin][0]
    print('Best fit: Chi2='+str(Chi2)+', [log10Y,beta,log10M200,log10c,sigmavm]='+str(bestfitparams))
    bestfit=np.array([bestfitparams[0],bestfitparams[1],bestfitparams[2],bestfitparams[3],bestfitparams[4],Chi2])
    #Save results:
    np.savetxt('MCMC_results/Bestfitparams_M200csigmavm_'+filename+'_'+runname+'.dat',bestfit)
    return 'Done.'  


# # Test run

# In[53]:


print(np.array([[galnumber,names[galnumber]] for galnumber in range(0,len(names))]))


# In[42]:


#print('Start parallel job.')
#parallelstart = time.time()

##for galnum in range(0,len(names)):
##for galnum in range(0,1):
#or galnum in [4]:
#   #for DMprofile in ['NFW','Bl', 'Gn']:
#   for DMprofile in ['NFW']:
#       #for CoreGrowingCollapse in ['CoreGrowing']:
#       #_____Number of burn-in runs_____
#       nburnins=5
#       #_____Number of walkers (must be the same for burn in and finalrun because of the set up of the initial conditions)
#       #nwalkers=200
#       #nwalkers=224
#       nwalkers=12
#       #_____Chain lengths_____
#       #nsamples_burnin = 50
#       nsamples_burnin = 1
#       #nsamples_finalrun = 500
#       samples_finalrun = 1
#       #_____Print properties of MCMCs_____
#       burninlength = nwalkers*nsamples_burnin
#       chainlength = nwalkers*nsamples_finalrun
#       print(str(names[galnum])+': ')
#       print('Burn in: nburnins='+str(nburnins)+', burninlength='+str(burninlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_burnin]))
#       print('Final run: chainlength='+str(chainlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_finalrun]))
#       #_____Run MCMCs____
#       #print(MCMCNewM200cxsctn(galnum,DMprofile,CoreGrowingCollapse,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun))
#       #print(MCMCNewM200cxsctn(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun))
#       print(MCMCNewM200csigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun))
#       
#       
#arallelend = time.time()
#rint((parallelend - parallelstart))
#rint('Parallel job done.')


# In[2]:


parallelstart = time.time()

#_____Number of burn-in runs_____
nburnins=5
#_____Number of walkers (must be the same for burn in and finalrun because of the set up of the initial conditions)
#nwalkers=200
nwalkers=224
#nwalkers=12
#_____Chain lengths_____
nsamples_burnin = 50
#nsamples_burnin = 1
nsamples_finalrun = 500
#nsamples_finalrun = 1
#_____Print properties of MCMCs_____
burninlength = nwalkers*nsamples_burnin
chainlength = nwalkers*nsamples_finalrun
print(str(names[galnum])+': ')
print('Burn in: nburnins='+str(nburnins)+', burninlength='+str(burninlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_burnin]))
print('Final run: chainlength='+str(chainlength)+', [nwalkers,nsamples]='+str([nwalkers,nsamples_finalrun]))

#_____Run MCMCs____
print(MCMCNewM200csigmavm(galnum,DMprofile,nburnins,nwalkers,nsamples_burnin,nsamples_finalrun))
        
        
parallelend = time.time()
tparallel=parallelend - parallelstart
print('tparallel='+str(tparallel))


# In[8]:


end = time.time()
ttot=end - start
print('ttot='+str(ttot))

