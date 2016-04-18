"""
Reproduce Pulsar Timing Array Fisher matrix for a binary source
"""

import numpy as np
import numpy.random as npr
from math import sin, cos
import matplotlib.pyplot as plt
import scipy.integrate

class PTA:
    def __init__(self, N = 3, T = 10.0, SNR = 10.0):
        """
        Pulsar timing array with N pulsars sampled for T years
        """
        year = 3.1536e7   #year in seconds
        self.T = T * year  #Length of timing in seconds

        #Normalise noise by SNR per pulsar
        #Currently this normalised to total SNR
        self.SNR = SNR / np.sqrt(N)

        #Define parameters of GW source
        f = 50.0e-9    #GW frequency in Hz
        chirpM = np.power(10.0, 8.5)  #chirp mass in Msol
        D = 1.0    #Distance in kpc
        R = self.R(chirpM, f, D)
        
        # PArameter order is R, theta, phi, psi, inc, f, Phi0 
        self.params = self.randomSource(R, f)
        self.labels = ("R", "theta", "phi", "psi", "inc", "f", "Phi0")

        #Set constant time sampling between t=0 and t=T
        # want to ensure sample each GW cycle many times
        print "GW period is ", 1.0 / f / year, "years"
        npercycle = 256
        self.deltat = 1.0 / f / npercycle
        nsample = int(self.T / self.deltat)
        self.t = np.linspace(0.0, self.T , nsample)

        #Create set of pulsars in the array
        self.pulsars = self.uniformPulsars(N)
        
        #Build Fisher Matrix from all pulsars
        self.fisher = self.fisherMatrix()

        #Calculate errors
        self.errors = self.reportErrors()


    def resetWithNewSource(self):
        """
        Recalculate Fisher errors for a new randomly located source.
        """
        self.params = self.randomSource()
        self.fisher = self.fisherMatrix()
        self.errors = self.reportErrors()
        

    def randomSource(self, R = None, f = None):
        """
        Random source properties. Currently R and f are fixed
        """
        # Parameter order is R, theta, phi, psi, inc, f, Phi0

        if R is None:
            R = self.params[0]
        if f is None:
            f = self.params[5]

        # Uniform distribution of locations on sky
        mu = npr.uniform(-1.0, 1.0)
        theta = np.arccos(mu)
        phi = npr.uniform(-np.pi, np.pi)

        #Gravitational polarisation varies over 2pi (I think)
        psi = npr.uniform(-np.pi, np.pi)

        #inclination angle needs to be uniform in cos(inc)
        muinc = npr.uniform(-1.0, 1.0)
        inc = np.arccos(muinc)

        #Initial phase over 2pi
        Phi0 = npr.uniform(-np.pi, np.pi)
        
        params = np.array([R, theta, phi, psi, inc, f, Phi0])
        return params


    def timingResidual(self, t, params, pulsar):
        """
        Timing residuals at Earth from common pulsar term Eq(31)
        """

        #Extract source parameters from params tuple
        R, theta, phi, psi, inc, f, Phi0 = params

        #Extract pulsar position from pulsar tuple
        thetaA, phiA = pulsar

        a = 1.0 + np.power(np.cos(inc), 2.0)   #Eq(27a)
        b = -2.0 * np.cos(inc)                 #Eq(27b)

        #Source direction vectors for wave principal axes
        m = np.array([sin(phi) * cos(psi) - sin(psi) * cos(phi) * cos(theta),
                      -cos(phi) * cos(psi) - sin(psi) * sin(phi) * cos(theta),
                      sin(psi) * sin(theta)])
        
        n = np.array([-sin(phi) * sin(psi) - cos(psi) * cos(phi) * cos(theta),
                      cos(phi) * sin(psi) - cos(psi) * sin(phi) * cos(theta),
                      cos(psi) * sin(theta)])

        # Wave propogation direction Omega = m x n
        Omega = np.array([-sin(theta) * cos(phi),
                          -sin(theta) * sin(phi),
                          -cos(theta)])

        #position vector for pulsar
        p = np.array([sin(thetaA) * cos(phiA),
                      sin(thetaA) * sin(phiA),
                      cos(thetaA)])

        #Calculate "antennae beam patterns" for pulsar
        Fplus = self.Fplus(Omega, m, p, n)
        Fcross = self.Fcross(Omega, m, p, n)        

        # Phi from Eq (23)
        # Restrict to case of constant frequency for the moment
        # Note that need to include zero point here (I think)
        Phi = 2.0 * np.pi * f * t

        #calculate timing residual for single pulsar from Eq (31)
        rEarth = a * Fplus * ( np.sin(Phi) - np.sin(Phi0) )
        rEarth -= b * Fcross * ( np.cos(Phi) - np.cos(Phi0) )
        rEarth *= R
        
        return rEarth
    

    def Fplus(self, Omega, m , p, n):
        """ Fplus(Omega) Eq(8a)
        Omega, m, n, p should be direction vectors in cartesian coordinates
        """
        F = np.power(np.dot(m, p), 2.0) - np.power(np.dot(n, p), 2.0)

        #If pulsar lies in direction of source will get divide by zero
        try:
            F /= 2.0 * (1.0 + np.dot(Omega, p))
        except:
            F = 0.0
                
        return F


    def Fcross(self, Omega, m , p, n):
        """ Fcross(Omega) Eq(8b)
        Omega, m, n, p should be direction vectors in cartesian coordinates
        """
        F = np.dot(m, p) * np.dot(n, p)

        #If pulsar lies in direction of source will get divide by zero
        try:
            F /= 1.0 + np.dot(Omega, p)
        except:
            F = 0.0
        
        return F
    

    def R(self, chirpM, f, D):
        """
        GW amplitude from Eq (32)
        chirpM = chirp mass = (m1*m2)^3/5 / (m1 + m2)^1/5 , [chirpM] = mass
        D = luminosity distance to GW source
        f = frequency (taken to be constant)
        """

        # amplitude of GW strain from Eq (26)
        AGW = 2.0 * np.power(chirpM, 5.0/3.0) * np.power(np.pi * f, 2.0/3.0)
        AGW /= D

        #convert from geometric units to dimensionless number
        #multiply by G * Msol / c^2 in units of kpc since D in kpc
        AGW *= 4.78679e-17
        #print "AGW = ", AGW

        #finally get the parameter needed for Fisher matrix Eq (32)
        R = AGW / (2.0 * np.pi * f)
        
        return R


    def innerProduct(self, x, y, S0):
        """
        Inner product of two time series from Eq (43b)

        Assume: time series begins at t=0 and ends at t=T with
        constant time sampling deltat
        """
        
        #For constant time sampling at rate deltat integral is
        #just dot product multipied by time interval
        #Requires time sampling to be small compared to GW period
        #to get sensible results. Differs from Simpson at mildly
        #significant level though, so shouldn't use!
        
        inner = np.dot(x, y) * self.deltat
        
        #More accurate but slower via Simpson's rule
        #inner = scipy.integrate.simps(x * y, dx = self.deltat)

        
        #Then normalise correctly
        inner *= 2.0 / S0

        return inner


    def residualDerivatives(self, t, params, pulsar, steps = None):
        """
        Calculate derivatives of timing residuals. Use a five point stencil
        to get good accuracy on the derivatives
        """

        # Base set of parameters
        R, theta, phi, psi, inc, f, Phi0 = params

        #7 parameters - establish derivative step size first
        # relative steps for dimensional parameters
        # absolute steps for angles
        if steps is None:
            dR = 0.01 * R
            dtheta = 0.01
            dphi = 0.01
            dpsi = 0.01
            dinc = 0.01
            df = 0.001 * f
            dPhi0 = 0.01
        else:
            dR, dtheta, dphi, dpsi, dinc, df, dPhi0 = steps
        
        dparams = (dR, dtheta, dphi, dpsi, dinc, df, dPhi0)

        #Calculate two sided derivatives. Form is same for all
        #so do this as a simple loop.
        derivatives = []
        for i, param in enumerate(params):

            paramsP = params.copy()
            paramsP[i] = paramsP[i] + dparams[i]
            rP = self.timingResidual(t, paramsP, pulsar)

            paramsPP = params.copy()
            paramsPP[i] = paramsPP[i] + 2.0 * dparams[i]
            rPP = self.timingResidual(t, paramsPP, pulsar)

            paramsM = params.copy()
            paramsM[i] = paramsM[i] - dparams[i]
            rM = self.timingResidual(t, paramsM, pulsar)
            
            paramsMM = params.copy()
            paramsMM[i] = paramsMM[i] - 2.0 * dparams[i]
            rMM = self.timingResidual(t, paramsMM, pulsar)

            #Four point derivative
            derivative = -rPP + 8.0 * rP - 8.0 * rM + rMM
            derivative /= 12.0 * dparams[i]

            derivatives.append(derivative)

        return derivatives
    

    def checkDerivatives(self, indx = 0):
        """
        Test behaviour of derivatives
        """

        t = self.t
        params = self.params
        pulsar = self.pulsars[0]
        R, theta, phi, psi, inc, f, Phi0 = params

        factor = 10.0
        for i in range(10):
            factor /= 2.0

            dR = 0.01 * R * factor
            dtheta = 0.01 * factor
            dphi = 0.01 * factor
            dpsi = 0.01 * factor
            dinc = 0.01 * factor
            df = 0.001 * f * factor
            dPhi0 = 0.01 * factor
            steps = (dR, dtheta, dphi, dpsi, dinc, df, dPhi0)

            derivatives = self.residualDerivatives(t, params, pulsar, steps)
            print steps[indx], factor
            print np.sum(derivatives[indx])
        

    def fisherMatrixSingle(self, params, pulsar, SNR):
        """
        Calculate Fisher Matrix for an individual pulsar observed with
        signal-to-nose ratio of SNR
        """

        #Normalise noise on pulsar by SNR per pulsar
        r = self.timingResidual(self.t, params, pulsar)
        S0 = self.innerProduct(r, r, 1.0) / np.power(SNR, 2.0)
        #print "normalisation = ", S0
        #print "check = ", self.innerProduct(r, r, S0)

        # Calculate derivatives of timing residuals with parameters
        derivatives = self.residualDerivatives(self.t, params, pulsar)

        #Fisher Matrix comes from inner product of pairs of derivatives
        nparam = len(params)
        fisher = np.zeros([nparam, nparam])

        for i in range(nparam):
            for j in range(i+1):
                fisher[i][j] = self.innerProduct(
                    derivatives[i], derivatives[j], S0)
                #use symmetry to fill in lower triangle
                fisher[j][i] = fisher[i][j]
                
        return fisher

    def fisherMatrix(self, params = None, pulsars = None, SNR =None):
        """
        Full fisher matrix from many pulsars
        """

        if params is None:
            params = self.params
        if pulsars is None:
            pulsars = self.pulsars
        if SNR is None:
            SNR = self.SNR
        
        #Build Fisher Matrix from all pulsars
        fisher = np.zeros([len(params), len(params)])

        for pulsar in pulsars:
            fisher += self.fisherMatrixSingle(params, pulsar, SNR)

        return fisher
    

    def uniformPulsars(self, N):
        """
        Random realisation of N uniformly distributed pulsars
        """

        pulsars = []
        for i in range(N):
            pulsar = self.randomPulsarLocationUniform()
            pulsars.append(pulsar)

        return pulsars

    def updateNumberPulsars(self, Nnew, SNR = 10.0):
        """
        Add more pulsars to an existing array to get to Nnew
        """

        Nold = len(self.pulsars)

        if Nnew > Nold:
            for i in range(Nnew - Nold):
                pulsar = self.randomPulsarLocationUniform()
                self.pulsars.append(pulsar)
        else:
            print "Nnew > Nold for number of pulsars"

        
        #Normalise noise by SNR per pulsar
        #Currently this normalised to total SNR so have to adjust if
        #you change number of pulsars
        self.SNR = SNR / np.sqrt(Nnew)
        

    def randomPulsarLocationUniform(self):
        """
        Draw a random pulsar location from a uniform distribution
        """
        mu = npr.uniform(-1.0, 1.0)
        phi = npr.uniform(-np.pi, np.pi)
        pulsar = np.array([np.arccos(mu), phi])      

        return pulsar
    

    def reportErrors(self, fisher = None, params = None):
        """
        Output Fisher errors
        """
        if params is None:
            params = self.params
        if fisher is None:
            fisher = self.fisher
            
        ifisher = np.linalg.inv(fisher)
        errors = []
        for i, param in enumerate(params):
            #print self.labels[i], param, np.sqrt(ifisher[i][i])
            errors.append(np.sqrt(ifisher[i][i]))

        #Also calculate solid angle error, which is complicated
        theta = params[1]
        phi = params[2]

        #Need Fisher errors on theta and phi
        dtheta = errors[1]
        dphi = errors[2]

        #Need correlation coefficient for theta-phi
        cThetaPhi = ifisher[1][2] / (dtheta * dphi)

        #Error for dOmega is a little ambiguous. Sesana doesn't
        #agree with the Cutler reference they cite and Sesana
        #formula often involves sqrt of negative number.
        #Sesana Eq (39) expression also seems to be missing dtheta and
        #dphi factors in the second term
        
        #I'll solve this by taking absolute value 
        dOmega = np.power(sin(theta) * dtheta * dphi, 2.0)
        dOmega -= np.power(sin(theta) * cThetaPhi * dtheta * dphi, 2.0)
        dOmega = 2.0 * np.pi * np.sqrt(np.abs(dOmega))

        #om1 = dOmega
        #Cutler formulation Eq(3.7) makes more sense and gives similar values
        #dOmega = (1.0 - cThetaPhi) * sin(theta) * dtheta * dphi
        #dOmega = 2.0 * np.pi * dOmega
        #print "Test:", dOmega, om1, om1/dOmega
        
        #Finally convert from steradians to sq.deg.
        dOmega *= np.power(180.0 / np.pi, 2.0)

        #And add this to errors. Slight concern that could get confused
        #since labels don't match here
        errors.append(dOmega)

        return errors

    def correlationMatrix(self, fisher = None, make_fig = False, save_fig = False):
        """
        Calculate the matrix of correlation coefficients i.e. inverse fisher
        matrix elements normalised by the diagonal elements
        """

        if fisher is None:
            fisher = self.fisher
            
        ifisher = np.linalg.inv(fisher)

        norms = 1.0 / np.sqrt(np.diag(ifisher))

        #Next line is a little odd, but seems to work to
        #calculate (38b) in single step
        correlation_matrix = (ifisher * norms).T * norms

        #Optionally display a visual representation of the correlation matrix
        if make_fig:
            #Black and white chess board plot of correlation coefficients.
            #vmin/vmax fix colour range between -1 and 1
            plt.imshow(correlation_matrix, interpolation='none', cmap = 'gray', vmin= -1, vmax = 1)
            plt.colorbar()
            locs, labels = plt.xticks()
            plt.xticks(np.arange(len(self.labels)),self.labels)
            plt.yticks(np.arange(len(self.labels)),self.labels)

            #Add a little information, although really needs a sky plot to be
            #fully informative
            titlestr = "%i Pulsars and source at (%f, %f)" % (
                len(self.pulsars), self.params[1], self.params[2])
            plt.title(titlestr)

            if save_fig:
                filename = "correlation_matrix.eps"
                plt.savefig(filename)
            else:
                plt.show()
                
            plt.close()

        return correlation_matrix

    def skyplot(self):
        """
        Display pulsars on the sky. Internal coordinate system is (theta, phi)
        which is presumable right ascencion and declination
        """

        data = np.array(self.pulsars)
        Y = data[:, 0]   #theta
        X = data[:, 1]   #phi

        #Need Y between [-pi/2, pi/2], but internally calculated in [0, pi]
        #by the arccos function. Subtract pi/2 to put on spherical coordinates
        # cos(0) is straight up in coordinates, corresponding to 90N on sphere
        Y = Y - np.pi/2.0

        plt.figure()
        ax = plt.subplot(111, projection = 'mollweide')
        ax.scatter(X, Y)
        ax.grid(True)

        #Add GW source
        theta = self.params[1] - np.pi/2
        phi = self.params[2]
        ax.scatter(phi, theta, color = 'r', marker = '^', s=100)

        plt.show()


        
