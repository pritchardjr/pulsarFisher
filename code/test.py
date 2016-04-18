"""
Try to reproduce the Sesana error table
"""

import numpy as np
import pulsarFisher as PF

def testSingle(N = 3, nsources = 1000, PTA = None):
    """
    Run a single realisation of a PTA with nsource randomly located
    GW sources
    """
    if PTA is None:
        PTA = PF.PTA(N=N)
    else:
        PTA.updateNumberPulsars(Nnew = N)

    print len(PTA.pulsars)

    #Monte Carlo over many randomly located GW sources
    errors = []
    for i in range(nsources):
        PTA.resetWithNewSource()
        errors.append(PTA.errors)

    #Get median errors
    errors = np.array(errors)
    labels = list(PTA.labels)
    labels.append("dOmega")
    print errors.shape

    mederrors = []
    for i in range(errors.shape[1]):
        print i
        svec = sorted(errors[:,i])
        mederror = np.median(svec)

        #now get 25% and 75% intervals
        lowindx = int(0.25 * len(svec))
        highindx = int(0.75 * len(svec))
        lowval = svec[lowindx]
        highval = svec[highindx]
        mederrors.append(mederror)
        print labels[i], mederror, np.array([lowval, highval]) - mederror

    return errors, mederrors, PTA


def test():

    nsources = 100
    Ns = [3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]

    filename = "result.log"
    f = open(filename, "w")

    data = []

    # initalise PTA with a single pulsar source
    # will add sources on each loop to match values in Ns
    PTA = PF.PTA(N=1)
    
    for N in Ns:
        errors, mederrors, PTA = testSingle(N, nsources, PTA)
        R = PTA.params[0]
        dRR = mederrors[0] / R
        dOmega = mederrors[7]
        result = (N, dOmega, dRR, mederrors[4], mederrors[3], mederrors[5]/1.0e-10, mederrors[6])

        resultstr = "%i, %f, %f, %f, %f, %f, %f \n" % result
        print resultstr
        f.write(resultstr)

    f.close()
    
        
        


    
