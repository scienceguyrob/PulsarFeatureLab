"""

**************************************************************************
| PFDFeatureExtractor.py                                                 |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Contains feature extraction methods used by only PFD files.            |
| This code runs on python 2.4 or later.                                 |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************
 
"""

# Numpy Imports:
from numpy import where
from numpy import asarray
from numpy import arange
from numpy import sqrt
from numpy import exp
from numpy import pi
from numpy import array
from numpy import corrcoef

import numpy as Num
import numpy.fft as FFT

from scipy.special import i0
from scipy.optimize import leastsq
from scipy import stats  # BWS

import matplotlib.pyplot as plt

# Custom file Imports:
from FeatureExtractor import FeatureExtractor

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

isintorlong = lambda x: type(x) == type(0) or type(x) == type(0L)

class PFDFeatureExtractor(FeatureExtractor):
    """                
    Contains the functions used to generate the scores that describe the key features of
    a pulsar candidate.
    
    """
    
    # ****************************************************************************************************
    #
    # Constructor.
    #
    # ****************************************************************************************************
    
    def __init__(self,debugFlag):
        """
        Default constructor.
        
        Parameters:
        
        debugFlag     -    the debugging flag. If set to True, then detailed
                           debugging messages will be printed to the terminal
                           during execution.
        """
        FeatureExtractor.__init__(self,debugFlag)
        
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Eatough et al., MNRAS 407, 4, 2010.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PFD specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Bates et al., MNRAS 427, 2, 2012.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PFD specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Thornton., PhD Thesis, Univ. Manchester, 2013.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    def getCandidateParameters(self,profile):
        """
        Features 12-15 of those described in Thornton., PhD Thesis, Univ. Manchester, 2013.
        
        Feature 12. The candidate period.
                 
        Feature 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Feature 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Feature 15. The best pulse width.
        
        Parameters:
        profile    -    the PFDFile candidate object NOT profile data.
        
        Returns:
        The candidate period.
        The best signal-to-noise value obtained for the candidate. Higher values desired.
        The best dispersion measure (dm) obtained for the candidate.
        The best pulse width.
        
        """
        
        # Please note that the parameter passed in to this function is actually an
        # instance of the PFDFile class. This is done to keep the code similar for both
        # PFD and PHCX files. However this means we may get confused when we see that
        # the parameter passed in is called profile - this is the PFDFile object. Thus
        # to access the profile we must call profile.profile. I know this may seem
        # confusing, but it is done on purpose to ensure that the code in the PFDFile
        # and PHCXFile scripts is as similar as possible. Despite the fact that these
        # formats are very different.
         
        # Score 12
        self.period = profile.bary_p1 *1000
        
        # Score 13
        avg = profile.profile.mean()
        var = profile.profile.var()
        sigma = sqrt(var)

        # Don't we need to worry about the contribution from the pulse
        # itself here?  - BWS 20140314 - How many iterations...?

        snrprofile = []
        goodbins = 0
        nbin = 0
        sum = 0
        while nbin < len(profile.profile):
            if profile.profile[nbin] > avg - 3 * sigma and profile.profile[nbin] < avg + 3 * sigma:
                snrprofile.append(profile.profile[nbin])
            nbin += 1

        snr_profile = array(snrprofile)

        avg = snr_profile.mean()
        var = snr_profile.var()
        
        self.snr = ((profile.profile-avg)/sqrt(var)).sum()
        if self.snr < 0:
            self.snr = 0.1

        # Score 14
        self.dm = profile.bestdm
    
        # Calculate the width of the pulse BWS 20140316

        peak = profile.profile.argmax() # Finds the index of the largest value across the x-axis.
        xData = array(range(len(profile.profile)))

        # Rotate profile to put it in the centre 
        shift = peak - len(profile.profile) / 2
        rot_profile = self.fft_rotate(profile.profile,shift) - min(profile.profile)
        
        # Determine the pulse width, assume that it can be gotten by finding extrema
        # of the Half maximum points. 
        peak = rot_profile.argmax()
        #print "Peak:" + str(peak)
        halfmax_profile = max(rot_profile) / 2
        left_lim = peak
        while left_lim > 0:
            #print rot_profile[left_lim], halfmax_profile
            if rot_profile[left_lim] < halfmax_profile:
                break
            else:
                left_lim -= 1
        #print "LL:" + str(left_lim)
        right_lim = peak
        while right_lim < len(rot_profile):
            #print rot_profile[right_lim], halfmax_profile
            if rot_profile[right_lim] < halfmax_profile:
                break
            else:
                right_lim += 1
        #print "RL:" + str(right_lim)
        
        if(self.debug):
            plt.plot(xData,rot_profile,left_lim,rot_profile[left_lim], 'o',right_lim,rot_profile[right_lim],'o',peak,halfmax_profile,'o')
            plt.show()

        self.width = (1.0 * (right_lim - left_lim - 1.0)) / len(rot_profile);

        return [self.period,self.snr,self.dm,self.width]
        
    
    # ****************************************************************************************************
    #
    # DM Curve Fittings
    #
    # ****************************************************************************************************
        
    def getDMFittings(self,data):
        """
        Features 1-4 of those described in Thornton., PhD Thesis, Univ. Manchester, 2013.
        
        Computes the dispersion measure curve fitting parameters. There are four scores computed:
        
        Feature 16. This feature computes SNR / SQRT( (P-W) / W ).
                 
        Feature 17. Difference between fitting factor Prop, and 1. If the candidate is a pulsar,
                  then prop should be equal to 1.
        
        Feature 18. Difference between best DM value and optimised DM value from fit. This difference
                  should be small for a legitimate pulsar signal. 
                 
        Feature 19. Chi squared value from DM curve fit, smaller values indicate a smaller fit. Thus
                  smaller values will be possessed by legitimate signals.
        
        Parameters:
        rawData    -    the raw candidate xml data.
        profile    -    the profile data.
        
        Returns:
        SNR / SQRT( (P-W) / W ).
        Difference between fitting factor Prop, and 1.
        Difference between best DM value and optimized DM value from fit.
        Chi squared value from DM curve fit, smaller values indicate a smaller fit.
        
        """
        
        # Calculates the residuals.
        def __residuals(paras, x, y):     
            Amp,Prop,Shift,Up = paras
            weff = sqrt(wint + pow(Prop*kdm*abs((self.dm + Shift)-x)*df/pow(f,3),2))
            for wind in range(len(weff)):
                if ( weff[wind] > self.period ):
                    weff[wind] = self.period
            SNR  = Up+Amp*sqrt((self.period-weff)/weff)
            err  = y - SNR
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            Amp,Prop,Shift,Up = paras
            weff = sqrt(wint + pow(Prop*kdm*abs((self.dm + Shift)-x)*df/pow(f,3),2))
            for wind in range(len(weff)):
                if ( weff[wind] > self.period ):
                    weff[wind] = self.period
            SNR  = Up+Amp*sqrt((self.period-weff)/weff)
            return SNR
        
        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values,dm_index = data.plot_chi2_vs_DM(lodm, hidm)
            
        # Extract DM curve.
        curve=[]
        curve.append(y_values)
        curve.append(range(len(y_values)))
            
        yData = curve[0]
        yData = 255./max(yData)*yData
        length_all = len(y_values)
        length = len(yData)
                    
        # Get start and end DM value and calculate step width.
        dm_start,dm_end = float(dm_index[1]),float(dm_index[len(dm_index)-1])
        dm_step = abs(dm_start-dm_end)/length_all
        
        # SNR and pulse parameters.
        wint = (self.width * self.period)**2
        kdm = 8.3*10**6
        df = 32
        f = 135
        
        peak = self.snr/sqrt((self.period-sqrt(wint))/sqrt(wint))
        
        # Scale x-data.
        xData = []
        for i in range(length):
            xData.append(dm_start+curve[1][i]*dm_step)    
        xData = array(xData)
        
        # Calculate theoretic dm-curve from best values.
        _help = []
        for i in range(length):
            weff = sqrt(wint + pow(kdm*abs(self.dm-xData[i])*df/pow(f,3),2))
            if weff > self.period:
                weff = self.period
            SNR = sqrt((self.period-weff)/weff)
            _help.append(float(SNR))
            
        theo = (255./max(_help))*array(_help)
        
        # Start parameter for fit.
        Amp = (255./max(_help))
        Prop,Shift  = 1,0
        p0 = (Amp,Prop,Shift,0)
        plsq = leastsq(__residuals, p0, args=(xData,yData))
        fit = __evaluate(xData, plsq[0])

        if(self.debug):
            plt.plot(xData,fit,xData,yData,xData,theo)
            plt.title("DM Curve, theoretical curve and fit.")
            plt.show()
            
        # Chi square calculation.
        chi_fit,chi_theo = 0,0
        ndeg = 0
        for i in range(length):
            if theo[i] > 0:
                chi_fit  += (yData[i]-fit[i])**2  / fit[i]
                chi_theo += (yData[i]-theo[i])**2 / theo[i]
                ndeg += 1
                
        chi_fit  =  chi_fit/ndeg
        chi_theo = chi_theo/ndeg
        
        #print "CHISQ: " + str(chi_fit) + " " + str(chi_theo)

        diffBetweenFittingFactor = abs(1-plsq[0][1])
        diffBetweenBestAndOptimisedDM = plsq[0][2]
        return peak, diffBetweenFittingFactor, diffBetweenBestAndOptimisedDM , chi_theo
    
    def getDMCurveData(self,data):
        """
        Extracts the DM curve data from the PFD file.
        
        """
        
        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values,dm_index = data.plot_chi2_vs_DM(lodm, hidm)
        
        return y_values
    
    def getDMCurveDataNormalised(self,data):
        """
        Extracts the DM curve data from the PFD file.
        
        """
        
        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values,dm_index = data.plot_chi2_vs_DM(lodm, hidm)
            
        # Extract DM curve.
        curve=[]
        curve.append(y_values)
        curve.append(range(len(y_values)))
            
        yData = curve[0]
        yData = 255./max(yData)*yData
        
        return yData

    def getDMSNRCurveData(self, data):
        """
        Extracts the DM curve data from the PFD file.

        """

        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values, dm_index = data.plot_SNR_vs_DM(lodm, hidm)

        return (y_values, dm_index)
    
    # ****************************************************************************************************
    #
    # Sub-band scores
    #
    # ****************************************************************************************************
    
    def getSubbandParameters(self,data=None,profile=None):
        """
        Features 20-22 of those described in Thornton., PhD Thesis, Univ. Manchester, 2013.
        
        Computes the sub-band features. There are three computed:
        
        Feature 20. RMS of peak positions in all sub-bands. Smaller values should be possessed by
                  legitimate pulsar signals.
                 
        Feature 21. Average correlation coefficient for each pair of sub-bands. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Feature 22. Sum of correlation coefficients between sub-bands and profile. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Parameters:
        data       -    the raw candidate data.
        profile    -    a numpy.ndarray containing profile data.
        
        Returns:
        RMS of peak positions in all sub-bands.
        Average correlation coefficient for each pair of sub-bands.
        Sum of correlation coefficients between sub-bands and profile.
        
        """
        
        if(data==None and profile==None):
            return [0.0,0.0,0.0]
        
        # First, sub-bands.
        subbands = data.plot_subbands()
        prof_bins = data.proflen
        band_subbands = data.nsub
        
        RMS,mean_corr = self.getSubband_features(subbands, prof_bins, band_subbands, self.width)
        correlation = self.getProfileCorr(subbands, band_subbands, profile)
        
        # Now calculate integral of correlation coefficients.
        correlation_integral = 0
        for i in range( len( correlation ) ):
            correlation_integral += correlation[i]
                    
        return [RMS,mean_corr,correlation_integral]
    
    # ******************************************************************************************
 
    def getProfileCorr(self,subbands, band_subbands, profile):
        """
        Calculates the correlation of the profile with the subbands, -integrals.
        
        Parameters:
        subbands         -    the sub-band data.
        band_subbands    -    the number of sub-bands.
        bestWidth        -    the best pulse width.
        
        Returns:
        
        A list with the correlation data in decimal format.            
        
        """
        
        corrlist = []
        for j in range(band_subbands):
            coef = abs(corrcoef(subbands[j],profile))
            if coef[0][1] > 0.0055:
                corrlist.append(coef[0][1])
        
        return array(corrlist)

    # ******************************************************************************************

    def subband_correlation(self,subbands,profile):
        """
        Calculates the correlation coefficient of each subband and the pulse profile.

        Parameters:
        subbands - the subband data
        profile - the pulse profile

        Returns:

        A list of correlation coefficient of each subband and the pulse profile
        """
        corrlist1 = []
        maxmin1 = []


        # Scrunch subband from to 36, 32, 40 or 30 depending on data in order of preference
        # I hope there are no data with less than 30 subbands
        scrunchsubbands = []
        noofsubbands = [36,32,40,30]
        remainder = []
        for i in range(len(noofsubbands)):
            remainder.append(len(subbands)%noofsubbands[i])

        scrunchfactor=len(subbands)/noofsubbands[Num.argmin(remainder)]
        for i in range(noofsubbands[Num.argmin(remainder)]):
            scrunchsubbands.append(Num.sum(subbands[(i*scrunchfactor):(i*scrunchfactor)+scrunchfactor],axis=0))
        subbands = scrunchsubbands
        
        #defunct LOTAAS exclusive scrunching
        #if len(subbands) == 288:
        #    for j in range(36):
        #        scrunchsubbands.append(
        #            subbands[(j * 8) + 0] + subbands[(j * 8) + 1] + subbands[(j * 8) + 2] + subbands[(j * 8) + 3] +
        #            subbands[(j * 8) + 4] + subbands[(j * 8) + 5] + subbands[(j * 8) + 6] + subbands[(j * 8) + 7])
        #    subbands = scrunchsubbands

        # remove empty subbands and calculate the coefficients
        for j in range(len(subbands)):
            maxmin1.append(max(subbands[j]) - min(subbands[j]))

        maxchange1 = max(maxmin1)

        for j in range(len(subbands)):
            if max(subbands[j]) - min(subbands[j]) > 0.05 * maxchange1:
                coef1 = (corrcoef(subbands[j], profile))
                corrlist1.append(coef1[0][1])

        return corrlist1

    # ******************************************************************************************

    def subint_correlation(self,subints,profile):
        """
        Calculates the correlation coefficient of each subint and the pulse profile.

        Parameters:
        subints - the subint data
        profile - the pulse profile

        Returns:

        A list of correlation coefficient of each subint and the pulse profile
        """
        corrlist2 = []
        maxmin2 = []

        # scrunch subints to 36, 32, 40 or 30 in order of preference for more than 30 subints

        if len(subints) >= 30:
            scrunchsubints = []
            
            noofsubints = [36,32,40,30]
            remainder = []
            for i in range(len(noofsubints)):
                remainder.append(len(subints)%noofsubints[i])
            scrunchfactor=len(subints)/noofsubints[Num.argmin(remainder)]
  
            for j in range(noofsubints[Num.argmin(remainder)]):
                scrunchsubints.append(Num.sum(subints[(j*scrunchfactor):(j*scrunchfactor)+scrunchfactor],axis=0))
        
            subints = scrunchsubints

        # remove empty subints
        for j in range(len(subints)):
            maxmin2.append(max(subints[j]) - min(subints[j]))

        maxchange2 = max(maxmin2)

        for j in range(len(subints)):
            if max(subints[j]) - min(subints[j]) > 0.05 * maxchange2:
                coef2 = (corrcoef(subints[j], profile))
                corrlist2.append(coef2[0][1])

        return corrlist2

    # ****************************************************************************************************
    #
    # Other Utility Functions
    #
    # ****************************************************************************************************
    
    def delay_from_DM(self,DM, freq_emitted):
        """
        Return the delay in seconds caused by dispersion, given
        a Dispersion Measure (DM) in cm-3 pc, and the emitted
        frequency (freq_emitted) of the pulsar in MHz.
        """
        if (type(freq_emitted)==type(0.0)):
            if (freq_emitted > 0.0):
                return DM/(0.000241*freq_emitted*freq_emitted)
            else:
                return 0.0
        else:
            return where(freq_emitted > 0.0,DM/(0.000241*freq_emitted*freq_emitted), 0.0)
        
    # ****************************************************************************************************
    
    def fft_rotate(self,arr, bins):
        """
        Return array 'arr' rotated by 'bins' places to the left.  The
        rotation is done in the Fourier domain using the Shift Theorem.
        'bins' can be fractional.  The resulting vector will have the
        same length as the original.
        """
        arr = asarray(arr)
        freqs = arange(arr.size/2+1, dtype=Num.float)
        phasor = exp(complex(0.0, (2.0*pi)) * freqs * bins / float(arr.size))
        return Num.fft.irfft(phasor * Num.fft.rfft(arr))

    # ****************************************************************************************************

    def delay_from_foffsets(self, df, dfd, dfdd, times):
        """
        Return the delays in phase caused by offsets in
        frequency (df), and two frequency derivatives (dfd, dfdd)
        at the given times in seconds.
        """
        f_delays = df * times
        fd_delays = dfd * times ** 2 / 2.0
        fdd_delays = dfdd * times ** 3 / 6.0
        return (f_delays + fd_delays + fdd_delays)
    
    # ****************************************************************************************************
    
    def span(self,Min, Max, Number):
        """
        span(Min, Max, Number):
        Create a range of 'Num' floats given inclusive 'Min' and 'Max' values.
        """
        assert isintorlong(Number)
        if isintorlong(Min) and isintorlong(Max) and (Max-Min) % (Number-1) != 0:
            Max = float(Max) # force floating points
        
        return Min+(Max-Min)*Num.arange(Number)/(Number-1)
        
    # ****************************************************************************************************
    
    def rotate(self,arr, bins):
        """
        Return an array rotated by 'bins' places to the left
        """
        bins = bins % len(arr)
        if bins==0:
            return arr
        else:
            return Num.concatenate((arr[int(bins):], arr[:int(bins)]))
    
    # ****************************************************************************************************
    
    def interp_rotate(self,arr, bins, zoomfact=10):
        """
        Return a sinc-interpolated array rotated by 'bins' places to the left.
        'bins' can be fractional and will be rounded to the closest
        whole-number of interpolated bins.  The resulting vector will
        have the same length as the oiginal.
        """
        newlen = len(arr)*zoomfact
        rotbins = int(Num.floor(bins*zoomfact+0.5)) % newlen
        newarr = self.periodic_interp(arr, zoomfact)
        return self.rotate(newarr, rotbins)[::zoomfact]

    # ****************************************************************************************************
    
    def periodic_interp(self,data, zoomfact, window='hanning', alpha=6.0):
        """
        Return a periodic, windowed, sinc-interpolation of the data which
        is oversampled by a factor of 'zoomfact'.
        """
        zoomfact = int(zoomfact)
        if (zoomfact < 1):
            #print "zoomfact must be >= 1."
            return 0.0
        elif zoomfact==1:
            return data
        
        newN = len(data)*zoomfact
        # Space out the data
        comb = Num.zeros((zoomfact, len(data)), dtype='d')
        comb[0] += data
        comb = Num.reshape(Num.transpose(comb), (newN,))
        # Compute the offsets
        xs = Num.zeros(newN, dtype='d')
        xs[:newN/2+1] = Num.arange(newN/2+1, dtype='d')/zoomfact
        xs[-newN/2:]  = xs[::-1][newN/2-1:-1]
        # Calculate the sinc times window for the kernel
        if window.lower()=="kaiser":
            win = _window_function[window](xs, len(data)/2, alpha)
        else:
            win = _window_function[window](xs, len(data)/2)
        kernel = win * self.sinc(xs)
        
        if (0):
            print "would have plotted."
        return FFT.irfft(FFT.rfft(kernel) * FFT.rfft(comb))
    
    # ****************************************************************************************************
    
    def sinc(self,xs):
        """
        Return the sinc function [i.e. sin(pi * xs)/(pi * xs)] for the values xs.
        """
        pxs = Num.pi*xs
        return Num.where(Num.fabs(pxs)<1e-3, 1.0-pxs*pxs/6.0, Num.sin(pxs)/pxs)
    
    # ****************************************************************************************************
    
    # The code below is a little bit of a mess. But There was little I could do to
    # clean in up, since this is PRESTO code being retro-fitted to work for our purposes.
    
def kaiser_window(xs, halfwidth, alpha):
    """
        Return the kaiser window function for the values 'xs' when the
            the half-width of the window should be 'haldwidth' with
            the folloff parameter 'alpha'.  The following values are
            particularly interesting:

            alpha
            -----
            0           Rectangular Window
            5           Similar to Hamming window
            6           Similar to Hanning window
            8.6         Almost identical to the Blackman window 
    """
    win = i0(alpha*Num.sqrt(1.0-(xs/halfwidth)**2.0))/i0(alpha)
    return Num.where(Num.fabs(xs)<=halfwidth, win, 0.0)

def hanning_window(xs, halfwidth):
    """
    hanning_window(xs, halfwidth):
        Return the Hanning window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    win =  0.5 + 0.5*Num.cos(Num.pi*xs/halfwidth)
    return Num.where(Num.fabs(xs)<=halfwidth, win, 0.0)

def hamming_window(xs, halfwidth):
    """
    hamming_window(xs, halfwidth):
        Return the Hamming window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    win =  0.54 + 0.46*Num.cos(Num.pi*xs/halfwidth)
    return Num.where(Num.fabs(xs)<=halfwidth, win, 0.0)

def blackman_window(xs, halfwidth):
    """
    blackman_window(xs, halfwidth):
        Return the Blackman window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    rat = Num.pi*xs/halfwidth
    win =  0.42 + 0.5*Num.cos(rat) + 0.08*Num.cos(2.0*rat) 
    return Num.where(Num.fabs(xs)<=halfwidth, win, 0.0)

def rectangular_window(xs, halfwidth):
    """
    rectangular_window(xs, halfwidth):
        Return a rectangular window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    return Num.where(Num.fabs(xs)<=halfwidth, 1.0, 0.0)

_window_function = {"rectangular": rectangular_window,
                    "none": rectangular_window,
                    "hanning": hanning_window,
                    "hamming": hamming_window,
                    "blackman": blackman_window,
                    "kaiser": kaiser_window}

    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Lee et al., MNRAS 433, 1, 2013.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PFD specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Morello et al., MNRAS 443, 2, 2014. 
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PFD specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
        
