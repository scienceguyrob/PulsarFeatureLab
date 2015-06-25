"""

**************************************************************************
| PHCXFeatureExtractor.py                                                |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Contains feature extraction methods used by only PHCX files.           |
| This code runs on python 2.4 or later.                                 |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************
 
"""

# Numpy Imports:
from numpy import array
from numpy import corrcoef
from numpy import sqrt
from numpy import mean
from numpy import std

from scipy.optimize import leastsq
from scipy.optimize import minimize

import matplotlib.pyplot as plt 

# Custom file Imports:
from FeatureExtractor import FeatureExtractor

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class PHCXFeatureExtractor(FeatureExtractor):
    """                
    Contains the functions used to generate the features that describe the key features of
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
    
    # Please add PHCX specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Bates et al., MNRAS 427, 2, 2012.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PHCX specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Thornton., PhD Thesis, Univ. Manchester, 2013.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    def getCandidateParameters(self,profile,section):
        """
        Features 12-15 of those described in Thornton., PhD Thesis, Univ. Manchester, 2013.
        
        Feature 12. The candidate period.
                 
        Feature 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Feature 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Feature 15. The best pulse width.
        
        Parameters:
        profile    -    the raw xml candidate data.
        section    -    A phcx file specific variable, used to identify the section of the xml data to read.
                        Value should be 1 for standard PHCX and 0 for SUPERB PHCX files.
        
        Returns:
        The candidate period.
        The best signal-to-noise value obtained for the candidate. Higher values desired.
        The best dispersion measure (dm) obtained for the candidate.
        The best pulse width.
        
        """
        
        self.snr = float(profile.getElementsByTagName('Snr')[section].childNodes[0].data)
        self.dm = float(profile.getElementsByTagName('Dm')[section].childNodes[0].data)
        self.period = float(profile.getElementsByTagName('BaryPeriod')[section].childNodes[0].data) * 1000
        self.width = float(profile.getElementsByTagName('Width')[section].childNodes[0].data)
        
        return [self.period,self.snr,self.dm,self.width]
        
    
    # ****************************************************************************************************
    #
    # DM Curve Fittings
    #
    # ****************************************************************************************************
    
    def getDMFittings(self,data,section):
        """
        Features 16-19 of those described in Thornton., PhD Thesis, Univ. Manchester, 2013.
        
        Computes the dispersion measure curve fitting parameters. There are four computed:
        
        Feature 16. This feature computes SNR / SQRT( (P-W) / W ).
                 
        Feature 17. Difference between fitting factor Prop, and 1. If the candidate is a pulsar,
                  then prop should be equal to 1.
        
        Feature 18. Difference between best DM value and optimised DM value from fit. This difference
                  should be small for a legitimate pulsar signal. 
                 
        Feature 19. Chi squared value from DM curve fit, smaller values indicate a smaller fit. Thus
                  smaller values will be possessed by legitimate signals.
        
        Parameters:
        data       -    the raw candidate xml data.
        section    -    A phcx file specific variable, used to identify the section of the xml data to read.
                        Value should be 1 for standard PHCX and 0 for SUPERB PHCX files.
        
        Returns:
        SNR / SQRT( (P-W) / W ).
        Difference between fitting factor Prop, and 1.
        Difference between best DM value and optimised DM value from fit.
        Chi squared value from DM curve fit, smaller values indicate a smaller fit.
        
        """
        
        # Calculates the residuals.
        def __residuals(paras, x, y):     
            Amp,Prop,Shift = paras
            weff = sqrt(wint + pow(Prop*kdm*abs((self.dm + Shift)-x)*df/pow(f,3),2))
            SNR  = Amp*sqrt((self.period-weff)/weff)
            err  = y - SNR
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            Amp,Prop,Shift = paras
            weff = sqrt(wint + pow(Prop*kdm*abs((self.dm + Shift)-x)*df/pow(f,3),2))
            SNR  = Amp*sqrt((self.period-weff)/weff)
            return SNR
        
        # Extract DM curve.
        dm_curve_all = array(self.getDM_FFT(data,section))
        curve = self.dm_curve(dm_curve_all)
        yData = curve[0]
        length_all = len(dm_curve_all)
        length = len(yData)
        
        # Extract x-scale for DM curve.
        read_data = list(data.getElementsByTagName('DmIndex')[section].childNodes[0].data)
        dm_index,temp = [],''
        for i in range(len(read_data)):
            if (read_data[i] != "\n"):
                temp += (read_data[i])
            else:
                dm_index.append(temp)
                temp = ''
                
        # Get start and end DM value and calculate step width.
        dm_start,dm_end = float(dm_index[1]),float(dm_index[len(dm_index)-1])
        dm_step = abs(dm_start-dm_end)/length_all
        
        # SNR and pulse parameters.
        wint = (self.width * self.period)**2
        kdm = 8.3*10**6
        df = 400
        f = 1374
        
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
            SNR = sqrt((self.period-weff)/weff)
            _help.append(float(SNR))
            
        theo = (255./max(_help))*array(_help)
        
        # Start parameter for fit.
        Amp = (255./max(_help))
        Prop,Shift  = 1,0
        p0 = (Amp,Prop,Shift)
        plsq = leastsq(__residuals, p0, args=(xData,yData))
        fit = __evaluate(xData, plsq[0])
        
        if(self.debug):
            plt.plot(xData,fit,xData,yData,xData,theo)
            plt.title("DM Curve, theoretical curve and fit.")
            plt.legend( ('Fit to DM', 'DM', 'Theoretical') )
            plt.show()
            
        # Chi square calculation.
        chi_fit,chi_theo = 0,0
        for i in range(length):
            if fit[i] >= 1.:
                chi_fit  += (yData[i]-fit[i])**2
                chi_theo += (yData[i]-theo[i])**2
                
        chi_fit  =  chi_fit/length
        chi_theo = chi_theo/length
        
        diffBetweenFittingFactor = abs(1-plsq[0][1])
        diffBetweenBestAndOptimisedDM = plsq[0][2]
        return peak, diffBetweenFittingFactor, diffBetweenBestAndOptimisedDM , chi_theo
    
    # ******************************************************************************************
    
    def dm_curve(self,data):
        """
        Extracts the DM curve from the DM data block in the phcx file.
        
        Parameters:
        data    -    a numpy.ndarray containing the DM data.
        
        Returns:
        
        An array describing the curve.
        
        """
        
        result,x,temp = [],[],[]
        for i in range(len(data)):
            if (i+1)%128 == 0:
                result.append(max(temp))
                x.append(i - 128)
                temp = []
            else:
                temp.append(data[i])
        
        return array(result),array(x)
        
    # ******************************************************************************************
    
    def getDM_FFT(self,xmldata,section):
        """
        Extracts the DM curve from the DM data block in the phcx file.
        
        Parameters:
        xmldata    -    a numpy.ndarray containing the DM data in decimal format.
        section    -    the section of the xml file to find the DM data within. This
                        is required sine there are two DM sections in the phcx file.
        
        Returns:
        
        An array containing the DM curve data in decimal format.
        
        """
        
        # Extract data.
        dec_value = []
        block = xmldata.getElementsByTagName('DataBlock') # gets all of the bits with the title 'section'.
        points = block[section].childNodes[0].data
        
        # Transform data from hexadecimal to decimal values.
        x,y=0,0
        while x < len(points):
            if points[x] != "\n":
                try:
                    hex_value = points[x:x+2]
                    dec_value.append(int(hex_value,16)) # now the profile (shape, unscaled) is stored in dec_value
                    x = x+2
                    y = y+1
                except ValueError:
                    break
            else:
                x = x+1
                
        return dec_value
 
    # ****************************************************************************************************
    #
    # Sub-band features
    #
    # ****************************************************************************************************
    
    def getSubbandParameters(self,section,data=None,profile=None):
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
        section    -    A phcx file specific variable, used to identify the section of the xml data to read.
                        Value should be 1 for standard PHCX and 0 for SUPERB PHCX files.
                        
        Returns:
        RMS of peak positions in all sub-bands.
        Average correlation coefficient for each pair of sub-bands.
        Sum of correlation coefficients between sub-bands and profile.
        
        """
        
        if(data==None and profile==None):
            return [0.0,0.0,0.0]
        
        block_bands = data.getElementsByTagName("SubBands")
        frequency = block_bands[section].childNodes[0].data
        prof_bins = int(block_bands[section].getAttribute("nBins"))
        band_subbands = int(block_bands[section].getAttribute("nSub"))
        subbands = self.hexToDec(frequency, band_subbands, prof_bins)
        bestWidth = float(data.getElementsByTagName('Width')[section].childNodes[0].data)
        
        RMS,mean_corr = self.getSubband_features(subbands, prof_bins, band_subbands, bestWidth)
        correlation = self.getProfileCorr(data,profile,"Bands",section)
        
        # Now calculate integral of correlation coefficients.
        correlation_integral = 0
        for i in range( len( correlation ) ):
            correlation_integral += correlation[i]
                    
        return [RMS,mean_corr,correlation_integral]
    
    # ******************************************************************************************
 
    def hexToDec(self,listData,nsub,nbin):
        """
        Converts hexadecimal data to decimal data.
        
        Parameters:
        list    -    a numpy.ndarray containing the DM data in hexadecimal format.
        nsub    -    number of sub-bands.
        nbin    -    number of bins.
        
        Returns:
        
        A list with the data in decimal format.            
        
        """
        x,y = 0,0
        newlist = []
        while x < len(listData):
            if listData[x] != "\n":
                try:
                    hexValue = listData[x:x+2]
                    newlist.append(int(hexValue,16))
                    x += 2
                    y += 1
                except ValueError:
                    break
            else:
                x += 1
                
        a = array(newlist).reshape(nsub,nbin)
        
        return a
    
    # ******************************************************************************************
 
    def getProfileCorr(self,xmldata,p,pattern,section):
        """
        Calculates the correlation of the profile with the subbands, -integrals.
        
        Parameters:
        xmldata    -    a numpy.ndarray containing the DM data in hexadecimal format.
        p          -    the profile data.
        pattern    -    the section of the phcx xml file to look in, should be 'Bands'.
        section    -    A phcx file specific variable, used to identify the section of the xml data to read.
                        Value should be 1 for standard PHCX and 0 for SUPERB PHCX files.
        Returns:
        
        A list with the correlation data in decimal format.            
        
        """
        
        block_bands = xmldata.getElementsByTagName('Sub'+pattern)
        frequency = block_bands[section].childNodes[0].data
        nbin_bands = int(block_bands[section].getAttribute("nBins"))
        nsub_bands = int(block_bands[section].getAttribute("nSub"))
        allbands = self.hexToDec(frequency, nsub_bands, nbin_bands)
        
        corrlist = []
        for j in range(nsub_bands):
            coef = abs(corrcoef(allbands[j],p))
            if coef[0][1] > 0.0055:
                corrlist.append(coef[0][1])
                
        return array(corrlist)
    
    # ******************************************************************************************
          
    def getSubbandData(self,xmldata,profileIndex):
        """
        Returns sub-band data.
        
        Parameters:
        xmldata    -    the xml data read in from the phcx file.
        profileIndex    -    index of the <Profile/> tag to read in the xml data.
        
        Returns:
        A list data type containing 128 integer data points.
        """
        
        block_bands = xmldata.getElementsByTagName('SubBands')
        frequency = block_bands[1].childNodes[0].data
        nbin_bands = int(block_bands[1].getAttribute("nBins"))
        nsub_bands = int(block_bands[1].getAttribute("nSub"))
        allbands = self.hexToDec(frequency, nsub_bands, nbin_bands)
        
        # OK so the allbands variable contains data of size 16x128
        data = array(allbands)
        # Post processing of data, convert to a single vector of data.
        sum_= [0] * 128 # empty array.
        for i in range(0, len(data)):
            
            #print "ROW:",i,"\n"
            row = array(data[i])
            #print row
            
            mean_ = mean(row)
            
            #print "Mean: ", mean_
            row = row - mean_
            #print "normalised row:\n",row
            sum_+=row
            
        #print "Summed data...\n"
        for i in range(0, len(sum_)):
            if sum_[i]<0:
                sum_[i]=0.0
                
        #print sum        
        #print "Returning sub-band data..."    
        return sum_
    
    # ******************************************************************************************
          
    def getSubintData(self,xmldata,profileIndex):
        """
        Returns sub integration data.
        
        Parameters:
        xmldata    -    the xml data read in from the phcx file.
        profileIndex    -    index of the <Profile/> tag to read in the xml data.
        
        Returns:
        A list data type containing 128 integer data points.
        """
        block_bands = xmldata.getElementsByTagName('SubIntegrations')
        frequency = block_bands[1].childNodes[0].data
        nbin_bands = int(block_bands[1].getAttribute("nBins"))
        nsub_bands = int(block_bands[1].getAttribute("nSub"))
        allInts = self.hexToDec(frequency, nsub_bands, nbin_bands)
        
        # OK so the allbands variable contains data of size 32x128
        data = array(allInts)
        # Post processing of data, convert to a single vector of data.
        sum_= [0] * 128 # empty array.
        for i in range(0, len(data)):
            
            #print "ROW:",i,"\n"
            row = array(data[i])
            #print row
            
            mean_ = mean(row)
            
            #print "Mean: ", mean_
            row = row - mean_
            #print "normalised row:\n",row
            sum_+=row
            
        #print "Summed data...\n"
        for i in range(0, len(sum_)):
            if sum_[i]<0:
                sum_[i]=0.0
                
        #print sum        
        #print "Returning sub-band data..."    
        return sum_
    
    # ****************************************************************************************************
    
    def getDMPlaneCurveData(self,data,section):
        """
        Gets the DM curve data.
        
        Returns:
        A float array containing DM curve data
        
        """
        
        # Extract DM curve.
        dm_curve_all = array(self.getDM_FFT(data,section))
        curve = self.dm_curve(dm_curve_all)
        yData = curve[0]
        return yData
    
    def getDMCurveData(self,data,section):
        """
        Gets the DM curve data.
        
        Returns:
        A float array containing DM curve data
        
        """
        
        # Extract DM curve.
        dm_curve_all = array(self.getDM_FFT(data,section))
        return dm_curve_all
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Lee et al., MNRAS 433, 1, 2013.
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PHCX specific feature extraction code for this work as appropriate.
    
    # ****************************************************************************************************
    #
    # Feature Extraction functions --> Morello et al., MNRAS 443, 2, 2014. 
    #           |          |          |          |          |          |           |
    #           v          v          v          v          v          v           v
    # ****************************************************************************************************
    
    # Please add PHCX specific feature extraction code for this work as appropriate.
    
    # ******************************************************************************************
    
    # ADDING A NEW FEATURE? Add the code that computes and extracts the new features here.
    # Put feature code here that works ONLY for PHCX files.