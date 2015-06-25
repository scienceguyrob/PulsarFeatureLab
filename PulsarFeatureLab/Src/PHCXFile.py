"""

**************************************************************************
| PHCXFile.py                                                            |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Represents an individual PHCX candidate. This code runs on python      |
| 2.4 or later.                                                          |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************
 
"""

# Numpy/Scipy Imports:
from numpy import array
from numpy import std
from numpy import mean

# Standard library Imports:
import gzip,sys

# XML processing Imports:
from xml.dom import minidom

# Custom file Imports:
import Utilities
from PHCXFeatureExtractor import PHCXFeatureExtractor

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class PHCX(Utilities.Utilities):
    """                
    Represents an individual pulsar candidate.
    
    """
    
    # ****************************************************************************************************
    #
    # Constructor.
    #
    # ****************************************************************************************************
    
    def __init__(self,debugFlag,candidateName,xmlProfileIndex):
        """
        Default constructor.
        
        Parameters:
        
        debugFlag     -    the debugging flag. If set to True, then detailed
                           debugging messages will be printed to the terminal
                           during execution.
        candidateName -    the name for the candidate, typically the file path.
        """
        Utilities.Utilities.__init__(self,debugFlag)
        self.fe = PHCXFeatureExtractor(self.debug)
        self.epsilon = 0.000005 # Used during feature comparison.
         
        self.cand = candidateName           # The name of the candidate.
        self.profileIndex = xmlProfileIndex # A phcx file specific variable, used to identify the section of the xml data to read.
        self.profile      = []              # The decimal profile data.
        self.rawdata      = []              # The raw data read in from the file, in this case xml.
        self.features     = []
        self.load()

    # ****************************************************************************************************
           
    def load(self):
        """
        Attempts to load candidate profile data from the file, performs file consistency checks if the
        debug flag is set to true.
        
        Parameters:
        N/A
        
        Return:
        N/A
        """
        
        # Read data directly from phcx file.
        if(".gz" in self.cand):
            data = gzip.open(self.cand,'rb')
        else:
            data = infile = open(self.cand, "rb")
                
        self.rawdata = minidom.parse(data) # strip off xml data
        data.close()
            
        # Explicit debugging required.
        if(self.debug):
            
            # If candidate file is invalid in some way...
            if(self.isValid()==False):
                
                print "Invalid PHCX candidate: ",self.cand
                raise Exception("Invalid PHCX candidate: PHCXFile.py (Line 106).")
            
            # Candidate file is valid.
            else:
                print "Candidate file valid."
                # Extracts data from this part of a candidate file. It contains details
                # of the profile in hexadecimal format. The data is extracted from the part
                # of the candidate file which resembles:
                # <Profile nBins='128' format='02X' min='-0.000310' max='0.000519'>
                #
                # Call to ph.getprofile() below will return a LIST data type of 128 integer data points.
                # Phcx files actually contain two profile sections (i.e. there are two <Profile>...</Profile> 
                # sections in the file) which can be read using the XML dom code by specifying the index of the
                # profile section to use. The first section profileIndex = 0 pertains to a profile obtained after the FFT,
                # the second, profileIndex = 1, to a profile that has been period and DM searched using PDMPD. We choose 1 here
                # as it should have a better SNR .... maybe.
                self.profile = array(self.getprofile())
                         
        # Just go directly to feature generation without checks.
        else:
            self.out( "Candidate validity checks skipped.","")
            # See comment above to understand what happens with this call.
            self.profile = array(self.getprofile())
    
    # ****************************************************************************************************
    
    def getprofile(self):
        """
        Returns a list of 128 integer data points representing a pulse profile.
        Takes two parameters: the xml data and the profile index to use. 
        The xml data contains two distinct profile sections (i.e. there are two <Profile>...</Profile> 
        sections in the file) which are indexed. The first section with profileIndex = 0 pertains to a
        profile obtained after the FFT, the second, profileIndex = 1, to a profile that has been period
        and DM searched using PDMPD.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing 128 integer data points.
        """
        # First obtain desired block of xml data.
        block = self.rawdata.getElementsByTagName('Profile')
        
        # Get raw hexadecimal data from the block
        points = block[self.profileIndex].childNodes[0].data
        
        # The format of the hexadecimal data is 02X, i.e. hexadecimal value with 2 digits.
        decimal_profile = []
        index = 0 # The index at which hexadecimal conversion will be performed.
        
        while index < len(points):
            if points[index] != "\n":
                try:
                    hex_value = points[index:index+2]
                    #print "Hex value:\t", hex_value
                    decimal_profile.append(int(hex_value,16)) # now the profile (shape, unscaled) is stored in dec_value
                    #print "Decimal value:\t",int(hex_value,16)
                    index = index+2 # Skip two characters to next hexadecimal number since format is 02X.
                except ValueError:
                    if points[index] =="\t":# There is a tab at the end of the xml data. So break the loop normally here.
                        break
                    else: # Unexpected error, report to user. 
                        print "Unexpected value error obtaining profile data for: ",self.cand
                        break
            else:
                index = index+1
                
        return decimal_profile
                
    # ****************************************************************************************************
        
    def isValid(self):
        """
        Tests the xml data loaded from a phcx file for well-formedness, and invalid values.
        To understand the code here its best to take a look at a phcx xml file, to see the
        underlying structure. Alternatively I've generated a xml schema file which summarizes
        the structure (should be in same folder as this file) called: phcx.xsd.xml .
        
        Parameters:
        N/A
        
        Returns:
        True if the xml data is well formed and valid, else false.
        """
        
        # Read out data blocks.
        profile_block = self.rawdata.getElementsByTagName('Profile')
        subband_block = self.rawdata.getElementsByTagName('SubBands')
        datablock_block = self.rawdata.getElementsByTagName('DataBlock')
        
        # Test length of data in blocks. These should be equal to 2, since there
        # are two profile blocks, two sub-band blocks and two data blocks in the
        # xml file.
        if ( len(profile_block) == len(subband_block) == len(datablock_block) == 2 ):
            
            # There are two sections in the XML file:
            #<Section name='FFT'>...</Section>
            #<Section name='FFT-pdmpd'>...</Section>
            #
            # The first section (index=0) contains the raw FFT data, the second (index=1)
            # contains data that has been period and DM searched using a separate tool.
            # Mike Keith should know more about this tool called "pdmpd". Here
            # data from both these sections is extracted to determine its length.
            
            # From <Section name='FFT'>...</Section>
            subband_points_fft   = subband_block[0].childNodes[0].data
            datablock_points_fft = datablock_block[0].childNodes[0].data
            
            # From <Section name='FFT-pdmpd'>...</Section>
            profile_points_opt   = profile_block[1].childNodes[0].data
            subband_points_opt   = subband_block[1].childNodes[0].data
            datablock_points_opt = datablock_block[1].childNodes[0].data
            
            # Note sure if the checks here are valid, i.e. if there are 99 profile points is that bad?
            if ( len(profile_points_opt)>100) & (len(subband_points_opt)>1000) & (len(subband_points_fft)>1000) & (len(datablock_points_opt)>1000 ):
                
                subband_bins = int(subband_block[1].getAttribute("nBins"))
                subband_subbands = int(subband_block[1].getAttribute("nSub"))
                dmindex = list(self.rawdata.getElementsByTagName('DmIndex')[1].childNodes[0].data)
                
                # Stored here so call to len() made only once.
                lengthDMIndex = len(dmindex) # This is the DM index from the <Section name='FFT'>...</Section> part of the xml file.
                
                if (subband_bins == 128) & (subband_subbands == 16) & (lengthDMIndex > 100):
                    
                    # Now check for NaN values.
                    bestWidth      = float(self.rawdata.getElementsByTagName('Width')[1].childNodes[0].data)
                    bestSNR        = float(self.rawdata.getElementsByTagName('Snr')[1].childNodes[0].data)
                    bestDM         = float(self.rawdata.getElementsByTagName('Dm')[1].childNodes[0].data)
                    bestBaryPeriod = float(self.rawdata.getElementsByTagName('BaryPeriod')[1].childNodes[0].data)
                    
                    if (bestWidth != "nan") & (bestSNR != "nan") & (bestDM != "nan") & (bestBaryPeriod != "nan"):
                        return True
                    else:
                        print "\tPHCX check 4 failed, NaN's present in: ",self.cand
                        
                        # Extra debugging info for anybody encountering errors.
                        if (bestWidth != "nan") :
                            self.out("\t\"Width\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                        if (bestSNR != "nan") :
                            self.out("\t\"Snr\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                        if (bestDM != "nan"):
                            self.out("\t\"Dm\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                        if (bestBaryPeriod != "nan"):
                            self.out("\t\"BaryPeriod\" value found in <Section name='FFT-pdmpd'>...</> is NaN in: ",self.cand)
                            
                        return False
                else:
                    self.out("\tPHCX check 3 failed, wrong number of bins, sub-bands in: ",self.cand)
                    
                    # Extra debugging info for anybody encountering errors.
                    if(subband_bins!=128):
                        self.outMultiple("\tNumber of sub-band bins != 128 there are instead ",subband_bins, " in: ",self.cand)
                    if(subband_subbands!=16):
                        self.outMultiple("\tNumber of sub-bands != 16 there are instead ",subband_subbands, " in: ",self.cand)
                    if(lengthDMIndex<100):
                        self.outMultiple("\tNumber of DM indexes < 100 there are instead ",lengthDMIndex, " in: ",self.cand)
                        
                    return False
            else:
                self.out("\tPHCX check 2 failed, not enough profile points, sub-band points in: ",self.cand)
                self.out("\tPoints in <Section name='FFT'>...</>","")
                self.outMultiple("\tSub-band points: ",len(subband_points_fft)," Data block points: ", len(datablock_points_fft))
                self.out("\tPoints in <Section name='FFT-pdmpd'>...</>")
                self.outMultiple("\tProfile points: ",len(profile_points_opt)," Sub-band points: ",len(subband_points_opt)," Data block points: ", len(datablock_points_opt))
                return False
        else:
            self.out("\tPHCX check 1 failed, profile, sub-band and data blocks of unequal size in: ",self.cand)
            return False
          
    # ****************************************************************************************************
        
    def computeFeatures(self,feature_type):
        """
        Builds the features using the PFDOperations.py file. Returns the features.
        
        Parameters:
        type               -    the type of features to generate.
        
                                feature_type = 1 generates 12 features from Eatough et al., MNRAS, 407, 4, 2010.
                                feature_type = 2 generates 22 features from Bates et al., MNRAS, 427, 2, 2012.
                                feature_type = 3 generates 22 features from Thornton, PhD Thesis, Univ. Manchester, 2013.
                                feature_type = 4 generates 6 features from Lee et al., MNRAS, 333, 1, 2013.
                                feature_type = 5 generates 6 features from Morello et al., MNRAS, 433, 2, 2014.
                                feature_type = 6 generates 8 features from Lyon et al.,2015.
                                feature_type = 7 obtains raw integrated (folded) profile data.
                                feature_type = 8 obtains raw DM-SNR Curve data.
        
        Returns:
        An array of candidate features as floating point values.
        """
        
        if(feature_type == 1):
            return self.computeType_1()
        elif(feature_type == 2):
            return self.computeType_2()
        elif(feature_type == 3):
            return self.computeType_3()
        elif(feature_type == 4):
            return self.computeType_4()
        elif(feature_type == 5):
            return self.computeType_5()
        elif(feature_type == 6):
            return self.computeType_6()
        elif(feature_type == 7):
            return self.computeType_7()
        elif(feature_type == 8):
            return self.computeType_8()
        else:
            raise Exception("Invalid features specified!")
                            
    # ****************************************************************************************************
    
    def computeType_1(self):
        """
        Generates 12 features from Eatough et al., MNRAS, 407, 4, 2010.
        
        Parameters:
        N/A
        
        Returns:
        An array of 12 candidate features as floating point values.
        """
        
        # The features described in this work have not been implemented.
        # It is hoped the authors of this work, or others, will implement the features
        # at some point, allowing for a full comparison between all features.
        #
        # Please add PHCX and PFD compatible feature extraction code for this work in
        # FeatureExtractor.py     - for code that applies to both PHXC and PFD files.
        # PHCXFeatureExtractor.py - for code that applies to PHXC files only.
        # PFDFeatureExtractor.py  - for code that applies to PFD files only.
        
        for count in range(1,23):
            self.features.append(0.0)
            
        return self.features
    
    # ****************************************************************************************************
    
    def computeType_2(self):
        """
        Generates 22 features from Bates et al., MNRAS, 427, 2, 2012.
        
        Parameters:
        N/A
        
        Returns:
        An array of 22 candidate features as floating point values.
        """
        
        # The features described in this work have not been implemented.
        # It is hoped the authors of this work, or others, will implement the features
        # at some point, allowing for a full comparison between all features.
        #
        # Please add PHCX and PFD compatible feature extraction code for this work in
        # FeatureExtractor.py     - for code that applies to both PHXC and PFD files.
        # PHCXFeatureExtractor.py - for code that applies to PHXC files only.
        # PFDFeatureExtractor.py  - for code that applies to PFD files only.
        
        for count in range(1,23):
            self.features.append(0.0)
            
        return self.features
    
    # ****************************************************************************************************    
    def computeType_3(self):
        """
        Generates 22 features from Thornton, PhD Thesis, Univ. Manchester, 2013.
        
        Features:
        
        Computes the sinusoid fitting features for the profile data. There are four features computed:
        
        Feature 1. Chi-Squared value for sine fit to raw profile. This attempts to fit a sine curve
                 to the pulse profile. The reason for doing this is that many forms of RFI are sinusoidal.
                 Thus the chi-squared value for such a fit should be low for RFI (indicating
                 a close fit) and high for a signal of interest (indicating a poor fit).
                 
        Feature 2. Chi-Squared value for sine-squared fit to amended profile. This attempts to fit a sine
                 squared curve to the pulse profile, on the understanding that a sine-squared curve is similar
                 to legitimate pulsar emission. Thus the chi-squared value for such a fit should be low for
                 RFI (indicating a close fit) and high for a signal of interest (indicating a poor fit).
                 
        Feature 3. Difference between maxima. This is the number of peaks the program identifies in the pulse
                 profile - 1. Too high a value may indicate that a candidate is caused by RFI. If there is only
                 one pulse in the profile this value should be zero.
                 
        Feature 4. Sum over residuals.  Given a pulse profile represented by an array of profile intensities P,
                 the sum over residuals subtracts ( (max-min) /2) from each value in P. A larger sum generally
                 means a higher SNR and hence other features will also be stronger, such as correlation between
                 sub-bands. Example,
                 
                 P = [ 10 , 13 , 17 , 50 , 20 , 10 , 5 ]
                 max = 50
                 min = 5
                 (abs(max-min))/2 = 22.5
                 so the sum over residuals is:
                 
                  = (22.5 - 10) + (22.5 - 13) + (22.5 - 17) + (22.5 - 50) + (22.5 - 20) + (22.5 - 10) + (22.5 - 5)
                  = 12.5 + 9.5 + 5.5 + (-27.5) + 2.5 + 12.5 + 17.5
                  = 32.5
        
        Computes the Gaussian fitting features for the profile data. There are seven features computed:
        
        Feature 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
                 This fits a two Gaussian curves to a histogram of the profile data. One of these
                 Gaussian fits has its mean value set to the value in the centre bin of the histogram,
                 the other is not constrained. Thus it is expected that for a candidate arising from noise,
                 these two fits will be very similar - the distance between them will be zero. However a
                 legitimate signal should be different giving rise to a higher feature value.
                 
        Feature 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
                 This computes the maximum height of the fixed Gaussian curve (mean fixed to the centre
                 bin) to the profile histogram, and the maximum height of the non-fixed Gaussian curve
                 to the profile histogram. This ratio will be equal to 1 for perfect noise, or close to zero
                 for legitimate pulsar emission.
        
        Feature 7. Distance between expectation values of derivative histogram and profile histogram. A histogram
                 of profile derivatives is computed. This finds the absolute value of the mean of the 
                 derivative histogram, minus the mean of the profile histogram. A value close to zero indicates 
                 a candidate arising from noise, a value greater than zero some form of legitimate signal.
        
        Feature 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. Describes the width of the
                 pulse, i.e. the width of the Gaussian fit of the pulse profile. Equal to 2*sqrt( 2 ln(2) )*sigma.
                 Not clear whether a higher or lower value is desirable.
        
        Feature 9. Chi squared value from Gaussian fit to pulse profile. Lower values are indicators of a close fit,
                 and a possible profile source.
        
        Feature 10. Smallest FWHM of double-Gaussian fit to pulse profile. Some pulsars have a doubly peaked
                  profile. This fits two Gaussians to the pulse profile, then computes the FWHM of this
                  double Gaussian fit. Not clear if higher or lower values are desired.
        
        Feature 11. Chi squared value from double Gaussian fit to pulse profile. Smaller values are indicators
                  of a close fit and possible pulsar source.
        
        Computes the candidate parameters. There are four features computed:
        
        Feature 12. The candidate period.
                 
        Feature 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Feature 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Feature 15. The best pulse width.
        
        Computes the dispersion measure curve fitting parameters:
        
        Feature 16. This feature computes SNR / SQRT( (P-W) / W ).
                 
        Feature 17. Difference between fitting factor Prop, and 1. If the candidate is a pulsar,
                  then prop should be equal to 1.
        
        Feature 18. Difference between best DM value and optimised DM value from fit. This difference
                  should be small for a legitimate pulsar signal. 
                 
        Feature 19. Chi squared value from DM curve fit, smaller values indicate a smaller fit. Thus
                  smaller values will be possessed by legitimate signals.
        
         Computes the sub-band features:
        
        Feature 20. RMS of peak positions in all sub-bands. Smaller values should be possessed by
                  legitimate pulsar signals.
                 
        Feature 21. Average correlation coefficient for each pair of sub-bands. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Feature 22. Sum of correlation coefficients between sub-bands and profile. Larger values should be
                  possessed by legitimate pulsar signals.
                                                
        Parameters:
        N/A
        
        Returns:
        An array of 22 candidate features as floating point values.
        """
        
        # Get features 1-4
        try:
            
            sin_fit = self.fe.getSinusoidFittings(self.profile)
            # Add first features.
            self.features.append(float(sin_fit[0])) # Feature 1.  Chi-Squared value for sine fit to raw profile.
            self.features.append(float(sin_fit[1])) # Feature 2.  Chi-Squared value for sine-squared fit to amended profile.
            self.features.append(float(sin_fit[2])) # Feature 3.  Difference between maxima.
            self.features.append(float(sin_fit[3])) # Feature 4.  Sum over residuals.
            
            if(self.debug==True):
                print "\nFeature 1. Chi-Squared value for sine fit to raw profile = ",sin_fit[0]
                print "Feature 2. Chi-Squared value for sine-squared fit to amended profile = ",sin_fit[1]
                print "Feature 3. Difference between maxima = ",sin_fit[2]
                print "Feature 4. Sum over residuals = ",sin_fit[3]
        
        # Get features 5-11
            guassian_fit = self.fe.getGaussianFittings(self.profile)
            
            self.features.append(float(guassian_fit[0]))# Feature 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
            self.features.append(float(guassian_fit[1]))# Feature 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
            self.features.append(float(guassian_fit[2]))# Feature 7. Distance between expectation values of derivative histogram and profile histogram.
            self.features.append(float(guassian_fit[3]))# Feature 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. 
            self.features.append(float(guassian_fit[4]))# Feature 9. Chi squared value from Gaussian fit to pulse profile.
            self.features.append(float(guassian_fit[5]))# Feature 10. Smallest FWHM of double-Gaussian fit to pulse profile. 
            self.features.append(float(guassian_fit[6]))# Feature 11. Chi squared value from double Gaussian fit to pulse profile.
            
            if(self.debug==True):
                print "\nFeature 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram = ", guassian_fit[0]
                print "Feature 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram = ",guassian_fit[1]
                print "Feature 7. Distance between expectation values of derivative histogram and profile histogram. = ",guassian_fit[2]
                print "Feature 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile = ", guassian_fit[3]
                print "Feature 9. Chi squared value from Gaussian fit to pulse profile = ",guassian_fit[4]
                print "Feature 10. Smallest FWHM of double-Gaussian fit to pulse profile = ", guassian_fit[5]
                print "Feature 11. Chi squared value from double Gaussian fit to pulse profile = ", guassian_fit[6]

        # Get features 12-15
            candidateParameters = self.fe.getCandidateParameters(self.rawdata,self.profileIndex)
            
            self.features.append(float(candidateParameters[0]))# Feature 12. Best period.
            self.features.append(self.filterFeature(13,float(candidateParameters[1])))# Feature 13. Best S/N value.
            self.features.append(self.filterFeature(14,float(candidateParameters[2])))# Feature 14. Best DM value.
            self.features.append(float(candidateParameters[3]))# Feature 15. Best pulse width.
            
            if(self.debug==True):
                print "\nFeature 12. Best period = "         , candidateParameters[0]
                print "Feature 13. Best S/N value = "        , candidateParameters[1], " Filtered value = ", self.filterFeature(13,float(candidateParameters[1]))
                print "Feature 14. Best DM value = "         , candidateParameters[2], " Filtered value = ", self.filterFeature(14,float(candidateParameters[2]))
                print "Feature 15. Best pulse width = "      , candidateParameters[3]
        
        # Get features 16-19
            DMCurveFitting = self.fe.getDMFittings(self.rawdata,self.profileIndex)
            
            self.features.append(float(DMCurveFitting[0]))# Feature 16. SNR / SQRT( (P-W)/W ).
            self.features.append(float(DMCurveFitting[1]))# Feature 17. Difference between fitting factor, Prop, and 1.
            self.features.append(self.filterFeature(18,float(DMCurveFitting[2])))# Feature 18. Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest).
            self.features.append(float(DMCurveFitting[3]))# Feature 19. Chi squared value from DM curve fit.
            
            if(self.debug==True):
                print "\nFeature 16. SNR / SQRT( (P-W) / W ) = " , DMCurveFitting[0]
                print "Feature 17. Difference between fitting factor, Prop, and 1 = " , DMCurveFitting[1]
                print "Feature 18. Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest) = ", DMCurveFitting[2], " Filtered value = ", self.filterFeature(18,float(DMCurveFitting[2]))
                print "Feature 19. Chi squared value from DM curve fit = " , DMCurveFitting[3]
        
        # Get features 20-22
            subbandFeatures = self.fe.getSubbandParameters(self.profileIndex,self.rawdata,self.profile)
            
            self.features.append(float(subbandFeatures[0]))# Feature 20. RMS of peak positions in all sub-bands.
            self.features.append(float(subbandFeatures[1]))# Feature 21. Average correlation coefficient for each pair of sub-bands.
            self.features.append(float(subbandFeatures[2]))# Feature 22. Sum of correlation coefficients between sub-bands and profile.
            
            if(self.debug==True):
                print "\nFeature 20. RMS of peak positions in all sub-bands = " , subbandFeatures[0]
                print "Feature 21. Average correlation coefficient for each pair of sub-bands = " , subbandFeatures[1]
                print "Feature 22. Sum of correlation coefficients between sub-bands and profile = " , subbandFeatures[2]
        
        except Exception as e: # catch *all* exceptions
            print "Error computing features\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Exception computing 22 features from Thornton, PhD Thesis, Univ. Manchester, 2013.")

        return self.features
    
    # ****************************************************************************************************
    
    def computeType_4(self):
        """
        Generates 6 features from Lee et al., MNRAS, 333, 1, 2013.
        
        Parameters:
        N/A
        
        Returns:
        An array of 6 candidate features as floating point values.
        """
        
        # The features described in this work have not been implemented.
        # It is hoped the authors of this work, or others, will implement the features
        # at some point, allowing for a full comparison between all features.
        #
        # Please add PHCX and PFD compatible feature extraction code for this work in
        # FeatureExtractor.py     - for code that applies to both PHXC and PFD files.
        # PHCXFeatureExtractor.py - for code that applies to PHXC files only.
        # PFDFeatureExtractor.py  - for code that applies to PFD files only.
        
        for count in range(1,7):
            self.features.append(0.0)
    
        return self.features
    
    # ****************************************************************************************************
    
    def computeType_5(self):
        """
        Generates 6 features from Morello et al., MNRAS, 433, 2, 2014.
        
        Parameters:
        N/A
        
        Returns:
        An array of 6 candidate features as floating point values.
        """
        
        # The features described in this work have not been implemented.
        # It is hoped the authors of this work, or others, will implement the features
        # at some point, allowing for a full comparison between all features.
        #
        # Please add PHCX and PFD compatible feature extraction code for this work in
        # FeatureExtractor.py     - for code that applies to both PHXC and PFD files.
        # PHCXFeatureExtractor.py - for code that applies to PHXC files only.
        # PFDFeatureExtractor.py  - for code that applies to PFD files only.
        
        for count in range(1,7):
            self.features.append(0.0)
    
        return self.features
    
    # ****************************************************************************************************
    
    def computeType_6(self):
        """
        Generates 8 features from Lyon et al.,2015.
        
        Feature 1. Mean of the integrated (folded) pulse profile.
        Feature 2. Standard deviation of the integrated (folded) pulse profile.
        Feature 3. Skewness of the integrated (folded) pulse profile.
        Feature 4. Excess kurtosis of the integrated (folded) pulse profile.
        Feature 5. Mean of the DM-SNR curve.
        Feature 6. Standard deviation of the DM-SNR curve.
        Feature 7. Skewness of the DM-SNR curve.
        Feature 8. Excess kurtosis of the DM-SNR curve.
        
        Parameters:
        N/A
        
        Returns:
        An array of 8 candidate features as floating point values.
        """
        
        try:
            
            # First compute profile stats.
            bins =[] 
            for intensity in self.profile:
                bins.append(float(intensity))
            
            mn = mean(bins)
            stdev = std(bins)
            skw = self.fe.skewness(bins)        
            kurt = self.fe.excess_kurtosis(bins) 
            
            if(self.debug==True):
                print "\nFeature 1. Mean of the integrated (folded) pulse profile = ",            str(mn)
                print "Feature 2. Standard deviation of the integrated (folded) pulse profile = ",str(stdev)
                print "Feature 3. Skewness of the integrated (folded) pulse profile = ",          str(skw)
                print "Feature 4. Excess Kurtosis of the integrated (folded) pulse profile = ",   str(kurt)
                
            self.features.append(mn)
            self.features.append(stdev)
            self.features.append(skw)
            self.features.append(kurt)
        
            # Now compute DM-SNR curve stats.
            bins=[]
            bins=self.fe.getDMCurveData(self.rawdata,self.profileIndex)
            
            mn = mean(bins)
            stdev = std(bins)
            skw = self.fe.skewness(bins)        
            kurt = self.fe.excess_kurtosis(bins)
            
            if(self.debug==True):
                print "\nFeature 5. Mean of the integrated SNR-DM Curve = ", str(mn)
                print "Feature 6. Standard deviation of the SNR-DM Curve = ",str(stdev)
                print "Feature 7. Skewness of the SNR-DM Curve = ",          str(skw)
                print "Feature 8. Excess Kurtosis of the SNR-DM Curve = ",   str(kurt)
                
            self.features.append(mn)
            self.features.append(stdev)
            self.features.append(skw)
            self.features.append(kurt) 
        
        except Exception as e: # catch *all* exceptions
            print "Error getting features from PHCX file\n\t", sys.exc_info()[0]
            print self.format_exception(e)
            raise Exception("Exception computing 8 features from Lyon et al.,2015.")
            return []
    
        return self.features
        
    # ****************************************************************************************************
    
    def computeType_7(self):
        """
        Obtain integrated (folded) profile data.
        
        Parameters:
        N/A
        
        Returns:
        An array of data.
        """
        
        for intensity in self.profile:
            self.features.append(float(intensity))
            
        return self.features
        
    # ****************************************************************************************************
    
    def computeType_8(self):
        """
        Obtain SNR-DM curve data.
        Parameters:
        N/A
        
        Returns:
        An array of data.
        """
        
        return self.fe.getDMCurveData(self.rawdata,self.profileIndex)
    
    # ******************************************************************************************
    
    def filterFeature(self,s,value):
        """
        Filters a returned Feature value, so that if it is outside an expected range,
        then it is corrected, and the corrected version returned.
        
        Parameter:
        s        -    index of the feature, i.e. 1,2,3,...,n.
        value    -    the value of the feature.
        
        Return:
        The Feature value if it is valid, else a formatted version of the Feature.
        """

        if(s==13):# SNR
            if(self.isEqual(value, 0.0, self.epsilon)==-1):
                return 0.0
            else:
                return value
            
        elif(s==14): # DM
            if(self.isEqual(value, 0.0, self.epsilon)==-1):
                return 0.0
            else:
                return value
            
        elif(s==18): # mod(DMfit - DMbest).
            return float(abs(value))
        else:
            return value
    
    # ******************************************************************************************
    
    def isEqual(self,a,b,epsln):
        """
        Used to compare two floats for equality. This code has to cope with some
        extreme possibilities, i.e. the comparison of two floats which are arbitrarily
        small or large.
        
        Parameters:
        a        -    the first floating point number.
        b        -    the second floating point number.
        epsln    -    the allowable error.
        
        Returns:
        
        A value of -1 if a < b, a value greater than 1 if a > b, else
        zero is returned.
        
        """
        
        # There are two possibilities - both numbers may have exponents,
        # neither may have exponents, or a combination may occur. We need
        # a valid way to compare numbers with these possibilities which fits
        # ALL scenarios. The decision here (right or wrong!) is to avoid
        # wasting time on the perfect solution, and just allow the user to
        # specify an epsilon value they are happy with. In this case we 
        # are assuming a change to the feature smaller than epsilon is 
        # effectively meaningless. 
        
        if( abs(a - b) > epsln):
            if( a < b):
                return -1
            else:
                return 1 
        else:
            return 0
    
    # *******************************************************************************************