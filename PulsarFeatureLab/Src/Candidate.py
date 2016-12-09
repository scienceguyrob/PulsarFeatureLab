"""

**************************************************************************
| Candidate.py                                                           |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Represents an individual pulsar candidate. This code runs on python    |
| 2.4 or later.                                                          |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************
 
"""

# Custom Imports
import PHCXFile as phcx
import PFDFile as pfd

# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class Candidate:
    """
    Represents an individual pulsar candidate.
    
    """
    
    # ****************************************************************************************************
    #
    # INIT FUNCTION
    #
    # ****************************************************************************************************
    def __init__(self,name="Unknown",path=""):
        """
        Represents an individual Pulsar candidate.
        
        Parameters:
        
        name    -     the primary name for this individual candidate.
                      The file name is typically used.
        path    -     the full path to the candidate file.
        
        """
        self.candidateName = name # Name of the candidate file, minus the path.
        self.candidatePath = path # The full path to the candidate.
        self.features = []        # Stores all candidate features.  
            
    # ****************************************************************************************************
    # 
    # FEATURE CALCULATIONS.
    #
    # ****************************************************************************************************
    
    def getFeatures(self,feature_type,candidate_type,verbose):
        """
        Calculates the features for this candidate. If the file name of
        this Candidate object contains .pfd, then the PFD file feature generation
        code should be executed. Likewise if the file name ends in PHCX, then
        PHCX file feature generation (either for Thornton gnuzipped '.phcx.gz' PHCX files, or
        Morello et al. PHCX files) should be executed. 
        
        Note that there is a subtle difference between the PHCX files used by Thornton
        and Morello et al. - in terms of where the profile data is stored in XML format. 
        Thus these files should be treated differently. For Thornton generated PHCX files,
        the xml data contains two distinct profile sections:
        
            i.e. there are two <Profile>...</Profile>  sections in the file.
        
        These section are indexed by the code that reads the xml. The first section
        with profileIndex = 0, corresponds to a profile obtained after the FFT.
        The second, profileIndex = 1, to a profile that has been period and DM searched
        using PDMPD. Since the latter is the correct data to use for Thronton produced
        PHCX files, we use the data at profileIndex = 1.
        
        For Morello et al. produced data, there is a similar situation, except that the data
        sections are in reverse order. So now we need to use profileIndex = 0 to get the profile
        data that has been period and DM searched using PDMPD. 
        
        The result of this is that the two types of PHCX file are treated slightly differently.
        
        If further data file formats need to be processed, then changes need
        to be made here in order to cope with them. For example, if a new file format
        called .x appears, then below a check must be added for .x files,
        along with a new script to deal with them. Also note that additional changes
        would also be needed for the PulsarFeatureLab.py command line argument processing
        code, and a slight modification to the process() function in the FeatureGenerator.py
        script.
        
        Parameters:
        feature_type         -  the type of features to generate.
        
                                feature_type = 1 generates 12 features from Eatough et al., MNRAS, 407, 4, 2010.
                                feature_type = 2 generates 22 features from Bates et al., MNRAS, 427, 2, 2012.
                                feature_type = 3 generates 22 features from Thornton, PhD Thesis, Univ. Manchester, 2013.
                                feature_type = 4 generates 6 features from Lee et al., MNRAS, 333, 1, 2013.
                                feature_type = 5 generates 6 features from Morello et al., MNRAS, 433, 2, 2014.
                                feature_type = 6 generates 8 features from Lyon et al.,2015.
                                feature_type = 7 obtains raw integrated (folded) profile data.
                                feature_type = 8 obtains raw DM-SNR Curve data.
        
        candidate_type     -    the type of candidate file being processed.
                                
                                candidate_type = 1 assumes PHCX candidates output by the pipeline described by
                                                 Morello et al., MNRAS 443, 2, 2014.
                                candidate_type = 2 assumes gnuzipped ('.gz') PHCX candidates produced by the
                                                 pipeline described by Thornton., PhD Thesis, Univ. Manchester, 2013.
                                candidate_type = 3 assumes PFD files output by the LOTAAS and similar surveys in the
                                                 presto PFD format.
                                                                         
        verbose            -    the verbose logging flag.
        
        Returns:
        
        The candidate features as an array of floats.
        """  
        
        if(".pfd" in self.candidateName and candidate_type == 3):
            c = pfd.PFD(verbose,self.candidateName)
            self.features = c.computeFeatures(feature_type)
            return self.features

        elif (".pfd.36scrunch" in self.candidateName and candidate_type == 3):
            c = pfd.PFD(verbose, self.candidateName)
            self.features = c.computeFeatures(feature_type)
            return self.features

        elif(".phcx.gz" in self.candidateName and candidate_type == 2):
            profileIndex = 1 # For xml file, read comments above.
            c = phcx.PHCX(verbose,self.candidateName,profileIndex)
            self.features = c.computeFeatures(feature_type)
            return self.features
        
        elif(".phcx" in self.candidateName and candidate_type == 1):
            profileIndex = 0 # For xml file, read comments above.
            c = phcx.PHCX(verbose,self.candidateName,profileIndex)
            self.features = c.computeFeatures(feature_type)
            return self.features
        
        else:
            raise Exception("Unknown candidate type: Candidate.py (Line 136).")
    
    # ****************************************************************************************************
    
    def getName(self):
        """
        Obtains the name of the candidate file, not the full path.
        
        
        Returns:
        
        The name of the candidate file.
        """
        return self.candidateName
    
    # ****************************************************************************************************
    
    def getPath(self):
        """
        Obtains the full path to the candidate.
        
        
        Returns:
        
        The full path to the candidate.
        """
        return self.candidatePath
    
    # ****************************************************************************************************
    
    def __str__(self):
        """
        Overridden method that provides a neater string representation
        of this class. This is useful when writing these objects to a file
        or the terminal.
        
        """
            
        return self.candidateName + "," + self.candidatePath
    
    # ****************************************************************************************************