"""

**************************************************************************
| DataProcessor.py                                                       |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Collects candidate files and initiates feature generation. This code   |
| runs on python 2.4 or later.                                           |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************
 
"""

# Standard library Imports:
import sys,os,fnmatch,datetime

# Custom file Imports:
import Utilities, Candidate


# ****************************************************************************************************
#
# CLASS DEFINITION
#
# ****************************************************************************************************

class DataProcessor(Utilities.Utilities):
    """                
    Searches for candidate files in the local directory, 
    or a directory specified by the user.
    
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
        
        debugFlag    -    the debugging flag. If set to true, then detailed
                          debugging messages will be printed to the terminal
                          during execution.
        """
        self.utils = Utilities.Utilities.__init__(self,debugFlag)
        self.gzPhcxRegex = "*.phcx.gz"   
        self.phcxRegex    = "*.phcx" 
        self.pfdRegex     = "*.pfd"   
        self.featureStore   = []     # Variable which stores the features created for a candidate.
        
    # ****************************************************************************************************
    
    def storeFeatureMeta(self,candidate,features):
        """
        Appends candidate features to a list held by this object. This stores 
        each feature in memory, as opposed to writing them out to a file each time.
        
        Parameters:
        
        candidate  -    The name of the candidate the features belong to.
        features   -    A float array of candidate features.
        
        Return:
        N/A
        """
        
        # Join features into single comma separated line.
        allFeatures =  str(",".join(map(str, features)))
        entry1 = allFeatures + ",%" + candidate
        entry2 = entry1.replace("nan","0") # Remove NaNs since these cause error for ML tools like WEKA
        entry3 = entry2.replace("inf","0") # Remove infinity values since these cause error for ML tools like WEKA
        self.featureStore.append(entry3)
        
    # ****************************************************************************************************
    
    def storeFeatureNoMeta(self,candidate,features):
        """
        Appends candidate features to a list held by this object. This records 
        each feature in memory as opposed to writing them out to a file each time.
        
        Parameters:
        
        candidate  -    The name of the candidate the features belong to.
        features   -    A float array of candidate features.
        
        Return:
        N/A
        """
        
        # Join features into single comma separated line.
        allFeatures =  str(",".join(map(str, features)))
        entry1 = allFeatures
        entry2 = entry1.replace("nan","0") # Remove NaNs since these cause error for ML tools like WEKA
        entry3 = entry2.replace("inf","0") # Remove infinity values since these cause error for ML tools like WEKA
        self.featureStore.append(entry3)
    
    # ****************************************************************************************************
    
    def process(self,directory,output,feature_type,candidate_type,verbose,meta,arff):
        """
        Processes pulsar candidates of the type specified by 'candidate_type'.
        Writes the features of each candidate found to a single file, 'output'.
        
        Parameters:
        
        directory          -    the directory containing the candidates to process.
        output             -    the file to write the features to.
        feature_type       -    the type of features to generate.
        
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
                                candidate_type = 4 assumes PHCX candidates output by the SKA SA pipeline.
                                                 
        verbose            -    debug logging flag, if true output statements will be verbose.
        meta               -    a flag that when set to true, indicates that meta information will be retained
                                in the output files produced by this code. So if meta is set to true, then each line
                                of features will have the full path to the candidate they belong to included. Otherwise
                                they will not, making it hard to find which features belong to which candidate.
        
        arff               -    a flag that when set to true, indicates that meta output data will be written in ARFF format.
       
        Return:
        
        N/A
        """
        
        # Used to monitor feature creation statistics.
        candidatesProcessed = 0;
        successes = 0;
        failures = 0;
        
        print "\n\t*************************"
        print "\t| Searching Recursively |"
        print "\t*************************"
        
        # Check the type of candidate file used.
        if (candidate_type == 1):
            print "\tSearching for candidates with file extension: ", self.phcxRegex
            fileTypeRegexes = [self.phcxRegex]
        elif(candidate_type == 2):
            print "\tSearching for candidates with file extension: ", self.gzPhcxRegex
            fileTypeRegexes = [self.gzPhcxRegex]
        elif(candidate_type == 3):
            print "\tSearching for candidates with file extension: ", self.pfdRegex
            fileTypeRegexes = [self.pfdRegex]
        elif(candidate_type == 4):
            print "\tSearching for candidates with file extension: ", self.phcxRegex
            fileTypeRegexes = [self.phcxRegex]
        else:
            print "\tNo candidate file type provided, exiting..."
            sys.exit()
        
        print "\tSearching: ", directory  
          
        start = datetime.datetime.now() # Used to measure feature generation time.
        
        # For each type of file this program recognises   
        for filetype in fileTypeRegexes:
            
            # Loop through the specified directory
            for root, subFolders, filenames in os.walk(directory):
                
                # If the file type matches one of those this program recognises
                for filename in fnmatch.filter(filenames, filetype):
                    
                    cand = os.path.join(root, filename) # Gets full path to the candidate.
                    
                    # If the file does not have the expected suffix (file extension), skip to the next.  
                    if(cand.endswith(filetype.replace("*",""))==False):
                        continue
                    
                    candidatesProcessed+=1
                    
                    if(candidatesProcessed%10000==0):# Every 10,000 candidates
                        
                        # This 'if' statement is used to provide useful feedback on feature
                        # generation. But it is also used to write the features collected so far,
                        # to the output file at set intervals. This helps a) reduce memory load, and
                        # b) reduce disc load (by writing out lots of features in one go, as opposed
                        # to one by one).
                        
                        print "\tCandidates processed: ", candidatesProcessed    

                        # Write out the features collected so far.
                        outputText=""
                        for s in self.featureStore:
                            outputText+=s+"\n"
        
                        self.appendToFile(output, outputText) # Write all 10,000 entries to the output file.
                        self.featureStore = []                # Clear the feature store, freeing up memory.
                        
                    try:
                        
                        # Create the candidate object.    
                        c = Candidate.Candidate(cand,str(directory+cand))
                        
                        # Get the features from the candidate.
                        features = c.getFeatures(feature_type,candidate_type,verbose)
                        
                        # If the user would like the output to be in ARFF format, then each candidate
                        # has to be associated with a label. Since this code cannot know the true label
                        # of a candidate, here the unknown label '?' is appended as a additional feature.
                        if(arff and feature_type > 0 and feature_type < 7):
                            features.append("?")
                        
                        # Store the features so it can later be written to the specified output file.    
                        if(meta):
                            # Store with meta information - basically this means including the candidate
                            #                               name (full path) with each feature set. This means that
                            #                               each set of features will be linked to a candidate,
                            #                               useful for certain investigations (i.e. why a specific 
                            #                               candidate achieved particular feature values).
                            self.storeFeatureMeta(cand, features)
                        else:
                            self.storeFeatureNoMeta(cand, features) # Store only the feature data.
                            
                    except Exception as e: # Catch *all* exceptions.
                        print "\tError reading candidate data :\n\t", sys.exc_info()[0]
                        print self.format_exception(e)
                        print "\t",cand, " did not have features generated."
                        failures+=1
                        continue
                        
                    successes+=1
        
        # Save any remaining features, since its possible that some features
        # were not written to the output file in the loop above.
        
        if(len(self.featureStore) > 0):
            
            outputText=""
            
            for s in self.featureStore:
    
                outputText+= s+"\n"
                
            self.appendToFile(output, outputText)
            self.featureStore = []
        
        # Finally get the time that the procedure finished.
        end = datetime.datetime.now()
        
        # Output feature generation statistics.        
        print "\tCompleted candidate search."
        
        print "\n\t******************************"
        print "\t| Feature Generation Results |"
        print "\t******************************"
        print "\tCandidates processed:\t",candidatesProcessed
        print "\tSuccesses:\t", successes
        print "\tFailures:\t", failures
        print "\tExecution time: ", str(end - start)
        
    # ****************************************************************************************************