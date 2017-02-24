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

import multiprocessing
import itertools

# Custom file Imports:
import Utilities, Candidate


#Helper functions defined outside the class in order to allow for mutliprocessing: 
#multiprocessing pickles objects, and this doesn't work well with classes.

def featureMeta(candidate,features):
    """
    Returns a string of features with the metadata attached. Strips bad data fields from the list
    
    Parameters:
    
    candidate  -    The name of the candidate the features belong to.
    features   -    A float array of candidate features.
    
    Return:
    modified version of the list
    """ 
    # Join features into single comma separated line.
    allFeatures =  str(",".join(map(str, features)))
    entry1 = allFeatures + ",%" + candidate
    entry2 = entry1.replace("nan","0") # Remove NaNs since these cause error for ML tools like WEKA
    entry3 = entry2.replace("inf","0") # Remove infinity values since these cause error for ML tools like WEKA
    return entry3
    
# ****************************************************************************************************

def featureNoMeta(candidate,features):
    """
    Returns a string of features without the metadata attached. Strips bad data fields from the list
    
    Parameters:
    
    candidate  -    The name of the candidate the features belong to.
    features   -    A float array of candidate features.
    
    Return:
    modified version of the list
    """
    
    # Join features into single comma separated line.
    allFeatures =  str(",".join(map(str, features)))
    entry1 = allFeatures
    entry2 = entry1.replace("nan","0") # Remove NaNs since these cause error for ML tools like WEKA
    entry3 = entry2.replace("inf","0") # Remove infinity values since these cause error for ML tools like WEKA
    return entry3

def worker( (name,path,feature_type,candidate_type,verbose,meta,arff) ):
    """
    Creates a candidate object to perform the processing. If succesful, returns the feature 
    and None,None. If the worker hits an exception,return the exception information and 
    the name of the candidate to the main thread so it can be printed.

    This needs to be defined outside the class so it can be done in parallel
    """
    try:
        c = Candidate.Candidate(name,path)
        features = c.getFeatures(feature_type, candidate_type, verbose)
        if (arff and feature_type > 0 and feature_type < 7):
            features.append('?')
        if meta:
            return featureMeta(path,features),None, name
        else:
            return featureNoMeta(path,features),None, name
    except Exception as e:
        #return exception information to the main thread
        return None,e, name

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

    def generate_orders(self, directory, fileTypeRegexes, feature_type,candidate_type,verbose,meta,arff):
        """
        Yields the parameters to be dispatched to the subprocess workers: the filename generated from 
        walking the directory, and the behaviour parameters (feature_type, candidate_type, verbose, meta, arff)
        """
        for filetype in fileTypeRegexes:
        # Loop through the specified directory
            for root, subFolders, filenames in os.walk(directory):
                
                # If the file type matches one of those this program recognises
                for filename in fnmatch.filter(filenames, filetype):
                    cand = os.path.join(root, filename)
                    #yield all the arguments that need to be passed into the worker
                    yield cand,os.path.join(directory,cand), feature_type, candidate_type, verbose, meta, arff 
                    
                    # If the file does not have the expected suffix (file extension), skip to the next.  
                    if(cand.endswith(filetype.replace("*",""))==False):
                        continue

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
        candidatesProcessed = 0
        successes = 0
        failures = 0
        
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
        
        #spawn a pool of workers to process the individual files 
        
        worker_pool = multiprocessing.Pool(multiprocessing.cpu_count()) #try to utilize all avaliable cores

        #create a generator that feeds the filenames of the candidates to process and the behaviour options to 
        #the processor workers
        orders = self.generate_orders(directory,fileTypeRegexes, feature_type,candidate_type,verbose,meta,arff)

        #dispatch the processes to the worker pool. Use imap_unordered because it doesn't matter what order 
        #we process the files in
        for feature,ex,name in worker_pool.imap_unordered(worker, orders):
            if feature is not None:
                self.featureStore.append(feature)
                successes += 1
            else:
                #worker hit an exception: display the kind of exception and the file name. 
                #Unfortunatly, multiprocessing makes it harder to give a better traceback
                print "\tError reading candidate data :\n\t" 
                print "\tEncountered exception: "
                print "\t",ex
                print "\t",name, " did not have features generated."
                failures += 1 
                
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
