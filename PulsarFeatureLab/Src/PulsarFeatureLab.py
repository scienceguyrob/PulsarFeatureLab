"""
    **************************************************************************
    |                                                                        |
    |                     Pulsar Feature Lab Version 1.0                     |
    |                                                                        |
    **************************************************************************
    | Description:                                                           |
    |                                                                        |
    | Generates machine learning features from pulsar candidate files in     |
    | CSV or ARFF format. This code runs on python 2.4 or later.             |
    **************************************************************************
    | Citation Request:                                                      |
    |                                                                        |
    | If you find this code useful, and use it as part of your research      |
    | please use the following citation:                                     |
    |                                                                        |
    | Lyon et al., MNRAS, 000, 1, 2015.                                      |
    **************************************************************************
    | Author: Rob Lyon                                                       |
    | Email : robert.lyon@postgrad.manchester.ac.uk                          |
    | web   : www.scienceguyrob.com                                          |
    **************************************************************************
    | Required Command Line Arguments:                                       |
    |                                                                        |
    | -d (string) path to the directory containing candidates.               |
    |                                                                        |
    | -f (string) path to the output file to create or append to.            |
    |                                                                        |
    | -t (int) type of features to generate:                                 |
    |                                                                        |
    |    1 - 12 features from Eatough et al., MNRAS 407, 4, 2010.            |
    |    2 - 22 features from Bates et al., MNRAS 427, 2, 2012.              |
    |    3 - 22 features from Thornton., PhD Thesis, Univ. Manchester, 2013. |
    |    4 - 6 features from  Lee et al., MNRAS 433, 1, 2013.                |
    |    5 - 6 features from Morello et al., MNRAS 443, 2, 2014.             |
    |    6 - 8 features from Lyon et al., MNRAS, 000, 0, 2015.               |
    |    7 - Raw integrated (folded) profile data.                           |
    |    8 - Raw DM-SNR Curve data.                                          |
    |                                                                        |
    | -c (int)  candidate file type:                                         |
    |                                                                        |
    |    1 - The PHCX candidates output by the pipeline described by         |
    |        Morello et al., MNRAS 443, 2, 2014.                             |
    |    2 - The gnuzipped ('.gz') PHCX candidates produced by the pipeline  |
    |        described by Thornton., PhD Thesis, Univ. Manchester, 2013.     |
    |    3 - The PFD files output by the LOTAAS and similar surveys in the   |
    |        presto PFD format.                                              |
    |    4 - The PHCX candidates output by the pipeline written by Monika    |
    |        Obrocka, and the tam at SKA SA.                                 |
    |                                                                        |
    **************************************************************************
    | Optional Command Line Arguments:                                       |
    |                                                                        |
    | --arff (boolean) flag that when provided, writes feature output in the |
    |      WEKA ARFF file format.                                            |
    |                                                                        |
    | --meta (boolean) flag that when provided, writes meta information, i.e.|
    |      the candidate file name, to the output file.                      |
    |                                                                        |
    | -v (boolean) verbose debugging flag.                                   |
    **************************************************************************
    | License:                                                               |
    |                                                                        |
    | Code made available under the GPLv3 (GNU General Public License), that |
    | allows you to copy, modify and redistribute the code as you see fit    |
    | (http://www.gnu.org/copyleft/gpl.html). Though a mention to the        |
    | original author using the citation above in derivative works, would be |
    | very much appreciated.                                                 |
    **************************************************************************
"""

# Command Line processing Imports:
from optparse import OptionParser

import sys

# Custom file Imports:
import Utilities, DataProcessor

# ******************************
#
# CLASS DEFINITION
#
# ******************************

class PulsarFeatureLab:
    """                
    Generates the features used to summarise and describe a pulsar candidate. 
    """
    
    # ******************************
    #
    # MAIN METHOD AND ENTRY POINT.
    #
    # ******************************

    def main(self,argv=None):
        """
        Main entry point for the Application. Processes command line
        input and begins creating the features.
    
        """
        
        # ****************************************
        #         Execution information
        # ****************************************
        
        print(__doc__)
        
        # ****************************************
        #    Command line argument processing
        # ****************************************
        
        # Python 2.4 argument processing.
        parser = OptionParser()

        # REQUIRED ARGUMENTS
        parser.add_option("-d", action="store", dest="dir",help='The path to the directory containing candidates (required).',default="")
        parser.add_option("-f", action="store", dest="out",help='The path to the output file to create or append to (required).',default="")
        parser.add_option("-t", type="int", dest="feature_type",help='The type of features to generate (required).',default=-1)
        parser.add_option("-c", type="int", dest="candidate_type",help='The input file type from which features will be generated (required).',default=-1)
        
        # OPTIONAL ARGUMENTS
        parser.add_option("--arff", action="store_true", dest="arff",help='Flag that when true writes the output in ARFF format (optional).',default=False)
        parser.add_option("--meta", action="store_true", dest="meta",help='Flag that when true writes meta information to the file (optional).',default=False)
        parser.add_option("-v", action="store_true", dest="verbose",help='Verbose debugging flag (optional).',default=False)

        (args,options) = parser.parse_args()# @UnusedVariable : Tells Eclipse IDE to ignore warning.
        
        # Update variables with command line parameters.
        
        self.dir = args.dir
        self.out = args.out
        self.feature_type = args.feature_type
        self.candidate_type = args.candidate_type
        
        self.arff = args.arff
        self.meta = args.meta
        self.verbose = args.verbose
        
        # Helper files.
        utils = Utilities.Utilities(self.verbose)
        dp    =  DataProcessor. DataProcessor(self.verbose)
        
        # Process -d argument if provided. 
        if(utils.dirExists(self.dir)):
            self.searchLocalDirectory = False
        else:
            print "You must supply a valid input directory with the -d flag."
            sys.exit()
        
        # Process -f argument if provided.
        if(utils.fileExists(self.out)):
            utils.clearFile(self.out)
        else:
            try:
                output = open(self.out, 'w') # First try to create file.
                output.close()
            except IOError:
                pass
            
            # Now check again if it exists.
            if(not utils.fileExists(self.out)):
                print "You must supply a valid output file path with the -d flag."
                sys.exit()
                
            
        # Process -t argument if provided. 
        if(self.feature_type < 1 or self.feature_type > 8):
            print "You must supply a valid type specifying the features to generate with the -t flag."
            print "1    -    Features from 'Selection of radio pulsar candidates using artificial neural networks', Eatough et al., MNRAS 407, 4, 2010."
            print "2    -    Features from 'The High Time Resolution Universe Pulsar Survey - VI. An artificial neural network and timing of 75 pulsars', Bates et al., MNRAS 427, 2, 2012."
            print "3    -    Features from 'The High Time Resolution Radio Sky', Thornton., PhD Thesis, Univ. Manchester, 2013."
            print "4    -    Features from 'PEACE: pulsar evaluation algorithm for candidate extraction - a software package for post-analysis processing of pulsar survey candidates', Lee et al., MNRAS 433, 1, 2013."
            print "5    -    Features from 'SPINN: a straightforward machine learning solution to the pulsar candidate selection problem', Morello et al., MNRAS 443, 2, 2014."
            print "6    -    Features from 'The features from the paper I'm writing', Lyon et al. 2015."
            print "7    -    Raw data from the integrated (folded) profile."
            print "8    -    Raw data from the DM-SNR curve ."
            
            sys.exit()
        
        # Process -c argument if provided. 
        if(self.candidate_type < 1 or self.candidate_type > 4):
            print "You must indicate the type of candidate file features will be extracted from with the -c flag."
            print "1    -    The PHCX candidates produced by pipeline described 'SPINN: a straightforward machine learning solution to the pulsar candidate selection problem', Morello et al., MNRAS 443, 2, 2014."
            print "2    -    The gnuzipped ('.gz') PHCX candidates produced by pipeline described in 'The High Time Resolution Radio Sky', Thornton., PhD Thesis, Univ. Manchester, 2013."
            print "3    -    The PFD files output by the LOTAAS and similar surveys (presto PFD format)."
            print "4    -    The PHCX files output by the SKA SA Pipelines (PHCX format)."
            sys.exit()
        
        # Process --arff argument if provided.
        if(self.arff == True):
            
            """
            ARFF is the standard file format, used by the WEKA data mining tool. Here we provide
            the option to write out candidate feature data to ARFF format, for direct use with WEKA and
            other tools. The format itself is simple. It simply requires a file header that describes
            the data it contains, and that each row of features be associated with a class label. For
            example, suppose we have the following feature data, each row extracted from 1 candidate:
            
            8,5,2,1,1,2,3,1
            6,5,3,1,1,1,4,2
            6,2,3,1,1,2,3,1
            8,5,2,1,1,2,3,1
            5,3,3,1,1,1,5,3
            
            Here there are 5 candidates described, by 8 individual features. To describe this in ARFF
            format we first:
            
            1. describe the file itself with the @relation tag.
            2. describe each feature used with the @attribute tag. 
            3. the possible class labels also with the @attribute tag.
            4. the @data tag indicates where the feature data begins.
            
            So the corresponding ARFF file for the data above would look like:
            
            @relation 5_candidates_described_by_8_features

            @attribute Feature_1 numeric
            @attribute Feature_2 numeric
            @attribute Feature_3 numeric
            @attribute Feature_4 numeric
            @attribute Feature_5 numeric
            @attribute Feature_6 numeric
            @attribute Feature_7 numeric
            @attribute Feature_8 numeric
            @attribute class {0,1}

            @data
            8,5,2,1,1,2,3,1,? % This is a comment. Here ? is the unknown label.
            6,5,3,1,1,1,4,2,?
            6,2,3,1,1,2,3,1,?
            8,5,2,1,1,2,3,1,?
            5,3,3,1,1,1,5,3,?
            
            Note that an extra class label (?) has been added to each row of data.
            This indicates that the true class of each row is unknown. Though if it
            was known to be positive (i.e. pulsar) then ? would be replaced with 1, and if it was
            known to be negative (i.e. non-pulsar) then ? would be replaced by 0.
            
            Comments in ARFF files are indicated by a %.
            """
            print "\tAppending ARFF Header information to output file."
            
            # Add class label information to the ARFF file, for those features we can
            # produce an ARFF file for.  
            if(self.feature_type > 0 and self.feature_type < 7):
                utils.appendToFile(self.out, "@relation Pulsar_Feature_Data_Type_"+ str(self.feature_type)+"\n")
            
            if(self.feature_type == 1):
                for count in range(1,13):
                    utils.appendToFile(self.out, "@attribute Feature_"+ str(count)+" numeric\n")
            elif(self.feature_type == 2 | self.feature_type == 3):
                for count in range(1,23):
                    utils.appendToFile(self.out, "@attribute Feature_"+ str(count)+" numeric\n")
            elif(self.feature_type == 4 | self.feature_type == 5):
                for count in range(1,7):
                    utils.appendToFile(self.out, "@attribute Feature_"+ str(count)+" numeric\n")
            elif(self.feature_type == 6):
                for count in range(1,9):
                    utils.appendToFile(self.out, "@attribute Feature_"+ str(count)+" numeric\n")
            elif(self.feature_type == 7 or self.feature_type == 8):
                print "\t NOTE:"
                print "\t + Cannot write integrated (folded) profile or DM-SNR curve data to ARFF format."
                print "\t + Different candidates (PHCX vs PFD) use different numbers of bins for their integrated profile"
                print "\t + and their DM-SNR curves, which can't be known in advance. Simply add an appropriate ARFF header"
                print "\t + to the output CSV file this application produces manually."
            
            # Add class label information to the ARFF file, for those features we can
            # produce an ARFF file for.    
            if(self.feature_type > 0 and self.feature_type < 7):
                utils.appendToFile(self.out, "@attribute class {0,1}\n@data\n")
            
        # ****************************************
        #   Print command line arguments & Run
        # ****************************************
              
        print "\n\t**************************"
        print "\t| Command Line Arguments |"
        print "\t**************************"
        print "\tDebug:",self.verbose
        print "\tCandidate directory:",self.dir
        print "\tCandidate type:",self.candidate_type
        print "\tOutput path:",self.out
        print "\tOutput meta information?:",self.meta
        print "\tOutput arff file?:",self.arff
        print "\tFeature Type:",self.feature_type,"\n\n" 
        
        dp.process(self.dir, self.out, self.feature_type,self.candidate_type,self.verbose,self.meta,self.arff)
            
        print "\tDone."
        print "\t**************************************************************************" # Used only for formatting purposes.
    
    # ****************************************************************************************************
      
if __name__ == '__main__':
    PulsarFeatureLab().main()