"""

**************************************************************************
| Utilities.py                                                           |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Contains support functions for the codebase. This code runs on python  |
| 2.4 or later.                                                          |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************
 

"""

# Python 2.4 imports.
import traceback
import sys, os

# ******************************************************************************************
#
# CLASS DEFINITION
#
# ******************************************************************************************

class Utilities(object):
    """
    Provides utility functions used when computing features.
    
    """
    
    # ******************************************************************************************
    #
    # Constructor.
    #
    # ******************************************************************************************
    
    def __init__(self,debugFlag):
        self.debug = debugFlag
        
    # ******************************************************************************************
    #
    # Functions.
    #
    # ******************************************************************************************
    
    def appendToFile(self,path,text):
        """
        Appends the provided text to the file at the specified path.
        
        Parameters:
        path    -    the path to the file to append text to.
        text    -    the text to append to the file.
        
        Returns:
        N/A
        """
        
        destinationFile = open(path,'a')
        destinationFile.write(str(text))
        destinationFile.close()
    
    # ******************************************************************************************
    
    def clearFile(self, path):
        """
        Clears the file at the specified path.
        
        Parameters:
        path    -    the path to the file to append text to.
        
        Returns:
        N/A
        """
        open(path, 'w').close()
    
    # ******************************************************************************************
        
    def fileExists(self,path):
        """
        Checks a file exists, returns true if it does, else false.
        
        Parameters:
        path    -    the path to the file to look for.
        
        Returns:
        
        True if the file exists, else false.
        """
        
        try:
            fh = open(path)
            fh.close()
            return True
        except IOError:
            return False
    
    # ******************************************************************************************
    
    def dirExists(self,path):
        """
        Checks a directory exists, returns true if it does, else false.
        
        Parameters:
        path    -    the path to the directory to look for.
        
        Returns:
        
        True if the file exists, else false.
        """
        
        try:
            if(os.path.isdir(path)):
                return True
            else:
                return False
        except IOError:
            return False
    
    # ******************************************************************************************
            
    def format_exception(self,e):
        """
        Formats error messages.
        
        Parameters:
        e    -    the exception.
        
        Returns:
        
        The formatted exception string.
        """
        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))
        
        exception_str = "\nTraceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        
        # Removing the last \n
        exception_str = exception_str[:-1]
        
        return exception_str
    
    # ******************************************************************************************
    
    def out(self,message,parameter):
        """
        Writes a debug statement out if the debug flag is set to true.
        
        Parameters:
        message    -    the string message to write out
        parameter  -    an accompanying parameter to write out.
        
        Returns:
        N/A
        """
        
        if(self.debug):
            print message , parameter
            
    # ******************************************************************************************
    
    def outMutiple(self,parameters):
        """
        Writes a debug statement out if the debug flag is set to true.
        
        Parameters:
        parameters  -    the values to write out.
        
        Returns:
        N/A
        """
        
        if(self.debug):
            
            output =""
            for p in parameters:
                output+=str(p)
                
            print output
            
    # ******************************************************************************************
            