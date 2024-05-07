# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# # SPMpy file_loding_functions
#
# * Authors : Dr. Jewook Park(IBS-CVQS)
#     * *IBS-CVQS (Inistitute for Basic Science,Center for Van der Waals Quantum Solids), South Korea*
#     * email :  jewookpark@ibs.re.kr
#
# > **SPMpy** is a python package for scanning probe microscopy (SPM) data analysis, such as scanning tunneling microscopy and spectroscopy (STM/S) data and atomic force microscopy (AFM) images, which are inherently multidimensional. To analyze SPM data, SPMpy exploits recent image processing(a.k.a. Computer Vision) techniques. SPMpy data analysis functions utilize well-established Python packages, such as Numpy, PANDAS, matplotlib, Seaborn, holoview, etc. In addition, many parts are inspired by well-known SPM data analysis programs, for example, Wsxm and Gwyddion. Also, SPMpy is applying lessons from 'Fundamentals in Data Visualization'(https://clauswilke.com/dataviz/).
#
# >  **SPMpy** is an open-source project. (Github: https://github.com/Jewook-Park/SPMPY )
# > * Contributions, comments, ideas, and error reports are always welcome. Please use the Github page or email jewookpark@ibs.re.kr. Comments & remarks should be in Korean or English. 

# + [markdown] tags=[]
# # <font color=blue>files_in_folder</font>
# ## 0. file path checkup
#
# * working folder(path) checkup
#     * default = current path 
# * files_in_folder(path)
#     * change the path & show the file list  ( sxm + 3ds)
#         * to avoid unicodeerror, add 'r' in front of the file path 
#             * eg) path = r'D:\CALDES data - JEWOOK PARK research group'
#         * return 
#             * file_list_df columns  = [group,num,file_name,type]
#         

# +
####################################
# check the file location 
####################################
# use the pre-set path 
# or use get an input 
#test_path = r'D:\CALDES data - JEWOOK PARK research group\CloudStation\SPMs\ULT-SPM (Unisoku) - Jewook Park\Raw data\2022\2022 0105 Cu(111) Wtip17 LN2T'
#############################
# to avoid  "unicodeescape" error 
# add 'r' in front of the file path

#files_in_folder = input("copy&paste the file location:")
def files_in_folder(path): 
    """
    

    Parameters
    ----------
    path : str 
        folder path 
        * copy and paste the folder path
        * add 'r' to avoid unicodeerror 
        * eg) test_path = r'D:\CALDES data - JEWOOK PARK research group\...'
    Returns
    -------
    file_list_df : PANDAS DataFrame
        file list dataframe 

    """
    import os
    import glob
    import pandas as pd
    import numpy as np
    
    currentPath = os.getcwd() #get current path
    print ("Current Path = ", os.getcwd()) # print current path 
    #######################################
    files_in_folder = path
    # copy & paste the "SPM data file" location (folder(path)) 
    os.chdir(files_in_folder)
    print ("Changed Path = ", os.getcwd()) 
    # check the re-located path 
    ####################################

    ######################################
    # call all the sxm  files in path    #
    ######################################
    path = "./*"
    # pt_spec_file_list = (glob.glob('*.dat')) 
    sxm_file_list = (glob.glob('*.sxm')) 
    grid_file_list = (glob.glob('*.3ds')) 
    csv_file_list = (glob.glob('*.csv')) 
    gwy_file_list = (glob.glob('*.gwy')) 
    # using "glob"  all " *.sxm" files  in file_list
    #####################################
    ## sxm file
    file_list_sxm_df = pd.DataFrame([[
        file[:-7],file[-7:-4],file] 
                                     for file in sxm_file_list],
        columns =['group','num','file_name'])

    sxm_file_groups= list (set(file_list_sxm_df['group']))
    ## 3ds file
    file_list_3ds_df = pd.DataFrame([[
    file[:-7],file[-7:-4],file] 
                                 for file in grid_file_list],
    columns =['group','num','file_name'])
    ## csv file
    file_list_csv_df = pd.DataFrame([[
        file[:-7],file[-7:-4],file] 
                                     for file in csv_file_list],
        columns =['group','num','file_name'])
    ## gwy file
    file_list_gwy_df = pd.DataFrame([[
        file[:-4], np.nan, file] 
                                     for file in gwy_file_list],
        columns =['group','num','file_name'])   
    
    file_list_df = pd.concat ([file_list_sxm_df, file_list_3ds_df, file_list_csv_df, file_list_gwy_df],ignore_index= True)
    file_list_df['type'] = [file_name[-3:] for file_name in  file_list_df.file_name]
    print (file_list_df)

    
    #############################################################
    # to call all the files in sxm_file_groups[0]
    ##  file_list_df[file_list_df['group'] == sxm_file_groups[0]]
    #############################################################
    #print (file_list_sxm_df)
    #print (file_list_3ds_df)
    # indicates # of files in each group 
    for group in sxm_file_groups:
        print ('sxm file groups :  ', group, ':  # of files = ',
               len(file_list_sxm_df[file_list_sxm_df['group'] == group]) )
    if len(file_list_df[file_list_df['type'] == '3ds']) ==0 :
        print ('No GridSpectroscopy data')
    else :
        print ('# of GridSpectroscopy',
               list(set(file_list_df[file_list_df['type'] == '3ds'].group))[0], 
               ' = ',           
               file_list_df[file_list_df['type'] == '3ds'].group.count())

    return file_list_df


# + [markdown] tags=[]
# # <font color=blue>img2xr</font>
# ## 1. 2D image (topography & LDOS, *.sxm) to xarray  
#
# * input: nanonis 2D data (*.sxm) 
# * output : Xarray (_xr) with attributes
#     * nanonis (sxm)  $\to $ numpy $\to$ pd.DataFrame(_df) $\to$ xr.DataSet (_xr) 
# * Xarray attributes
#         * title
#         * X_spacing
#         * Y_spacing
#         * freq_X_spacing
#         * freq_Y_spacing

# + colab={"base_uri": "https://localhost:8080/", "height": 112, "resources": {"http://localhost:8080/nbextensions/google.colab/files.js": {"data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgZG8gewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwoKICAgICAgbGV0IHBlcmNlbnREb25lID0gZmlsZURhdGEuYnl0ZUxlbmd0aCA9PT0gMCA/CiAgICAgICAgICAxMDAgOgogICAgICAgICAgTWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCk7CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPSBgJHtwZXJjZW50RG9uZX0lIGRvbmVgOwoKICAgIH0gd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCk7CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK", "headers": [["content-type", "application/javascript"]], "ok": true, "status": 200, "status_text": ""}}} id="I91tRQmWf3Ry" outputId="4d722d2c-3419-45a1-cc42-c9ab6d7ca18d"
#####################
# conver the given sxm file in current path
# to  xarray DataSet (including attributes)

def img2xr (loading_sxm_file, center_offset = False):
    # import necessary module 
    import os
    import glob
    import numpy as np
    import pandas as pd
    import scipy as sp
    import math
    import matplotlib.pyplot as plt
    import re

    from warnings import warn

    try:
        import nanonispy as nap
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named nanonispy')
        # !pip install nanonispy
        import nanonispy as nap

    try:
        import xarray as xr
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named xarray')
        # #!pip install --upgrade scikit-image == 0.19.0.dev0
        # !pip install xarray 
        import xarray as xr

    try:
        import seaborn_image as isns
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named seaborn-image')
        # #!pip install --upgrade scikit-image == 0.19.0.dev0
        # !pip install --upgrade seaborn-image    
        import seaborn_image as isns



    ####################################
    # check the file location 
    ####################################
    currentPath = os.getcwd() #get current path
    print (os.getcwd()) # print current path 
    file_list = (glob.glob('*.sxm')) 
    file_list_df = pd.DataFrame(
        [[file[:-7],file[-7:-4],file] for file in file_list],
        columns =['group','num','file_name'])
    # check all the sxm files in the folder 
    if file_list_df.file_name.isin([loading_sxm_file]).sum() == 0: 
        print ('no sxm file')
    else : 
        print(file_list_df[file_list_df.file_name == loading_sxm_file])
    #######################################


    NF = nap.read.NanonisFile(loading_sxm_file)
    Scan = nap.read.Scan(NF.fname)
    #Scan.basename # file name only *.sxm 
    #Scan.header # heater dict 
    ##############################
    # Scan conditions from the header
    V_b = float(Scan.header['bias>bias (v)'])
    I_t = float(Scan.header['z-controller>setpoint'])

    [size_x,size_y] = Scan.header['scan_range']
    [cntr_x, cntr_y] = Scan.header['scan_offset']
    [dim_px,dim_py] = Scan.header['scan_pixels']
    [step_dx,step_dy] = [ size_x/dim_px, size_y/dim_py] 
    #pixel_size # size를 pixel로 나눔
    Rot_Rad = math.radians( float(Scan.header['scan_angle'])) 
    #str --> degree to radian 

    print ('scan direction (up/down): ', Scan.header['scan_dir'])
    ###   nX, nY --> x,y real scale  np array 
    nX = np.array([step_dx*(i+1/2) for i in range (0,dim_px)])
    nY = np.array([step_dy*(i+1/2) for i in range (0,dim_py)])
    # nX,nY for meshgrid (start from 1/2, not 0 )
    # dimesion맞춘 x, y steps # i 가 0부터 시작하니 1/2 더했음
    # In case of rotation ==0
    x = cntr_x - size_x + nX
    y = cntr_y - size_y + nY
    # real XY position in nm scale, Center position & scan_szie + XY position
    # center position  과 scan size을 고려한 x,y real 
    #########################################################################
    # np.meshgrid 
    x_mesh_0, y_mesh_0 = np.meshgrid(nX, nY)
    x_mesh = cntr_x - size_x + x_mesh_0
    y_mesh = cntr_y - size_y + y_mesh_0 
    # if there is rotation 
    x_mesh_r   =  np.cos(Rot_Rad)*x_mesh_0 + np.sin(Rot_Rad)*y_mesh_0  # "cloclwise"
    y_mesh_r   = -np.sin(Rot_Rad)*x_mesh_0 + np.cos(Rot_Rad)*y_mesh_0
    #########################################################################
    # image title 
    # image가 rotation되었을 경우는 따로 표시 rot !=0 
    if Rot_Rad ==0 : 
        image_title = Scan.basename[:-4] + '\n' + \
            str(round(size_x* 1E9 )) + ' nm x ' + \
                str(round(size_y* 1E9 )) + ' nm '  +\
                    ' V = '+ str(V_b) + ' V ' +\
                        ' I = ' + str(round(I_t *1E12)) + ' pA ' 
    else: 
        image_title = Scan.basename[:-4] + '\n' + \
            str(round(size_x* 1E9 )) + ' nm x ' + \
                str(round(size_y* 1E9 )) + ' nm '  +\
                    ' V = '+ str(V_b) + ' V ' +\
                        ' I = ' + str(round(I_t *1E12)) + ' pA ' +\
                            ' R = ' + str(math.degrees(Rot_Rad)) + 'deg'
    print(image_title)
    #########################################################################
    # scan channels in DataFrame

    #Scan.signals.keys()
    Scan.signals['Z'].keys()
    
    Scan.signals['Z']['forward'].shape
    z_fwd = Scan.signals['Z']['forward']
    z_bwd = Scan.signals['Z']['backward'][:,::-1]

    
    #Scan.signals['LI_Demod_1_X'].keys()
    
    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ])
    # 'LI' & 'X' in  channel name (signal.keys) 
    LIX_key = [s  for s in Scan.signals.keys()  if "LIX"  in s  if "X" in s ]
    # 0 is fwd, 1 is bwd 
    LIX_fwd  = Scan.signals[LIX_key[0]]['forward']
    LIX_bwd  = Scan.signals[LIX_key[0]]['backward'][:,::-1]

    #LIX_fwd = Scan.signals['LI_Demod_1_X']['forward']
    #LIX_bwd = Scan.signals['LI_Demod_1_X']['backward'][:,::-1]
    # LIX channel name varies w.r.t nanonis version 
    
    # same for LIY --> update later.. if needed 
    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ])
    # 'LI' & 'Y' in  channel name (signal.keys) 
    #LIY_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ]
    # 0 is fwd, 1 is bwd 
    #LIY_fwd, LIY_bwd = Gr.signals[LIY_keys[0]] ,Gr.signals[LIY_keys[1] ]
     
    
    
    #bwd channel : opposite data direction. 
    #bwd 는 x방향 순서가 반대다.  # 뒤집어 줘야 함. 
    ########################################
    if Scan.header['scan_dir'] == 'down':
        z_fwd = z_fwd[::-1,:]
        z_bwd = z_bwd[::-1,:]
        LIX_fwd = LIX_fwd[::-1,:]
        LIX_bwd = LIX_bwd[::-1,:]
    # if scan_direction == down, flip the data
    #scan 방향이 down 이면 y 방향 아래위 뒤집어준다
    ########################################
    z_fwd_df = pd.DataFrame(z_fwd)
    z_fwd_df.index.name ='row_y'
    z_fwd_df.columns.name ='col_x'

    z_bwd_df = pd.DataFrame(z_bwd)
    z_bwd_df.index.name ='row_y'
    z_bwd_df.columns.name ='col_x'

    LIX_fwd_df = pd.DataFrame(LIX_fwd)
    LIX_fwd_df.index.name ='row_y'
    LIX_fwd_df.columns.name ='col_x'

    LIX_bwd_df = pd.DataFrame(LIX_bwd)
    LIX_bwd_df.index.name ='row_y'
    LIX_bwd_df.columns.name ='col_x'
    # save data channels as DataFrame
    # z & LIX 를 df 로 저장 
    ########################################
    z_fwd_df = z_fwd_df.fillna(0)
    z_bwd_df = z_bwd_df.fillna(0)
    LIX_fwd_df = LIX_fwd_df.fillna(0)   
    LIX_bwd_df = LIX_bwd_df.fillna(0)
    # incompleted scan ==> np.nan in data point, ==> fillna()
    # scan 방향이 중간에 멈추었다면, 0 으로채운다.
    ########################################


    ############################
    # conver to DataFrame (PANDAS) 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    z_LIX_fNb_df = pd.concat([z_fwd_df.stack(),
                              z_bwd_df.stack(),
                              LIX_fwd_df.stack(),
                              LIX_bwd_df.stack()], axis = 1)
    # set colunm name for new DataFrame
    z_LIX_fNb_df.columns =['z_fwd','z_bwd', 'LIX_fwd','LIX_bwd']
    # z_LIX_fNb_df


    ############################
    # conver to xarray 
    ############################
    z_LIX_fNb_xr = z_LIX_fNb_df.to_xarray()
    # rename coord as "X", "Y" 
    z_LIX_fNb_xr = z_LIX_fNb_xr.rename(
        {"row_y": "Y", "col_x":"X"})
    # real size of XY 
    z_LIX_fNb_xr= z_LIX_fNb_xr.assign_coords(
        X = z_LIX_fNb_xr.X.values *step_dx, 
        Y = z_LIX_fNb_xr.Y.values *step_dy )
    # XY axis: 0 ~ size_XY

    ############################
    # check the XY ratio 
    ############################
    if  size_x == size_y : 
        pass
    else : 
        print ('size_x != size_y')
    # if xy size is not same, report it! 

    if step_dx != step_dy :
        xystep_ratio = step_dy/step_dx # check the XY pixel_ratio
        X_interp = np.linspace(
        z_LIX_fNb_xr.X[0], z_LIX_fNb_xr.X[-1], z_LIX_fNb_xr.X.shape[0]*1)
        step_dx = step_dx # step_dx check 

        Y_interp = np.linspace(
        z_LIX_fNb_xr.Y[0], z_LIX_fNb_xr.Y[-1], int(z_LIX_fNb_xr.Y.shape[0]*xystep_ratio)) 
        step_dy = step_dy/ xystep_ratio # step_dy check 

        # interpolation ratio should be int
        z_LIX_fNb_xr= z_LIX_fNb_xr.interp(X = X_interp, Y = Y_interp)
        print('step_dx/step_dy = ', xystep_ratio)
        print ('z_LIX_fNb_xr ==> reshaped')
    else: 
        z_LIX_fNb_xr =z_LIX_fNb_xr
        print('step_dx == step_dy')
    #print('z_LIX_fNb_xr', 'step_dx, step_dy = ',  z_LIX_fNb_xr.dims)
    print('z_LIX_fNb_xr', 'step_dx, step_dy = ', 
          re.findall('\{([^}]+)', str(z_LIX_fNb_xr.dims)))
    # regex practice


    ##########
    #################################
    # assigne attributes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    z_LIX_fNb_xr.attrs['title'] = image_title
    if 'Wtip' in image_title:
        z_LIX_fNb_xr.attrs['tip'] = 'W'
    elif 'Ni_tip' in image_title:
        z_LIX_fNb_xr.attrs['tip'] = 'Ni'
    elif 'Co_coated' in image_title:
        z_LIX_fNb_xr.attrs['tip'] = 'Co_coated'
    elif 'AFM' in image_title:
        z_LIX_fNb_xr.attrs['tip'] = 'AFM'
    else: 
        z_LIX_fNb_xr.attrs['tip'] = 'To Be Announced'
        print('tip material will be announced')
        
    if 'Cu(111)' in image_title:
        z_LIX_fNb_xr.attrs['sample'] = 'Cu(111)'
    elif 'Au(111)' in image_title:
        z_LIX_fNb_xr.attrs['sample'] = 'Au(111)'
    else: 
        z_LIX_fNb_xr.attrs['sample'] = 'To Be Announced'
        print('sample type will be announced')
    
    z_LIX_fNb_xr.attrs['image_size'] = [size_x,size_y]
    z_LIX_fNb_xr.attrs['X_spacing'] = step_dx
    z_LIX_fNb_xr.attrs['Y_spacing'] = step_dy    
    z_LIX_fNb_xr.attrs['freq_X_spacing'] = 1/step_dx
    z_LIX_fNb_xr.attrs['freq_Y_spacing'] = 1/step_dy

    # in case of real X Y ( center & size of XY)
    if center_offset == True:
        # move the scan center postion in real scanner field of view
        z_LIX_fNb_xr.assign_coords(X=(z_LIX_fNb_xr.X + cntr_x -  size_x/2))
        z_LIX_fNb_xr.assign_coords(Y=(z_LIX_fNb_xr.Y + cntr_y -  size_y/2))
    else :
        pass
        # (0,0) is the origin of image 


    #################################
    # test & how to use xr data 
    # z_LIX_fNb_xr  # xr dataset (with data array channels )
    #z_LIX_fNb_xr.z_fwd # select data channel
    #z_LIX_fNb_xr.data_vars # data channels check 
    #z_LIX_fNb_xr.z_fwd.values  # to call data array in nd array 
    #z_LIX_fNb_xr.dims # data channel dimension (coords) 
    #z_LIX_fNb_xr.coords # data  channel coordinates check 
    #z_LIX_fNb_xr.attrs # data  channel attributes check 

    return z_LIX_fNb_xr


# + [markdown] tags=[]
# # <font color=blue>grid2xr</font>
# ## 2.  GridSpectroscopy (*.3ds)  to xarray  
#
# * input: *.3ds file  ( grid 3d dataset )
# * output: Xarray (_xr) with attributes
#     * nanonis 3D data set (3ds)  $\to $ numpy $\to$ pd.DataFrame(_df) $\to$ xr.DataSet (_xr) 
# * Xarray attributes
#     * title
#     * X_spacing
#     * Y_spacing
#     * bias mV info
#     * freq_X_spacing
#     * freq_Y_spacing
#


# +
#griddata_file = file_list_df[file_list_df.type=='3ds'].iloc[0].file_name

def grid2xr(griddata_file, center_offset = True): 

    file = griddata_file
    #####################
    # conver the given 3ds file
    # to  xarray DataSet (check the attributes)

    import os
    import glob
    import numpy as np
    import numpy.fft as npf
    #import xarray as xr
    import pandas as pd
    import scipy as sp
    import matplotlib.pyplot as plt


    try:
        import nanonispy as nap
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named nanonispy')
        # !pip install nanonispy
        import nanonispy as nap

    try:
        import xarray as xr
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named xarray')
        # #!pip install --upgrade scikit-image == 0.19.0.dev0
        # !pip install xarray 
        import xarray as xr

    try:
        import seaborn_image as isns
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named seaborn-image')
        # #!pip install --upgrade scikit-image == 0.19.0.dev0
        # !pip install --upgrade seaborn-image    
        import seaborn_image as isns


    try:
        import xrft
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named xrft')
        # !pip install xrft 
        import xrft


    NF = nap.read.NanonisFile(file)
    Gr = nap.read.Grid(NF.fname)#
    ## Gr 로  해당 grid data 불러옴 # 중간에 끊기면 안됨
    channel_name = Gr.signals.keys()  # Gr 내의 data signals
    #print (channel_name)
    N = len(file);
    f_name = file[0:N-4]
    print (f_name) # Gr.basename




    #####################################
    #Header part
    #####################################
    #  Gr.header
    #####################################
    [dim_px,dim_py] = Gr.header['dim_px'] 
    [cntr_x, cntr_y] = Gr.header['pos_xy']
    [size_x,size_y] = Gr.header['size_xy']
    [step_dx,step_dy] = [ size_x/dim_px, size_y/dim_py] 
    #pixel_size # size를 pixel로 나눔

    ###   nX, nY --> x,y real scale  np array 
    nX = np.array([step_dx*(i+1/2) for i in range (0,dim_px)])# dimesion맞춘 xstep 
    nY = np.array([step_dy*(i+1/2) for i in range (0,dim_py)])# dimesion맞춘 ystep 

    x = cntr_x - size_x + nX
    y = cntr_y - size_y + nY
    # real XY position in nm scale, Center position & scan_szie + XY position
    # center position  과 scan size을 고려한 x,y real 
    #####################################
    # signal part
    # Gr.signals
    #####################################
    topography = Gr.signals['topo']
    params_v = Gr.signals['params'] 
    # params_v.shape = (dim_px,dim_py,15) 
    # 15: 3ds infos. 
    bias = Gr.signals['sweep_signal']
    # check the shape (# of 'original' bias points)
    I_fwd = Gr.signals['Current (A)'] # 3d set (dim_px,dim_py,bias)
    I_bwd = Gr.signals['Current [bwd] (A)'] # I bwd
    # sometimes, LI channel names are inconsistent depends on program ver. 
    # find 'LI Demod 1 X (A)'  or  'LI X 1 omega (A)'

    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ])
    # 'LI' & 'X' in  channel name (signal.keys) 
    LIX_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ]
    # 0 is fwd, 1 is bwd 
    LIX_fwd, LIX_bwd = Gr.signals[LIX_keys[0]] ,Gr.signals[LIX_keys[1] ]

    # same for LIY
    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ])
    # 'LI' & 'Y' in  channel name (signal.keys) 
    LIY_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ]
    # 0 is fwd, 1 is bwd 
    LIY_fwd, LIY_bwd = Gr.signals[LIY_keys[0]] ,Gr.signals[LIY_keys[1] ]


    ###########################################################
    #plt.imshow(topography) # toppography check
    #plt.imshow(I_fwd[:,:,0]) # LIX  check
    ###########################################################

    ##########################################################
    #		 Grid data 에 대한 Title 지정 
    #       grid size, pixel, bias condition 포함 #
    #############################################################
    # Gr.header.get('Bias>Bias (V)') # bias condition 
    # Gr.header.get('Z-Controller>Setpoint') # current set  condition
    # Gr.header.get('dim_px')  # jpixel dimension 
    title = Gr.basename +' ('  + str(
        float(Gr.header.get('Bias Spectroscopy>Sweep Start (V)'))
    ) +' V ~ ' +str( 
        float(Gr.header.get('Bias Spectroscopy>Sweep End (V)'))
    )+ ' V) \n at Bias = '+ Gr.header.get(
        'Bias>Bias (V)'
    )[0:-3]+' mV, I_t =  ' + Gr.header.get(
        'Z-Controller>Setpoint'
    )[0:-4]+ ' pA, '+str(
        Gr.header.get('dim_px')[0]
    )+' x '+str(
        Gr.header.get('dim_px')[1]
    )+' points'
    #############################################################       

    ### some times the topography does not look right. 
    # * then use the reshaping function 
    # only for asymmetry grid data set

    # eg) JW's MoS2 on HOPG exp. data 

    ###########################################################
    # topography  를 topography_reshape 으로 재지정.  
    ###########################################################
    topo_dimension_true = True


    if topo_dimension_true == True:
        topography_reshape = topography   
        #################################
        I_fwd_copy = I_fwd
        I_bwd_copy = I_bwd
        LIX_fwd_copy = LIX_fwd 
        LIX_bwd_copy = LIX_bwd 	
        # topography가 정상인 경우    
        #################################

        ##########################################################
        # 예를 들어  #  MoS2 on HOPG image 는 
        # 40 x 80 의 배열이 40x40 + 40 x40으로 되었음. 
        # x 한줄이 0-39: 1st line 40-79 : 2nd line임
        # 0-40, 19-59, 39-79 가 set로 움직임. 
        # 세로로 40X80 배열을 만들어서 
        # 0-39 를 2n으로 40-79 를 2n+1 으로 대입할것. 
        # topo # LIX f&b # I f&b #
        ##########################################################

    else: # topography가 비 정상인 경우  
        # topo_dimension_true == False 일경우 
        topography_reshape = np.transpose(np.copy(topography),(1,0)) 
        # 바꾼 dimension이 될 곳을 미리 만듬. 
        for x_indx, y_indx in enumerate (topography):
        # print(x_indx) # 0-39 # print(y_indx.shape)
            topography_reshape[2*x_indx,:] = y_indx[:40] # 앞쪽 절반은 첫번째줄로
            topography_reshape[2*x_indx+1,:] = y_indx[40:80] # 뒷쪽 절반은 두번째 줄로  
        #################################
        # 새로 바꾸는 정렬 방법을 확인하여 같은방식으로 I, LIX 에도 적용
        #################################
        #제대로 붙었는지 테스트
        plt.imshow(topography_reshape) # 80 * 40 OK
        # topography_reshape 정렬 끝 
        #################################
        I_fwd_copy = np.transpose(np.copy(I_fwd),(1,0,2))
        I_bwd_copy = np.transpose(np.copy(I_bwd),(1,0,2)) 
        # 바꾼 dimension이 될 곳을 미리 만듬. 	
        for x_indx, yNbias_plane in enumerate (I_fwd): 
            # 순서를바꿔서 두번째 index로 for loop
            print(x_indx) # 0-39 
            I_fwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            I_fwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로  

        for x_indx, yNbias_plane in enumerate (I_bwd): 
            # 순서를바꿔서 두번째 index로 for loop
            print(x_indx) # 0-39 
            I_bwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            I_bwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로plt.imshow(I_bwd_v[:,:,0]) # 40 * 256  cut 
        #################################
        # I reshape is done 
        #################################
        LIX_fwd_copy = np.transpose(np.copy(LIX_fwd),(1,0,2)) 
        LIX_bwd_copy = np.transpose(np.copy(LIX_bwd),(1,0,2)) 
        # 바꾼 dimension이 될 곳을 미리 만듬. 
        for x_indx, yNbias_plane in enumerate (LIX_fwd): 
            # 순서를바꿔서 두번째 index로 for loop
            LIX_fwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            LIX_fwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로  
        for x_indx, yNbias_plane in enumerate (LIX_bwd): 
            # 순서를바꿔서 두번째 index로 for loop
            LIX_bwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            LIX_bwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로
        #################################
        # LIX reshape is done 
        #################################
    #제대로 붙었는지 테스트

    #fig,axes = plt.subplots (nrows = 2,ncols = 3, figsize = (5,4),
    #                         sharex=True, sharey=True)
    #axs = axes.ravel()
    #axs[0].imshow(topography)
    #axs[0].set_title('topography')

    #axs[1].imshow(I_fwd_copy[:,:,0]) # 80 * 40 OK
    #axs[1].set_title('I_fwd[0]')
    #axs[2].imshow(LIX_fwd_copy[:,:,0]) # 80 * 40 OK
    #axs[2].set_title('LIX_fwd[0]')

    #axs[3].imshow(topography_reshape)
    #axs[3].set_title('topo_reshape')

    #axs[4].imshow(I_bwd_copy[:,:,0]) # 80 * 40 OK
    #axs[4].set_title('I_bwd[0]')

    #axs[5].imshow(LIX_bwd_copy[:,:,0]) # 80 * 40 OK
    #axs[5].set_title('LIX_bwd[0]')

    #fig.tight_layout()
    #plt.show()



    # after reshaping 

    topography = topography_reshape 
    #################################
    I_fwd = I_fwd_copy 
    I_bwd = I_bwd_copy 
    LIX_fwd  = LIX_fwd_copy 
    LIX_bwd  = LIX_bwd_copy
    ##########################################################





    ###########################
    # Bias segment check      #
    ###########################
    Segment = Gr.header['Bias>Bias (V)']
    # bias unit : '(V)' 

    if type(Segment) == str: # single segment case
        print ('No Segments\n'+ 'Grid data acquired at bias = '+  str(float(Segment)) + 'V')    
    ## No Segments # +  bias setting 

    ########################
    # bias interpolation to have bias 
    # bias_mV 는 bias 를 0 포함한 값으로interploation 
    # 3D data 가운데 x,y 를 꺼내서 bias interpolation 
    # e.g  256--> 양 끝점을 포함한 256+1 으로. (center 가 0가 되도록 )
        if len(bias)%2==0:
            bias_new = np.linspace(bias[0],bias[-1],num=(len(bias)+1)) 
            # 시작 부터 끝까지, 0포함 홀수, 전체크기+1 개의 단계로 세분화
        else:
            bias_new = np.linspace(bias[0],bias[-1],num=(len(bias))) 
            # bias_new  는 간격이 홀수개가 되도록 나눈 것임, 즉 0에가장 가까운 값이 하나뿐이도록 조정함 
        #새로 만든 bias_new 안에 있는지 없는지를 찾는다. 
        # '''    if np.amin(abs(bias_new)) < 1E-3: 
        ## 즉 0 이 아닌 값이 나오는 경우 이부분이 if가 필요한지 의문 ''' 
        nearest_zero_bias = np.where(abs(bias_new) == np.amin(abs(bias_new))) 
        # 0  에 가장가까운 값이 나온 곳의 index 를 찾음 
        bias_new = bias_new - bias_new[nearest_zero_bias] 
        # 가장 0에 가까운 값을 0으로 옮겼으니 항상 0 를 포함함. 
        #bias_new[np.where(bias_new == np.amin(abs(bias_new)))]=0

    ##############################################
    #'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn)'
    elif len(Segment) == 3:
        print('Number of Segments =' + str(len(Segment))) 
        Segments = np.array([[ float(Segments) 
                              for Segments in Seg.split(',') ] 
                             for Seg in Segment], dtype = np.float64)
        #  Segment 에서 한줄 씰 꺼내서, array, 한줄씩
        #','로 split한 문자들을 하나씩 float 변수로 바꿈, & np array화
        ### 현재 Nanonis version에서 bias 는 정확한 값이 아님. 
        Seg1 = np.linspace(Segments[0,0],Segments[0,1],int(Segments[0,-1]))
        Seg2 = np.linspace(Segments[1,0],Segments[1,1],int(Segments[1,-1]))
        Seg3 = np.linspace(Segments[2,0],Segments[2,1],int(Segments[2,-1]))
        # 겹치는  boundary 제외하고([1:]), Seg1, Seg2[1:], Seg3[1:] 합친다. 
        bias_Seg = np.append(np.append(Seg1,Seg2[1:]),Seg3[1:]) 
        # Seg1 에 Seg2[1:] 더하고, 거기에 다시한번 Se3[1:] 더함
        print ('bias_Seg size = ' + str(len(bias_Seg)))
        bias_Nsteps=int(int(Segments[1,-1])/
                        (Seg2[-1]-Seg2[0])*(bias_Seg[-1]-bias_Seg[0]))
        # 새로운 bias step 은 가장 작은 step size를 전체 영역에 적용함.    
        bias_Nsteps_size = (Seg2[-1]-Seg2[0])/(Segments[1,-1])
        # (Segments[1,0]-Segments[1,1])/int(Segments[1,-1]) # bias step size    
        Neg_bias=-1*np.arange(
            0,bias_Nsteps_size*bias_Nsteps/2, bias_Nsteps_size)
        Pos_bias=np.flip(
            np.arange(0,bias_Nsteps_size*bias_Nsteps/2,bias_Nsteps_size))
        bias_new = np.flip( np.append(Pos_bias,Neg_bias[1:])) 
        # segment 이후 bias 는 bias_new 로 재조립되었음	
        #여기에서 bias_new 를 홀수로 변환
        if len(bias_new)%2==0:
            bias_new = np.linspace(bias_new[0],bias_new[-1],num=(len(bias_new)+1)) 
        else:
            bias_new = np.linspace(bias_new[0],bias_new[-1],num=(len(bias_new))) 
        # 여기서 다시 bias_new가 0를 포함하는 확인해야함. 
        nearest_zero_bias = np.where(abs(bias_new) == np.amin(abs(bias_new))) 
        # 0  에 가장가까운 값이 나온 곳의 index 를 찾음 
        bias_new = bias_new - bias_new[nearest_zero_bias] 
        # 시작 부터 끝까지, 0포함 홀수
        print ('bias_new size = ' + str(len(bias_new)))
        # bias 
    # make a new list for Bias
    else:
        print ("Segment error /n code a 5 Sements case")
    #
    ######################################################################
    # Segment가 있더라도 모두 
    # 0을 포함한 홀수개의 같은 간격을 갖는  bias_new 로 bias 축을 바꿨음. 
    ######################################################################


    ######################################################################
    # bias_new 를 이용해서 interpolation
    # I_fwd, I_bwd, LIX_fwd, LIX_bwd
    # => I_fwd_interpolate
    #######################################################################

    def sweep_interpolation(np3Ddata, bias, bias_new):
        np3Ddata_interpolate = np.empty(
                    (np3Ddata.shape[0],np3Ddata.shape[1],bias_new.shape[0])) 
        # 원래값과 같지만, bias_new의 형태를 갖춘 empty interpolate 영역 만들고
        #xy dim 은 그대로 z 는 new bias dim으로        
        for x_i,np3Ddata_xi in enumerate(np3Ddata):
            for y_j,np3Ddata_xi_yj in enumerate(np3Ddata_xi):
                #print (np3Ddata_xi_yj.shape)
                Interpolation1D_i_f = sp.interpolate.interp1d(
                    bias,
                    np3Ddata_xi_yj,
                    fill_value = "extrapolate",
                    kind = 'cubic')
                np3Ddata_interpolate[x_i,y_j,:] = Interpolation1D_i_f(bias_new)
        return np3Ddata_interpolate

    I_fwd_interpolate = sweep_interpolation (I_fwd, bias, bias_new)
    I_bwd_interpolate = sweep_interpolation (I_bwd, bias, bias_new)
    LIX_fwd_interpolate = sweep_interpolation (LIX_fwd, bias, bias_new)
    LIX_bwd_interpolate = sweep_interpolation (LIX_bwd, bias, bias_new)

    ####################################################
    # interpolation 값과 bias_new 를 
    #
    # bias 방향 바꿈 그림 그릴때 헷갈리지 않도록. 
    ###################################################
    # Bias원래 방향에 맞춰서 둘중하나 선택 
    ###################################################
    if bias[0]>bias[-1]: 
        # 시작점이 마지막점보다 크면 (양수에서시작)
        # 변하는 것 없음. 	
        print ('start from POS bias')
        I_fwd = I_fwd_interpolate
        I_bwd = I_bwd_interpolate
        LIX_fwd = LIX_fwd_interpolate
        LIX_bwd = LIX_bwd_interpolate
        bias_mV = bias_new*1000
    else:  # 시작점이 마지막점보다 작으면 (음수에서시작)
        print ('start from NEG bias')
        I_fwd = np.flip(I_fwd_interpolate,2)
        I_bwd = np.flip(I_bwd_interpolate,2)
        LIX_fwd = np.flip(LIX_fwd_interpolate,2)
        LIX_bwd = np.flip(LIX_bwd_interpolate,2)
        bias_new_flip = np.flip(bias_new)
        # bias 0 는 POS , bias last 는 NEG bias로 바꿈. 
        bias_mV = bias_new_flip*1000
        print ('Flip => start from POS bias')
    ####################################################

    ###################################################
    # convert data XR DataSEt
    ####################################################
    

    # col = x 
    # row = y
    # I_fwd grid data ==> [Y, X, bias 로 대입]
    grid_xr = xr.Dataset(
        {
            "I_fwd" : (["Y","X","bias_mV"], I_fwd),
            "I_bwd" : (["Y","X","bias_mV"], I_bwd),
            "LIX_fwd" : (["Y","X","bias_mV"], LIX_fwd),
            "LIX_bwd" : (["Y","X","bias_mV"], LIX_bwd),
            "topography" : (["Y","X"], topography)
        },
        coords = {
            "X": (["X"], x),
            "Y": (["Y"], y),
            "bias_mV": (["bias_mV"], bias_mV)
        }
    )
    grid_xr.attrs["title"] = title
    #grid_xr.attrs['image_size'] = 
    #grid_xr.attrs['samlpe'] = 
    
    grid_xr.attrs['image_size']= [size_x,size_y]
    grid_xr.attrs['X_spacing']= step_dx
    grid_xr.attrs['Y_spacing']= step_dy    
    grid_xr.attrs['freq_X_spacing']= 1/step_dx
    grid_xr.attrs['freq_Y_spacing']= 1/step_dy

    # in case of real X Y ( center & size of XY)
    if center_offset == True:
        # move the scan center postion in real scanner field of view
        grid_xr.assign_coords( X = (grid_xr.X + cntr_x -  size_x/2))
        grid_xr.assign_coords( Y = (grid_xr.Y + cntr_y -  size_y/2))
    else :
        pass
        # (0,0) is the origin of image 
    
    
    
    return grid_xr

# -


# # <font color=blue>grid_line2xr</font>
# ## 3.  Line Spectroscopy (*.3ds)  to xarray  
#
# * input: *.3ds file (Line spectroscopy) 
# * output: Xarray (_xr) with attributes
#     * nanonis 3D data set (3ds)  $\to $ numpy $\to$ pd.DataFrame(_df) $\to$ xr.DataSet (_xr) 

# +
#griddata_file = file_list_df[file_list_df.type=='3ds'].iloc[0].file_name

def grid_line2xr(griddata_file, center_offset = True): 

    file = griddata_file
    #####################
    # conver the given 3ds file
    # to  xarray DataSet (check the attributes)

    import os
    import glob
    import numpy as np
    import numpy.fft as npf
    #import xarray as xr
    import pandas as pd
    import scipy as sp
    import matplotlib.pyplot as plt


    try:
        import nanonispy as nap
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named nanonispy')
        # !pip install nanonispy
        import nanonispy as nap

    try:
        import xarray as xr
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named xarray')
        # #!pip install --upgrade scikit-image == 0.19.0.dev0
        # !pip install xarray 
        import xarray as xr

    try:
        import seaborn_image as isns
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named seaborn-image')
        # #!pip install --upgrade scikit-image == 0.19.0.dev0
        # !pip install --upgrade seaborn-image    
        import seaborn_image as isns


    try:
        import xrft
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named xrft')
        # !pip install xrft 
        import xrft


    NF = nap.read.NanonisFile(file)
    Gr = nap.read.Grid(NF.fname)#
    ## Gr 로  해당 grid data 불러옴 # 중간에 끊기면 안됨
    channel_name = Gr.signals.keys()  # Gr 내의 data signals
    #print (channel_name)
    N = len(file);
    f_name = file[0:N-4]
    print (f_name) # Gr.basename




    #####################################
    #Header part
    #####################################
    #  Gr.header
    #####################################
    [dim_px,dim_py] = Gr.header['dim_px'] 
    [cntr_x, cntr_y] = Gr.header['pos_xy']
    [size_x,size_y] = Gr.header['size_xy']
    [step_dx,step_dy] = [ size_x/dim_px, size_y/dim_py] 
    #pixel_size # size를 pixel로 나눔

    ###   nX, nY --> x,y real scale  np array 
    nX = np.array([step_dx*(i+1/2) for i in range (0,dim_px)])# dimesion맞춘 xstep 
    nY = np.array([step_dy*(i+1/2) for i in range (0,dim_py)])# dimesion맞춘 ystep 

    x = cntr_x - size_x + nX
    y = cntr_y - size_y + nY
    # real XY position in nm scale, Center position & scan_szie + XY position
    # center position  과 scan size을 고려한 x,y real 
    #####################################
    # signal part
    # Gr.signals
    #####################################
    topography = Gr.signals['topo']
    params_v = Gr.signals['params'] 
    # params_v.shape = (dim_px,dim_py,15) 
    # 15: 3ds infos. 
    bias = Gr.signals['sweep_signal']
    # check the shape (# of 'original' bias points)
    I_fwd = Gr.signals['Current (A)'] # 3d set (dim_px,dim_py,bias)
    I_bwd = Gr.signals['Current [bwd] (A)'] # I bwd
    # sometimes, LI channel names are inconsistent depends on program ver. 
    # find 'LI Demod 1 X (A)'  or  'LI X 1 omega (A)'

    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ])
    # 'LI' & 'X' in  channel name (signal.keys) 
    LIX_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ]
    # 0 is fwd, 1 is bwd 
    LIX_fwd, LIX_bwd = Gr.signals[LIX_keys[0]] ,Gr.signals[LIX_keys[1] ]

    # same for LIY
    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ])
    # 'LI' & 'Y' in  channel name (signal.keys) 
    LIY_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ]
    # 0 is fwd, 1 is bwd 
    LIY_fwd, LIY_bwd = Gr.signals[LIY_keys[0]] ,Gr.signals[LIY_keys[1] ]


    ###########################################################
    #plt.imshow(topography) # toppography check
    #plt.imshow(I_fwd[:,:,0]) # LIX  check
    ###########################################################

    ##########################################################
    #		 Grid data 에 대한 Title 지정 
    #       grid size, pixel, bias condition 포함 #
    #############################################################
    # Gr.header.get('Bias>Bias (V)') # bias condition 
    # Gr.header.get('Z-Controller>Setpoint') # current set  condition
    # Gr.header.get('dim_px')  # jpixel dimension 
    title = Gr.basename +' ('  + str(
        float(Gr.header.get('Bias Spectroscopy>Sweep Start (V)'))
    ) +' V ~ ' +str( 
        float(Gr.header.get('Bias Spectroscopy>Sweep End (V)'))
    )+ ' V) \n at Bias = '+ Gr.header.get(
        'Bias>Bias (V)'
    )[0:-3]+' mV, I_t =  ' + Gr.header.get(
        'Z-Controller>Setpoint'
    )[0:-4]+ ' pA, '+str(
        Gr.header.get('dim_px')[0]
    )+' x '+str(
        Gr.header.get('dim_px')[1]
    )+' points'
    #############################################################       

    ### some times the topography does not look right. 
    # * then use the reshaping function 
    # only for asymmetry grid data set

    # eg) JW's MoS2 on HOPG exp. data 

    ###########################################################
    # topography  를 topography_reshape 으로 재지정.  
    ###########################################################
    topo_dimension_true = True


    if topo_dimension_true == True:
        topography_reshape = topography   
        #################################
        I_fwd_copy = I_fwd
        I_bwd_copy = I_bwd
        LIX_fwd_copy = LIX_fwd 
        LIX_bwd_copy = LIX_bwd 	
        # topography가 정상인 경우    
        #################################

        ##########################################################
        # 예를 들어  #  MoS2 on HOPG image 는 
        # 40 x 80 의 배열이 40x40 + 40 x40으로 되었음. 
        # x 한줄이 0-39: 1st line 40-79 : 2nd line임
        # 0-40, 19-59, 39-79 가 set로 움직임. 
        # 세로로 40X80 배열을 만들어서 
        # 0-39 를 2n으로 40-79 를 2n+1 으로 대입할것. 
        # topo # LIX f&b # I f&b #
        ##########################################################

    else: # topography가 비 정상인 경우  
        # topo_dimension_true == False 일경우 
        topography_reshape = np.transpose(np.copy(topography),(1,0)) 
        # 바꾼 dimension이 될 곳을 미리 만듬. 
        for x_indx, y_indx in enumerate (topography):
        # print(x_indx) # 0-39 # print(y_indx.shape)
            topography_reshape[2*x_indx,:] = y_indx[:40] # 앞쪽 절반은 첫번째줄로
            topography_reshape[2*x_indx+1,:] = y_indx[40:80] # 뒷쪽 절반은 두번째 줄로  
        #################################
        # 새로 바꾸는 정렬 방법을 확인하여 같은방식으로 I, LIX 에도 적용
        #################################
        #제대로 붙었는지 테스트
        plt.imshow(topography_reshape) # 80 * 40 OK
        # topography_reshape 정렬 끝 
        #################################
        I_fwd_copy = np.transpose(np.copy(I_fwd),(1,0,2))
        I_bwd_copy = np.transpose(np.copy(I_bwd),(1,0,2)) 
        # 바꾼 dimension이 될 곳을 미리 만듬. 	
        for x_indx, yNbias_plane in enumerate (I_fwd): 
            # 순서를바꿔서 두번째 index로 for loop
            print(x_indx) # 0-39 
            I_fwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            I_fwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로  

        for x_indx, yNbias_plane in enumerate (I_bwd): 
            # 순서를바꿔서 두번째 index로 for loop
            print(x_indx) # 0-39 
            I_bwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            I_bwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로plt.imshow(I_bwd_v[:,:,0]) # 40 * 256  cut 
        #################################
        # I reshape is done 
        #################################
        LIX_fwd_copy = np.transpose(np.copy(LIX_fwd),(1,0,2)) 
        LIX_bwd_copy = np.transpose(np.copy(LIX_bwd),(1,0,2)) 
        # 바꾼 dimension이 될 곳을 미리 만듬. 
        for x_indx, yNbias_plane in enumerate (LIX_fwd): 
            # 순서를바꿔서 두번째 index로 for loop
            LIX_fwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            LIX_fwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로  
        for x_indx, yNbias_plane in enumerate (LIX_bwd): 
            # 순서를바꿔서 두번째 index로 for loop
            LIX_bwd_copy[2*x_indx,:,:] = yNbias_plane[:40,:] 
            # 앞쪽 절반은 첫번째줄로
            LIX_bwd_copy[2*x_indx+1,:,:] = yNbias_plane[40:80,:] 
            # 뒷쪽 절반은 두번째 줄로
        #################################
        # LIX reshape is done 
        #################################
    #제대로 붙었는지 테스트

    #fig,axes = plt.subplots (nrows = 2,ncols = 3, figsize = (5,4),
    #                         sharex=True, sharey=True)
    #axs = axes.ravel()
    #axs[0].imshow(topography)
    #axs[0].set_title('topography')

    #axs[1].imshow(I_fwd_copy[:,:,0]) # 80 * 40 OK
    #axs[1].set_title('I_fwd[0]')
    #axs[2].imshow(LIX_fwd_copy[:,:,0]) # 80 * 40 OK
    #axs[2].set_title('LIX_fwd[0]')

    #axs[3].imshow(topography_reshape)
    #axs[3].set_title('topo_reshape')

    #axs[4].imshow(I_bwd_copy[:,:,0]) # 80 * 40 OK
    #axs[4].set_title('I_bwd[0]')

    #axs[5].imshow(LIX_bwd_copy[:,:,0]) # 80 * 40 OK
    #axs[5].set_title('LIX_bwd[0]')

    #fig.tight_layout()
    #plt.show()



    # after reshaping 

    topography = topography_reshape 
    #################################
    I_fwd = I_fwd_copy 
    I_bwd = I_bwd_copy 
    LIX_fwd  = LIX_fwd_copy 
    LIX_bwd  = LIX_bwd_copy
    ##########################################################





    ###########################
    # Bias segment check      #
    ###########################
    Segment = Gr.header['Bias>Bias (V)']
    # bias unit : '(V)' 

    if type(Segment) == str: # single segment case
        print ('No Segments\n'+ 'Grid data acquired at bias = '+  str(float(Segment)) + 'V')    
    ## No Segments # +  bias setting 

    ########################
    # bias interpolation to have bias 
    # bias_mV 는 bias 를 0 포함한 값으로interploation 
    # 3D data 가운데 x,y 를 꺼내서 bias interpolation 
    # e.g  256--> 양 끝점을 포함한 256+1 으로. (center 가 0가 되도록 )
        if len(bias)%2==0:
            bias_new = np.linspace(bias[0],bias[-1],num=(len(bias)+1)) 
            # 시작 부터 끝까지, 0포함 홀수, 전체크기+1 개의 단계로 세분화
        else:
            bias_new = np.linspace(bias[0],bias[-1],num=(len(bias))) 
            # bias_new  는 간격이 홀수개가 되도록 나눈 것임, 즉 0에가장 가까운 값이 하나뿐이도록 조정함 
        #새로 만든 bias_new 안에 있는지 없는지를 찾는다. 
        # '''    if np.amin(abs(bias_new)) < 1E-3: 
        ## 즉 0 이 아닌 값이 나오는 경우 이부분이 if가 필요한지 의문 ''' 
        nearest_zero_bias = np.where(abs(bias_new) == np.amin(abs(bias_new))) 
        # 0  에 가장가까운 값이 나온 곳의 index 를 찾음 
        bias_new = bias_new - bias_new[nearest_zero_bias] 
        # 가장 0에 가까운 값을 0으로 옮겼으니 항상 0 를 포함함. 
        #bias_new[np.where(bias_new == np.amin(abs(bias_new)))]=0

    ##############################################
    #'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn)'
    elif len(Segment) == 3:
        print('Number of Segments =' + str(len(Segment))) 
        Segments = np.array([[ float(Segments) 
                              for Segments in Seg.split(',') ] 
                             for Seg in Segment], dtype = np.float64)
        #  Segment 에서 한줄 씰 꺼내서, array, 한줄씩
        #','로 split한 문자들을 하나씩 float 변수로 바꿈, & np array화
        ### 현재 Nanonis version에서 bias 는 정확한 값이 아님. 
        Seg1 = np.linspace(Segments[0,0],Segments[0,1],int(Segments[0,-1]))
        Seg2 = np.linspace(Segments[1,0],Segments[1,1],int(Segments[1,-1]))
        Seg3 = np.linspace(Segments[2,0],Segments[2,1],int(Segments[2,-1]))
        # 겹치는  boundary 제외하고([1:]), Seg1, Seg2[1:], Seg3[1:] 합친다. 
        bias_Seg = np.append(np.append(Seg1,Seg2[1:]),Seg3[1:]) 
        # Seg1 에 Seg2[1:] 더하고, 거기에 다시한번 Se3[1:] 더함
        print ('bias_Seg size = ' + str(len(bias_Seg)))
        bias_Nsteps=int(int(Segments[1,-1])/
                        (Seg2[-1]-Seg2[0])*(bias_Seg[-1]-bias_Seg[0]))
        # 새로운 bias step 은 가장 작은 step size를 전체 영역에 적용함.    
        bias_Nsteps_size = (Seg2[-1]-Seg2[0])/(Segments[1,-1])
        # (Segments[1,0]-Segments[1,1])/int(Segments[1,-1]) # bias step size    
        Neg_bias=-1*np.arange(
            0,bias_Nsteps_size*bias_Nsteps/2, bias_Nsteps_size)
        Pos_bias=np.flip(
            np.arange(0,bias_Nsteps_size*bias_Nsteps/2,bias_Nsteps_size))
        bias_new = np.flip( np.append(Pos_bias,Neg_bias[1:])) 
        # segment 이후 bias 는 bias_new 로 재조립되었음	
        #여기에서 bias_new 를 홀수로 변환
        if len(bias_new)%2==0:
            bias_new = np.linspace(bias_new[0],bias_new[-1],num=(len(bias_new)+1)) 
        else:
            bias_new = np.linspace(bias_new[0],bias_new[-1],num=(len(bias_new))) 
        # 여기서 다시 bias_new가 0를 포함하는 확인해야함. 
        nearest_zero_bias = np.where(abs(bias_new) == np.amin(abs(bias_new))) 
        # 0  에 가장가까운 값이 나온 곳의 index 를 찾음 
        bias_new = bias_new - bias_new[nearest_zero_bias] 
        # 시작 부터 끝까지, 0포함 홀수
        print ('bias_new size = ' + str(len(bias_new)))
        # bias 
    # make a new list for Bias
    else:
        print ("Segment error /n code a 5 Sements case")
    #
    ######################################################################
    # Segment가 있더라도 모두 
    # 0을 포함한 홀수개의 같은 간격을 갖는  bias_new 로 bias 축을 바꿨음. 
    ######################################################################


    ######################################################################
    # bias_new 를 이용해서 interpolation
    # I_fwd, I_bwd, LIX_fwd, LIX_bwd
    # => I_fwd_interpolate
    #######################################################################

    def sweep_interpolation(np3Ddata, bias, bias_new):
        np3Ddata_interpolate = np.empty(
                    (np3Ddata.shape[0],np3Ddata.shape[1],bias_new.shape[0])) 
        # 원래값과 같지만, bias_new의 형태를 갖춘 empty interpolate 영역 만들고
        #xy dim 은 그대로 z 는 new bias dim으로        
        for x_i,np3Ddata_xi in enumerate(np3Ddata):
            for y_j,np3Ddata_xi_yj in enumerate(np3Ddata_xi):
                #print (np3Ddata_xi_yj.shape)
                Interpolation1D_i_f = sp.interpolate.interp1d(
                    bias,
                    np3Ddata_xi_yj,
                    fill_value = "extrapolate",
                    kind = 'cubic')
                np3Ddata_interpolate[x_i,y_j,:] = Interpolation1D_i_f(bias_new)
        return np3Ddata_interpolate

    I_fwd_interpolate = sweep_interpolation (I_fwd, bias, bias_new)
    I_bwd_interpolate = sweep_interpolation (I_bwd, bias, bias_new)
    LIX_fwd_interpolate = sweep_interpolation (LIX_fwd, bias, bias_new)
    LIX_bwd_interpolate = sweep_interpolation (LIX_bwd, bias, bias_new)

    ####################################################
    # interpolation 값과 bias_new 를 
    #
    # bias 방향 바꿈 그림 그릴때 헷갈리지 않도록. 
    ###################################################
    # Bias원래 방향에 맞춰서 둘중하나 선택 
    ###################################################
    if bias[0]>bias[-1]: 
        # 시작점이 마지막점보다 크면 (양수에서시작)
        # 변하는 것 없음. 	
        print ('start from POS bias')
        I_fwd = I_fwd_interpolate
        I_bwd = I_bwd_interpolate
        LIX_fwd = LIX_fwd_interpolate
        LIX_bwd = LIX_bwd_interpolate
        bias_mV = bias_new*1000
    else:  # 시작점이 마지막점보다 작으면 (음수에서시작)
        print ('start from NEG bias')
        I_fwd = np.flip(I_fwd_interpolate,2)
        I_bwd = np.flip(I_bwd_interpolate,2)
        LIX_fwd = np.flip(LIX_fwd_interpolate,2)
        LIX_bwd = np.flip(LIX_bwd_interpolate,2)
        bias_new_flip = np.flip(bias_new)
        # bias 0 는 POS , bias last 는 NEG bias로 바꿈. 
        bias_mV = bias_new_flip*1000
        print ('Flip => start from POS bias')
    ####################################################

    ###################################################
    # convert data XR DataSEt
    ####################################################


    grid_xr = xr.Dataset(
        {
            "I_fwd" : (["Y","X","bias_mV"], I_fwd),
            "I_bwd" : (["Y","X","bias_mV"], I_bwd),
            "LIX_fwd" : (["Y","X","bias_mV"], LIX_fwd),
            "LIX_bwd" : (["Y","X","bias_mV"], LIX_bwd),
            "topography" : (["Y","X"], topography)
        },
        coords = {
            "X": (["X"], x),
            "Y": (["Y"], y),
            "bias_mV": (["bias_mV"], bias_mV)
        }
    )
    grid_xr.attrs["title"] = title
    #grid_xr.attrs['image_size'] = 
    #grid_xr.attrs['samlpe'] = 
    
    grid_xr.attrs['image_size']= [size_x,size_y]
    grid_xr.attrs['X_spacing']= step_dx
    grid_xr.attrs['Y_spacing']= step_dy    
    #grid_xr.attrs['freq_X_spacing']= 1/step_dx
    #grid_xr.attrs['freq_Y_spacing']= 1/step_dy

    # in case of real X Y ( center & size of XY)
    if center_offset == True:
        # move the scan center postion in real scanner field of view
        grid_xr.assign_coords( X = (grid_xr.X + cntr_x -  size_x/2))
        grid_xr.assign_coords( Y = (grid_xr.Y + cntr_y -  size_y/2))
    else :
        pass
        # (0,0) is the origin of image 
    
    return grid_xr

# -


# # <font color=blue>gwy_image2df & gwy_df_channel2xr</font>
# ## 4. Gwyddion 2D image to PANDAS Dataframe or Xarray
# ### 4.1. **gwy_image2df** : gwy file name $\to$ PANDAS DataFrame
# * input: *.gwy file 
# * output: PANDAS DataFrame
#     * gwyddion 2D image data (*gwy)  $\to $ numpy $\to$ pd.DataFrame(_df) 
#
# ### 4.2. **gwy_df_channel2xr** : Choose a data channe in gwy_df $\to$ Xarray DataArray
# * input: gwy_df dataframe & channel number ( N=0)
# * output: Xarray DataSet 
#     * pd.DataFrame(_df)  $\to $ xarray Dataset (_xr)
#
#

# +
def gwy_image2df (gwy_file_name):
    import pandas as pd
    try:
        import gwyfile
    except ModuleNotFoundError:
        warn('ModuleNotFoundError: No module named gwyfile')
        # !pip install gwyfile
        import gwyfile
    gwyfile_df = pd.DataFrame(gwyfile.util.get_datafields(gwyfile.load(gwy_file_name)))
    # convert all gwy file channels to pd.DataFrame
    pd.set_option('display.float_format', '{:.3e}'.format)
    return gwyfile_df

#gwy_df = gwyImage2df( file_list_df.file_name[1])



# +

def gwy_df_channel2xr (gwy_df, ch_N=0): 
    import pandas as pd
    #convert a channel data to xr DataArray format
    chN_df = gwy_df.iloc[:,ch_N]
    chNdf_temp = pd.DataFrame(chN_df.data.reshape((chN_df.yres, chN_df.xres))).stack()
    chNdf_temp = chNdf_temp.rename_axis (['Y','X'])
    x_step = chN_df.xreal / chN_df.xres 
    y_step = chN_df.yreal / chN_df.yres 
    chNxr = chNdf_temp.to_xarray()
    chNxr = chNxr.assign_coords(X = chNxr.X.values * x_step, 
                                Y = chNxr.Y.values * y_step )
    return chNxr

# gwy_ch_xr = gwy_df_channel2xr(gwy_df, ch_N=3)

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # updated 2021 0607
#
# ##  * To Be Continued
# -

