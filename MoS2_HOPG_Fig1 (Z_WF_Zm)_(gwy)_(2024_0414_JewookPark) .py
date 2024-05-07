# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Understanding Interlayer Coupling in Few-Layer $MoS_{2}$  through Stacking Configuration Control
# * Stacking dependent bandgap of $MoS_{2}$ on (highly )HOPG substrates 
#
# # <font color=blue>Fig 1- AFM + KPFM data analysis (Fig 1- large scan)</font>
#
# ## ($MoS_{2}$ islands on HOPG substrates) STS data analysis data analysis 
#
# > * file loading : **SPMpy_file_loading_funcs**
#     > * Loading gwyddion file (*.gwy) loading (after image treatments with Gwyddion)
#
# > * 3D data analysis functions : **SPMpy_3D_data_analysis_funcs**
# > * 2D data analysis functions : **SPMpy_2D_data_analysis_funcs**
#
#
# * Authors : Dr. Jewook Park(ORNL, & IBS )
#     * *Center for Nanophase Materials Sciences, Oak Ridge National Laboratory, Oak Ridge, Tennessee 37831, USA *
#     * *Center for van der Waals Quantum Solid, Institute for Basic Science (IBS), Pohang 37673, Korea* 
#     * *Center for Artificial Low Dimensional Electronic Systems, Institute for Basic Science (IBS), Pohang, Korea* 
#     * email :  parkj1@ornl.gov
#
# > **SPMpy** is a python package for scanning probe microscopy (SPM) data analysis, such as scanning tunneling microscopy and spectroscopy (STM/S) data and atomic force microscopy (AFM) images, which are inherently multidimensional. To analyze SPM data, SPMpy exploits recent image processing(a.k.a. Computer Vision) techniques. SPMpy data analysis functions utilize well-established Python packages, such as Numpy, PANDAS, matplotlib, Seaborn, holoview, etc. In addition, many parts are inspired by well-known SPM data analysis programs, for example, Wsxm and Gwyddion. Also, SPMpy is trying to apply lessons from 'Fundamentals in Data Visualization'(https://clauswilke.com/dataviz/).
#
# >  **SPMpy** is an open-source project. (Github: https://github.com/jewook-park/SPMpy_ORNL )
# > * Contributions, comments, ideas, and error reports are always welcome. Please use the Github page or email jewookpark@ibs.re.kr. Comments & remarks should be in Korean or English. 
#
#

# %% [markdown]
# ###  0.Environment Preparation 
# * Import modules and functions 
#     * 0.0. Import necessary packages
#     * 0.1. loading **SPMpy** functions  

# %% id="Qm1zLaTHbSpK"
########################################
    #    * Step 1-1
    #    : Import necessary packages 
    #        import modules        
#########################################

import os
import glob
import numpy as np
import pandas as pd
import scipy as sp
from warnings import warn
from scipy import signal

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns
import skimage
import seaborn_image as isns

# some packages may be yet to be installed
# please install "seaborn-image" via conda to avoid install error 
# conda install -c conda-forge seaborn-image
try:
     from pptx import Presentation
except ModuleNotFoundError:
    warn('ModuleNotFoundError: No module named Presentation')
    # !pip install python-pptx  
    from pptx import Presentation
    from pptx.util import Inches, Pt

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
    
    # !pip install xarray 
    import xarray as xr
    
try:
    import xrft
except ModuleNotFoundError:
    warn('ModuleNotFoundError: No module named xrft')
    # !pip install xrft 
    import xrft
    
    
try:
    import holoviews as hv
except ModuleNotFoundError:
    warn('ModuleNotFoundError: No module named holoviews')
    # !pip install holoviews 
    import holoviews as hv
    
try:
    import hvplot.xarray
    import hvplot.pandas 
except ModuleNotFoundError:
    warn('ModuleNotFoundError: No module named hvplot')
    # !pip install hvplot
    import hvplot.xarray
    import hvplot.pandas 

try:
    import gwyfile
except ModuleNotFoundError:
    warn('ModuleNotFoundError: No module named gwyfile')
    # !pip install gwyfile
    import gwyfile
 
    
from SPMpy_file_loading_funcs import *
from SPMpy_2D_data_analysis_funcs import *
from SPMpy_3D_data_analysis_funcs import *




# %% [markdown]
# ###  1.1.Load the files and select dataset(xr) 
# * Checkup Current Working Directory
# * use **files_in_folder**
#     * file_list_df columns  = [group,num,file_name,type]

# %%
# check the sxm(or gwy) files in the given folder
#target_path = r'C:\IBS CALDES data\IBS Epitaxial vdW Quantum Solid\Papers\Preparation of pyramid and screw MoS2 on HOPG paper\Figure Preparation\Figure 1\2022 0619 Fig1 KPFM dataset'
target_path = r'C:\Users\gkp\OneDrive - Oak Ridge National Laboratory\Papers\Preparation of pyramid and screw MoS2 on HOPG paper\Figure Preparation\Figure 1\2022 0619 Fig1 KPFM dataset'
file_list_df = files_in_folder(target_path)
## Loading Image Channels
file_list_df


# %% [markdown]
# ### 1.2. select gwy_image files from file_list_df
# > I used the zoom in data set from the gwyddion. 
#
#
# ### 1.3. select W & S type islands data in Xarray format 
# > * W_xr = W_topo + W_WF + W_topo_zm + W_WF_zm
# > * S_xr = S_topo + S_WF + S_topo_zm + S_WF_zm
#

# %%
#gwy_image2df( file_list_df.file_name[1])
# gwy_image2df( file_list_df.file_name[2])
# test the file name
gwy_image2df( file_list_df.file_name[3])

# %%
# choose the channel according to the gwy file 

#######################################################
# gwy file, work function (WF) channel name is not correct.  change +- sign to correct.!
#########################################################


# choose the Wedding-cake 
W_topo_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[1]), 0)
W_WF_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[1]), 1)

W_topo_zm_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[1]), 2)
W_WF_zm_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[1]), 3)
# choose the Zoom in 3 #   (for the ZM2 , channel 2&3)


# choose the Spiral 
S_topo_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[3]), 2)
S_WF_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[3]),3)

S_topo_zm_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[3]),1)
S_WF_zm_xr  = gwy_df_channel2xr (gwy_image2df( file_list_df.file_name[3]), 0)
# choose the Zoom in 3 #   (for the ZM2 , channel 2&3)

# Add "name"  for "xr.merge"
W_topo_xr.name  = "W_topo"
W_WF_xr.name  = "W_WF"
W_topo_zm_xr.name  = "W_topo_zm"
W_WF_zm_xr.name  = "W_WF_zm"
S_topo_xr.name  = "S_topo"
S_WF_xr.name  = "S_WF"
S_topo_zm_xr.name  = "S_topo_zm"
S_WF_zm_xr.name  = "S_WF_zm"
# set a channel name for xr merge 

#######################################################
# gwy file, work function (WF) channel is always  not correct. 
# sometimes change +- sign to correct it accordingly.!
#########################################################

W_0_xr = xr.merge([W_topo_xr, W_WF_xr])
W_zm_xr = xr.merge([W_topo_zm_xr,W_WF_zm_xr])

S_0_xr = xr.merge([S_topo_xr, S_WF_xr])
S_zm_xr = xr.merge([S_topo_zm_xr, S_WF_zm_xr])


#Gaussian smoothing 
W_0_xr_g = filter_gaussian_xr(W_0_xr, sigma=3)
S_0_xr_g = filter_gaussian_xr(S_0_xr, sigma=3)

W_zm_xr_g = filter_gaussian_xr(W_zm_xr, sigma=3)
S_zm_xr_g = filter_gaussian_xr(S_zm_xr, sigma=3)



# %%
print (W_0_xr)
print  (W_zm_xr)
print (S_0_xr)
print  (S_zm_xr)

# %% [markdown]
# ### 2. Plot dataset 
#
# * 2.1. figure 1 AB, Topo &WF together
# * 2.2. figure 1 AB, Topo only, savefig (pdf or svg)

# %%
# %matplotlib inline

fig,axes = plt.subplots(ncols=2,nrows =2, figsize  =(8,8))
axs = axes.ravel()
isns.imshow(W_0_xr.W_topo*1E9,
            robust = True, perc=(0.5,99.5),
            cmap = 'viridis',  
            dx=1, 
            units="nm", 
            cbar_label = 'z (nm)',
            ax = axs[0])
axs[0].set_title('W(Z)')
isns.imshow(W_0_xr.W_WF*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            cbar_label = '$\Phi$ (mV)',
            ax = axs[2])
axs[2].set_title('W(WF)')
isns.imshow(S_0_xr.S_topo*1E9,
            robust = True,perc=(0.5,99.5),
            cmap = 'viridis',
            dx=1, units="nm", 
            cbar_label = 'z (nm)', 
            ax = axs[1])
axs[1].set_title('S(Z)')
isns.imshow(S_0_xr.S_WF*1E3, 
            robust = True,
            cmap = 'Greens_r', 
            cbar_label = '$\Phi$ (mV)',
            ax = axs[3])
axs[3].set_title('S(WF)')

plt.tight_layout()
plt.savefig('W_S_Z_WF.svg')
plt.show()
#isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_topo.unstack(),robust = True, cmap = 'copper',  dx=1, units="nm")
#isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_WF.unstack(),robust = True, cmap = 'Blues' )


# %% [markdown]
# ## Add CPD data channel 

# %%
W_zm_xr['W_CPD_zm']= -1*W_zm_xr.W_WF_zm
S_zm_xr['S_CPD_zm']= -1*S_zm_xr.S_WF_zm
# use the sige changes as an CPD  
# here, we ignored offset due tip work function to compare layer dependence

S_zm_xr
W_zm_xr

# %%
# %matplotlib inline

fig,axes = plt.subplots(ncols=2,nrows =2, figsize  =(8,8))
axs = axes.ravel()
isns.imshow(W_zm_xr.W_topo_zm*1E9,
            robust = True, perc=(0.5,99.5),
            cmap = 'viridis',  
            dx=1, 
            units="nm", 
            cbar_label = 'z (nm)',
            ax = axs[0])
axs[0].set_title('W(Z)')
isns.imshow(W_zm_xr.W_CPD_zm*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            cbar_label = 'CPD (mV)',
            ax = axs[2])
axs[2].set_title('W(CPD)')
isns.imshow(S_zm_xr.S_topo_zm*1E9,
            robust = True,perc=(0.5,99.5),
            cmap = 'viridis',
            dx=1, units="nm", 
            cbar_label = 'z (nm)', 
            ax = axs[1])
axs[1].set_title('S(Z)')
isns.imshow(S_zm_xr.S_CPD_zm*1E3, 
            robust = True,
            cmap = 'Greens_r', 
            cbar_label = 'CPD (mV)',
            ax = axs[3])
axs[3].set_title('S(CPD)')

plt.tight_layout()
plt.savefig('W_S_Z_CPD.svg')
plt.show()



# %%

#### save W_z and S_z
fig,ax = plt.subplots(1,1, figsize  =(4,3))
isns.imshow(W_0_xr.W_topo*1E9,
            robust = True, 
            height = 4, 
            cmap = 'viridis', 
            dx=1, 
            units="nm", 
            cbar_label = 'z (nm)', 
            ax = ax)
plt.tight_layout()
plt.savefig('W_z.svg', dpi = 300)

fig,ax = plt.subplots(1,1, figsize  =(4,3))
isns.imshow(S_0_xr.S_topo*1E9,
            robust = True, 
            height = 4, 
            cmap = 'viridis', 
            dx=1, 
            units="nm", 
            cbar_label = 'z (nm)',
            ax = ax)
plt.tight_layout()
plt.savefig('S_z.svg', dpi = 300)

isns.set_context(mode="paper", fontfamily="sans-serif")

fig,axes = plt.subplots(2,2, figsize  =(8,6))
axs = axes.ravel()
isns.imshow(W_zm_xr.W_topo_zm*1E9,
            robust = True,
            aspect = 'equal', 
            cmap = 'viridis',  
            dx=1, 
            units="nm", 
            cbar_label = 'z (nm)',
            fontsize='xx-large',
            ax = axs[0])
axs[0].set_title('W(Z)')
isns.imshow(W_zm_xr.W_WF_zm*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            cbar_label = '$\Phi$ (mV)',
            dx=1, units="nm", 
            fontsize='xx-large',
            ax = axs[2])
axs[2].set_title('W(WF)')
isns.imshow(S_zm_xr.S_topo_zm*1E9,
            robust = True,
            aspect = 'equal', 
            cmap = 'viridis', 
            dx=1, units="nm", 
            cbar_label = 'z (nm)',
            fontsize='xx-large',
            ax = axs[1])
axs[1].set_title('S(Z)')
isns.imshow(S_zm_xr.S_WF_zm*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            cbar_label = '$\Phi$ (mV)', 
            dx=1, units="nm", 
            ax = axs[3])
axs[3].set_title('S(WF)')

plt.tight_layout()
plt.savefig('W_S_Z_WF_z.svg')
plt.show()



# %% [markdown]
# ## Thresholding for Zoom-in 
#

# %% [markdown]
# ### W_zm_xr
#     * W_zm_xr $\to$ W_zm_topo_th  $\to$ assign terraces $\to$  W_zm_height_xr.height 
#     * $\to$ W_zm_xr.height $\to$ W_zm_height_xr_sobel$\to$  height_sobel.rolling 
#     * $\to$W_zm_xr.edge 

# %% [markdown]
# ### Wedding-cake terraces
#
# |HOPG | 1ML | 2ML | 3ML | 4ML |
# | :- | -: | :-: | :-: | :-: |
# |0   | 1 |   2|   3 |  4 |

# %%
W_zm_topo_th = threshold_multiotsu_xr(W_zm_xr, multiclasses=5)
#W_zm_topo_th.W_topo_zm.plot(cmap = 'copper')

W_zm_topo_th['height'] = W_zm_topo_th.W_topo_zm.copy()
W_zm_th_df = W_zm_topo_th.height.to_dataframe()
W_zm_height = W_zm_th_df.copy()
W_zm_height[W_zm_th_df ==0 ] = 0
W_zm_height[W_zm_th_df ==1 ] = 1
W_zm_height[W_zm_th_df ==2 ] = 2
W_zm_height[W_zm_th_df ==3 ] = 3
W_zm_height[W_zm_th_df ==4 ] = 4
W_zm_height[W_zm_th_df ==5 ] = 5
W_zm_height_xr = W_zm_height.to_xarray()

#W_zm_height_xr.height.plot()
#check the terrace 

# %% [markdown]
# ### edge finding with sobel 

# %%
W_zm_xr = W_zm_xr.merge(W_zm_height_xr)
# W_zm_xr  has a height info now 

W_zm_height_xr_sobel =  threshold_mean_xr(filter_sobel_xr(W_zm_height_xr))
#W_xr_th_sobel.W_topo_gaussian_sobel)
#S_xr_th_sobel = FilterSobel_xr(S_xr_th)

#ThrshldMean_xr
isns.imshow(W_zm_height_xr_sobel.height_sobel)

W_zm_xr['edge']  = W_zm_height_xr_sobel.height_sobel.rolling(X = 5, Y= 5,  min_periods = 2 , center  = True).mean().notnull()
#W_zm_xr.edge.plot() # check 

#isns.imshow(W_zm_height_xr_sobel.height_sobel)
#W_zm_xr['edge'] 
#W_zm_xr

# %% [markdown]
# ### check the boundary based on topography 

# %%
fig,axes =  plt.subplots(2,1)
axs = axes.ravel()
isns.imshow(W_zm_topo_th.W_topo_zm, ax =  axs[0], cmap = 'copper')
#isns.imshow(W_zm_height_xr_sobel.height_sobel, ax = axs[1], cmap = 'copper')
isns.imshow(W_zm_xr.edge, ax = axs[1], cmap = 'copper')
plt.show()
# choose the multiOtsu results to assigne the terrace 


# %% [markdown]
#     
# ### S_zm_xr

# %% [markdown]
# ### Spiral terraces
# |HOPG | 1ML | 2ML | 3ML | 4ML| 5ML |
# | :- | -: | :-: | :-: | :-: |:-: |
# |0   | 1 |   2|   3 |  4 |  5 |

# %%
## call saved file 
"""
S_zm_topo_th = threshold_multiotsu_xr(S_zm_xr, multiclasses=6)
"""


#S_zm_topo_th.S_topo_zm.plot()
#S_zm_topo_th.S_topo_zm.plot()

# %%
"""
S_zm_topo_th.to_netcdf("S_zm_topo_th.nc")

"""


# S_zm_topo_th take too long time to calcuate
# save it as nc format

# %%
#
S_zm_topo_th = xr.open_dataset("S_zm_topo_th.nc")

# %%
# remove the defect manually

S_zm_topo_th['height'] = S_zm_topo_th.S_topo_zm.copy()
S_zm_th_df = S_zm_topo_th.height.to_dataframe()
S_zm_height = S_zm_th_df.copy()

S_zm_th_df['set_X_range'] = S_zm_th_df.index.get_level_values(level=1)>1E-7
# multi index 에서 x에 해당하는  value 범위지정. 

S_zm_height[S_zm_th_df.height ==0 ] = 0
S_zm_height[S_zm_th_df.height ==1 ] = 1
S_zm_height[S_zm_th_df.height ==2 ] = 2
S_zm_height[S_zm_th_df.height ==3 ] = 3
S_zm_height[S_zm_th_df.height ==4 ] = 4
S_zm_height[(S_zm_th_df.height == 5 ) & (S_zm_th_df.set_X_range == False ) ] = 5
S_zm_height[(S_zm_th_df.height == 5 ) & (S_zm_th_df.set_X_range == True ) ] = 6
S_zm_height_xr = S_zm_height.to_xarray()

#S_zm_height_xr.height.plot()
#check the terrace 

# %%
S_zm_xr = S_zm_xr.merge(S_zm_height_xr)
# S_zm_xr  has a height info now 

S_zm_height_xr_sobel =  threshold_mean_xr(filter_sobel_xr(S_zm_height_xr))

#ThrshldMean_xr
isns.imshow(S_zm_height_xr_sobel.height_sobel)


S_zm_xr['edge']  = S_zm_height_xr_sobel.height_sobel.rolling(X = 7, Y= 7,  min_periods = 2 , center  = True).mean().notnull()
#S_zm_xr.edge.plot()

S_zm_xr

# %%
fig,axes =  plt.subplots(2,1)
axs = axes.ravel()
isns.imshow(S_zm_topo_th.S_topo_zm, ax =  axs[0], cmap = 'copper')
isns.imshow(S_zm_xr.edge, ax = axs[1], cmap = 'copper')
plt.show()
# choose the multiOtsu results to assigne the terrace 



# %%
#S_zm_xr
#W_zm_xr


W_xr_df = W_zm_xr.to_dataframe()
S_xr_df = S_zm_xr.to_dataframe()

# PANDAS masks for W
Wmask_0 = (~ W_xr_df.edge ) & (W_xr_df.height == 0) 
Wmask_1 = (~ W_xr_df.edge ) & (W_xr_df.height == 1)
Wmask_2 = (~ W_xr_df.edge ) & (W_xr_df.height == 2)
Wmask_3 = (~ W_xr_df.edge ) & (W_xr_df.height == 3)
Wmask_4 = (~ W_xr_df.edge ) & (W_xr_df.height == 4)

W_xr_df['layer'] = 'boundary'
W_xr_df.layer[Wmask_0] = 'HOPG'
W_xr_df.layer[Wmask_1] = '1ML'
W_xr_df.layer[Wmask_2] = '2ML'
W_xr_df.layer[Wmask_3] = '3ML'
W_xr_df.layer[Wmask_4] = '4ML'
W_xr_df.layer.astype(str)

# PANDAS masks for S
Smask_0 = ( ~S_xr_df.edge ) & (S_xr_df.height == 0) 
Smask_1 = ( ~S_xr_df.edge ) & (S_xr_df.height == 1)
Smask_2 = ( ~S_xr_df.edge ) & (S_xr_df.height == 2)
Smask_3 = ( ~S_xr_df.edge ) & (S_xr_df.height == 3)
Smask_4 = ( ~S_xr_df.edge ) & (S_xr_df.height == 4)
Smask_5 = ( ~S_xr_df.edge ) & (S_xr_df.height == 5)

S_xr_df['layer'] = 'boundary'
S_xr_df.layer[Smask_0] = 'HOPG'
S_xr_df.layer[Smask_1] = '1ML'
S_xr_df.layer[Smask_2] = '2ML'
S_xr_df.layer[Smask_3] = '3ML'
S_xr_df.layer[Smask_4] = '4ML'
S_xr_df.layer[Smask_5] = '5ML'
S_xr_df.layer.astype(str)

#####
# xarray 
W_df_xr = W_xr_df.to_xarray()
S_df_xr = S_xr_df.to_xarray()


# %%
W_zm_xr

# %%
#W_df_xr
#S_df_xr

fig,axes = plt.subplots(2,2, figsize  =(8,6))
axs = axes.ravel()
isns.set_scalebar(color = "k", location='lower left')
isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_topo_zm.unstack()*1E9,
            robust = True,
            aspect = 'equal', 
            cmap = 'viridis',  
            dx=1, 
            units="nm", 
            cbar_label = 'z (nm)', 
            ax = axs[0])
axs[0].set_title('W(Z)')


isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_CPD_zm.unstack()*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            cbar_label = 'CPD (mV)', 
            dx=1, units="nm", 
            ax = axs[2])
axs[2].set_title('W(CPD)')
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_topo_zm.unstack()*1E9,
            robust = True,
            aspect = 'equal', 
            cmap = 'viridis', 
            dx=1, units="nm", 
            cbar_label = 'z (nm)',
            ax = axs[1])
axs[1].set_title('S(Z)')
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_CPD_zm.unstack()*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            cbar_label = 'CPD (mV)',
            dx=1, units="nm", 
            ax = axs[3])
axs[3].set_title('S(CPD)')

plt.tight_layout()
plt.savefig('W_S_Z_CPD_z_remove edge.svg')
plt.show()



'''
isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_topo_zm.unstack(),robust = True, cmap = 'copper',  dx=1, units="nm")
isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_WF_zm.unstack(),robust = True, cmap = 'Blues' )

# hv plot is not working well. 
# use the isns instead, 
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_topo_zm.unstack(),robust = True, cmap = 'copper' ,  dx=1, units="nm")
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_WF_zm.unstack(),robust = True, cmap = 'Blues')

'''


# %%
#W_df_xr
#S_df_xr

fig,axes = plt.subplots(1,2, figsize  =(8,6))
axs = axes.ravel()
isns.set_scalebar(color = "k", location='lower left')

isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_WF_zm.unstack()*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            #cbar_label = '$\Phi_{sample}$ (mV)', 
            dx=1, units="nm", 
            ax = axs[0])
axs[0].set_title('W: $\Phi_{MoS_{2}}$ - $\Phi_{HOPG}$ (mV)')
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_WF_zm.unstack()*1E3,
            robust = True, 
            cmap = 'Greens_r', 
            #cbar_label = '$\Phi_{sample}$ (mV)', 
            dx=1, units="nm", 
            ax = axs[1])
axs[1].set_title('W+S: $\Phi_{MoS_{2}}$ - $\Phi_{HOPG}$ (mV) W+S')

# plt.tight_layout()
plt.savefig('W_S_Z_CPD_z_remove edge.svg')
plt.show()



'''
isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_topo_zm.unstack(),robust = True, cmap = 'copper',  dx=1, units="nm")
isns.imshow(W_xr_df[W_xr_df.layer != 'boundary'].W_WF_zm.unstack(),robust = True, cmap = 'Blues' )

# hv plot is not working well. 
# use the isns instead, 
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_topo_zm.unstack(),robust = True, cmap = 'copper' ,  dx=1, units="nm")
isns.imshow(S_xr_df[S_xr_df.layer != 'boundary'].S_WF_zm.unstack(),robust = True, cmap = 'Blues')

'''


# %%
#W_df_xr
#S_df_xr

############
# extract statistics from W_df_xr & S_df_xr
############
W_layers_name = [ 'HOPG', '1ML', '2ML', '3ML','4ML','boundary']

W_layers_stats = W_df_xr.groupby('layer').mean().W_topo_zm.to_dataframe()
W_layers_stats['W_WF'] = W_df_xr.groupby('layer').mean().W_WF_zm.to_dataframe()
W_layers_stats['W_CPD'] = W_df_xr.groupby('layer').mean().W_CPD_zm.to_dataframe()
#W_layers_stats
# W_topo, W_WF mean & std 


S_layers_name = [ 'HOPG', '1ML', '2ML', '3ML','4ML','5ML','boundary']

S_layers_stats = S_df_xr.groupby('layer').mean().S_topo_zm.to_dataframe()
S_layers_stats['S_WF'] = S_df_xr.groupby('layer').mean().S_WF_zm.to_dataframe()
S_layers_stats['S_CPD'] = S_df_xr.groupby('layer').mean().S_CPD_zm.to_dataframe()
#S_layers_stats
# S_topo, S_WF mean & std 
############
S_layers_stats

# %%
W_layers_stats

# %%
###################
# Confidential Interval calcuation based on scipy t-test
###################

###################
# W layers
###################
# topography 
W_layer_stats_topo = np.array([])

for layer_name in W_layers_name: 
    layer_mean =  W_df_xr.groupby('layer')[layer_name].W_topo_zm.mean().values
    layer_std =  W_df_xr.groupby('layer')[layer_name].W_topo_zm.std().values
    layer_CI95 = sp.stats.t.interval(0.95,
                                    W_df_xr.groupby('layer').count().sel(
                                        layer = layer_name
                                    ).W_topo_zm.values - 1,
                                    layer_mean,
                                    layer_std)
    W_layer_stats_topo = np.append (W_layer_stats_topo, 
                                    np.array([layer_mean, 
                                              layer_std,
                                              layer_CI95[0], 
                                              layer_CI95[1]]),
                                    axis = 0)
    
W_layer_stats_topo = W_layer_stats_topo.reshape(-1,4)
# layer_stats shape check 

W_layers_stats[['W_topo_mean',
                'W_topo_std',
                'W_topo_ci95_0',
                'W_topo_ci95_1']] = W_layer_stats_topo

###################
# Work Function results 
W_layer_stats_WF = np.array([])

for layer_name in W_layers_name: 
    layer_mean =  W_df_xr.groupby('layer')[layer_name].W_WF_zm.mean().values
    layer_std =  W_df_xr.groupby('layer')[layer_name].W_WF_zm.std().values
    layer_CI95 = sp.stats.t.interval(0.95,
                                    W_df_xr.groupby('layer').count().sel(
                                        layer = layer_name
                                    ).W_WF_zm.values - 1,
                                    layer_mean,
                                    layer_std)
    W_layer_stats_WF = np.append (W_layer_stats_WF,
                                  np.array([layer_mean, 
                                            layer_std, 
                                            layer_CI95[0], 
                                            layer_CI95[1]]), 
                                  axis = 0)
    
W_layer_stats_WF = W_layer_stats_WF.reshape(-1,4)
# layer_stats shape check 

W_layers_stats[['W_WF_mean',
                'W_WF_std',
                'W_WF_ci95_0',
                'W_WF_ci95_1']] = W_layer_stats_WF
#W_layers = W_layers.drop([ 'HOPG', 'boundary'])

W_layers_stats = W_layers_stats.drop(['W_topo_zm', 'W_WF'],
                                     axis =1)
# index sorting result  ==> drop inconsistent columns 
W_layers_stats


# %%
###################
# Confidential Interval calcuation based on scipy t-test
###################

###################
# S layers
###################
# topography 
S_layer_stats_topo = np.array([])

for layer_name in S_layers_name: 
    layer_mean =  S_df_xr.groupby('layer')[layer_name].S_topo_zm.mean().values
    layer_std =  S_df_xr.groupby('layer')[layer_name].S_topo_zm.std().values
    layer_CI95 = sp.stats.t.interval(0.95,
                                    S_df_xr.groupby('layer').count().sel(
                                        layer = layer_name
                                    ).S_topo_zm.values - 1,
                                    layer_mean,
                                    layer_std)
    S_layer_stats_topo = np.append (S_layer_stats_topo, 
                                    np.array([layer_mean,
                                              layer_std, 
                                              layer_CI95[0], 
                                              layer_CI95[1]]), 
                                    axis = 0)
    
S_layer_stats_topo = S_layer_stats_topo.reshape(-1,4)
# layer_stats shape check 

S_layers_stats[['S_topo_mean',
                'S_topo_std',
                'S_topo_ci95_0',
                'S_topo_ci95_1']] = S_layer_stats_topo

###################
# Work Function results 
S_layer_stats_WF = np.array([])

for layer_name in S_layers_name: 
    layer_mean =  S_df_xr.groupby('layer')[layer_name].S_WF_zm.mean().values
    layer_std =  S_df_xr.groupby('layer')[layer_name].S_WF_zm.std().values
    layer_CI95 = sp.stats.t.interval(0.95,
                                    S_df_xr.groupby('layer').count().sel(
                                        layer = layer_name
                                    ).S_WF_zm.values - 1,
                                    layer_mean,
                                    layer_std)
    S_layer_stats_WF = np.append (S_layer_stats_WF, 
                                  np.array([layer_mean, 
                                            layer_std, 
                                            layer_CI95[0], 
                                            layer_CI95[1]]), 
                                  axis = 0)
    
S_layer_stats_WF = S_layer_stats_WF.reshape(-1,4)
# layer_stats shape check 
S_layer_stats_WF

S_layers_stats[['S_WF_mean','S_WF_std','S_WF_ci95_0','S_WF_ci95_1']] = S_layer_stats_WF
#W_layers = W_layers.drop([ 'HOPG', 'boundary'])

S_layers_stats = S_layers_stats.drop(['S_topo_zm', 'S_WF'], axis =1)
# index sorting result  ==> drop inconsistent columns 
S_layers_stats

# %%
W_layers_stats_1234 = W_layers_stats.drop([ 'HOPG', 'boundary'])#.reset_index()
S_layers_stats_12345 = S_layers_stats.drop([ 'HOPG', 'boundary'])#.reset_index()
W_layers_stats_1234_offset = W_layers_stats_1234.copy(deep=  True )
W_layers_stats_1234_offset.W_topo_mean = W_layers_stats_1234.W_topo_mean -  W_layers_stats_1234.loc['1ML'].W_topo_mean
W_layers_stats_1234_offset.W_topo_mean = W_layers_stats_1234.W_topo_mean -  W_layers_stats_1234.loc['1ML'].W_topo_mean

W_layers_stats_1234_offset

# %%
W_xr_df

# %%
#matplotlib box plox

# %%
W_xr_df_01234 = W_xr_df[(W_xr_df.layer != 'boundary') & (W_xr_df.layer != 'HOPG')]
S_xr_df_012345 = S_xr_df[(S_xr_df.layer != 'boundary') & (S_xr_df.layer != 'HOPG')]

fig, axes = plt.subplots(1,2, figsize = (8,4))
axs = axes.ravel()
sns.boxplot(data =  W_xr_df_01234,
            x = 'layer', 
            y = 'W_topo_zm',
            ax = axs[0], 
            order = ['1ML','2ML','3ML','4ML'])
sns.boxplot(data =  W_xr_df_01234,
            x = 'layer', 
            y = 'W_WF_zm',
            ax = axs[1],
            order = [ '1ML','2ML','3ML','4ML'])
sns.boxplot(data =  S_xr_df_012345,
            x = 'layer', 
            y = 'S_topo_zm', 
            ax = axs[0], 
            order = [ '1ML','2ML','3ML','4ML','5ML'])
sns.boxplot(data =  S_xr_df_012345, 
            x = 'layer', 
            y = 'S_WF_zm', 
            ax = axs[1], 
            order = [ '1ML','2ML','3ML','4ML','5ML'])

plt.show()

# %%
W_xr_df_01234_offset = W_xr_df_01234.copy(deep = True)
W_xr_df_01234_offset.W_topo_zm = W_xr_df_01234.W_topo_zm -  W_xr_df_01234.groupby('layer').mean().W_topo_zm.loc['1ML']
W_xr_df_01234_offset.W_WF_zm = W_xr_df_01234.W_WF_zm -  W_xr_df_01234.groupby('layer').mean().W_WF_zm.loc['1ML']


S_xr_df_012345_offset = S_xr_df_012345.copy(deep = True)
S_xr_df_012345_offset.S_topo_zm = S_xr_df_012345.S_topo_zm -  S_xr_df_012345.groupby('layer').mean().S_topo_zm.loc['1ML']
S_xr_df_012345_offset.S_WF_zm = S_xr_df_012345.S_WF_zm -  S_xr_df_012345.groupby('layer').mean().S_WF_zm.loc['1ML']


fig, axes = plt.subplots(1,2, figsize = (8,4))
axs = axes.ravel()
sns.boxplot(data =  W_xr_df_01234_offset, x = 'layer', y = 'W_topo_zm', ax = axs[0], order = ['1ML','2ML','3ML','4ML'])
sns.boxplot(data =  W_xr_df_01234_offset, x = 'layer', y = 'W_WF_zm', ax = axs[1], order = [ '1ML','2ML','3ML','4ML'])
sns.boxplot(data =  S_xr_df_012345_offset, x = 'layer', y = 'S_topo_zm', ax = axs[0], order = [ '1ML','2ML','3ML','4ML','5ML'])
sns.boxplot(data =  S_xr_df_012345_offset, x = 'layer', y = 'S_WF_zm', ax = axs[1], order = ['1ML','2ML','3ML','4ML','5ML'])


plt.show()


# %%
# delete above figures & make it  simple format

# %%
#W_S_df = pd.merge(W_xr_df_01234_offset,S_xr_df_012345_offset)
# it takes too long time 
#W_S_df

# %%
W_xr_df_01234_offset['stacking'] = 'W'
S_xr_df_012345_offset['stacking'] = 'S'
W_S_df = pd.merge(W_xr_df_01234_offset,S_xr_df_012345_offset)

# %%
W_xr_df_01234_offset.drop (['height','edge' ], axis =1)

# %%
W_S_df = pd.merge(W_xr_df_01234_offset.drop (['height','edge' ], axis =1),
                  S_xr_df_012345_offset.drop (['height','edge' ], axis =1), how = 'outer')



# %%
# topo & WF in 1 column 
W_S_df['topo'] = W_S_df.W_topo_zm
W_S_df.topo[W_S_df.topo.isnull()] = W_S_df.S_topo_zm

W_S_df['WF'] = W_S_df.W_WF_zm
W_S_df.WF[W_S_df.WF.isnull()] = W_S_df.S_WF_zm

# %%
sns.boxplot(data =  W_S_df, x = 'layer', y = 'topo', hue = 'stacking',order = [ '1ML','2ML','3ML','4ML','5ML'],showfliers = False)

# %%
fig,ax  = plt.subplots(figsize=(6, 6))

ax =sns.boxplot(data =  W_S_df, x = 'layer', y = 'WF', hue = 'stacking',order = [ '1ML','2ML','3ML','4ML','5ML'],showfliers = False)

boxs = ax.patches
boxs[-3].set_facecolor("yellow")
boxs[-3].set_edgecolor("k")

# %%
W_S_df['topo_nm'] = W_S_df.topo* 1E9
W_S_df['WF_mV'] = W_S_df.WF* 1E3


#sns.set_style("darkgrid")
sns.set_style("whitegrid")
sns.set_context("paper")


fig, axes = plt.subplots(1,2, figsize = (8,4))
axs = axes.ravel()

sns.boxplot(data =  W_S_df,
            x = 'layer',
            y = 'topo_nm', 
            hue = 'stacking',
            order = [ '1ML','2ML','3ML','4ML','5ML'],
            showfliers = False,#True,
            linewidth=2,
            #whis = 2.5,
            #notch=True,
            ax = axs[0],
            color='g')
#axs[0].set_xticklabels(axs[0].get_xmajorticklabels(), fontsize = 'large')
#axs[0].set_yticklabels(axs[0].get_ymajorticklabels(), fontsize = 'large')

boxs_z = axs[0].patches

for bar_i in range(len(boxs_z)):
    if bar_i in [1,4,6,8]: # 짝수  in W_patch_N:  # W 
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:blue")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [3,5] :
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [7,9,10]:
        boxs_z[bar_i].set_facecolor("tab:orange")
        boxs_z[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [0]:
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:blue")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [11]:
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)       


sns.boxplot(data =  W_S_df,
            x = 'layer', 
            y = 'WF_mV',
            hue = 'stacking',
            order = [ '1ML','2ML','3ML','4ML','5ML'],
            showfliers = False,#True,
            #whis = 2.5,
            #notch=True,
            ax = axs[1],
            color='g')#,palette=my_palette)


boxs_wf = axs[1].patches

for bar_i in range(len(boxs_wf)):
    if bar_i in [1,4,6,8]: # 짝수  in W_patch_N:  # W 
        boxs_wf[bar_i].set_facecolor("white")
        boxs_wf[bar_i].set_edgecolor("tab:blue")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [3,5] :
        boxs_wf[bar_i].set_facecolor("white")
        boxs_wf[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [7,9,10] : 
        boxs_wf[bar_i].set_facecolor("tab:orange")
        boxs_wf[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)




#axs[1].set_xticklabels(axs[1].get_xmajorticklabels(), fontsize = 'large')
#axs[1].set_yticklabels(axs[1].get_ymajorticklabels(), fontsize = 'large')

#axs[1].patches[10].set_facecolor("darkred")
#axs[0].patches[1].set_facecolor("white")
#axs[0].patches[1].set_edgecolor("green")

#handles0, labels0 = axs[0].get_legend_handles_labels()    
#handles1, labels1  = axs[1].get_legend_handles_labels() 


legend_a0 = axs[0].legend(handles=[boxs_z[1]], labels = ['W island'],
                         loc='upper left',
                         bbox_to_anchor=(0.05, 0.95),
                         framealpha = 0.5)
legend_a1 = axs[0].legend(handles=[boxs_z[3],boxs_z[7]], labels = ['W+S island (W)','W+S island (S)'],
                         loc='upper left',
                         bbox_to_anchor=(0.05, 0.85),framealpha = 0.5)
legend_b0 = axs[1].legend(handles=[boxs_wf[1]], labels = ['W island'],
                         loc='upper right',
                         bbox_to_anchor=(0.95, 0.95),framealpha = 0.5)
legend_b1 = axs[1].legend(handles=[boxs_wf[3],boxs_wf[7]],
                         labels = ['W+S island (W)','W+S island (S)'],
                         loc='upper right',
                         bbox_to_anchor=(0.95, 0.85),framealpha = 0.5)

axs[0].add_artist(legend_a0)
axs[0].add_artist(legend_a1)
axs[1].add_artist(legend_b0)
axs[1].add_artist(legend_b1)   

###
# Associate manually the artists to a label.
axs[0].set_ylabel('Height (nm)')
axs[1].set_ylabel(r"$\Delta \Phi$ =  $\Phi_{layer}$  - $\Phi_{1}$    (mV)")

plt.tight_layout()
plt.savefig('W_S_Z_WF_boxplot.svg')
plt.show()

# %%
W_S_df['topo_nm'] = W_S_df.topo* 1E9
W_S_df['WF_mV'] = W_S_df.WF* 1E3


#sns.set_style("darkgrid")
sns.set_style("whitegrid")
sns.set_context("paper")


fig, axes = plt.subplots(1,2, figsize = (7.5,2.5))
axs = axes.ravel()

sns.boxplot(data =  W_S_df,
            x = 'layer',
            y = 'topo_nm', 
            hue = 'stacking',
            order = [ '1ML','2ML','3ML','4ML','5ML'],
            showfliers = False,#True,
            linewidth = 1,
            #whis = 2.5,
            #notch=True,
            ax = axs[0],
            color='g')
#axs[0].set_xticklabels(axs[0].get_xmajorticklabels(), fontsize = 'large')
#axs[0].set_yticklabels(axs[0].get_ymajorticklabels(), fontsize = 'large')


# change errorbar colors 
for line_i, line in enumerate (axs[0].get_lines()):
    #print( line_i)
    if line_i%10 in [0,1,2,3,4]:
        # 5 lines per points, 
        line.set_color('tab:blue')
        if line_i>= 40 :
            line.set_color('tab:orange')  
    else :
        line.set_color('tab:orange')  
    
for line_i, line in enumerate (axs[0].get_lines()):
    if line_i in [9,19,29,39,44]:
        line.set_color('orange') 
    # seperate assign for mean value line 
        
        
boxs_z = axs[0].patches

for bar_i in range(len(boxs_z)):
    if bar_i in [1,4,6,8]: # 짝수  in W_patch_N:  # W 
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:blue")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [3,5] :
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [7,9,10]:
        boxs_z[bar_i].set_facecolor("tab:orange")
        boxs_z[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [0]:
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:blue")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [11]:
        boxs_z[bar_i].set_facecolor("white")
        boxs_z[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)       


sns.boxplot(data =  W_S_df,
            x = 'layer', 
            y = 'WF_mV',
            hue = 'stacking',
            order = [ '1ML','2ML','3ML','4ML','5ML'],
            showfliers = False,#True,
            linewidth = 1,
            #whis = 2.5,
            #notch=True,
            ax = axs[1],
            color='g')#,palette=my_palette)


# change errorbar colors 
for line_i, line in enumerate (axs[1].get_lines()):
    #print( line_i)
    if line_i%10 in [0,1,2,3,4]:
        # 5 lines per points, 
        line.set_color('tab:blue')
        if line_i>= 40 :
            line.set_color('tab:orange')  
    else :
        line.set_color('tab:orange')  
    
for line_i, line in enumerate (axs[1].get_lines()):
    if line_i in [9,19,29,39,44]:
        line.set_color('orange') 
    # seperate assign for mean value line 

    
    
boxs_wf = axs[1].patches

for bar_i in range(len(boxs_wf)):
    if bar_i in [1,4,6,8]: # 짝수  in W_patch_N:  # W 
        boxs_wf[bar_i].set_facecolor("white")
        boxs_wf[bar_i].set_edgecolor("tab:blue")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [3,5] :
        boxs_wf[bar_i].set_facecolor("white")
        boxs_wf[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)
    elif bar_i in [7,9,10] : 
        boxs_wf[bar_i].set_facecolor("tab:orange")
        boxs_wf[bar_i].set_edgecolor("tab:orange")
        #barsA[bar_i].set_linewidth(1)




#axs[1].set_xticklabels(axs[1].get_xmajorticklabels(), fontsize = 'large')
#axs[1].set_yticklabels(axs[1].get_ymajorticklabels(), fontsize = 'large')

#axs[1].patches[10].set_facecolor("darkred")
#axs[0].patches[1].set_facecolor("white")
#axs[0].patches[1].set_edgecolor("green")

#handles0, labels0 = axs[0].get_legend_handles_labels()    
#handles1, labels1  = axs[1].get_legend_handles_labels() 


legend_a0 = axs[0].legend(handles=[boxs_z[1]], labels = ['W island'],
                         loc='upper left',
                         bbox_to_anchor=(0.05, 1),
                         framealpha = 0.5)
legend_a1 = axs[0].legend(handles=[boxs_z[3],boxs_z[7]], labels = ['W+S island (W)','W+S island (S)'],
                         loc='lower right',
                         bbox_to_anchor=(0.95, 0.02),framealpha = 0.5)
legend_b0 = axs[1].legend(handles=[boxs_wf[1]], labels = ['W island'],
                         loc='upper left',
                         bbox_to_anchor=(0.05, 1),framealpha = 0.5)
legend_b1 = axs[1].legend(handles=[boxs_wf[3],boxs_wf[7]],
                         labels = ['W+S island (W)','W+S island (S)'],
                         loc='lower right',
                         bbox_to_anchor=(0.95, 0.02),framealpha = 0.5)

axs[0].add_artist(legend_a0)
axs[0].add_artist(legend_a1)
axs[1].add_artist(legend_b0)
axs[1].add_artist(legend_b1)   

###
# Associate manually the artists to a label.
axs[0].set_ylabel('Height (nm)')
axs[1].set_ylabel(r"$\Delta \Phi$ =  $\Phi_{layer}$  - $\Phi_{1}$    (mV)")

plt.tight_layout()
plt.savefig('W_S_Z_WF_boxplot.svg')
plt.show()

# %%
sns.violinplot(data =  W_S_df,
            x = 'layer', 
            y = 'topo_nm',
            hue = 'stacking',
            order = [ '1ML','2ML','3ML','4ML','5ML'],split=False, capsize=.01,  ci=95, errwidth=2)

# %%

# %%

# %%

# %%
