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
# # Interlayer Coupling Dependent Bandgap of  $MoS_{2}$ Islands (_working title_)
# * Stacking dependent bandgap of $MoS_{2}$ on (highly )HOPG substrates 
#
# # <font color=blue>Fig 4- STM/S data analysis (Fig 4 W islands )</font>
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
# * Authors : Dr. Jewook Park(IBS)
#     * *IBS-VdWQS (Inistitute for Basic Science,Center for Van der Waals Quantum Solids), South Korea*
#     * email :  jewookpark@ibs.re.kr
#
# > **SPMpy** is a python package for scanning probe microscopy (SPM) data analysis, such as scanning tunneling microscopy and spectroscopy (STM/S) data and atomic force microscopy (AFM) images, which are inherently multidimensional. To analyze SPM data, SPMpy exploits recent image processing(a.k.a. Computer Vision) techniques. SPMpy data analysis functions utilize well-established Python packages, such as Numpy, PANDAS, matplotlib, Seaborn, holoview, etc. In addition, many parts are inspired by well-known SPM data analysis programs, for example, Wsxm and Gwyddion. Also, SPMpy is trying to apply lessons from 'Fundamentals in Data Visualization'(https://clauswilke.com/dataviz/).
#
# >  **SPMpy** is an open-source project. (Github: https://github.com/Jewook-Park/SPMPY )
# > * Contributions, comments, ideas, and error reports are always welcome. Please use the Github page or email jewookpark@ibs.re.kr. Comments & remarks should be in Korean or English. 
#

# + [markdown] tags=[]
# ### Import necessary packages & loading **SPMpy** functions  
# * file loading : SPMpy_file_loading_funcs
# * 3D data analysis functions : SPMpy_3D_data_analysis_funcs
# * 2D data analysis functions : SPMpy_2D_data_analysis_funcs

# + id="Qm1zLaTHbSpK" tags=[]
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
    import seaborn_image as isns
except ModuleNotFoundError:
    warn('ModuleNotFoundError: No module named seaborn_image')
    # !conda install -c conda-forge seaborn-image
    import seaborn_image as isns
    
    
    
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



####
# isns setting 
isns.set_image(origin = 'lower')




# + [markdown] tags=[]
# ### Checking up the file path
# ### 0.2. Grid DATA Import 
# ### $\Longrightarrow$    **files_in_folder**
# * working folder(path) checkup
#     * files_in_folder(path)
#     *  sxm file $\to $  **img2xr**  $\to $  a xarray format
#     *  sxm file $\to $   **grid2xr**  $\to $  a xarray format
#     *  sxm file $\to $  **grid_line2xr** $\to $  a xarray format
#     
# ## Choose the grid data file to analyze
# * using **grid2xr** function 
#     * '_fb' : add fwd/bwd data average 
#         * grid_topo : 2D data 
#         * grid_3D : 3D data
#     * **I_fb** : I, (forwad + backward sweep )/2
#     * **LIX_fb** : LIX, (forwad + backward sweep )/2
#     * **dIdV** : dI/dV (using xr differentiate _class_ )
#     * **LIX_unit_calc** : LIX_fb *  LIX_coefficient (for unit calibration) 
#
# * after **grid_3D_gap** function 
#     *  2D data :  CBM, VBM position assignment $\leftarrow$ based on **I** or **LIX**
#         * CBM_I_mV, VBM_I_mV, gap_size_I 
#         * CBM_LIX_mV, VBM_LIX_mV, gap_size_LIX
#     *  3D data : LDOS_fb $\leftarrow$ after unit calc & offset adjust
#         * I_fb, LIX_fb, LDOS_fb, LDOS_fb_CB, LDOS_fb_VB      
#         * **I_fb** : I, (forwad + backward sweep )/2
#         * **LIX_fb** : LIX, (forwad + backward sweep )/2
#         * **LDOS_fb** : LIX_fb *  LIX_coefficient (for unit calibration)  + offset adjustment (according to LIX  at I=0)
#         * **LDOS_fb_CB** : based on LIX assignment
#         * **LDOS_fb_VB** : based on LIX assignment
#         

# +
# check the sxm files in the given folder
target_path = r'C:\IBS CALDES data\IBS Epitaxial vdW Quantum Solid\Papers\Preparation of pyramid and screw MoS2 on HOPG paper\Figure Preparation\Figure4  heterogeneities'
#target_path = r'C:\Users\jewoo\ownCloud\IBS Epitaxial vdW Quantum Solid\Papers\Preparation of pyramid and screw MoS2 on HOPG paper\Figure Preparation\Figure4  heterogeneities'
file_list_df = files_in_folder(target_path)
## Loading Image Channels
#file_list_df


##############
# choose the Topography of ROI 
W_2009 = file_list_df[(file_list_df.type == 'gwy') ]
#W_2009

W_2009_topo = gwy_image2df(W_2009.file_name.values[0])
W_2009_topo
# -


W_2009_topo_zm =  gwy_df_channel2xr (W_2009_topo,3)
W_2009_topo_zm


# +
isns.set_image(origin = 'lower')

fig, ax = plt.subplots(figsize = (5,3))
isns.imshow(W_2009_topo_zm*1E9,
            robust = True, perc=(0.5,99.5),
            cmap = 'viridis',  
            dx=320/512, 
            units="nm", 
            cbar_label = 'z (nm)',
            ax = ax)
plt.savefig('W2009_ROI topo.svg')
plt.show()


# +
################
# Choose the STS for  W islands 
    
W_001 = file_list_df[
    (file_list_df.type == '3ds') & 
    (file_list_df.file_name.str.contains('Grid Spectroscopy002'))]
grid_xr = grid2xr(W_001.file_name.values[0])
# -


grid_xr

# + jp-MarkdownHeadingCollapsed=true tags=[]
grid_xr = grid_xr.assign_coords({'X': grid_xr.X -  grid_xr.X.min()})
grid_xr = grid_xr.assign_coords({'Y': grid_xr.Y -  grid_xr.Y.min()})

# grid data to xr 
grid_xr['I_fb'] = (grid_xr.I_fwd + grid_xr.I_fwd)/2
grid_xr['LIX_fb'] = (grid_xr.LIX_fwd + grid_xr.LIX_fwd)/2
# add 'I' & 'LIX' channel "fb = [fwb+bwd] / 2 " 

grid_topo = grid_xr[['topography']]
# topography 
grid_3D = grid_xr[['I_fb','LIX_fb']]
# averaged I & LIX 


grid_3D_gap = grid_3D_Gap(grid_3D)
# assign gap from STS

# + [markdown] tags=[]
# ### Pretreatments 
# * In case of **Energy gap** in dI/dV 
# * (I or LIX) is almost Zero. less than measurment error bar
#     * I_min_pA= 1E-11 // LIX_min_pA= 1E-12
#     * find gap size & adjust LIX offset, based on I or LIX
#         * to prevent ripples in dI/dV polyfit 
#

# +
grid_3D_gap

# check the STS results 

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ### interfactive plot with hvplot
#
# -

from hvplot import hvPlot
import holoviews as hv
import hvplot.xarray  # noqa
import panel as pn
import panel.widgets as pnw
import ipywidgets as ipw
from holoviews import opts
from holoviews.streams import Stream, param


# + [markdown] tags=[]
# ### LDOS unit calibration 
# * check the (X,Y) = (1,0) (now I am using the line STS)
# -

(hv.Curve(grid_3D.isel(X = 1, Y =0).LIX_fb, label = 'LIX_fb').opts(axiswise=True)\
*hv.Curve(grid_3D.isel(X = 1, Y =0).dIdV,  label = 'dIdV').opts(axiswise=True)\
*hv.Curve(grid_3D.isel(X = 1, Y =0).LIX_unit_calc, label = 'LIX_unit_calc').opts(axiswise=True)).opts(legend_position='top_left').relabel('grid_3D')\
+ hv.Curve(grid_3D_gap.isel(X = 1, Y =0).LDOS_fb,  label = 'grid_3D_gap.LDOS_fb').opts(axiswise=True, ylabel = 'dI/dV [A/V]')


# + [markdown] tags=[]
# ### 3D plot & slicing 
# * use the holoview 
#     * or hvplot(with panel- widget) interactive plot  $\to$  event handling  $\to$  later 
#     * mayavi? $\to$ later
#
# ### Topography**  check. 

# + jp-MarkdownHeadingCollapsed=true tags=[]
isns.set_image(origin = 'lower')

isns.imshow( plane_fit_y(grid_topo.topography) )
# check the how to show the 2D image 

isns.imshow(grid_topo.topography*1E9,
            robust = True,
            aspect = 'equal', 
            cmap = 'viridis', 
            dx=1, units="nm", 
            cbar_label = 'z (nm)',
            fontsize='xx-large')
# -

# # Thresholdin with multi Otsu
#
#
#

grid_topo_th = threshold_multiotsu_xr(grid_topo, multiclasses=6)
#W_zm_topo_th.W_topo_zm.plot(cmap = 'copper')

# ## Save Threshold data (xr: to_netcdf)
#

# +
### if it takes too long, 
# * split the area 2 *3 
# * threshold_multiotsu_xr(grid_topo, multiclasses=2)

## Save Threshold data pkl
grid_topo_th.to_netcdf("save_after_multiotsu_GridSpectroscopy002.nc")

# +
## Load saved Threshold data  (xr: to_netcdf)
# -

grid_topo_th = xr.open_dataset("save_after_multiotsu_GridSpectroscopy002.nc")
### 

# +


grid_topo = grid_xr[['topography']]
# to avoid restart the MultiOtsu, 
# reset the grid_topo here


isns.imshow(grid_topo_th.topography)

### check the histogram 
#grid_topo.topography.plot.hist(bins = 200)
grid_topo['height'] = grid_topo_th.topography
# CAUTION! 
# * 5ML+6NL => 4


# -

# ### find edges (using Sobel filter)

# +
#grid_topo.topography[grid_topo.height == 0]
#isns.imshow((grid_topo.height == 5))

grid_topo_sobel = filter_sobel_xr(grid_topo)
grid_topo_sobel

# + [markdown] tags=[]
# ## Assign the terrace indicator 


# +
grid_topo =  grid_topo.merge( grid_topo_sobel.height_sobel)
grid_topo['terrace'] = grid_topo.height.copy()

#grid_topo['terrace'].where((grid_topo.height_sobel!=0)& (grid_topo.height==0)) 


# edge finding 
mask_stepedges = (grid_topo.height_sobel!=0)
# terrace numbering 

mask_HOPG = (grid_topo.height_sobel==0)& (grid_topo.height==0)
mask_1ML = (grid_topo.height_sobel==0)& (grid_topo.height==1)
mask_2ML = (grid_topo.height_sobel==0)& (grid_topo.height==2)
mask_3ML = (grid_topo.height_sobel==0)& (grid_topo.height==3)
mask_4ML = (grid_topo.height_sobel==0)& (grid_topo.height==4)
mask_56ML = (grid_topo.height_sobel==0)& (grid_topo.height==5)

grid_topo['terrace'] = xr.where(mask_stepedges, 'Edge', grid_topo['terrace'])

grid_topo['terrace'] = xr.where(mask_HOPG, 'HOPG', grid_topo['terrace'])
grid_topo['terrace'] = xr.where(mask_1ML, 'W1ML', grid_topo['terrace'])
grid_topo['terrace'] = xr.where(mask_2ML, 'W2ML', grid_topo['terrace'])
grid_topo['terrace'] = xr.where(mask_3ML, 'W3ML', grid_topo['terrace'])
grid_topo['terrace'] = xr.where(mask_4ML, 'W4ML', grid_topo['terrace'])
grid_topo['terrace'] = xr.where(mask_56ML, 'W56ML', grid_topo['terrace'])

grid_topo['terrace']


#grid_topo['terrace'].isin('HOPG')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### Terrace selection 
# * use the where & mean for xr, instead of groupby 
# * Assign the terrace based on Multi Otsu results.
# #### Plotting layer dependent STS (seaborn) 
# * Use the PANDAS format & melting 
# * Seaborn plot (with ci=0.95)
#
# -

W1ML_df = grid_3D.where(grid_topo.terrace == 'W1ML').LIX_unit_calc.to_dataframe()
W2ML_df = grid_3D.where(grid_topo.terrace == 'W2ML').LIX_unit_calc.to_dataframe()
W3ML_df = grid_3D.where(grid_topo.terrace == 'W3ML').LIX_unit_calc.to_dataframe()
W4ML_df = grid_3D.where(grid_topo.terrace == 'W4ML').LIX_unit_calc.to_dataframe()
W5ML_df = grid_3D.where(grid_topo.terrace == 'W56ML').LIX_unit_calc.to_dataframe()

# +
W_df = pd.concat ( [W1ML_df,W2ML_df,W3ML_df,W4ML_df,W5ML_df],axis =1)
W_df.columns = ['W1','W2','W3','W4','W5']

W_df.groupby('bias_mV').mean()

##############
W_df = W_df.reset_index()
W_df_melt = W_df.melt(id_vars = ['Y','X','bias_mV'], value_vars = ['W1','W2','W3','W4','W5'])
W_df_melt
#################
sns.lineplot(data = W_df_melt, x = 'bias_mV', y= 'value', hue ='variable')

# + [markdown] tags=[]
# #### check the mean curves without CI values + seaborn plot 
# * concat the STS curves 
# * the same as seaborn result
#
# -

W1ML_mean_df = grid_3D.where(grid_topo.terrace == 'W1ML').LIX_unit_calc.mean(dim = ["X","Y"]).to_dataframe()
W2ML_mean_df = grid_3D.where(grid_topo.terrace == 'W2ML').LIX_unit_calc.mean(dim = ["X","Y"]).to_dataframe()
W3ML_mean_df = grid_3D.where(grid_topo.terrace == 'W3ML').LIX_unit_calc.mean(dim = ["X","Y"]).to_dataframe()
W4ML_mean_df = grid_3D.where(grid_topo.terrace == 'W4ML').LIX_unit_calc.mean(dim = ["X","Y"]).to_dataframe()
W5ML_mean_df = grid_3D.where(grid_topo.terrace == 'W56ML').LIX_unit_calc.mean(dim = ["X","Y"]).to_dataframe()
############
W_mean_df = pd.concat ( [W1ML_mean_df,W2ML_mean_df,W3ML_mean_df,W4ML_mean_df,W5ML_mean_df],axis =1)
W_mean_df.columns = ['W1','W2','W3','W4','W5']
W_mean_df
##############
W_mean_df = W_mean_df.reset_index()
W_mean_df_melt = W_mean_df.melt(id_vars = ['bias_mV'], value_vars = ['W1','W2','W3','W4','W5'])
W_mean_df_melt
################
sns.lineplot(data = W_mean_df_melt, x = 'bias_mV', y= 'value', hue ='variable')

# + [markdown] tags=[]
# #### Check 2nd derivative to find peaks 
# * convert to Xarray 
# * calc derivative 
# * smoothing with 'savgol_filter'
# -

for ch in W_mean_df.set_index('bias_mV'): 
    W_mean_df[ch] = sp.signal.savgol_filter(W_mean_df[ch],
                                            window_length = 7,
                                            polyorder = 3)

# +
W_mean_df_1deriv = W_mean_df.set_index('bias_mV').to_xarray().differentiate(coord = 'bias_mV')
W_mean_df_2deriv = W_mean_df_1deriv.differentiate(coord = 'bias_mV')
W_mean_df_2deriv_sg =  W_mean_df_2deriv.copy()

for ch in W_mean_df_2deriv_sg:
    W_mean_df_2deriv_sg[ch].values =  sp.signal.savgol_filter(W_mean_df_2deriv[ch].values,
                                                              window_length = 7,
                                                              polyorder = 3)

W_mean_df_2deriv_sg

###########

# find the peaks& dips 

for ch in W_mean_df_2deriv_sg:
    W_mean_df_2deriv_sg[ch+'_pks'] = xr.DataArray(sp.signal.find_peaks(W_mean_df_2deriv_sg[ch].values))
    W_mean_df_2deriv_sg[ch+'_dps'] = xr.DataArray(sp.signal.find_peaks(-1*W_mean_df_2deriv_sg[ch].values, distance  = 10))
    
W_mean_df_2deriv_sg


########################


# + [markdown] tags=[]
# #### find dips in 2nd derivative for peaks in STS 
# * use dps 
# * Define LIX resolution limit 
# * (manually )Select peaks 

# +
# pks N dps
W2_1_dps = W_mean_df_2deriv_sg.W1_dps.data.item()[0]
W2_2_dps = W_mean_df_2deriv_sg.W2_dps.data.item()[0]
W2_3_dps = W_mean_df_2deriv_sg.W3_dps.data.item()[0]
W2_4_dps = W_mean_df_2deriv_sg.W4_dps.data.item()[0]
W2_5_dps = W_mean_df_2deriv_sg.W5_dps.data.item()[0]

# set LDOS limit 
LIX_limit= 4E-15

# LDOS value at the peak is larger than LIX_limit
#(W_mean_df.W1[W1_1_dps] > LIX_limit)
W2_1_dps[(W_mean_df.W1[W2_1_dps] > LIX_limit).values].shape
W2_2_dps[(W_mean_df.W2[W2_2_dps] > LIX_limit).values].shape
W2_3_dps[(W_mean_df.W3[W2_3_dps] > LIX_limit).values].shape
W2_4_dps[(W_mean_df.W4[W2_4_dps] > LIX_limit).values].shape
W2_5_dps[(W_mean_df.W5[W2_5_dps] > LIX_limit).values].shape





# + [markdown] tags=[]
# ### Find peak positions 
# * using 2nd derivative of each curves
# * __Local minimums in d2(LDOS)/dV2) = Peaks in LDOS__
# * delete the minor peak ( comes from ripples in the gap region)

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### W1 Dips

# +

fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W1',
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_1_dps],
                y = W_mean_df.W1[W2_1_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W1', 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_1_dps], 
                y = W_mean_df_2deriv_sg.W1[W2_1_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
W2_1_dps_slct = W2_1_dps[[0,1,2,-4, -3,-2,-1]]
#print(W2_1_dps_slct)


sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W1',
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_1_dps_slct],
                y = W_mean_df.W1[W2_1_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W1', 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_1_dps_slct].values, 
                y = W_mean_df_2deriv_sg.W1[W2_1_dps_slct].values,
                ax=axs1tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')

plt.suptitle('W2_2ML_peaks')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### W2 Dips

# +
fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W2',
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_2_dps],
                y = W_mean_df.W2[W2_2_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W2', 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_2_dps], 
                y = W_mean_df_2deriv_sg.W2[W2_2_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
W2_2_dps_slct = W2_2_dps[[1, 2,-3,-2]]
#print(W2_2_dps_slct)


sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W2',
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_2_dps_slct],
                y = W_mean_df.W2[W2_2_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W2', 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_2_dps_slct].values, 
                y = W_mean_df_2deriv_sg.W2[W2_2_dps_slct].values,
                ax=axs1tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')

plt.suptitle('W2_2ML_peaks')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### W3 Dips

# +
print(W2_3_dps)
fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W3',
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_3_dps],
                y = W_mean_df.W3[W2_3_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W3', 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_3_dps], 
                y = W_mean_df_2deriv_sg.W3[W2_3_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
W2_3_dps_slct = W2_3_dps[[0,1,2,  -4,-3,-2,-1]]
#print(W2_3_dps_slct)


sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W3',
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_3_dps_slct],
                y = W_mean_df.W3[W2_3_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W3', 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_3_dps_slct].values, 
                y = W_mean_df_2deriv_sg.W3[W2_3_dps_slct].values,
                ax=axs1tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')

plt.suptitle('W2_3ML_peaks')

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ##### W4 Dips

# +
print(W2_4_dps)
fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W4',
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_4_dps],
                y = W_mean_df.W4[W2_4_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W4', 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_4_dps], 
                y = W_mean_df_2deriv_sg.W4[W2_4_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
W2_4_dps_slct = W2_4_dps[[0,1,2, -5,-4,-3,-2]]
#print(W2_4_dps_slct)


sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W4',
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_4_dps_slct],
                y = W_mean_df.W4[W2_4_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W4', 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_4_dps_slct].values, 
                y = W_mean_df_2deriv_sg.W4[W2_4_dps_slct].values,
                ax=axs1tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')

plt.suptitle('W2_4ML_peaks')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##### W5 Dips

# +
print(W2_5_dps)
fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W5',
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_5_dps],
                y = W_mean_df.W5[W2_5_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W5', 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_5_dps], 
                y = W_mean_df_2deriv_sg.W5[W2_5_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
W2_5_dps_slct = W2_5_dps[[0,1, -4,-3, -2,-1]]
#print(W2_4_dps_slct)


sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W5',
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_5_dps_slct],
                y = W_mean_df.W5[W2_5_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = W_mean_df_2deriv_sg.to_dataframe(),
             x =  'bias_mV', y = 'W5', 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(x = W_mean_df_2deriv_sg.bias_mV[W2_5_dps_slct].values, 
                y = W_mean_df_2deriv_sg.W5[W2_5_dps_slct].values,
                ax=axs1tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')

plt.suptitle('W2_4ML_peaks')

# + [markdown] tags=[]
# ###  plot peaks & curves
#

# +
# W stacks 

fig,axs = plt.subplots(figsize = (6,4))

offset_2ML = 2E-12
offset_3ML = 4E-12
offset_4ML = 6E-12
offset_5ML = 8E-12

sns.lineplot(data = W_mean_df,
             x =  'bias_mV', y = 'W1',
             ax=axs, color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_1_dps_slct],
                y = W_mean_df.W1[W2_1_dps_slct],
                ax=axs, color  = 'tab:blue')

sns.lineplot(data = W_mean_df+offset_2ML,
             x =  'bias_mV', y = 'W2',
             ax=axs, color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_2_dps_slct],
                y = W_mean_df.W2[W2_2_dps_slct]+offset_2ML,
                ax=axs, color  = 'tab:blue')

sns.lineplot(data = W_mean_df+offset_3ML,
             x =  'bias_mV', y = 'W3',
             ax=axs, color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_3_dps_slct],
                y = W_mean_df.W3[W2_3_dps_slct]+offset_3ML,
                ax=axs, color  = 'tab:blue')
                
sns.lineplot(data = W_mean_df+offset_4ML,
             x =  'bias_mV', y = 'W4',
             ax=axs, color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_4_dps_slct],
                y = W_mean_df.W4[W2_4_dps_slct]+offset_4ML,
                ax=axs, color  = 'tab:blue')

sns.lineplot(data = W_mean_df+offset_5ML,
             x =  'bias_mV', y = 'W5',
             ax=axs, color  = 'tab:blue')
sns.scatterplot(x =  W_mean_df.bias_mV[W2_5_dps_slct],
                y = W_mean_df.W5[W2_5_dps_slct]+offset_5ML,
                ax=axs, color  = 'tab:blue')



axs.set_ylabel('dI/dV (nA/V)', color='tab:blue')

# -

# ## Rotation  grid_3D data
#
#

# %matplotlib qt5


# +
l_pf_start, l_pf_end, _ = line_profile_xr_GUI(grid_topo)
# l_pf_start (cx1,ry1) & l_pf_end (cx2,ry2)

l_slope = np.degrees(2*math.pi+math.atan2(l_pf_end[1]-l_pf_start[1],l_pf_end[0]-l_pf_start[0]))
    
l_slope

# -

# ### Set a horzontal line 

# +
isns.set_image(origin = 'lower')

# %matplotlib inline
isns.imshow(grid_topo.topography)
# -

# a## Padding 
# ### Rotation 

# + [markdown] tags=[]
# ## line profile  GUI 
#
# -

grid_topo

grid_topo_r = grid_topo[['topography','height','height_sobel']]
# terrace name is not rotatable
# l_slope
grid_topo_rot = rotate_2D_xr(grid_topo_r,l_slope)
#rotate_3D_xr(grid_3D_gap,l_slope)

grid_3D_gap

grid_3D_gap_rot = rotate_3D_xr(grid_3D_gap,l_slope)
grid_3D_gap_rot

isns.imshow(grid_3D_gap_rot.LDOS_fb.isel(bias_mV=0))

# ##  Holoview plot 
#

# + [markdown] tags=[]
# ##  Bias_mV slicing Examples 
#
# ### W2ML 

# +
###############
# bias_mV slicing
grid_3D_hv = hv.Dataset(grid_3D_gap_rot.LDOS_fb)

dmap_plane  = ["X","Y"]
dmap = grid_3D_hv.to(hv.Image,
                     kdims = dmap_plane,
                     dynamic = True )
dmap.opts(colorbar = True,
          cmap = 'bwr',
          frame_width = 400,
          aspect = 1).relabel('XY plane slicing: ')
fig = hv.render(dmap)
dmap

# +
###############
# holoview slicing 
###############

# choose the channel []

grid_3D_gap_rot_zm = grid_3D_gap_rot.where(grid_3D_gap_rot.X>1.2E-7,drop=True
                                          ).where(grid_3D_gap_rot.X<2.1E-7,drop=True
                                                 ).where(grid_3D_gap_rot.Y>1.48E-7,drop=True
                                                        ).where(grid_3D_gap_rot.Y<1.85E-7,drop=True)
grid_3D_look_hv = hv.Dataset(grid_3D_gap_rot_zm.LDOS_fb)
#grid_3D_look_hv = hv.Dataset(grid_3D_gap_rot.LDOS_fb)
# convert xr dataset as a holoview dataset 
hv.extension('bokeh')
# -

###############
# bias_mV slicing
dmap_plane  = ["X","Y"]
dmap = grid_3D_look_hv.to(hv.Image,
                     kdims = dmap_plane,
                     dynamic = True )
dmap.opts(colorbar = True,
          cmap = 'bwr',
          frame_width = 400,
          aspect = 1).relabel('XY plane slicing: ')
fig = hv.render(dmap)
dmap

# +
#grid_3D_gap_rot_zm.mean (dim = 'Y').LDOS_fb.plot()
grid_3D_gap_rot_zm.mean (dim = 'Y').LDOS_fb

y_avg_line = grid_3D_gap_rot_zm.mean (dim = 'Y').LDOS_fb.to_dataframe().reset_index()
y_avg_line

# +
### add offset to the LDOS_fb for sns plot 

# +
#sns.lineplot(data = y_avg_line, x =  'bias_mV', y='LDOS_fb', hue='X')
y_avg_line['LDOS_offset'] =y_avg_line.LDOS_fb +  (y_avg_line.X /  y_avg_line.X.min()-1)*2E-12
y_avg_line['X_nm'] =((y_avg_line.X /  y_avg_line.X.min() -1 )*100 ).astype('int64')

#y_avg_line['LDOS_offset'] =y_avg_line.X /  y_avg_line.X.min()* 0.2E-12
y_avg_line
# -

y_avg_line

sns.lineplot(data = y_avg_line, x =  'bias_mV', y='LDOS_offset', hue='X_nm', palette ='RdPu',legend="full")

# +
fig,ax = plt.subplots(figsize = (6,4))
sns.lineplot(data = y_avg_line, x =  'bias_mV', y='LDOS_offset', hue='X_nm', palette ='RdPu',legend="full")
#ax.set_ylim(top = 3E-12)
ax.set_ylabel('dI/dV (A/V)')
ax.set_xlabel('Bias (mV)')
boundary_line0 = ax.get_lines()[12]
boundary_line0.set(color = 'b')
boundary_line1 = ax.get_lines()[13]
boundary_line1.set(color = 'b')
boundary_line2 = ax.get_lines()[14]
boundary_line2.set(color = 'b')

boundary_line2 = ax.get_lines()[32]
boundary_line2.set(color = 'b')

handles0, labels0 =  ax.get_legend_handles_labels()

ax.legend (handles = handles0[0:25:4][::-1], labels  = ['60 nm','45 nm','30 nm','15 nm','0 nm'], loc = 'upper center')
plt.savefig('W_lineSTS.svg')
plt.show()
# -



grid_3D_rot = rotate_3D_xr(grid_3D,l_slope)
grid_3D_rot.LIX_unit_calc.isel(bias_mV = 0).plot()

grid_3D_rot.LIX_unit_calc.where(grid_3D_rot.Y>1.45E-7, 
                                drop = True).where(grid_3D_rot.Y<1.6E-7,
                                                   drop = True).where(grid_3D_rot.X>1E-7, 
                                                                      drop = True).where(grid_3D_rot.X<2.2E-7, 
                                                                                         drop = True).isel(bias_mV = 0).plot()

grid_3D_zm_line = grid_3D_rot.where(grid_3D_rot.Y>1.5E-7, 
                                    drop = True).where(grid_3D_rot.Y<1.65E-7,
                                                       drop = True).where(grid_3D_rot.X>1E-7, 
                                                                          drop = True).where(grid_3D_rot.X<2.2E-7, 
                                                                                             drop = True).where(grid_3D_rot.bias_mV<800, 
                                                                                             drop = True)



grid_topo_rot = rotate_2D_xr(grid_topo[['topography','height']],l_slope)

grid_topo_zm_line =  grid_topo_rot.where(grid_topo_rot.Y>1.5E-7, 
                                    drop = True).where(grid_topo_rot.Y<1.65E-7,
                                                       drop = True).where(grid_topo_rot.X>1E-7, 
                                                                          drop = True).where(grid_topo_rot.X<2.2E-7, 
                                                                                             drop = True)

grid_topo_zm_line

grid_3D_zm_line

grid_3D_zm_line_x = grid3D_line_avg_pks(grid_3D_zm_line, average_in= 'Y', distance =10, threshold = 50E-15 )

grid_3D_zm_line_x = grid_3D_zm_line_x.assign_coords(X =  grid_3D_zm_line_x.X.values  -  grid_3D_zm_line_x.X.min().values)


# + tags=[]
def  grid_lineNpks_offset(xr_data_l_pks, 
                          ch_l_name = 'LIX_unit_calc',
                          plot_y_offset= 2E-11, 
                          peak_LIX_min = 1E-13, 
                          fig_size = (6,8), 
                          legend_title = None):
    # add peak point one-by-one (no palett func in sns)
    #  after find peak & padding
    # use choose the channel to offset-plot 
    # use the plot_y_offset to adjust the offset values 
    ch_l_name = ch_l_name
    ch_l_pk_name = ch_l_name +'_peaks_pad'
    line_direction = xr_data_l_pks.line_direction
    plot_y_offset  =  plot_y_offset
    
    sns_color_palette = "rocket"
    #color map for fig
    
    #xr_data_l_pks
    ### prepare XR dataframe for line spectroscopy plot 
    xr_data_l_pks_ch_slct = xr_data_l_pks[[ch_l_name,ch_l_pk_name]]
    # choose the 2 channels from 2nd derivative (to maintain the coords info) 


    #line_direction check again 
    
    if xr_data_l_pks.line_direction == 'Y': 
        spacing = xr_data_l_pks_ch_slct.Y_spacing
    elif xr_data_l_pks.line_direction == 'X': 
        spacing = xr_data_l_pks_ch_slct.X_spacing
    else : 
        print('check direction & X or Y spacing for offset') 

    xr_data_l_pks_ch_slct['offset'] = (xr_data_l_pks_ch_slct[line_direction] - xr_data_l_pks_ch_slct[line_direction].min())/spacing
    # prepare offset index channnel 
    print (' plot_y_offset  to adjust line-by-line spacing')

    xr_data_l_pks_ch_slct[ch_l_name+'_offset'] = xr_data_l_pks_ch_slct[ch_l_name] + plot_y_offset * xr_data_l_pks_ch_slct['offset']
    # offset the curve b
    print (xr_data_l_pks_ch_slct)
    

    ch_l_name_df_list = [] 
    ch_l_name_pks_df_list = []
    # prepare empty list to append dataframes in the for - loop (y_i or x_i)

    #line_direction check again 
    #########################
    # line_diection check
    if xr_data_l_pks_ch_slct.line_direction == 'Y': 
        lines  = xr_data_l_pks_ch_slct.Y

        for y_i, y_points in enumerate (lines):

            # set min peak height (LIX amplitude =  resolution limit)

            y_i_pks  = xr_data_l_pks_ch_slct[ch_l_pk_name].isel(Y = y_i).dropna(dim='peaks').astype('int32')
            # at (i_th )Y position, select peak index for bias_mV
            real_pks_mask = (xr_data_l_pks_ch_slct.isel(Y = y_i, bias_mV = y_i_pks.values)[ch_l_name] > peak_LIX_min).values
            # prepare a 'temp' mask for each Y position 
            y_i_pks_slct =  y_i_pks.where(real_pks_mask).dropna(dim='peaks').astype('int32')
            # y_i_pks_slct with mask selection  

            ch_l_name_y_i_df = xr_data_l_pks_ch_slct[ch_l_name+'_offset'].isel(Y = y_i).to_dataframe()
            # LIX_offset  at Y_i position 
            ch_l_name_df_list.append(ch_l_name_y_i_df)
            
            ch_l_name_y_i_pks_df = xr_data_l_pks_ch_slct.isel(Y = y_i, bias_mV = y_i_pks_slct.values)[ch_l_name+'_offset'].to_dataframe()
            # selected peaks with offest Y 
            ch_l_name_pks_df_list.append(ch_l_name_y_i_pks_df)
            
            # data at selected Y, & peak position, LIX_offset
            
    #########################
    # line_diection check

    elif xr_data_l_pks_ch_slct.line_direction == 'X': 
        lines = xr_data_l_pks_ch_slct.X

        for x_i, x_points in enumerate (lines):

            # set min peak height (LIX amplitude =  resolution limit)

            x_i_pks  = xr_data_l_pks_ch_slct[ch_l_pk_name].isel(X = x_i).dropna(dim='peaks').astype('int32')
            # at (i_th )X position, select peak index for bias_mV
            real_pks_mask = (xr_data_l_pks_ch_slct.isel(X = x_i, bias_mV = x_i_pks.values)[ch_l_name] > peak_LIX_min).values
            # prepare a 'temp' mask for each X position 
            x_i_pks_slct =  x_i_pks.where(real_pks_mask).dropna(dim='peaks').astype('int32')
            # x_i_pks_slct with mask selection  

            ch_l_name_x_i_df = xr_data_l_pks_ch_slct[ch_l_name+'_offset'].isel(X = x_i).to_dataframe()
            # LIX_offset  at X_i position 
            ch_l_name_df_list.append(ch_l_name_x_i_df)
            ch_l_name_x_i_pks_df = xr_data_l_pks_ch_slct.isel(X = x_i, bias_mV = x_i_pks_slct.values)[ch_l_name+'_offset'].to_dataframe()
            ch_l_name_pks_df_list.append(ch_l_name_x_i_pks_df)
            
            # selected peaks with offest X 
            
    else : 
        print('check direction & X or Y spacing for offset') 
    
    ch_l_name_df = pd.concat(ch_l_name_df_list).reset_index()
    ch_l_name_pks_df = pd.concat(ch_l_name_pks_df_list).reset_index()
    
    fig,ax = plt.subplots(figsize = fig_size)

    sns.lineplot(data = ch_l_name_df,
                         x ='bias_mV', 
                         y = ch_l_name+'_offset',
                         palette = "rocket",
                         hue = xr_data_l_pks.line_direction,
                         ax = ax)

    sns.scatterplot(data = ch_l_name_pks_df,
                            x ='bias_mV',
                            y = ch_l_name+'_offset',
                            palette ="rocket",
                            hue = xr_data_l_pks.line_direction,
                    s=20,
                            ax = ax)
    # legend control!( cut the handles 1/2)
    ax.set_xlabel('Bias (mV)')   
    #ax.set_ylabel(ch_l_name+'_offset')   
    ax.set_ylabel('LDOS')   
    handles0, labels0 = ax.get_legend_handles_labels()
    labels1 = [ str(float(label)*100) for label in labels0[:int(len(labels0)//2)] ] 
    # convert the line length as nm
    print(labels1)
    ax.legend(handles0[:int(len(handles0)//2)],
              labels1, title = legend_title)
    # use the half of legends (line + scatter) --> use lines only
    #plt.show()
    return xr_data_l_pks_ch_slct, ch_l_name_df, ch_l_name_pks_df, fig
# +
grid_3D_zm_line_x_l_pk_LIX_slct, grid_3D_zm_line_x_l_LIX_df, grid_3D_zm_line_x_l_pk_LIX_df, fig =  grid_lineNpks_offset(grid_3D_zm_line_x,
                                                                                                                        fig_size = (3,5), 
                                                                                                                        legend_title= 'L (nm)',
                                                                                                                       plot_y_offset= 0.1E-12)
fig.savefig('grid_3D_zm_line_xL_0.svg')


# -

cmap_rocket  = sns.color_palette("rocket", as_cmap=True)

    
    #cmap_rocket.figsave('cmap_rocket.png')

fig,ax = plt.subplots()
isns.imshow(cmap_rocket.colors, cmap ="rocket", ax=ax)
plt.savefig('rocket_cmap.svg')

grid_topo_zm_line



# +
grid_topo_zm_line = grid_topo_zm_line.assign_coords(X =  grid_topo_zm_line.X.values  -  grid_topo_zm_line.X.min().values)
# set X  coords start from 0 

grid_topo_zm_line = plane_fit_surface_xr (grid_topo_zm_line)
# use the surface line substraction to compenstae tilting Y 

grid_topo_zm_line['topography'] = grid_topo_zm_line.topography - grid_topo_zm_line.topography.min()
# set z topography  min = 0
grid_topo_zm_df = grid_topo_zm_line.mean(dim= 'Y').topography.to_dataframe().reset_index()

grid_topo_zm_df

# +


fig,ax =plt.subplots(figsize = (1.5,4))
ax.plot(grid_topo_zm_df.topography*1E10, grid_topo_zm_df.X*1E9) 
ax.set_xlabel('Z ($\AA$)')
ax.set_ylabel('L (nm)')
ax.set_ylim(top = 150)
ax.yaxis.set_label_position("right")
ax.yaxis.set_ticks_position("right")
plt.tight_layout()
plt.savefig('grid_topo_zm_line_xL_0.svg')
plt.show()
# -




