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
# # <font color=blue>Fig 4- STM/S data analysis (Fig 4 S islands )</font>
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
S_1035 = file_list_df[(file_list_df.type == 'gwy') ]
#W_2009

S_1035_topo = gwy_image2df(S_1035.file_name.values[0])
S_1035_topo
# -


S_1035_topo_zm =  gwy_df_channel2xr (S_1035_topo,0)
S_1035_topo_zm


# +
isns.set_image(origin = 'lower')

fig, ax = plt.subplots(figsize = (4,4))
isns.imshow(S_1035_topo_zm*1E9,
            robust = True, perc=(0.5,99.5),
            cmap = 'viridis',  
            dx=80/1024, 
            units="nm", 
            cbar_label = 'z (nm)',
            ax = ax)
plt.savefig('S_1035_topo_zm topo.svg')
plt.show()


# +
################
# Choose the STS for  W islands 
    
S_grids = file_list_df[
    (file_list_df.type == '3ds') & 
    (file_list_df.file_name.str.contains('line'))]
S_grids
# -


# ## Grid 003 

grid_xr = grid2xr(S_grids.file_name.values[0])


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
# -

grid_topo_th = threshold_mean_xr(grid_topo)
grid_topo_zm = grid_topo.where(grid_topo_th.topography.isnull(), drop = True)


# + jp-MarkdownHeadingCollapsed=true tags=[]
isns.set_image(origin = 'lower')

#isns.imshow( plane_fit_y(grid_topo.topography) )
# check the how to show the 2D image 

isns.imshow(grid_topo_zm.topography*1E9,
            robust = True,
            aspect = 'equal', 
            cmap = 'viridis', 
            dx=80/10, units="nm", 
            cbar_label = 'z (nm)',
            fontsize='xx-large')
# -

# ### topography based terrace assign

grid_topo_zm_y_avg  = grid_topo_zm.mean (dim = 'Y')
# test y avg curve 
grid_topo_zm_y_avg.to_dataframe().plot()

# +
grid_topo_zm_y_avg_th =  threshold_multiotsu_xr( grid_topo_zm_y_avg, multiclasses= 3)

# xr.concat  : for the same data_vars names
# xr.concat ([grid_topo, grid_topo_zm_y_avg_th],dim ="X")
# -

# ### add terrace assignment data & merge

# +
# xr data set data vars name change 
grid_topo_zm_y_avg_th = grid_topo_zm_y_avg_th.rename_vars({'topography':'terraces'})

# MERGE the terrace data vars to the original dataset 

grid_topo_terraces = grid_topo.merge(grid_topo_zm_y_avg_th)
grid_topo_terraces
#xr.concat ([grid_topo, grid_topo_zm_y_avg_th], dim = "Y")#,join="override")
# -

grid_topo_terraces.terraces.plot()

# ### extract the layer dependent dataframe

# +
#grid_topo_terraces.terraces == 2
grid_3D_4ML_df = grid_3D.where(grid_topo_terraces.terraces == 2, 
                            drop = True).where(grid_3D.X<3E-8,
                                               drop = True).LIX_unit_calc.to_dataframe()
grid_3D_4ML_df = grid_3D_4ML_df.rename(columns = {'LIX_unit_calc':'S4ML'})
#grid_3D_4ML_df

grid_3D_3ML_df = grid_3D.where(grid_topo_terraces.terraces == 1, 
                            drop = True).where(grid_3D.X<7E-8,
                                               drop = True).LIX_unit_calc.to_dataframe()
grid_3D_3ML_df = grid_3D_3ML_df.rename(columns = {'LIX_unit_calc':'S3ML'})
#grid_3D_3ML_df

grid_3D_2ML_df = grid_3D.where(grid_topo_terraces.terraces == 0, 
                            drop = True).where(grid_3D.X>3E-8,
                                               drop = True).LIX_unit_calc.to_dataframe()
grid_3D_2ML_df = grid_3D_2ML_df.rename(columns = {'LIX_unit_calc':'S2ML'})
#grid_3D_3ML_df

grid_3D_S234_df =pd.concat([grid_3D_2ML_df,grid_3D_3ML_df,grid_3D_4ML_df], axis =1)
grid_3D_S234_df


# -

grid_3D_S234_df = grid_3D_S234_df.reset_index().melt(id_vars = ['Y','X','bias_mV'])
#grid_3D_S234_df
grid_3D_S234_df = grid_3D_S234_df.rename(columns = {'variable':'terrace','value':'dIdV'} )
grid_3D_S234_df

# ### plot with sns

fig,ax = plt.subplots(figsize = (6,4))
sns.lineplot(data = grid_3D_S234_df, x = 'bias_mV', y ='dIdV', hue = 'terrace', ax= ax)
ax.set_ylabel('dI/dV (A/V)')
ax.set_xlabel('Bias (mV)')
ax.set_yscale('log')
plt.show()


# ### extract the mean values w.r.t. layers

# +
##  layer dependent means
grid_3D_S234_df_mean = grid_3D_S234_df.groupby(['bias_mV', 'terrace']).dIdV.mean().unstack()
#grid_3D_S234_df_mean


##  layer dependent means ==> melt for sns plot 

grid_3D_S234_df_mean_melt = grid_3D_S234_df_mean.reset_index().melt(id_vars = 'bias_mV').rename(columns = {'value' : 'dIdV'})
#grid_3D_S234_df_mean_melt

# +
fig,ax = plt.subplots(figsize = (6,4))
sns.lineplot(data = grid_3D_S234_df_mean_melt, x = 'bias_mV', y = 'dIdV', hue = 'terrace', ax =ax)

#grid_3D_S234_df.groupby(['bias_mV', 'terrace']).dIdV.mean().unstack().plot( ax= ax)
ax.set_ylabel('dI/dV (A/V)')
ax.set_xlabel('Bias (mV)')
ax.set_yscale('log')
plt.show()

# -

# ### find peaks 

# + [markdown] tags=[]
# #### check the mean curves without CI values + seaborn plot 
# * concat the STS curves 
# * the same as seaborn result
#
#
# #### Check 2nd derivative to find peaks 
# * convert to Xarray 
# * calc derivative 
# * smoothing with 'savgol_filter'
# -

grid_3D_S234_df_mean

# + [markdown] tags=[]
#

# + [markdown] tags=[]
# ### Find peak positions 
# * using 2nd derivative of each curves
# * __Local minimums in d2(LDOS)/dV2) = Peaks in LDOS__
# * delete the minor peak ( comes from ripples in the gap region)
#
# #### find dips in 2nd derivative for peaks in STS 
# * use dps 
# * Define LIX resolution limit 
# * (manually )Select peaks 
#
# -

# SG smoothing  first 
for ch in grid_3D_S234_df_mean: 
    grid_3D_S234_df_mean[ch] = sp.signal.savgol_filter(grid_3D_S234_df_mean[ch],
                                            window_length = 7,
                                            polyorder = 3)

grid_3D_S234_df_mean_2deriv_sg

# +
# 2nd derivative 
grid_3D_S234_df_mean_1deriv = grid_3D_S234_df_mean.to_xarray().differentiate(coord = 'bias_mV')
grid_3D_S234_df_mean_2deriv = grid_3D_S234_df_mean_1deriv.differentiate(coord = 'bias_mV')
grid_3D_S234_df_mean_2deriv_sg =  grid_3D_S234_df_mean_2deriv.copy()

# smoothing after 2nd derivative 
for ch in grid_3D_S234_df_mean_2deriv:
    grid_3D_S234_df_mean_2deriv_sg[ch].values =  sp.signal.savgol_filter(grid_3D_S234_df_mean_2deriv[ch].values,
                                                              window_length = 7,
                                                              polyorder = 3)

#grid_3D_S234_df_mean_2deriv_sg

###########

# find the peaks& dips 

for ch in grid_3D_S234_df_mean_2deriv_sg:
    grid_3D_S234_df_mean_2deriv_sg[ch+'_pks'] = xr.DataArray(sp.signal.find_peaks(grid_3D_S234_df_mean_2deriv_sg[ch].values))
    grid_3D_S234_df_mean_2deriv_sg[ch+'_dps'] = xr.DataArray(sp.signal.find_peaks(-1*grid_3D_S234_df_mean_2deriv_sg[ch].values, distance  = 10))
    
grid_3D_S234_df_mean_2deriv_sg

########################


# +
# pks N dps
S2ML_dps = grid_3D_S234_df_mean_2deriv_sg.S2ML_dps.data.item()[0]
S3ML_dps = grid_3D_S234_df_mean_2deriv_sg.S3ML_dps.data.item()[0]
S4ML_dps = grid_3D_S234_df_mean_2deriv_sg.S4ML_dps.data.item()[0]

# set LDOS limit 
LIX_limit= 1E-15
# -


# LDOS value at the peak is larger than LIX_limit
#(W_mean_df.W1[W1_1_dps] > LIX_limit)
S2ML_dps = S2ML_dps[grid_3D_S234_df_mean.iloc[S2ML_dps].S2ML > LIX_limit]
S3ML_dps = S3ML_dps[grid_3D_S234_df_mean.iloc[S3ML_dps].S3ML > LIX_limit]
S4ML_dps = S4ML_dps[grid_3D_S234_df_mean.iloc[S4ML_dps].S4ML > LIX_limit]



# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ##### S2ML Dips

# +
S2ML_mean = grid_3D_S234_df_mean.S2ML
S2ML_mean_2deriv = grid_3D_S234_df_mean_2deriv_sg.S2ML.to_pandas()
#S2ML_mean

fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = S2ML_mean,
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(data = S2ML_mean.iloc[S2ML_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = S2ML_mean_2deriv, 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(data = S2ML_mean_2deriv.iloc[S2ML_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
S2ML_dps_slct = S2ML_dps[[0,1,2]]
#print(W2_1_dps_slct)

sns.lineplot(data = S2ML_mean,
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(data = S2ML_mean.iloc[S2ML_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = S2ML_mean_2deriv, 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(data = S2ML_mean_2deriv.iloc[S2ML_dps_slct],
                ax=axs1tw, color  = 'grey')
axs1tw.set_ylabel('d2(LODS)/dV2', color='grey')


plt.suptitle('S2_2ML_peaks')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ##### S3ML Dips

# +
S3ML_mean = grid_3D_S234_df_mean.S3ML
S3ML_mean_2deriv = grid_3D_S234_df_mean_2deriv_sg.S3ML.to_pandas()
#S3ML_mean

fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = S3ML_mean,
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(data = S3ML_mean.iloc[S3ML_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = S3ML_mean_2deriv, 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(data = S3ML_mean_2deriv.iloc[S3ML_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
S3ML_dps_slct = S3ML_dps[[0,1,2,3]]
#print(W2_1_dps_slct)

sns.lineplot(data = S3ML_mean,
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(data = S3ML_mean.iloc[S3ML_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = S3ML_mean_2deriv, 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(data = S3ML_mean_2deriv.iloc[S3ML_dps_slct],
                ax=axs1tw, color  = 'grey')
axs1tw.set_ylabel('d2(LODS)/dV2', color='grey')


plt.suptitle('S2_3ML_peaks')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ##### S4ML Dips

# +
S4ML_mean = grid_3D_S234_df_mean.S4ML
S4ML_mean_2deriv = grid_3D_S234_df_mean_2deriv_sg.S4ML.to_pandas()
#S3ML_mean

fig,axes = plt.subplots(2,1, figsize = (6,8))
axs = axes.ravel()

axs0tw = axs[0].twinx()
axs1tw = axs[1].twinx()
# double Y setting 

sns.lineplot(data = S4ML_mean,
             ax=axs[0], color  = 'tab:blue')
sns.scatterplot(data = S4ML_mean.iloc[S4ML_dps],
                ax=axs[0], color  = 'tab:blue')
axs[0].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = S4ML_mean_2deriv, 
             ax=axs0tw, color  = 'grey')
sns.scatterplot(data = S4ML_mean_2deriv.iloc[S4ML_dps],
                ax=axs0tw, color  = 'grey')
axs0tw.set_ylabel('d2(LODS)/dV2', color='grey')


# selected peak points 
S4ML_dps_slct = S4ML_dps[[0,1,2,3,4]]


sns.lineplot(data = S4ML_mean,
             ax=axs[1], color  = 'tab:blue')
sns.scatterplot(data = S4ML_mean.iloc[S4ML_dps_slct],
                ax=axs[1], color  = 'tab:blue')
axs[1].set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = S4ML_mean_2deriv, 
             ax=axs1tw, color  = 'grey')
sns.scatterplot(data = S4ML_mean_2deriv.iloc[S4ML_dps_slct],
                ax=axs1tw, color  = 'grey')
axs1tw.set_ylabel('d2(LODS)/dV2', color='grey')


plt.suptitle('S2_4ML_peaks')

# + [markdown] tags=[]
# ###  plot peaks & curves
#
# -

S3ML_mean.iloc[S3ML_dps]+offset_3ML

# +
# W stacks 

fig,axs = plt.subplots(figsize = (6,4))


offset_2ML = 0E-11
offset_3ML = 5E-11
offset_4ML = 10E-11


sns.lineplot(data = S2ML_mean*1E9,
             ax=axs, color  = 'tab:blue')
sns.scatterplot(data = S2ML_mean.iloc[S2ML_dps]*1E9,
                ax=axs, color  = 'tab:blue')
axs.set_ylabel('LDOS', color='tab:blue')



sns.lineplot(data = (S3ML_mean+offset_3ML)*1E9,
             ax=axs, color  = 'tab:blue')
sns.scatterplot(data = (S3ML_mean.iloc[S3ML_dps]+offset_3ML)*1E9,
                ax=axs, color  = 'tab:blue')
axs.set_ylabel('LDOS', color='tab:blue')

sns.lineplot(data = (S4ML_mean+offset_4ML)*1E9,
             ax=axs, color  = 'tab:blue')
sns.scatterplot(data = (S4ML_mean.iloc[S4ML_dps]+offset_4ML)*1E9,
                ax=axs, color  = 'tab:blue')
axs.set_ylabel('LDOS', color='tab:blue')

axs.set_ylabel('dI/dV (nA/V)', color='tab:blue')
axs.set_xlabel('Bias (mV)', color='tab:blue')

