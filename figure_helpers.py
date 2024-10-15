import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    ''' Loads all data

    Returns
    -------
    output: tuple
        (fig2_db, sel_df, fig3all_db, fig3_db, t1df, t2df, database, soc_context_df,df, scdf)
        
    Notes
    -----
    fig2_db
    sel_df
    fig3all_db: all stimuli
    fig3_db: only BOS
    t1df, t2df,
    database: mda through mdd
    
    soc_context_df: mdy and mdx
    
    df: combined `soc_context_df` and `database` aggregate by tetrode
    
    scdf: filtered out high outliers from soc_context_df
    '''
    
    path = r"../../../data/fig_data"

    # Fig 2
    fig2_db = pd.read_csv(os.path.join(path, r'Fig1Stats.csv'))

    # Fig 3
    sel_df = pd.read_csv(os.path.join(path, 'selectivityIndex.csv'))

    # Not relative to BOS (for all other anlyses)
    fig3all_db = pd.read_csv(os.path.join(path, 'Fig3All_lofilt.csv'))

    # Relative to BOS (for d')
    fig3_db = pd.read_csv(os.path.join(path, 'Fig3BOST_lofilt.csv')) 

    t1df = pd.read_csv(os.path.join(path,'t1MapsFR.csv'), header=None)
    t2df = pd.read_csv(os.path.join(path,'t2MapsFR.csv'), header=None)

    # Fig 4
    database = pd.read_pickle(os.path.join(path, 'database.pkl'))
    soc_context_df =pd.read_pickle(os.path.join(path, 'soc_context_df.pkl'))

    #------------------------------------------------------------------
    # Filter tetrodes for each mda, mdb, and mdd
    database = database[  ((database.subject=='mda') & (database.tetrode==4))
                        | ((database.subject=='mdb') & (database.tetrode==1))
                        | ((database.subject=='mdd') & (database.tetrode==4))]

    # Get equivalent dataframe for subject mdy
    # note: could be for mdy and mdx if preferred
    soc_context_df = soc_context_df[
        (
            (soc_context_df.tetrode==2) 
            & (soc_context_df.date=='12 4 18')
            & (soc_context_df.subject=='mdy')
        ) | ((soc_context_df.tetrode==2) 
            & (soc_context_df.date=='11 29 18')
            & (soc_context_df.subject=='mdy')
        )
    ]

    # Data cleaning: inconsistent column names
    database.loc[database.stimulus=='BOS_REV_new', 'stimulus']='BOS REV'
    database.loc[database.stimulus=='BOS_REV', 'stimulus']='BOS REV'
    database.loc[database.stimulus=='BOS_new', 'stimulus']='BOS'

    # Combine mda, mdb, mdd data with mdy data to form df
    df = pd.concat([
        soc_context_df[soc_context_df.context=='solitary'].drop('context',1),
        database
    ])
    
    # Aggregate by tetrode
    df = df.groupby(['subject','stimulus','period','frequency','stimulus_index','tetrode']).agg(np.mean).reset_index()
    

    def quant(x):
        # Split, *apply*, combine on upper outlier removal
        return x[x['power'] < x['power'].quantile(0.85)]


    # For larger dataset
    df = df.groupby(['subject','stimulus','period','frequency','tetrode']).apply(quant).reset_index(drop=True)

    # And for soccial context dataset as well
    scdf = soc_context_df.groupby(
        ['subject','stimulus','period','frequency','tetrode']
    ).apply(quant).reset_index(drop=True)
    
    return (fig2_db, sel_df, fig3all_db, fig3_db, t1df, t2df, database, soc_context_df, df)

    

def interaction(data, respvar, axes, pal=['r','b']):
    '''Makes a plot of the interaction effects, comprising a strip 
    plot and a point plot using plot_each.
    
    Parameters
    ----------
    Same as maineffect, above
    
    Returns
    -------
    output: (matplotlib.pyplot.tuple)
        Axes in a tuple of subplot axes
    '''
    for ind in range(2): # first narrow then broad
        plot_each(axes[ind], [pal[ind]], data[data['broad']==ind], respvar)
    return axes


def maineffect(data, respvar, axis=None, pal=['r', 'b']):
    '''Makes a plot of the main effect comprising a 
    strip plot and a point plot using plot_each
    
    Unlike interaction(), maineffect() generates two plots next to each other.
    
    Parameters
    ----------
    data: (pandas.DataFrame)
        Dataframe containing measurements across cells across treatments
    
    Returns
    -------
    axis: (matplotlib.pyplot.axis)
        Single axis that has been passed through plot_each
    '''
    
    if axis == None:
        _, axis = plt.subplots()
    plot_each(axis, pal, data, respvar)
    return axis


def plot_each(axis, pal, df, respvar):
    '''Produces a superimposed strip/point plot
    comparing treatment conditions
    
    Parameters
    ----------
    axis: (matplotlib.pyplot.axis)
        Axis on which to plot
    
    pal: list
        List of matplotlib arguments for colors (e.g., 'r', 'b')
        
    df: pandas.DataFrame
        Dataframe containing information about response variable
        respvar across cells across treatment conditions
    
    respvar: String
        Response variable to be plotted
    '''
    both_cell_types = len(df.broad.unique()) > 1
    
    sns.stripplot(x='treatments', y=respvar, hue='broad', data=df,
                  ax=axis, palette=pal, dodge=True, size=3.5)
    
    sns.pointplot(x='treatments', y=respvar, hue='broad', data=df,
                  ax=axis, palette=pal, markers='_', dodge=True,
                  errwidth=.4, join=False)
    
    axis.legend_.remove()



def organize_axes(axis, ylab, subfig_label, xticklab=['sol aud','soc aud'],
                  sig_apostrophes=None, yticklabs=None, annotation=None):
    '''    
    Parameters
    ----------
    axis: matplotlib.pyplot.plt
        Axis on which to act
    
    ylab: String
        y-axis label
    
    subfig_label: String
        Subfigure letter (e.g., the c in Fig 1c).
        
    xticklab: list(String)
        List of strings for labeling conditions on x axis
    
    sig_apostrophes: String
        Significance apostrophes
        
    yticklabs: String
        List of strings for labeling conditions on y axis

    annotation: String
        Additional label for data, usually "filtered" or "unfiltered"
    '''
    ylim = axis.get_ylim()
    xlim = axis.get_xlim()
    
    # Tidy axes
    if xticklab == '':
        axis.set_xticks([])
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        axis.tick_params(axis=u'both', which=u'both',length=0)
        sns.despine(ax=axis, bottom=True, left=True)
    
    # Set y axis tick labels, if any
    if yticklabs is not None:
        axis.set_yticklabels(yticklabs)
        plt.yticks(rotation=0)
        
    # refine y axis label 
    if ylab is None:
        ylab = respvar
    
    axis.set(xlabel='', ylabel=ylab, xticklabels=xticklab)

    # add subfig label
    axis.text(
        xlim[0] - (xlim[1]-xlim[0])*0.1,
        ylim[1] + (ylim[1]-ylim[0])*0.15,
        subfig_label, fontsize=16, fontweight='bold',
        va='top', ha='right')
    
    if axis.legend_ is not None:
        axis.legend_.remove()    
        
    sns.despine(ax=axis)
    
    # Add significance apostrophes in figure
    if sig_apostrophes is not None:
        y_height = (abs(ylim[0])+abs(ylim[1])) * 0.95 - abs(ylim[0]) # hits 95% mark
        axis.text(sum(xlim)/2, y_height, sig_apostrophes, fontsize=16, fontweight='bold', va='top', ha='center')

    # Position annotation in figure
    if annotation is not None:
        y_height = (abs(ylim[0])+abs(ylim[1])) * 0.75 - abs(ylim[0]) # hits 75% mark
        x_pos = (xlim[1]-xlim[0]) * 0.5
        axis.text(x_pos, y_height, annotation)


def heatmaps(t1df, t2df, units, t1axes, t2axes):
    ''' 
    Docstring
    
    Parameters
    ----------
    t1df, t2df: (pandas.DataFrame)
        dataframe with pre, post
    units: 
        indices for units in the dataframes
    t1axes, t2axes: (matplotlib.pyplot.axis)
        to plot on        
    '''
    for i in range(len(units)):
        unit_ind = units[i]

        curr_t1df = t1df[unit_ind:unit_ind + 4]
        curr_t2df = t2df[unit_ind:unit_ind + 4]
        
        # Get global max and min for both treatments to bound the colormaps
        '''Note: the reason I do not put this into a for loop iterating
        over t1 and t2 is this line of code: I want the color map for t1 and
        t2 to depend on the bounds on t1, as it is more or less guaranteed
        to be larger than that of t2.'''
        vmin, vmax = curr_t1df.min().min(), t1df[0:4].max().max()

        # Get t1 and t2 (treatment 1/2) axes
        t1ax, t2ax = t1axes[i], t2axes[i]
        
        # Common keyword arguments for heatmap
        kwargs = {'cmap':'viridis', 'vmin':vmin, 'vmax':vmax, 'annot':True, 'cbar':False}
        
        sns.heatmap(curr_t1df, ax=t1ax, **kwargs)
        sns.heatmap(curr_t2df, ax=t2ax, **kwargs)
        
        sns.despine(ax=t1ax, left=True, bottom=True)
        sns.despine(ax=t2ax, left=True, bottom=True)


def full_despine(ax=None):
    ''' Completes a full despining of the figure by 
    removing y ticks, x ticks, y labels, x labels, x
    tick labels and y tick labels'''
    
    if ax == None:
        ax = plt.gca()       
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel('')
    sns.despine(left=True, bottom=True)
    ax.set_xlabel('')
    ax.set_xticklabels([''])
    ax.set_xticks([])
    

def show_freq(fq, df):
    '''Compare frequency power at frequency fq for each frequency 
    in a dataframe in each condition: pre, during, post
    
    Parameters
    ----------
    fq: (int)
        frequency to be plotted 
    
    df: (pandas.DataFrame)
        dataframe containing power for each frequency at each condition.
    '''
    
    plt.figure(figsize=(9, 7))
    
    # Get all unique subjects
    subs = pd.unique(df.subject)
    
    # List of stimuli
    stims = ['BOS','BOS REV', 'conspecific', 'whitenoise']
    
    # Iterating through combinations of subjects and stimuli
    for i, sub in enumerate(subs):
        for j, stim in enumerate(stims):
            
            ax = plt.subplot(4, len(subs), j*len(subs) + (i+1))
            subdf = df[
                (df.frequency==fq) 
                & (df.subject==sub) 
                & (df.stimulus == stim)
            ]
            
            # Sometimes the dataframe is empty. If not empty,
            if len(subdf) > 0:
                
                # Generate stripplot
                sns.stripplot(
                    data=subdf,
                    x='subject',
                    y='power',
                    hue='period',
                    ax=ax,
                    alpha=.85,
                    dodge=True,
                    size=2,
                    hue_order=['pre','during','post']
                )
                
                # Remove legend
                ax.legend_.remove()
            
            # Row and column labels
            if i == 0:
                # Stimuli go on the left
                ax.set_title(stim)
                
            if j == 0:
                # Subject goes on the top
                ax.set_title(sub)

            if i == 0 and j == 0:
                # Top left shows subject and stimulus
                ax.set_title(stim + ', ' + sub)
            
            # Fully despine & clean the current figure
            full_despine()

            
def demarcate_freq_bands(facetgrid):
    ''' Demarcate frequency bands of interest '''
    # For each column of axes in facet grid:
    for ax_col in facetgrid.axes:
        
        # For each axis (row) in the current column:
        for ax in ax_col:
            
            # Get y limit
            ylim = ax.get_ylim()
            
            # For each of these ranges, highlight:
            for rang in [[8,10],[22,24],[33,35]]:
                ax.fill_between(
                    rang,
                    ylim[0], ylim[1],
                    color='g',
                    alpha=0.2
                )
                
def medfilt(d):
    '''Median filter
    Take the median of all stimulus presentations. This function is not being used right now.
    Parameters
    ----------
    d: pandas.DataFrame
        Input data
        
    Returns
    -------
    output: pandas.DataFrame
        Median-filtered data
    '''
    return d.groupby(
        ['subject','stimulus','period','frequency']
    ).agg(
        np.median
    ).reset_index()