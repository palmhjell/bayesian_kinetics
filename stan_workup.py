"""
Includes helper functions for analyzing and visualizing the outputs of
a Bayesian approach to enzyme kinetics in Stan.

Many of these approaches are derived from Justin Bois's courses at Caltech,
specifically bebi103 'Data Analysis in the Biological Sciences', with code
available at https://github.com/justinbois/bebi103.

These are often very specific to the data used in this analysis and not
all that generalizable. Certain naming schemes within functions should
be updated if used.
"""
# Data processing
import numpy as np
import pandas as pd

# Stats
from statsmodels.distributions.empirical_distribution import ECDF

# Data viz
import bokeh.io
import bokeh.plotting


def summarize(data, name=None, units=None, plot=True):
    """Summarizes an array of posterior sample draws.
    
    Parameters:
    -----------
    data: pandas Series or numpy array
        A 1D array containing MCMC posterior sample draws
    name: string
        Name of the parameter; automatically pulled if
        data is a pd.Series and name=None
    units: string
        The units associated with the parameter
    plot: bool, default True
        Whether or not to display an empirical cumulative
        distribution (ECDF) plot
    
    Returns:
    --------
    summary: DataFrame
        Contains the summary stats of the samples
    """
    if isinstance(data, pd.Series) and name is None:
        name = data.name
    if units is not None:
        name = f'{name} ({units})'
        
    vals = np.round(np.quantile(data, [0.5, 0.025, 0.975]), 3)
    
    df = pd.DataFrame({
        'value': name,
        'median': vals[0],
        '95% CR': f'[{vals[1]}, {vals[2]}]'
    },
        index=[0])
    
    if plot:
        ecdf = ECDF(data.values)
        p = bokeh.plotting.figure(plot_width=400, plot_height=300)
        p.xaxis.axis_label = name
        p.yaxis.axis_label = 'ECDF'
        p.circle(ecdf.x, ecdf.y)
        bokeh.io.show(p)
        
    return df


def predictive_regression(
    df,
    ppc_name,
    percentiles=[95, 75, 50, 25],
    x_label='[indole] (µM)',
    y_label='k (per sec)',
    width=500,
    height=400,
    colors=[],
):
    """Displays the credible regions of the posterior, given
    that the model contains posterior predictive checks (ppcs).
    In this instance, ppcs are obtained on the range (1, n+1),
    where 'n' corresponds to the concentration of substrate.
    
    Parameters:
    -----------
    df: stan_samples.to_dataframe() object
        DataFrame obtained from the sampling that contains
        the ppc values
    ppc_name: string
        Name of base ppc variable in stan code. Samples are
        written in the form ppc_name[i]
    percentiles: list
        Percentile regions to display, extending from the median
        value. Takes values from 0 to 100 (probably exclusive...)
    x_label: string
        Name of the x-axis values for the regression
    y_label: string
        Name of the y-axis values (ppc samples) for the regression
    width: int
        Plot width
    height: int
        Plot height
    colors: list-like
        Colors for the percentiles, from first to last. Not really
        validated against the percentiles list here, so keep that in
        mind.
    """
    # Build ptiles (percentile ranges)
    ptiles = [pt for pt in percentiles if pt > 0]
    ptiles = ([50 - pt/2 for pt in percentiles] + [50]
                + [50 + pt/2 for pt in percentiles[::-1]])
    ptiles = np.array(ptiles) / 100
    ptiles_str = [str(pt) for pt in ptiles]
    
    # Get ppc dataframe quantiles
    ppc_string = f'{ppc_name}['
    columns = [col for col in df.columns if ppc_string in col]
    df_ppc = df[columns].quantile(ptiles).reset_index().melt(id_vars='index', var_name='ppc', value_name=y_label)
    
    # Convert to indole concentration (or whatever x-axis label)
    df_ppc[x_label] = df_ppc['ppc'].apply(lambda x: int(x.replace(ppc_string, '').replace(']', '')) - 1)
    
    # Set new index
    df_ppc.set_index(['index', 'ppc'], inplace=True)
    df_ppc.index.names = ['quantile', 'ppc']
    
    # Prepare values
    concs = df_ppc[x_label].unique()
    
    if not colors:
        colors = [
            '#bfd3bf',
            '#99b899',
            '#739d73',
            '#4d824d',
            '#266826',
            '#004D00',
        ]
    
    # Set up plot
    p = bokeh.plotting.figure(plot_width=width, plot_height=height)
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    
    # Add credible regions
    for i, ptile in enumerate(ptiles_str[:len(ptiles_str)//2]):
        p.varea(
            x=concs,
            y1=df_ppc.loc[float(ptile)][y_label],
            y2=df_ppc.loc[float(ptiles_str[-i-1])][y_label],
            color=colors[i],
        )
        
    # Plot the median
    p.line(
        x=concs,
        y=df_ppc.loc[0.5][y_label],
        line_width=2,
        color=colors[-1],
    )
    
    return p