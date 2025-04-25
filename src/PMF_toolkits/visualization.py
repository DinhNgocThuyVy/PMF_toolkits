"""
Primary plotting interface for PMF results visualization.

This class provides all plotting capabilities for PMF results, accessible through
either pmf.plot or pmf.visualization. The plot interface is recommended for
consistency with common Python plotting libraries.

Examples
--------
>>> pmf = PMF(site="urban_site")
>>> # Plot factor profiles
>>> pmf.plot.plot_factor_profiles()
>>> # Plot contributions timeseries
>>> pmf.plot.plot_contributions_timeseries()
>>> # Plot seasonal contributions
>>> pmf.plot.plot_seasonal_contributions()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Optional, Union, Any, Tuple
import warnings
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

from .utils import (get_sourceColor, format_xaxis_timeseries, add_season, 
                   pretty_specie, get_sourcesCategories)

class PMFVisualization:
    """
    Primary visualization interface for PMF analysis results.
    
    This class handles all plotting functionality and is accessible through
    either pmf.plot (recommended) or pmf.visualization.
    
    Parameters
    ----------
    pmf : PMF
        Parent PMF object containing the data to visualize
    savedir : str, default="./"
        Directory to save plot files
    """
    
    def __init__(self, pmf, savedir: str = "./"):
        self.pmf = pmf
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['ytick.major.size'] = 4
        plt.rcParams['axes.linewidth'] = 1

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the dataframe has a DatetimeIndex.
        
        If the index is not already a DatetimeIndex, tries to convert it.
        If that fails, looks for a Date column to use as index.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check/convert
            
        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex
            
        Raises
        ------
        ValueError
            If a DatetimeIndex cannot be created
        """
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided")
            
        # If already a DatetimeIndex, return as is
        if isinstance(df.index, pd.DatetimeIndex):
            return df
            
        # Try to convert the current index to datetime
        try:
            df = df.copy()  # Avoid modifying the original
            df.index = pd.to_datetime(df.index, errors='coerce')
            # Drop NaT values that couldn't be converted
            df = df[df.index.notnull()]
            if len(df) == 0:
                raise ValueError("All index values were invalid dates")
            return df
        except Exception as e:
            # If index conversion failed, look for a Date column
            if 'Date' in df.columns:
                try:
                    df = df.copy()
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # Drop NaT values
                    df = df.dropna(subset=['Date'])
                    if len(df) == 0:
                        raise ValueError("All Date values were invalid")
                    df = df.set_index('Date')
                    return df
                except Exception as inner_e:
                    raise ValueError(f"Failed to convert Date column to DatetimeIndex: {str(inner_e)}")
            else:
                raise ValueError("DataFrame has no DatetimeIndex and no Date column")

    def _get_profiles_data(self, constrained: bool = True, 
                          profile_stats: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        pmf = self.pmf
        profiles = pmf.dfprofiles_c if constrained else pmf.dfprofiles_b
        
        if profiles is None:
            raise ValueError(f"{'Constrained' if constrained else 'Base'} profiles not available")
        
        stats = None
        if profile_stats:
            try:
                dfBS = pmf.dfBS_profile_c if constrained else pmf.dfBS_profile_b
                if dfBS is not None:
                    stats = pd.DataFrame(index=profiles.index, 
                                       columns=pd.MultiIndex.from_product(
                                           [profiles.columns, ["P05", "P25", "median", "P75", "P95"]]
                                       ))
                    
                    for prf in profiles.columns:
                        if isinstance(dfBS.index, pd.MultiIndex):
                            for sp in profiles.index:
                                try:
                                    boots = dfBS.xs((sp, prf), level=("Specie", "Profile"), axis=0, drop_level=False)
                                    if not boots.empty:
                                        stats.loc[sp, (prf, "P05")] = boots.quantile(0.05, axis=1).iloc[0]
                                        stats.loc[sp, (prf, "P25")] = boots.quantile(0.25, axis=1).iloc[0]
                                        stats.loc[sp, (prf, "median")] = boots.quantile(0.5, axis=1).iloc[0]
                                        stats.loc[sp, (prf, "P75")] = boots.quantile(0.75, axis=1).iloc[0]
                                        stats.loc[sp, (prf, "P95")] = boots.quantile(0.95, axis=1).iloc[0]
                                except (KeyError, ValueError):
                                    pass
            except Exception as e:
                stats = None
                
        return profiles, stats
    
    def _save_figure(self, fig: plt.Figure, filename: str, close: bool = True, **kwargs) -> None:
        kwargs.setdefault('bbox_inches', 'tight')
        kwargs.setdefault('dpi', 300)
        kwargs.setdefault('facecolor', 'white')
        
        if not any(filename.endswith(ext) for ext in ['.png', '.pdf', '.jpg', '.svg']):
            filename += '.png'
            
        fig.savefig(os.path.join(self.savedir, filename), **kwargs)
        
        if close:
            plt.close(fig)
    
    def _get_colors_for_profiles(self, profiles: List[str]) -> Dict[str, str]:
        colors = {}
        for p in profiles:
            categories = get_sourcesCategories([p])
            color = get_sourceColor(categories[0] if categories else p)
            colors[p] = color
        return colors

    def plot_factor_profiles(self, constrained: bool = True, 
                           profiles: Optional[List[str]] = None,
                           species: Optional[List[str]] = None,
                           normalize: bool = False,
                           bootstrap: bool = False,
                           plot_total: bool = False,
                           horizontal: bool = False,
                           figsize: Optional[Tuple[float, float]] = None,
                           title: Optional[str] = None,
                           log_scale: bool = True,
                           filename: Optional[str] = None,
                           colors: Optional[Dict[str, str]] = None,
                           ncols: int = 1) -> plt.Figure:
        """Plot factor profiles with uncertainty ranges from bootstrap if available."""
        pmf = self.pmf
        df, stats = self._get_profiles_data(constrained=constrained, profile_stats=bootstrap)
        
        profiles_to_plot = profiles or pmf.profiles
        profiles_to_plot = [p for p in profiles_to_plot if p in df.columns]
        
        if not profiles_to_plot:
            raise ValueError("No valid profiles to plot")
            
        if species:
            df = df.loc[df.index.isin(species)]
        
        if not plot_total and pmf.totalVar and pmf.totalVar in df.index:
            df = df.drop(pmf.totalVar)
            
        if normalize and pmf.totalVar and pmf.totalVar in df.index:
            df = df.div(df.loc[pmf.totalVar], axis=1)
            if pmf.totalVar in df.index:
                df = df.drop(pmf.totalVar)
                
        nplots = len(profiles_to_plot)
        nrows = int(np.ceil(nplots / ncols))
        
        if figsize is None:
            if horizontal:
                figsize = (8 * ncols, max(3, 0.25 * len(df.index) + 1) * nrows)
            else:
                figsize = (8 * ncols, 5 * nrows)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, 
                                constrained_layout=True, squeeze=False)
        axes = axes.flatten()
        
        if colors is None:
            colors = self._get_colors_for_profiles(profiles_to_plot)
            
        for i, profile in enumerate(profiles_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            color = colors.get(profile, 'tab:blue')
            profile_data = df[profile]#.sort_values(ascending=False)
            
            if horizontal:
                ax.barh(profile_data.index, profile_data, color=color, alpha=0.7)
                
                if bootstrap and stats is not None:
                    try:
                        err_low = profile_data - stats.loc[profile_data.index, (profile, 'P05')]
                        err_high = stats.loc[profile_data.index, (profile, 'P95')] - profile_data
                        
                        ax.errorbar(profile_data, profile_data.index,
                                   xerr=np.vstack([err_low, err_high]),
                                   fmt='none', ecolor='black', capsize=3)
                    except:
                        pass
                    
                ax.set_xscale('log' if log_scale else 'linear')
                ax.set_xlabel('Concentration')
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                
            else:
                ax.bar(profile_data.index, profile_data, color=color, alpha=0.7)
                
                if bootstrap and stats is not None:
                    try:
                        err_low = profile_data - stats.loc[profile_data.index, (profile, 'P05')]
                        err_high = stats.loc[profile_data.index, (profile, 'P95')] - profile_data
                        
                        ax.errorbar(profile_data.index, profile_data,
                                   yerr=np.vstack([err_low, err_high]),
                                   fmt='none', ecolor='black', capsize=3)
                    except:
                        pass
                    
                ax.set_yscale('log' if log_scale else 'linear')
                ax.set_ylabel('Concentration')
                ax.tick_params(axis='x', rotation=90)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
            ax.set_title(profile)
            
        for i in range(len(profiles_to_plot), len(axes)):
            axes[i].axis('off')
            
        if title:
            fig.suptitle(title, fontsize=14)
            
        if filename:
            self._save_figure(fig, filename)
            
        return fig

    def plot_profile_comparison(self, profile1: str, profile2: str,
                              constrained: bool = True,
                              species: Optional[List[str]] = None,
                              normalize: bool = False,
                              colors: Optional[List[str]] = None,
                              figsize: Tuple[float, float] = (10, 6),
                              title: Optional[str] = None,
                              correlation: bool = True,
                              log_scale: bool = False,
                              filename: Optional[str] = None) -> plt.Figure:
        """Compare two factor profiles with correlation analysis."""
        pmf = self.pmf
        df, _ = self._get_profiles_data(constrained=constrained)
        
        if profile1 not in df.columns or profile2 not in df.columns:
            raise ValueError(f"One or both profiles not found: {profile1}, {profile2}")
            
        if species:
            df = df.loc[df.index.isin(species)]
            
        if normalize and pmf.totalVar and pmf.totalVar in df.index:
            df = pmf.to_relative_mass(constrained)
            
        if correlation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            if not colors:
                colors = ['#1f77b4', '#ff7f0e']
                
            p1data = df[profile1].sort_values(ascending=False)
            p2data = df[profile2].sort_values(ascending=False)
            
            ax1.barh(p1data.index, p1data, color=colors[0], alpha=0.7, label=profile1)
            ax1.barh(p2data.index, p2data, color=colors[1], alpha=0.3, label=profile2)
            
            if log_scale:
                ax1.set_xscale('log')
                
            ax1.set_xlabel('Concentration')
            ax1.legend(loc='upper right')
            ax1.grid(axis='x', linestyle='--', alpha=0.3)
            
            common_species = df.index[df[profile1].notnull() & df[profile2].notnull()]
            x = df.loc[common_species, profile1]
            y = df.loc[common_species, profile2]
            
            slope, intercept, r_value, p_value, stderr = linregress(x, y)
            r_squared = r_value**2
            
            ax2.scatter(x, y, alpha=0.8)
            
            x_pred = np.linspace(x.min(), x.max(), 100)
            y_pred = intercept + slope * x_pred
            ax2.plot(x_pred, y_pred, 'r-', alpha=0.7)
            
            ax2.text(0.05, 0.95, f'$R^2 = {r_squared:.3f}$\n$y = {slope:.3f}x {intercept:+.3f}$',
                    transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            if log_scale:
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                
            ax2.set_xlabel(profile1)
            ax2.set_ylabel(profile2)
            ax2.grid(True, alpha=0.3)
            
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            if not colors:
                colors = ['#1f77b4', '#ff7f0e']
                
            p1data = df[profile1].sort_values(ascending=False)
            p2data = df[profile2].reindex(p1data.index)
            
            x = np.arange(len(p1data))
            width = 0.35
            
            ax.bar(x - width/2, p1data, width, color=colors[0], label=profile1)
            ax.bar(x + width/2, p2data, width, color=colors[1], label=profile2)
            
            ax.set_xticks(x)
            ax.set_xticklabels(p1data.index, rotation=90)
            if log_scale:
                ax.set_yscale('log')
                
            ax.set_ylabel('Concentration')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
        if title:
            fig.suptitle(title, fontsize=14)
            
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig
    def plot_station_averages(self, constrained: bool = True,
                          specie: Optional[str] = None,
                          normalize: bool = False,
                          profiles: Optional[List[str]] = None,
                          stacked: bool = True,
                          figsize: Tuple[float, float] = (8, 4),
                          dpi: int = 200,
                          title: Optional[str] = None,
                          filename: Optional[str] = None) -> plt.Figure:
        """
        Plot average factor contributions by station for multi-site data.
        
        This method creates a stacked bar plot showing the average contributions
        of each factor across different stations in a multi-site analysis.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained profiles.
        specie : str, optional
            Species to analyze. If None, uses totalVar.
        normalize : bool, default=False
            Whether to normalize contributions to percentage.
        profiles : List[str], optional
            Specific profiles to include. If None, includes all.
        stacked : bool, default=True
            Whether to create a stacked bar plot.
        figsize : Tuple[float, float], default=(8, 4)
            Figure size (width, height) in inches.
        dpi : int, default=200
            Resolution for the figure.
        title : str, optional
            Figure title.
        filename : str, optional
            File path to save the figure.
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        pmf = self.pmf
        
        # Check if we're using multi-site data
        if not hasattr(pmf.read, 'multisites') or not pmf.read.multisites:
            raise ValueError("This method is only for multi-site data.")
        
        # Get profiles and contributions data
        if constrained:
            if pmf.dfprofiles_c is None or pmf.dfcontrib_c is None:
                raise ValueError("Constrained profiles or contributions not available")
            dfprofiles = pmf.dfprofiles_c
            dfcontrib = pmf.dfcontrib_c
        else:
            if pmf.dfprofiles_b is None or pmf.dfcontrib_b is None:
                raise ValueError("Base profiles or contributions not available")
            dfprofiles = pmf.dfprofiles_b
            dfcontrib = pmf.dfcontrib_b
        
        # Get species to analyze
        specie = specie or pmf.totalVar
        if specie is None:
            raise ValueError("No species or total variable specified")
            
        if specie not in dfprofiles.index:
            raise ValueError(f"Species '{specie}' not found in profiles")
        
        # Reset index to ensure Station and Date are columns
        dfcontrib_reset = dfcontrib.reset_index()
        if "Station" not in dfcontrib_reset.columns:
            raise ValueError("Multi-site data structure is invalid. 'Station' column is missing.")
        
        # Set up multi-index with Station and Date
        try:
            dfcontrib_idx = dfcontrib_reset.set_index(["Station", "Date"])
        except KeyError:
            # Try to handle case where Date might be the index already
            if "Date" not in dfcontrib_reset.columns and isinstance(dfcontrib_reset.index, pd.DatetimeIndex):
                dfcontrib_reset["Date"] = dfcontrib_reset.index
                dfcontrib_idx = dfcontrib_reset.set_index(["Station", "Date"])
            else:
                raise ValueError("Required columns not found in multi-site data.")
        
        # Calculate absolute contributions (μg/m³)
        dfcontrib_absolu = (dfprofiles.loc[specie] * dfcontrib_idx).sort_index(axis=1)
        
        # Filter profiles if needed
        if profiles:
            available_profiles = [p for p in profiles if p in dfcontrib_absolu.columns]
            if not available_profiles:
                raise ValueError("No specified profiles found in data")
            dfcontrib_absolu = dfcontrib_absolu[available_profiles]
        
        # Calculate average by station
        dfcontrib_absolu_reset = dfcontrib_absolu.reset_index()
        dfcontrib_Year = dfcontrib_absolu_reset.groupby("Station").mean()
        
        # Replace numeric station names with 'Average' if needed (common in some datasets)
        if any(isinstance(idx, (int, float)) for idx in dfcontrib_Year.index):
            dfcontrib_Year = dfcontrib_Year.rename(index={0: 'Average'})
        
        # Drop rows with all NaN values
        dfcontrib_Year = dfcontrib_Year.dropna(how="all")
        
        # Create normalized version if requested
        if normalize:
            dfcontrib_Year_plot = dfcontrib_Year.copy()
            for station in dfcontrib_Year.index:
                station_sum = dfcontrib_Year.loc[station].sum()
                if station_sum > 0:  # Avoid division by zero
                    dfcontrib_Year_plot.loc[station] = dfcontrib_Year.loc[station] / station_sum * 100
        else:
            dfcontrib_Year_plot = dfcontrib_Year
        
        # Get factor list for plotting
        Factor = dfcontrib_Year_plot.columns
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get colors for factors
        colors = {p: get_sourceColor(p) for p in Factor}
        
        # Create stacked bar plot
        if stacked:
            dfcontrib_Year_plot.plot(
                kind="bar", 
                stacked=True,
                ax=ax,
                color=[colors.get(p, None) for p in Factor]
            )
        else:
            dfcontrib_Year_plot.plot(
                kind="bar", 
                ax=ax,
                color=[colors.get(p, None) for p in Factor]
            )
        
        # Add separator line if there are multiple stations
        if len(dfcontrib_Year_plot) > 1:
            plt.axvline(x=0.5, color='grey', ls='--', lw=1)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1., 1.0))
        
        # Set axis grid
        ax.xaxis.grid(alpha=0.2, color='gray', linestyle='dashed')
        ax.set_axisbelow(True)
        ax.yaxis.grid(alpha=0.2)
        
        # Set y-axis label
        if normalize:
            plt.ylabel(f"Contribution to {specie} (%)")
        else:
            plt.ylabel(f"Contribution to {specie} (µg m$^{{-3}}$)")
        
        # Hide the right and top spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f"Average contributions by station{' (normalized)' if normalize else ''}")
        
        plt.tight_layout()
        
        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)
        
        return fig, dfcontrib_Year
        
    def plot_species_contributions(self, species: str, profiles_to_plot: Optional[List[str]] = None,
                             threshold: float = 0.0, figsize: Tuple[int, int] = (10, 6),
                             ax: Optional[plt.Axes] = None, 
                             filename: Optional[str] = None,
                             constrained: bool = True,
                             use_colors: bool = True,
                             **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the contribution of different factors to a specific species.
        
        Parameters
        ----------
        species : str
            Species to plot
        profiles_to_plot : list of str, optional
            List of profiles to include (default: all profiles)
        threshold : float, default=0.0
            Minimum contribution threshold to include a profile (in %)
        figsize : tuple, default=(10, 6)
            Figure size
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        filename : str, optional
            Path to save the figure
        constrained : bool, default=True
            Use constrained or base run
        use_colors : bool, default=True
            Use predefined source colors
        **kwargs : dict
            Additional keyword arguments passed to matplotlib.pyplot.bar
            
        Returns
        -------
        tuple
            Figure and axes objects
        """
        pmf = self.pmf
        if not pmf.ensure_data_loaded():
            raise ValueError("Required data not loaded.")
            
        df = pmf.get_total_species_sum(constrained=constrained)
        
        if species not in df.index:
            available_species = df.index.tolist()
            raise ValueError(
                f"Species '{species}' not found. Available species: "
                f"{', '.join(available_species[:10])}" + 
                ("..." if len(available_species) > 10 else "")
            )
        
        # Get profiles to plot with default handling
        if profiles_to_plot is None:
            profiles_to_plot = pmf.profiles
            
        # Sort profiles alphabetically
        profiles_to_plot = sorted(profiles_to_plot)
        
        # Filter profiles by contribution threshold
        if threshold > 0:
            profiles_to_plot = [p for p in profiles_to_plot 
                            if p in df.columns and df.loc[species, p] >= threshold]
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Extract data for the selected species and profiles
        species_data = df.loc[species, profiles_to_plot]
        
        # Get source colors if requested
        if use_colors:
            from .utils import get_sourceColor
            colors = [get_sourceColor(p) for p in profiles_to_plot]
        else:
            colors = None
            
        # Plot as bar chart
        bars = ax.bar(range(len(profiles_to_plot)), species_data, color=colors, **kwargs)
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', rotation=0,
                fontsize=9)
        
        # Set labels and title
        ax.set_ylabel(f"% of {species}", fontsize=12)
        ax.set_title(f"Contribution to {pretty_specie(species)}", fontsize=14)
        ax.set_xticks(range(len(profiles_to_plot)))
        ax.set_xticklabels(profiles_to_plot, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
            
        return fig, ax
    
    def plot_contributions_timeseries(self, constrained: bool = True,
                                    specie: Optional[str] = None,
                                    profiles: Optional[List[str]] = None,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None,
                                    stacked: bool = True,
                                    colors: Optional[Dict[str, str]] = None,
                                    figsize: Tuple[float, float] = (10, 6),
                                    title: Optional[str] = None,
                                    filename: Optional[str] = None) -> plt.Figure:
        """Plot factor contributions over time."""
        pmf = self.pmf
        
        try:
            df_contrib = pmf.to_cubic_meter(
                specie=specie,
                constrained=constrained,
                profiles=profiles
            )
        except Exception as e:
            raise ValueError(f"Error retrieving contribution data: {str(e)}")
            
        if df_contrib is None or df_contrib.empty:
            raise ValueError("No contribution data available")
            
        if start_date or end_date:
            if start_date:
                df_contrib = df_contrib[df_contrib.index >= start_date]
            if end_date:
                df_contrib = df_contrib[df_contrib.index <= end_date]
                
        profiles_to_plot = profiles or df_contrib.columns.tolist()
        
        # Get source colors
        from .utils import get_sourceColor
        colors = [get_sourceColor(p) for p in profiles_to_plot]
            
        fig, ax = plt.subplots(figsize=figsize)
        
        if stacked:
            ax.stackplot(
                df_contrib.index, 
                [df_contrib[p] for p in profiles_to_plot],
                labels=profiles_to_plot,
                colors=[get_sourceColor(p) for p in profiles_to_plot],
                alpha=0.7
            )
        else:
            for p in profiles_to_plot:
                ax.plot(
                    df_contrib.index, 
                    df_contrib[p],
                    label=p,
                    color=colors.get(p, None),
                    alpha=0.7
                )
                
        format_xaxis_timeseries(ax)
        
        specie_label = specie or pmf.totalVar or "PM"
        ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
        
        if len(profiles_to_plot) > 10:
            ax.legend(
                loc='upper left', 
                bbox_to_anchor=(0.5, 0), 
                ncol=5 if len(profiles_to_plot) > 10 else 1
            )
        else:
            ax.legend()
            
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig
    
    def plot_seasonal_contributions(self, specie: Optional[str] = None, 
                              annual: bool = True, normalize: bool = True, 
                              constrained: bool = True, figsize: Tuple[int, int] = (10, 6),
                              stacked: bool = False, ax: Optional[plt.Axes] = None,
                              profiles_to_plot: Optional[List[str]] = None,
                              filename: Optional[str] = None,
                              **kwargs) -> plt.Figure:
        """
        Plot seasonal contributions as bar chart.
        
        Parameters
        ----------
        specie : str, optional
            Species to analyze, defaults to totalVar
        annual : bool, default=True
            Whether to include annual average
        normalize : bool, default=True
            Whether to normalize to 100%
        constrained : bool, default=True
            Whether to use constrained or base run
        figsize : tuple, default=(10, 6)
            Figure size
        stacked : bool, default=False
            Whether to show stacked bars or grouped bars
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        profiles_to_plot : list of str, optional
            Specific profiles to include (default: all profiles)
        filename : str, optional
            Path to save the figure
        **kwargs : dict
            Additional keyword arguments passed to matplotlib.pyplot.bar
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        pmf = self.pmf
        if not pmf.ensure_data_loaded():
            raise ValueError("Required data not loaded.")
            
        # Calculate seasonal contribution
        try:
            df = pmf.get_seasonal_contribution(
                specie=specie, 
                annual=annual, 
                normalize=normalize, 
                constrained=constrained
            )
        except Exception as e:
            print(f"Error calculating seasonal contribution: {str(e)}")
            raise
            
        # Filter profiles if specified
        if profiles_to_plot is not None:
            # Sort profiles alphabetically
            profiles_to_plot = sorted(profiles_to_plot)
            df = df[profiles_to_plot]
        else:
            # Sort all profiles alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Get source colors
        from .utils import get_sourceColor
        colors = [get_sourceColor(p) for p in df.columns]
        
        # Plot the data
        if stacked:
            # Stacked bar chart
            df.plot(kind='bar', stacked=True, ax=ax, color=colors, **kwargs)
            
            # Add percentage labels on stacked bars
            bottom_values = np.zeros(len(df))
            for i, col in enumerate(df.columns):
                for j, val in enumerate(df[col]):
                    if val > 0.03:  # Only show labels for bars that are big enough
                        ax.text(j, bottom_values[j] + val/2, f'{val:.1%}' if normalize else f'{val:.1f}',
                            ha='center', va='center', fontsize=8)
                    bottom_values[j] += val
        else:
            # Grouped bar chart
            df.plot(kind='bar', ax=ax, color=colors, **kwargs)
            
            # Add percentage labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%' if normalize else '%.1f', 
                            fontsize=8, padding=3)
        
        # Set labels and title
        ax.set_ylabel('Contribution' + (' (%)' if normalize else ''))
        ax.set_title(f'Seasonal contribution' + 
                    (f' of {specie}' if specie else '') +
                    (' (normalized)' if normalize else ''))
        
        # Adjust legend
        ax.legend(title='Sources', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_contributions_pie(self, constrained: bool = True,
                             specie: Optional[str] = None,
                             profiles: Optional[List[str]] = None,
                             explode: Union[float, List[float]] = 0.0,
                             colors: Optional[Dict[str, str]] = None,
                             figsize: Tuple[float, float] = (8, 8),
                             title: Optional[str] = None,
                             filename: Optional[str] = None) -> plt.Figure:
        """Create a pie chart of factor contributions."""
        pmf = self.pmf
        
        try:
            df_contrib = pmf.to_cubic_meter(
                specie=specie,
                constrained=constrained,
                profiles=profiles
            )
        except Exception as e:
            raise ValueError(f"Error retrieving contribution data: {str(e)}")
            
        if df_contrib is None or df_contrib.empty:
            raise ValueError("No contribution data available")
            
        contrib_avg = df_contrib.mean()
        contrib_avg = contrib_avg.sort_values(ascending=False)
        contrib_avg = contrib_avg[contrib_avg > 0]
        
        # Sort profiles alphabetically
        contrib_avg = contrib_avg.sort_index(axis = 0)
        
        if len(contrib_avg) == 0:
            raise ValueError("No positive contributions found")
            
        if colors is None:
            colors = self._get_colors_for_profiles(contrib_avg.index)
            
        if isinstance(explode, (int, float)):
            explode = [explode] * len(contrib_avg)
            
        if len(explode) != len(contrib_avg):
            explode = explode[:len(contrib_avg)] if len(explode) > len(contrib_avg) else explode + [0.0] * (len(contrib_avg) - len(explode))
            
        fig, ax = plt.subplots(figsize=figsize)
        
        pie_colors = [colors.get(p, 'tab:blue') for p in contrib_avg.index]
        wedges, texts = ax.pie(
            contrib_avg, 
            labels=None, 
            explode=explode, 
            colors=pie_colors,
            startangle=90,
            shadow=False
        )
        
        total = contrib_avg.sum()
        labels = [f"{p} ({100*v/total:.1f}%)" for p, v in contrib_avg.items()]
        ax.legend(
            wedges, 
            labels,
            loc='center left',
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        specie_label = specie or pmf.totalVar or "PM"
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Contributions to {specie_label}', fontsize=14)
            
        center_text = f"Total:\n{total:.2f} µg m$^{-3}$"
        ax.text(0, 0, center_text, ha='center', va='top', fontsize=12)
        ax.axis('equal')
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig
    
    def plot_weekly_pattern(self, constrained: bool = True,
                          specie: Optional[str] = None,
                          profiles: Optional[List[str]] = None,
                          normalize: bool = False,
                          colors: Optional[Dict[str, str]] = None,
                          figsize: Tuple[float, float] = (10, 6),
                          title: Optional[str] = None,
                          filename: Optional[str] = None) -> plt.Figure:
        """Plot weekly pattern of factor contributions."""
        pmf = self.pmf
        
        try:
            df_contrib = pmf.to_cubic_meter(
                specie=specie,
                constrained=constrained,
                profiles=profiles
            )
        except Exception as e:
            raise ValueError(f"Error retrieving contribution data: {str(e)}")
            
        if df_contrib is None or df_contrib.empty:
            raise ValueError("No contribution data available")
            
        if profiles:
            valid_profiles = [p for p in profiles if p in df_contrib.columns]
            if not valid_profiles:
                raise ValueError(f"None of the specified profiles found: {profiles}")
            df_contrib = df_contrib[valid_profiles]
            
        df_contrib = df_contrib.copy()
        df_contrib['weekday'] = df_contrib.index.dayofweek
        df_contrib['weekday_name'] = df_contrib.index.day_name()
        
        if colors is None:
            colors = self._get_colors_for_profiles(df_contrib.columns[:-2])
            
        fig, ax = plt.subplots(figsize=figsize)
        
        weekly_pattern = df_contrib.groupby('weekday').mean()
        weekly_pattern = weekly_pattern.drop(columns=['weekday'])
        
        if normalize:
            for col in weekly_pattern.columns:
                if col != 'weekday_name':
                    weekly_pattern[col] = weekly_pattern[col] / weekly_pattern[col].mean()
            
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(range(7))
        
        x = np.arange(len(day_names))
        for i, profile in enumerate(weekly_pattern.columns):
            if profile == 'weekday_name':
                continue
                
            ax.plot(
                x,
                weekly_pattern[profile],
                '-o',
                label=profile,
                color=colors.get(profile, None),
                alpha=0.8,
                lw=2
            )
            
        ax.set_xticks(x)
        ax.set_xticklabels(day_names)
        ax.tick_params(axis='x', rotation=45)
        
        specie_label = specie or pmf.totalVar or "PM"
        if normalize:
            ax.set_ylabel('Normalized contribution\n(ratio to average)')
        else:
            ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
            
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best')
        
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig

    def plot_diurnal_pattern(self, constrained: bool = True,
                           specie: Optional[str] = None,
                           profiles: Optional[List[str]] = None,
                           by_season: bool = False,
                           normalize: bool = False,
                           colors: Optional[Dict[str, str]] = None,
                           figsize: Optional[Tuple[float, float]] = None,
                           title: Optional[str] = None,
                           filename: Optional[str] = None) -> plt.Figure:
        """Plot diurnal (hourly) pattern of factor contributions."""
        pmf = self.pmf
        
        try:
            df_contrib = pmf.to_cubic_meter(
                specie=specie,
                constrained=constrained,
                profiles=profiles
            )
        except Exception as e:
            raise ValueError(f"Error retrieving contribution data: {str(e)}")
            
        if df_contrib is None or df_contrib.empty:
            raise ValueError("No contribution data available")
            
        if profiles:
            valid_profiles = [p for p in profiles if p in df_contrib.columns]
            if not valid_profiles:
                raise ValueError(f"None of the specified profiles found: {profiles}")
            df_contrib = df_contrib[valid_profiles]
            
        if not hasattr(df_contrib.index, 'hour') or len(df_contrib) < 24:
            raise ValueError("Hourly data not available or insufficient for diurnal pattern")
            
        df_contrib = df_contrib.copy()
        df_contrib['hour'] = df_contrib.index.hour
        df_contrib['month'] = df_contrib.index.month
        
        month_to_season = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 
            9: "Fall", 10: "Fall", 11: "Fall"
        }
        df_contrib['season'] = df_contrib['month'].map(month_to_season)
        
        if colors is None:
            colors = self._get_colors_for_profiles(df_contrib.columns[:-3])
            
        if by_season:
            seasons = ["Winter", "Spring", "Summer", "Fall"]
            if len(df_contrib.columns) <= 4:
                nrows, ncols = 2, 2
            else:
                nrows, ncols = 4, 1
        else:
            nrows, ncols = 1, 1
            
        if figsize is None:
            if by_season:
                figsize = (5 * ncols, 3 * nrows)
            else:
                figsize = (10, 6)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, 
                               squeeze=False, sharex=True)
        axes = axes.flatten()
        
        if by_season:
            for i, season in enumerate(seasons):
                ax = axes[i]
                
                season_data = df_contrib[df_contrib['season'] == season]
                hourly_pattern = season_data.groupby('hour').mean()
                hourly_pattern = hourly_pattern.drop(columns=['hour', 'month'])
                
                if normalize:
                    for col in hourly_pattern.columns:
                        if col != 'season':
                            hourly_pattern[col] = hourly_pattern[col] / hourly_pattern[col].mean()
                
                for profile in hourly_pattern.columns:
                    if profile == 'season':
                        continue
                        
                    ax.plot(
                        hourly_pattern.index,
                        hourly_pattern[profile],
                        '-o',
                        label=profile,
                        color=colors.get(profile, None),
                        alpha=0.8,
                        markersize=4
                    )
                    
                ax.set_title(season)
                
                ax.set_xticks(range(0, 24, 3))
                ax.set_xlim(-0.5, 23.5)
                
                specie_label = specie or pmf.totalVar or "PM"
                if normalize:
                    ax.set_ylabel('Normalized contribution')
                else:
                    ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
                    
                ax.grid(True, linestyle='--', alpha=0.3)
                
                if i == 0:
                    ax.legend(loc='best')
                    
        else:
            ax = axes[0]
            
            hourly_pattern = df_contrib.groupby('hour').mean()
            hourly_pattern = hourly_pattern.drop(columns=['hour', 'month', 'season'])
            
            if normalize:
                for col in hourly_pattern.columns:
                    hourly_pattern[col] = hourly_pattern[col] / hourly_pattern[col].mean()
            
            for profile in hourly_pattern.columns:
                ax.plot(
                    hourly_pattern.index,
                    hourly_pattern[profile],
                    '-o',
                    label=profile,
                    color=colors.get(profile, None),
                    alpha=0.8,
                    lw=2
                )
                
            ax.set_xticks(range(0, 24, 3))
            ax.set_xlim(-0.5, 23.5)
            ax.set_xlabel('Hour of day')
            
            specie_label = specie or pmf.totalVar or "PM"
            if normalize:
                ax.set_ylabel('Normalized contribution\n(ratio to average)')
            else:
                ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
                
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if len(hourly_pattern.columns) > 8:
                ax.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=min(5, len(hourly_pattern.columns))
                )
            else:
                ax.legend(loc='best')
            
        if title:
            fig.suptitle(title, fontsize=14)
            
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig
    
    def plot_bootstrap_mapping(self, constrained: bool = True,
                             figsize: Tuple[float, float] = (8, 6),
                             title: Optional[str] = None,
                             filename: Optional[str] = None) -> plt.Figure:
        """Plot bootstrap factor mapping statistics."""
        pmf = self.pmf
        
        if constrained:
            if pmf.dfbootstrap_mapping_c is None:
                raise ValueError("Constrained bootstrap mapping data not available")
            mapping_data = pmf.dfbootstrap_mapping_c
        else:
            if pmf.dfbootstrap_mapping_b is None:
                raise ValueError("Base bootstrap mapping data not available")
            mapping_data = pmf.dfbootstrap_mapping_b
            
        if "unmapped" in mapping_data.columns:
            mapping_data = mapping_data.drop(columns=["unmapped"])
        mapping_data = mapping_data.apply(pd.to_numeric)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            mapping_data,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar_kws={'label': 'Number of bootstrap runs'},
            ax=ax
        )
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Bootstrap Factor Mapping', fontsize=14)
            
        ax.set_xlabel('Base Run Factors')
        ax.set_ylabel('Bootstrap Factors')
        
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig

            
    def plot_contribution_uncertainty_bootstrap(self, factor: str, 
                                             constrained: bool = True,
                                             percentiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
                                             figsize: Tuple[float, float] = (12, 6),
                                             title: Optional[str] = None,
                                             filename: Optional[str] = None) -> plt.Figure:
        """Plot time series with bootstrap uncertainty intervals."""
        pmf = self.pmf
        
        # Get bootstrap results
        dfBS = pmf.dfBS_profile_c if constrained else pmf.dfBS_profile_b
        if dfBS is None:
            raise ValueError(f"Bootstrap data not available")
            
        # Get contributions for the selected factor
        try:
            contrib = pmf.to_cubic_meter(profiles=[factor], constrained=constrained)
        except Exception as e:
            raise ValueError(f"Error retrieving contribution data: {str(e)}")
            
        if contrib is None or contrib.empty:
            raise ValueError("No contribution data available")
            
        if factor not in contrib.columns:
            raise ValueError(f"Factor '{factor}' not found in contribution data")
            
        # Extract the bootstrap data for this factor
        try:
            if isinstance(dfBS.index, pd.MultiIndex):
                profile_exists = factor in dfBS.index.get_level_values("Profile").unique()
                if not profile_exists:
                    raise ValueError(f"Factor '{factor}' not found in bootstrap data")
                    
                bs_data = dfBS.xs(factor, level="Profile")
                
                # Create percentile curves for each date
                bs_perc = []
                for p in percentiles:
                    bs_perc.append(bs_data.quantile(p, axis=1))
                    
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Plot the original time series
                ax.plot(contrib.index, contrib[factor], 'k-', lw=2, label=f'{factor} (Base Run)')
                
                # Plot uncertainty intervals
                ax.fill_between(
                    contrib.index,
                    bs_perc[0],  # 5th percentile
                    bs_perc[-1],  # 95th percentile
                    alpha=0.3,
                    color='skyblue', 
                    label=f'90% Confidence Interval'
                )
                
                # Format time axis
                format_xaxis_timeseries(ax)
                
                # Add labels and legend
                specie_label = pmf.totalVar or "PM"
                ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.3)
                
                if title:
                    ax.set_title(title, fontsize=14)
                else:
                    ax.set_title(f'Time Series with Bootstrap Uncertainty for {factor}', fontsize=14)
                    
                plt.tight_layout()
                
                if filename:
                    self._save_figure(fig, filename)
                    
                return fig
                
            else:
                raise ValueError("Bootstrap data structure not recognized")
            
        except Exception as e:
            raise ValueError(f"Error processing bootstrap data: {str(e)}")


    def plot_per_microgram(self, df: Optional[pd.DataFrame] = None, 
                        constrained: bool = True, 
                        profiles: Optional[List[str]] = None, 
                        species: Optional[List[str]] = None,
                        figsize: Optional[Tuple[float, float]] = None,
                        title: Optional[str] = None,
                        filename: Optional[str] = None) -> plt.Figure:
        """
        Plot factor profiles normalized by total mass (μg/μg).
        
        This visualizes the mass fraction of each species in each factor,
        showing the chemical fingerprint independent of total mass.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            Bootstrap profile data. If None, uses internal profile data.
        constrained : bool, default=True
            Whether to use constrained profiles.
        profiles : List[str], optional
            Specific profiles to plot. If None, plots all.
        species : List[str], optional
            Specific species to include. If None, includes all.
        figsize : Tuple[float, float], optional
            Figure size (width, height) in inches.
        title : str, optional
            Figure title.
        filename : str, optional
            File path to save the figure.
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        pmf = self.pmf
        
        # Get appropriate profile data
        if df is None:
            # Use standard profiles
            dfprofiles = pmf.dfprofiles_c if constrained else pmf.dfprofiles_b
            if dfprofiles is None:
                raise ValueError(f"No {'constrained' if constrained else 'base'} profiles available")
                
            # Check if totalVar is set and exists
            if not pmf.totalVar or pmf.totalVar not in dfprofiles.index:
                available = dfprofiles.index.tolist()
                raise ValueError(f"Total variable '{pmf.totalVar}' not found in profile data. Available: {available}")
                
            # Get default profiles if not specified
            if profiles is None:
                profiles = pmf.profiles or dfprofiles.columns.tolist()
                
            # Create normalized data
            normalized_data = {}
            for profile in profiles:
                if profile not in dfprofiles.columns:
                    print(f"Warning: Profile '{profile}' not found in profile data, skipping")
                    continue
                    
                # Normalize by totalVar
                denominator = dfprofiles.loc[pmf.totalVar, profile]
                if np.isclose(denominator, 0):
                    print(f"Warning: Total variable is zero for profile '{profile}', skipping")
                    continue
                    
                normalized_profile = dfprofiles[profile] / denominator
                normalized_data[profile] = normalized_profile
                
        else:
            # Using bootstrap data - ensure it has MultiIndex with Specie and Profile levels
            if not isinstance(df.index, pd.MultiIndex) or not all(name in df.index.names for name in ['Specie', 'Profile']):
                raise ValueError("Bootstrap data must have a MultiIndex with levels 'Specie' and 'Profile'")
                
            # Check if totalVar exists in the data
            if pmf.totalVar not in df.index.get_level_values('Specie'):
                raise ValueError(f"Total variable '{pmf.totalVar}' not found in bootstrap data")
                
            # Get default profiles if not specified
            if profiles is None:
                profiles = list(df.index.get_level_values('Profile').unique())
                
            # Create normalized data
            normalized_data = {}
            for profile in profiles:
                try:
                    # Get the totalVar value for this profile
                    totalvar_value = df.xs((pmf.totalVar, profile), level=('Specie', 'Profile'))
                    
                    if np.all(np.isclose(totalvar_value, 0)):
                        print(f"Warning: Total variable is zero for profile '{profile}' in bootstrap data, skipping")
                        continue
                        
                    # Get all data for this profile and normalize
                    profile_data = df.xs(profile, level='Profile')
                    normalized = profile_data.div(totalvar_value.values, axis=1)
                    normalized_data[profile] = normalized
                    
                except KeyError:
                    print(f"Warning: Profile '{profile}' not found in bootstrap data, skipping")
        
        # Filter species if specified
        if species:
            for profile in normalized_data:
                if isinstance(normalized_data[profile], pd.DataFrame):
                    normalized_data[profile] = normalized_data[profile].loc[normalized_data[profile].index.isin(species)]
                else:
                    normalized_data[profile] = normalized_data[profile][normalized_data[profile].index.isin(species)]
        
        if not normalized_data:
            raise ValueError("No valid profiles available for plotting")
        
        # Set figure size
        if figsize is None:
            figsize = (12, len(normalized_data) * 4)
            
        # Create figure with subplots for each profile
        fig, axes = plt.subplots(len(normalized_data), 1, figsize=figsize, sharex=True
                                 )
        if len(normalized_data) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # Plot each profile
        for ax, (profile, data) in zip(axes, normalized_data.items()):
            if isinstance(data, pd.DataFrame):  # Bootstrap data
                # Create boxplot
                plot_data = data.T.melt(var_name='Specie', ignore_index=False)
                sns.boxplot(data=plot_data, x='Specie', y='value', color='grey', ax=ax)
                
                # Add reference line if we have base/constrained profiles
                ref_profiles = pmf.dfprofiles_c if constrained else pmf.dfprofiles_b
                if ref_profiles is not None and profile in ref_profiles:
                    ref_data = ref_profiles[profile] / ref_profiles.loc[pmf.totalVar, profile]
                    ref_data = ref_data.reindex(data.index).reset_index()
                    sns.stripplot(data=ref_data, x='index', y=profile, ax=ax, 
                                jitter=False, color='red', size=8)
            else:  # Regular profile data
                # Simple bar chart for non-bootstrap data
                data_df = data.reset_index()
                data_df.columns = ['Specie', 'Value']
                sns.barplot(data=data_df, x='Specie', y='Value', ax=ax, color=get_sourceColor(profile))
            
            # Format the axis
            ax.set_yscale('log')
            ax.set_ylim(1e-6, 3)
            ax.set_ylabel('μg/μg')
            ax.set_title(profile)
            
            # Format x-axis labels
            ax.set_xticklabels(
                pretty_specie([t.get_text() for t in ax.get_xticklabels()]),
                rotation=90
            )
        
        # Add legend if bootstrap data
        if isinstance(list(normalized_data.values())[0], pd.DataFrame):
            ref_artist = plt.Line2D((0, 1), (0, 0), color='red', marker='o', linestyle='')
            bs_artist = plt.Rectangle((0, 0), 0, 0, color="grey")
            axes[0].legend([ref_artist, bs_artist], ["Reference run", "Bootstrap"], 
                        loc="upper left", bbox_to_anchor=(1., 1.), frameon=False)
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Profiles Normalized by Total Mass (μg/μg)', fontsize=16)
        
        plt.tight_layout()
        if len(normalized_data) > 1:
            plt.subplots_adjust(hspace=0.4)
        
        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)
        
        return fig

    def plot_stacked_profiles(self, constrained: bool = True,
                             species: Optional[List[str]] = None,
                             profiles: Optional[List[str]] = None,
                             figsize: Optional[Tuple[float, float]] = None,
                             title: Optional[str] = None,
                             filename: Optional[str] = None) -> plt.Figure:
        """
        Plot stacked bar chart of profiles showing species composition.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained profiles.
        species : List[str], optional
            Specific species to include. If None, includes all.
        profiles : List[str], optional
            Specific profiles to include. If None, includes all.
        figsize : Tuple[float, float], optional
            Figure size (width, height) in inches.
        title : str, optional
            Figure title.
        filename : str, optional
            File path to save the figure.
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        pmf = self.pmf
        
        # Get profiles data
        if constrained:
            if pmf.dfprofiles_c is None:
                raise ValueError("Constrained profiles not available")
            df_profiles = pmf.dfprofiles_c.copy()
        else:
            if pmf.dfprofiles_b is None:
                raise ValueError("Base profiles not available")
            df_profiles = pmf.dfprofiles_b.copy()
            
        # Filter profiles if specified
        if profiles:
            available_profiles = [p for p in profiles if p in df_profiles.columns]
            if not available_profiles:
                raise ValueError("No specified profiles found in data")
            df_profiles = df_profiles[available_profiles]
        else:
            available_profiles = df_profiles.columns.tolist()
            
        # Filter species if specified and remove total variable
        if species:
            df_profiles = df_profiles.loc[species]
        elif pmf.totalVar in df_profiles.index:
            df_profiles = df_profiles.drop(pmf.totalVar)
            
        # Set figsize if not provided
        if figsize is None:
            figsize = (12, 10)
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot stacked bar chart
        df_profiles.plot(kind='bar', stacked=True, ax=ax)
        
        # Format plot
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Species')
        plt.xticks(rotation=90)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Stacked profiles: species composition')
            
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
            
        return fig

    def plot_polluted_contribution(self, constrained: bool = True, 
                                  threshold: Optional[float] = None,
                                  specie: Optional[str] = None,
                                  normalize: bool = True,
                                  figsize: Tuple[float, float] = (10, 6),
                                  title: Optional[str] = None,
                                  filename: Optional[str] = None) -> plt.Figure:
        """
        Plot contribution of factors during polluted days.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained results.
        threshold : float, optional
            Pollution threshold. If None, uses 75th percentile.
        specie : str, optional
            Species to analyze. If None, uses totalVar.
        normalize : bool, default=True
            Whether to normalize to percentage.
        figsize : Tuple[float, float], default=(10, 6)
            Figure size (width, height) in inches.
        title : str, optional
            Figure title.
        filename : str, optional
            File path to save the figure.
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        pmf = self.pmf
        
        # Get contribution data
        if constrained:
            if pmf.dfcontrib_c is None:
                raise ValueError("Constrained contributions not available")
            dfcontrib = pmf.dfcontrib_c
            dfprofiles = pmf.dfprofiles_c
        else:
            if pmf.dfcontrib_b is None:
                raise ValueError("Base contributions not available")
            dfcontrib = pmf.dfcontrib_b
            dfprofiles = pmf.dfprofiles_b
            
        # Set species to analyze
        specie = specie or pmf.totalVar
        if specie is None:
            raise ValueError("No species or total variable specified")
            
        if specie not in dfprofiles.index:
            raise ValueError(f"Species '{specie}' not found in profiles")
            
        # Convert to concentration
        df = dfcontrib.copy() * dfprofiles.loc[specie]
        df = df.sort_index(axis = 0)
        # Calculate threshold if not provided
        if threshold is None:
            total = df.sum(axis=1)
            threshold = total.quantile(0.75)
            print(f"Using 75th percentile as threshold: {threshold:.2f}")
            
        # Get polluted days
        total = df.sum(axis=1)
        polluted_days = total[total > threshold].index
        
        if len(polluted_days) == 0:
            raise ValueError("No days exceed the threshold")
            
        # Filter for polluted days
        polluted_contrib = df.loc[polluted_days]
        
        # Calculate mean contribution
        polluted_mean = polluted_contrib.mean()
        
        # Normalize if requested
        if normalize:
            polluted_mean = polluted_mean / polluted_mean.sum() * 100
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar colors
        colors = {p: get_sourceColor(p) for p in polluted_mean.index}
        
        # Plot bar chart
        polluted_mean.plot(kind='bar', ax=ax, color=[colors[p] for p in polluted_mean.index])
        
        # Format plot
        if normalize:
            ax.set_ylabel('Contribution (%)')
            ax.set_ylim(0, max(polluted_mean) * 1.1)
        else:
            ax.set_ylabel(f'{specie} (µg m$^{-3}$)')
            
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Factor contributions during polluted days (>{threshold:.2f})')
            
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
            
        return fig

    def plot_samples_sources_contribution(self, constrained: bool = True, 
                                        specie: Optional[str] = None,
                                        start_date: Optional[str] = None,
                                        end_date: Optional[str] = None,
                                        n_samples: int = 30,
                                        figsize: Optional[Tuple[float, float]] = None,
                                        title: Optional[str] = None,
                                        filename: Optional[str] = None) -> plt.Figure:
        """
        Plot bar chart of factor contributions for each sample across time.

        Parameters
        ----------
        constrained : bool, default=True
            Use constrained or base run
        specie : str, optional
            Species to analyze (uses totalVar if None)
        start_date : str, optional
            Start date filter in format 'YYYY-MM-DD'
        end_date : str, optional
            End date filter in format 'YYYY-MM-DD'
        n_samples : int, default=30
            Maximum number of samples to display
        figsize : tuple of float, optional
            Figure size (width, height) in inches
        title : str, optional
            Figure title
        filename : str, optional
            File path to save the figure

        Returns
        -------
        plt.Figure
            Figure containing the stacked bar plot
        """
        pmf = self.pmf

        try:
            # Get contribution data
            df_contrib = pmf.to_cubic_meter(
                specie=specie,
                constrained=constrained
            )
            
            if df_contrib is None or df_contrib.empty:
                raise ValueError("No contribution data available")
            
            # Filter by date range if specified
            if start_date or end_date:
                if start_date:
                    df_contrib = df_contrib[df_contrib.index >= start_date]
                if end_date:
                    df_contrib = df_contrib[df_contrib.index <= end_date]
            
            # Set figsize
            if figsize is None:
                figsize = (12, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Limit the number of samples if needed
            if len(df_contrib) > n_samples:
                print(f"Limiting display to {n_samples} samples")
                # Take evenly spaced samples
                step = max(1, len(df_contrib) // n_samples)
                indices = list(range(0, len(df_contrib), step))[:n_samples]
                df_contrib = df_contrib.iloc[indices]
            
            # Get colors for factors
            profiles = df_contrib.columns.tolist()
            colors = self._get_colors_for_profiles(profiles)
            
            # Create the stacked bar plot - use positions instead of datetime objects
            positions = np.arange(len(df_contrib))
            bottom = np.zeros(len(df_contrib))
            
            for profile in profiles:
                ax.bar(positions, df_contrib[profile], bottom=bottom, 
                    label=profile, color=colors.get(profile, None), width=0.8)
                bottom += df_contrib[profile].values
            
            # Format x-axis labels - convert dates to strings explicitly
            if isinstance(df_contrib.index, pd.DatetimeIndex):
                date_labels = [d.strftime('%Y-%m-%d') for d in df_contrib.index]
                ax.set_xticks(positions)
                ax.set_xticklabels(date_labels, rotation=45, ha='right')
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels(df_contrib.index, rotation=45, ha='right')
            
            # Set labels
            specie_label = specie or pmf.totalVar or "PM"
            ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
            
            # Set title
            if title:
                ax.set_title(title, fontsize=14)
            else:
                ax.set_title(f"Factor contributions by sample", fontsize=14)
            
            # Add legend out of the main plot
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
            
            # Adjust layout
            fig.subplots_adjust(top=0.90, bottom=0.2, left=0.07, right=0.85)
            
            # Save figure if filename provided
            if filename:
                self._save_figure(fig, filename)
            
            return fig
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error plotting sample contributions: {str(e)}")

            
    def plot_contribution_summary(self, constrained: bool = True,
                                specie: Optional[str] = None,
                                figsize: Tuple[float, float] = (12, 10),
                                title: Optional[str] = None,
                                filename: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive contribution summary with pie chart and time series.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained results.
        specie : str, optional
            Species to analyze. If None, uses totalVar.
        figsize : Tuple[float, float], default=(12, 10)
            Figure size (width, height) in inches.
        title : str, optional
            Figure title.
        filename : str, optional
            File path to save the figure.
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        pmf = self.pmf
        
        # Convert to concentration
        try:
            df_contrib = pmf.to_cubic_meter(constrained=constrained, specie=specie)
        except Exception as e:
            raise ValueError(f"Failed to convert contributions: {str(e)}")
        df_contrib = df_contrib.sort_index(axis = 0)    
        # Create figure with 2 subplots
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1.5])
        
        # Get colors for factors
        colors = {p: get_sourceColor(p) for p in df_contrib.columns}
        
        # Pie chart of average contributions (top left)
        ax_pie = fig.add_subplot(gs[0, 0])
        df_contrib.mean().plot(
            kind='pie', 
            ax=ax_pie, 
            colors=[colors[p] for p in df_contrib.columns],
            #autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        ax_pie.set_title('Average Contribution')
        ax_pie.set_ylabel('')
        
        # Bar chart of average contributions (top right)
        ax_bar = fig.add_subplot(gs[0, 1])
        df_contrib.mean().plot(
            kind='bar',
            ax=ax_bar,
            color=[colors[p] for p in df_contrib.columns]
        )
        profiles_to_plot = df_contrib.columns

        ax_bar.set_title('Average Contribution')
        ax_bar.set_ylabel(f'{specie or pmf.totalVar or "PM"} (µg m$^{-3}$)')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')
        
        # Time series of stacked contributions (bottom)
        ax_ts = fig.add_subplot(gs[1, :])
        ax_ts.stackplot(
                df_contrib.index, 
                [df_contrib[p] for p in profiles_to_plot],
                labels=profiles_to_plot,
                colors=[get_sourceColor(p) for p in profiles_to_plot],
                alpha=0.7
            )
        ax_ts.set_title('Time Series of Contributions')
        ax_ts.set_ylabel(f'{specie or pmf.totalVar or "PM"} (µg m$^{-3}$)')
        ax_ts.grid(linestyle='--', alpha=0.7)
        format_xaxis_timeseries(ax_ts)
        
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Factor Contribution Summary', fontsize=16)
            
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.92)
            
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
            
        return fig

    def plot_similarity_matrix(self, constrained: bool = True,
                             metric: str = 'correlation',
                             figsize: Optional[Tuple[float, float]] = None,
                             title: Optional[str] = None,
                             filename: Optional[str] = None) -> plt.Figure:
        """
        Plot similarity matrix between factors.
        
        Parameters
        ----------
        constrained : bool, default=True
            Whether to use constrained results.
        metric : str, default='correlation'
            Similarity metric to use: 'correlation', 'timeseries', or 'profile'.
        figsize : Tuple[float, float], optional
            Figure size (width, height) in inches.
        title : str, optional
            Figure title.
        filename : str, optional
            File path to save the figure.
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        pmf = self.pmf
        
        # Get profiles and contributions
        if constrained:
            profiles = pmf.dfprofiles_c
            contrib = pmf.dfcontrib_c
        else:
            profiles = pmf.dfprofiles_b
            contrib = pmf.dfcontrib_b
            
        if profiles is None or contrib is None:
            raise ValueError("Profile or contribution data not available")
            
        # Calculate similarity matrix based on metric
        if metric == 'correlation' or metric == 'timeseries':
            # Correlation between time series
            similarity = contrib.corr()
        elif metric == 'profile':
            # Correlation between profiles
            similarity = profiles.T.corr()
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
            
        # Set figsize if not provided
        if figsize is None:
            n_profiles = len(profiles.columns)
            figsize = (n_profiles * 0.8 + 2, n_profiles * 0.8 + 1)
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        cmap = plt.cm.RdBu_r
        mask = np.triu(np.ones_like(similarity, dtype=bool), k=1)
        
        sns.heatmap(
            similarity, 
            annot=True, 
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            center=0,
            mask=mask,
            square=True,
            linewidths=0.5,
            ax=ax,
            fmt='.2f'
        )
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Factor Similarity Matrix ({metric})')
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
            
        return fig

    def plot_source_profile(self, profile: str, constrained: bool = True, log_scale: bool = True, 
                            limit_species: Optional[int] = None, include_zero: bool = False,
                            figsize: Tuple[int, int] = (10, 6), show_uncertainty: str = 'BS',
                            filename: Optional[str] = None) -> plt.Figure:
        """
        Plot source profile with concentrations on y-axis and optional uncertainty display.
        
        Parameters
        ----------
        profile : str
            Profile name to plot
        constrained : bool, default=True
            Whether to use constrained or base run
        log_scale : bool, default=True
            Whether to use log scale for y-axis
        limit_species : int, optional
            Maximum number of species to show, by decreasing contribution
        include_zero : bool, default=False
            Whether to include species with zero contribution
        figsize : tuple, default=(10, 6)
            Figure size
        show_uncertainty : str, default='BS'
            Type of uncertainty to display: 'BS' (Bootstrap), 'DISP', 'none'
        filename : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        pmf = self.pmf
        if not pmf.ensure_data_loaded():
            raise ValueError("Required data not loaded.")
            
        if profile not in pmf.profiles:
            raise ValueError(f"Profile '{profile}' not found. Available profiles: {pmf.profiles}")
        
        # Get profile data
        profiles_df = pmf.dfprofiles_c if constrained else pmf.dfprofiles_b
        if profiles_df is None:
            raise ValueError(f"{'Constrained' if constrained else 'Base'} profiles data not loaded")
        
        # Get uncertainty data
        uncertainty_df = None
        if show_uncertainty.upper() in ['BS', 'DISP']:
            unc_summary = pmf.df_uncertainties_summary_c if constrained else pmf.df_uncertainties_summary_b
            if unc_summary is not None and not unc_summary.empty:
                try:
                    # Filter for the requested profile
                    uncertainty_df = unc_summary.loc[profile]
                except (KeyError, ValueError):
                    print(f"Warning: No uncertainty data found for profile '{profile}'")
        
        # Get profile data as a series
        profile_data = profiles_df[profile]
        
        # Filter out total variable
        if pmf.totalVar in profile_data.index:
            profile_data = profile_data.drop(pmf.totalVar)
        
        # Filter out zeros if requested
        if not include_zero:
            profile_data = profile_data[profile_data > 0]
        
        # Sort species by contribution
        profile_data = profile_data.sort_values(ascending=False)
        
        # Limit to top N species if requested
        if limit_species:
            profile_data = profile_data.head(limit_species)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract uncertainty data if available
        error_low = None
        error_high = None
        
        if uncertainty_df is not None and show_uncertainty.upper() != 'NONE':
            try:
                # Get uncertainty ranges for the selected species
                species_in_plot = profile_data.index
                
                if show_uncertainty.upper() == 'BS':
                    # Use bootstrap confidence intervals
                    if 'BS 5th' in uncertainty_df.columns and 'BS 95th' in uncertainty_df.columns:
                        bs_low = uncertainty_df.loc[species_in_plot, 'BS 5th']
                        bs_high = uncertainty_df.loc[species_in_plot, 'BS 95th']
                        
                        # Calculate absolute error margins from base value
                        error_low = profile_data - bs_low
                        error_high = bs_high - profile_data
                        
                        # Avoid negative error bars for both low and high
                        error_low = error_low.clip(lower=0)
                        error_high = error_high.clip(lower=0)
                elif show_uncertainty.upper() == 'DISP':
                    # Use DISP ranges
                    if 'DISP Min' in uncertainty_df.columns and 'DISP Max' in uncertainty_df.columns:
                        disp_min = uncertainty_df.loc[species_in_plot, 'DISP Min']
                        disp_max = uncertainty_df.loc[species_in_plot, 'DISP Max']
                        
                        # Calculate absolute error margins
                        error_low = profile_data - disp_min
                        error_high = disp_max - profile_data
                        
                        # Avoid negative error bars for both low and high
                        error_low = error_low.clip(lower=0)
                        error_high = error_high.clip(lower=0)
            except (KeyError, ValueError) as e:
                print(f"Warning: Error extracting uncertainty data: {str(e)}")
                error_low = error_high = None
        
        # Create horizontal bar plot with optional error bars
        y_pos = np.arange(len(profile_data))
        
        # Plot bars with concentration on y-axis (horizontal bars)
        bars = ax.bar(y_pos, profile_data, align='center')
        
        # Add error bars if available
        if error_low is not None and error_high is not None:
            # Convert Series to arrays to ensure proper shape
            err_low_array = np.array(error_low)
            err_high_array = np.array(error_high)
            
            # Create 2D array for asymmetric xerr: shape (2, N)
            xerr = np.vstack((err_low_array, err_high_array))
            
            # Plot error bars
            ax.errorbar(
                profile_data, y_pos, 
                xerr=xerr,
                fmt='none', ecolor='black', capsize=5
            )
        
        # Set y-tick labels to species names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(profile_data.index)
        
        # Set log scale if requested
        if log_scale:
            ax.set_xscale('log')
        
        # Add grid lines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Concentration')
        ax.set_title(f'Source Profile: {profile}' + 
                    (f'\n{show_uncertainty} uncertainty' if show_uncertainty.upper() != 'NONE' else ''))
        
        # Add source color as a small rectangle in the corner
        from .utils import get_sourceColor
        source_color = get_sourceColor(profile)
        if source_color:
            color_rect = plt.Rectangle((0.01, 0.01), 0.05, 0.05, transform=ax.transAxes, 
                                    facecolor=source_color, edgecolor='black')
            ax.add_patch(color_rect)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            self._save_figure(fig, filename)
        
        return fig

    
    def plot_time_series(self, constrained: bool = True,
                        specie: Optional[str] = None,
                        profiles: Optional[List[str]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        stacked: bool = False,
                        normalize: bool = False,
                        rolling_mean: Optional[int] = None,
                        colors: Optional[Dict[str, str]] = None,
                        figsize: Tuple[float, float] = (10, 6),
                        title: Optional[str] = None,
                        filename: Optional[str] = None) -> plt.Figure:
        """
        Plot factor contributions over time.
        
        Parameters
        ----------
        constrained : bool, default=True
            Use constrained or base run
        specie : str, optional
            Species to analyze (uses totalVar if None)
        profiles : list of str, optional
            Specific profiles/factors to plot, all if None
        start_date : str, optional
            Start date filter in format 'YYYY-MM-DD'
        end_date : str, optional
            End date filter in format 'YYYY-MM-DD'
        stacked : bool, default=False
            Whether to create a stacked plot (True) or line plot (False)
        normalize : bool, default=False
            Show relative contributions (as percentages)
        rolling_mean : int, optional
            Window size for rolling average
        colors : Dict[str, str], optional
            Colors to use for each profile
        figsize : Tuple[float, float], default=(10, 6)
            Figure size (width, height) in inches
        title : str, optional
            Figure title
        filename : str, optional
            File path to save the figure
        
        Returns
        -------
        plt.Figure
            Figure containing the time series plots
        """
        pmf = self.pmf
        
        try:
            df_contrib = pmf.to_cubic_meter(
                specie=specie,
                constrained=constrained,
                profiles=profiles
            )
        except Exception as e:
            raise ValueError(f"Error retrieving contribution data: {str(e)}")
            
        if df_contrib is None or df_contrib.empty:
            raise ValueError("No contribution data available")

        # Ensure we have a DatetimeIndex
        try:
            df_contrib = self._ensure_datetime_index(df_contrib)
        except Exception as e:
            print(f"Warning: Could not ensure DatetimeIndex: {str(e)}")
            # Continue anyway to see if it works with the current index
            
        # Filter by date range
        if start_date or end_date:
            if start_date:
                df_contrib = df_contrib[df_contrib.index >= start_date]
            if end_date:
                df_contrib = df_contrib[df_contrib.index <= end_date]
        
        # Apply rolling mean if specified
        if rolling_mean is not None and rolling_mean > 1:
            df_contrib = df_contrib.rolling(window=rolling_mean, min_periods=1).mean()
        
        # Normalize if requested
        if normalize:
            df_contrib = df_contrib.div(df_contrib.sum(axis=1), axis=0) * 100
                
        profiles_to_plot = profiles or df_contrib.columns.tolist()
        
        if colors is None:
            colors = self._get_colors_for_profiles(profiles_to_plot)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        if stacked:
            ax.stackplot(
                df_contrib.index, 
                [df_contrib[p] for p in profiles_to_plot],
                labels=profiles_to_plot,
                colors=[colors.get(p, 'tab:blue') for p in profiles_to_plot],
                alpha=0.7
            )
        else:
            for p in profiles_to_plot:
                ax.plot(
                    df_contrib.index, 
                    df_contrib[p],
                    label=p,
                    color=colors.get(p, None),
                    alpha=0.7
                )
                
        format_xaxis_timeseries(ax)
        
        # Set y-axis label based on normalization setting
        if normalize:
            ax.set_ylabel('Contribution (%)')
        else:
            specie_label = specie or pmf.totalVar or "PM"
            ax.set_ylabel(f'{specie_label} (µg m$^{-3}$)')
        
        # Handle legend placement based on number of profiles
        if len(profiles_to_plot) > 10:
            ax.legend(
                loc='upper left', 
                bbox_to_anchor=(1.01, 1), 
                ncol=2 if len(profiles_to_plot) > 10 else 1
            )
        else:
            ax.legend()
            
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set title (include rolling mean information if applicable)
        if title:
            ax.set_title(title, fontsize=14)
        elif rolling_mean:
            rolling_text = f" ({rolling_mean}-day rolling mean)"
            ax.set_title(f"Factor contribution time series{rolling_text}", fontsize=14)
            
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
            
        return fig
    
    def plot_total_species_sum(self, station_list, source, figsize=(12, 4), dpi=200, 
                    filename: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot chemical profile of a source across multiple stations with bootstrap uncertainties.
        
        This method creates a comparative visualization showing both the reference run values
        and bootstrap uncertainties for a specific source profile across different stations.
        
        Parameters
        ----------
        station_list : list
            List of PMF objects representing different stations or runs
        source : str
            Source profile name to plot
        figsize : tuple, default=(12, 4)
            Figure size in inches
        dpi : int, default=200
            Resolution for the figure
        filename : str, optional
            File path to save the figure
            
        Returns
        -------
        tuple
            Figure and axes objects
            
        Notes
        -----
        Each PMF object in station_list should have:
        - A 'name' attribute identifying the station/run
        - get_total_species_sum() method returning percentage contribution by species
        - dfBS_profile_c attribute containing bootstrap results
        """
        profile = pd.DataFrame()  
        BS_df = pd.DataFrame()
        
        # Get common species across stations
        if len(station_list) > 1:
            Species = station_list[0].get_total_species_sum().index.intersection(
                station_list[1].get_total_species_sum().index)
        else:
            Species = station_list[0].get_total_species_sum().index
        
        # Process each station
        for station in station_list:
            # Get reference run data
            p = pd.DataFrame(station.get_total_species_sum()[source].loc[Species])
            p["Site"] = "Ref_run " + station.name
            profile = pd.concat([profile, p], axis=0)

            # Process bootstrap data
            df = station.dfBS_profile_c
            # Skip if bootstrap data is missing
            if df is None:
                print(f"Warning: No bootstrap data available for {station.name}")
                continue
                
            sumsp = pd.DataFrame(index=["Sum"])
            for sp in Species:
                sumsp.loc["Sum", sp] = df.loc[sp].mean(axis=1).sum()
                
            # Calculate normalized percentages from bootstrap runs
            try:
                d = df.xs(source, level="Profile").divide(sumsp.iloc[0], axis=0) * 100
                d.index.names = ["Specie"]
                d = d.reindex(Species).unstack().reset_index()
                d["Site"] = "BS_run " + station.name
                BS_df = pd.concat([BS_df, d], axis=0)
            except Exception as e:
                print(f"Error processing bootstrap data for {station.name}: {str(e)}")

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot bootstrap data as barplots with error bars
        if not BS_df.empty:
            sns.barplot(data=BS_df, x="Specie", y=0, hue="Site", ci="sd", ax=ax, palette="Blues")
        
        # Plot reference runs as points
        if not profile.empty:
            sns.stripplot(data=profile.reset_index(), x="Specie", y=source, hue="Site", 
                        marker="o", dodge=True, ax=ax, jitter=False, palette="autumn")
        
        # Formatting
        plt.xticks(rotation=90)
        ax.set_title("Chemical profile of " + source, fontsize=12)
        ax.set_xlabel("Species", fontsize=10)
        ax.set_ylabel("% of species", fontsize=10)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)
        
        return fig, ax