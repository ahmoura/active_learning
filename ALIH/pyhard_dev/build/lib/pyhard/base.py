import io
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Union, Optional, Tuple
from numbers import Number

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from bokeh.themes import Theme
from holoviews import opts, streams
from shapely.geometry import Point, Polygon

from .context import Workspace


dark_color = '#292929'
light_color = '#ffffff'
footprint_colors = {'good': '#58D68D', 'best': '#9B59B6'}

dark_theme = Theme(
    json={
        'attrs': {
            'Figure': {
                'background_fill_color': dark_color,
                'border_fill_color': dark_color,
                'outline_line_color': '#444444',
            },
            'Title': {
                'text_color': 'white',
                'text_font_size': '14pt'
            },
            'Grid': {
                'grid_line_dash': [6, 4],
                'grid_line_alpha': .3,
            },

            'Axis': {
                'major_label_text_color': 'white',
                'axis_label_text_color': 'white',
                'major_tick_line_color': 'white',
                'minor_tick_line_color': 'white',
                'axis_line_color': "white"
            },

            'ColorBar': {
                'background_fill_color': dark_color,
                'major_label_text_color': 'white',
                'title_text_color': 'white'
            },

            'Plot': {
                'sizing_mode': 'stretch_both',
                'margin': (0, 0, 0, 0)
            }
        }
    })

light_theme = Theme(
    json={
        'attrs': {
            'Figure': {
                'background_fill_color': light_color,
                'border_fill_color': light_color,
                'outline_line_color': '#444444',
            },
            'Title': {
                'text_color': 'black',
                'text_font_size': '14pt'
            },
            'Grid': {
                'grid_line_dash': [6, 4],
                'grid_line_alpha': .9,
            },

            'Axis': {
                'major_label_text_color': 'black',
                'axis_label_text_color': 'black',
                'major_tick_line_color': 'black',
                'minor_tick_line_color': 'black',
                'axis_line_color': 'black'
            },

            'ColorBar': {
                'background_fill_color': light_color,
                'major_label_text_color': 'black',
                'title_text_color': 'black'
            },

            'Plot': {
                'sizing_mode': 'stretch_both',
                'margin': (0, 0, 0, 0)
            }
        }
    })

dark_template = f"""
{{% extends base %}}

{{% block title %}}Instance Hardness dashboard{{% endblock %}}

{{% block preamble %}}
<style>
  @import url(https://fonts.googleapis.com/css?family=Noto+Sans);
  body {{
    font-family: 'Noto Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    color: #fff;
    background: {dark_color};
  }}
</style>
{{% endblock %}}
"""

light_template = f"""
{{% extends base %}}

{{% block title %}}Instance Hardness dashboard{{% endblock %}}

{{% block preamble %}}
<style>
  @import url(https://fonts.googleapis.com/css?family=Noto+Sans);
  body {{
    font-family: 'Noto Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    color: #000;
    background: {light_color};
  }}
</style>
{{% endblock %}}
"""


class BaseMeasures(ABC):
    """
    Base class for measures (aka meta-features). Each measure should be implemented as a separate method.
    """

    _measures_dict: dict

    @property
    def logger(self):
        raise NotImplementedError

    def _call_method(self, name, **kwargs):
        return getattr(self, name)(**kwargs)

    def calculate_all(self, measures_list=None):
        if measures_list is None:
            measures_list = self._measures_dict.keys()
        elif isinstance(measures_list, list):
            measures_list = sorted(list(set(measures_list) & set(self._measures_dict.keys())))
        else:
            raise TypeError(f"Expected type list for parameter 'measures_list', not '{type(measures_list)}'")

        results = OrderedDict()
        for k in measures_list:
            self.logger.info(f"Calculating measure {repr(k)}")
            results[k] = self._call_method(self._measures_dict[k])

        df_measures = pd.DataFrame(results)
        return df_measures.add_prefix('feature_')


class BaseLearner(ABC):
    """
    Base class for learners (algorithms). This class provides methods for assessing performance in a pool of learners.
    """

    def __init__(self):
        self.predicted_proba = None

    @staticmethod
    def _call_function(module, name, **kwargs):
        return getattr(module, name)(**kwargs)

    @abstractmethod
    def score(self, metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        """
        Returns an array with scores for each instance, and an array with probabilities
        """
        pass

    def estimate_ih(self) -> np.ndarray:
        """
        Estimates the Instance Hardness value.
        """
        L = len(self.predicted_proba.columns)
        IH = 1 - (1/L)*self.predicted_proba.sum(axis=1).values
        return IH

    @abstractmethod
    def run(self, algo, metric: str, n_folds: int, n_iter: int, hyper_param_optm: bool, hpo_evals: int,
            hpo_timeout: int, hpo_name: str, verbose: bool, **kwargs) -> Union[np.ndarray, np.ndarray]:
        """
        Evaluates a single learner. Should return an array with mean score per instance, and an array with mean proba
        per instance (from sklearn method `predict_proba`).
        """
        pass

    @abstractmethod
    def run_all(self, metric: str, n_folds: int, n_iter: int, algo_list: Optional[list], parameters: Optional[list],
                hyper_param_optm: bool, hpo_evals: int, hpo_timeout: int, verbose: bool) -> pd.DataFrame:
        """
        Evaluates a pool of learners. Should return a dataframe whose columns are the algorithms and rows are the
        instances. Columns names must have the prefix `algo_`.
        """
        pass


class BaseApp:
    _tabs_id = ['Instance Space', 'Footprint performance', 'Selection explorer']
    _tabs_inner_id = ['Statistics', 'Features', 'Meta-features']

    data: pd.DataFrame

    w_color: pn.widgets.Select
    w_color_range: pn.widgets.RangeSlider
    w_checkbox: pn.widgets.Checkbox
    w_footprint_on: pn.widgets.Checkbox
    w_footprint_algo: pn.widgets.Select

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.bbox = None
        self.cmap = 'coolwarm'
        self._page_tabs = pn.Tabs(*[(name, None) for name in self._tabs_id], dynamic=False)
        self._inner_tabs = pn.Tabs(*[(name, None) for name in self._tabs_inner_id], dynamic=True, tabs_location='left')
        self._page_tabs[2] = (self._tabs_id[2], self._inner_tabs)

    def footprint_area(self, algo):
        try:
            border_points_good = self.workspace.footprints.xs([algo, 'good']).values
        except KeyError:
            border_points_good = np.array([[0, 0]])
        try:
            border_points_best = self.workspace.footprints.xs([algo, 'best']).values
        except KeyError:
            border_points_best = np.array([[0, 0]])

        border_good, border_best = border_points_good, border_points_best

        footprint_good = hv.Polygons([border_good.tolist()],
                                     label='Good Footprint').opts(line_width=1, line_alpha=0.2,
                                                                  line_color='black',
                                                                  fill_color=footprint_colors['good'],
                                                                  fill_alpha=.2,
                                                                  show_legend=True)

        footprint_best = hv.Polygons([border_best.tolist()],
                                     label='Best Footprint').opts(line_width=1, line_alpha=0.2,
                                                                  line_color='black',
                                                                  fill_color=footprint_colors['best'],
                                                                  fill_alpha=.2,
                                                                  show_legend=True)
        return footprint_good * footprint_best

    def select_instances(self):
        if self.bbox is None:
            return pd.Series(False, index=self.data.index)
        x, y = list(self.bbox.keys())
        if len(self.bbox[x]) == 2:
            V1 = np.column_stack([self.bbox[x], self.bbox[y]])
            V2 = V1.copy()
            V2[0, 1], V2[1, 1] = V1[1, 1], V1[0, 1]
            V = np.array([V1[0, :], V2[0, :], V1[1, :], V2[1, :]])
            contour = list(map(tuple, V))
        else:
            contour = list(map(tuple, np.column_stack([self.bbox[x], self.bbox[y]])))
        polygon = Polygon(contour)
        mask = self.data[[x, y]].apply(lambda p: polygon.contains(Point(p[0], p[1])), raw=True, axis=1)
        return mask

    @abstractmethod
    def data_space(self, color: str, range_limits: Tuple[Number, Number], autorange_on: bool) -> hv.Scatter:
        pass

    @abstractmethod
    def instance_space(self, color: str, range_limits: Tuple[Number, Number], autorange_on: bool) -> hv.Scatter:
        pass

    def get_pane(self) -> pn.Template:
        @pn.depends(color=self.w_color.param.value, lim=self.w_color_range.param.value,
                    autorange_on=self.w_checkbox.param.value)
        def update_plot1(color, lim, autorange_on):
            return self.data_space(color, lim, autorange_on)

        @pn.depends(color=self.w_color.param.value, lim=self.w_color_range.param.value,
                    autorange_on=self.w_checkbox.param.value)
        def update_plot2(color, lim, autorange_on):
            return self.instance_space(color, lim, autorange_on)

        def selection_callback1(bbox, region_element, selection_expr, resetting):
            self.bbox = bbox
            if resetting:
                self.bbox = None
            self.populate_tabs()
            return hv.Polygons([[[0, 0]]])

        @pn.depends(footprint=self.w_footprint_algo.param.value, fp_on=self.w_footprint_on.param.value)
        def selection_callback2(bbox, region_element, selection_expr, footprint, fp_on):
            self.bbox = bbox

            self.populate_tabs()

            if fp_on:
                return self.footprint_area(footprint)
            else:
                return (hv.Polygons([[[0, 0]]], label='Good Footprint').opts(fill_color=footprint_colors['good']) *
                        hv.Polygons([[[0, 0]]], label='Best Footprint').opts(fill_color=footprint_colors['best']))

        dmap1 = hv.DynamicMap(update_plot1)
        dmap2 = hv.DynamicMap(update_plot2)
        dmap1.opts(title='Principal Components')
        dmap2.opts(title='Instance Space')

        selection1 = hv.streams.SelectionExpr(source=dmap1)
        reset = hv.streams.PlotReset()
        sel1_dmap = hv.DynamicMap(selection_callback1, streams=[selection1, reset])

        selection2 = hv.streams.SelectionExpr(source=dmap2)
        sel2_dmap = hv.DynamicMap(selection_callback2, streams=[selection2])

        def file_cb():
            mask = self.select_instances()
            df = self.workspace.data[mask]
            sio = io.StringIO()
            df.to_csv(sio)
            sio.seek(0)
            return sio

        button = pn.widgets.FileDownload(embed=False, auto=True, callback=file_cb,
                                         filename='selection.csv',
                                         label='Save selected points',
                                         button_type='primary')

        layout = (dmap1 * sel1_dmap + dmap2 * sel2_dmap).cols(2).opts(
            opts.Layout(shared_axes=False, shared_datasource=True, framewise=True),
            opts.Polygons(show_legend=True, legend_position='bottom'))

        gspec = pn.GridSpec(sizing_mode='stretch_both', background=light_color, margin=0)
        gspec[0, 0] = pn.Column('## Color', self.w_color, '### Color Bar', self.w_checkbox, self.w_color_range,
                                pn.Row(pn.Spacer(), height=20),
                                '## Footprint', self.w_footprint_on, self.w_footprint_algo,
                                pn.Row(pn.Spacer(), height=20),
                                '## Selection', button,
                                background=light_color)
        gspec[0, 1:5] = layout
        self._page_tabs[0] = (self._tabs_id[0], gspec)
        self._page_tabs[1] = (self._tabs_id[1], pn.widgets.DataFrame(self.workspace.footprint_performance,
                                                                     name='Performance',
                                                                     disabled=True,
                                                                     sizing_mode='stretch_both'))
        tmpl = pn.Template(light_template)
        tmpl.add_panel(name='IS', panel=self._page_tabs)
        return tmpl

    @abstractmethod
    def populate_tabs(self):
        pass

    def show(self, port=5001, show=True):
        tmpl = self.get_pane()
        pn.serve(tmpl, port=port, show=show, title='Instance Hardness',
                 websocket_origin=[f'127.0.0.1:{port}', f'localhost:{port}'])
