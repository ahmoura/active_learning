import logging
import holoviews as hv
import panel as pn
from pathlib import Path
from flask import Flask, render_template, request, redirect
from jinja2 import Environment, FileSystemLoader
from pyhard import log_file
from pyhard.app import main_page, app_page
from pyhard.integrator import load_cached_files
from pyhard.visualization import AppClassification, Demo
from pyhard.base import light_theme, light_template

logger = logging.getLogger(__name__)

flask_app = Flask(__name__)

templates_path = Path(__file__).parent / "templates"
index = templates_path / "index_panel.html"

hv.extension('bokeh')
hv.renderer('bokeh').theme = light_theme

configurations = {'n_folds': 5, 'n_iter': 1, 'metric': 'logloss',
                  'hyper_param_optm': True, 'hpo_evals': 50,
                  'feat_select': True, 'max_n_features': 10, 'method': 'mrmr',
                  'measures_list': ['LSC', 'CB', 'DCP', 'Usefulness', 'N1', 'LSR', 'TD_P', 'F3', 'Harmfulness', 'F1',
                                    'MV', 'TD_U', 'CL', 'kDN', 'N2', 'F2', 'CLD', 'F4', 'DS'],
                  'algo_list': ['gradient_boosting', 'random_forest', 'svc_linear', 'gaussian_nb',
                                'logistic_regression', 'bagging', 'svc_rbf', 'mlp']}

measures = ['kDN', 'DS', 'DCP', 'TD_P', 'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 'LSR', 'Harmfulness',
            'Usefulness', 'F1', 'F2', 'F3', 'F4']

algorithms = {
    'Linear SVM': 'svc_linear',
    'RBF SVM': 'svc_rbf',
    'Random Forest': 'random_forest',
    'Gradient Boosting': 'gradient_boosting',
    'Multilayer Perceptron': 'mlp',
    'Bagging': 'bagging',
    'Naive Bayes': 'gaussian_nb',
    'Logistic Regression': 'logistic_regression'
}

methods = ['cife', 'cmim', 'disr', 'icap', 'jmi', 'mifs', 'mim', 'mrmr']


# -------------------------------------------- Flask routes --------------------------------------------

def process_form(form: dict):
    measures_form = list(set(measures).intersection(set(form.keys())))
    algos_form = list(set(algorithms.values()).intersection(set(form.keys())))

    for k in (measures_form + algos_form):
        del form[k]

    form['measures_list'] = measures_form
    form['algo_list'] = algos_form

    form['feat_select'] = True if form['feat_select'] == 'on' else False
    form['hyper_param_optm'] = True if form['hyper_param_optm'] == 'on' else False

    form['method'] = str.lower(form['method'])

    return form


def merge_dict(d1, d2):
    return {**d1, **d2}


def get_form_state():
    state = dict()

    # general
    if configurations['metric'] == 'logloss':
        state = merge_dict(state, {'logloss': 'checked', 'brier': ''})
    else:
        state = merge_dict(state, {'logloss': '', 'brier': 'checked'})
    state['n_folds'] = configurations['n_folds']
    state['n_iter'] = configurations['n_iter']

    # algorithms
    for algo in algorithms.values():
        state[algo] = 'checked' if algo in configurations['algo_list'] else ''

    # measures
    for measure in measures:
        state[measure] = 'checked' if measure in configurations['measures_list'] else ''

    # hyper-parameter optimization
    if configurations['hyper_param_optm']:
        state = merge_dict(state, {'hpo_on': 'checked', 'hpo_off': ''})
    else:
        state = merge_dict(state, {'hpo_on': '', 'hpo_off': 'checked'})
    state['hpo_evals'] = configurations['hpo_evals']

    # feature selection
    if configurations['feat_select']:
        state = merge_dict(state, {'fs_on': 'checked', 'fs_off': ''})
    else:
        state = merge_dict(state, {'fs_on': '', 'fs_off': 'checked'})
    state['max_n_features'] = configurations['max_n_features']
    state['method'] = configurations['method']
    for m in methods:
        state[m] = 'selected' if m == configurations['method'] else ''

    return state


@flask_app.route('/config', methods=['GET', 'POST'])
def set_configurations():
    global configurations

    def str2num(s):
        try:
            return int(s)
        except ValueError:
            return s

    if request.method == "POST":
        form = request.form.to_dict(flat=True)
        form = {k: str2num(v) for k, v in form.items()}
        configurations = process_form(form)
        main_page.set_parameters(configurations)
        logger.info(f"Setting configurations to: {configurations}")
        return redirect('/main')
    return render_template("configuration.html", state=get_form_state())


# -------------------------------------------- Panel routes --------------------------------------------

def panel_main():
    env = Environment(loader=FileSystemLoader(str(templates_path)))
    demo_template = env.get_template('main.html')
    tmpl = pn.Template(demo_template)

    main_page.reset_state()
    main_page.set_parameters(configurations)

    # tmpl.add_variable('app_title', '<h1>Main page</h1>')
    tmpl.add_panel(name='load', panel=main_page.w_file)
    tmpl.add_panel(name='run', panel=pn.Row(main_page.button_run, main_page.w_spinner))
    tmpl.add_panel(name='stop', panel=main_page.button_stop)
    tmpl.add_panel(name='hidden', panel=main_page.w_hidden)
    tmpl.add_panel(name='info', panel=pn.Row(main_page.info))
    return tmpl


@DeprecationWarning
def panel_app():
    gspec = pn.GridSpec(sizing_mode='stretch_both')

    app_page.button1.on_click(app_page.button1_callback)
    app_page.button2.on_click(app_page.button2_callback)

    # pn.Param(action_example.param, name='PyHard')
    gspec[0, 0:5] = pn.Column('## Input data', app_page.w_data,
                              app_page.pn.Row(app_page.button1, app_page.button2),
                              app_page.info_msg,
                              # pn.layout.VSpacer(max_height=10),
                              '## General', app_page.w_folds, app_page.w_iter, '**Algorithm performance metric**',
                              app_page.w_metric,
                              pn.layout.VSpacer(max_height=20), margin=(0, 0, 0, 20))

    gspec[0, 5:10] = pn.Column('## Feature selection', app_page.w_fs, app_page.w_n_feat, app_page.w_method,
                               pn.layout.VSpacer(max_height=20),
                               '## Hyper-parameter optimization', app_page.w_hpo, app_page.w_evals,
                               margin=(0, 30, 0, 30))

    gspec[0, 11:13] = pn.Column('## Measures', app_page.w_measures, max_width=200)
    gspec[0, 13:15] = pn.Column('## Algorithms', app_page.w_algo, max_width=200, margin=(0, 20, 0, 0))

    tmpl = pn.Template(light_template)
    tmpl.add_panel(name='Test', panel=gspec)
    return tmpl


def panel_viz():
    try:
        data, df_metadata, df_is, df_footprint, df_foot_perf = load_cached_files()
        data.index = df_metadata.index
        app = AppClassification(data, df_metadata, df_is, df_footprint, df_foot_perf)
        return app.get_pane()
    except FileNotFoundError:
        return pn.Row("## Instance space visualization will be shown here after running the process.")


def panel_log():
    with open(log_file, 'r') as f:
        text = f.read()
    text = text.replace('\n', '<br>')
    text = text.replace('ERROR', '<span style="color: #ff0000"> ERROR </span>')
    # text = text.replace('WARNING', '<span style="color: #ffff00"> WARNING </span>')
    # text = text.replace('INFO', '<span style="color: #00ff00"> WARNING </span>')

    env = Environment(loader=FileSystemLoader(str(templates_path)))
    log_template = env.get_template('log_template.html')
    tmpl = pn.Template(log_template)
    tmpl.add_variable('log', text)

    return tmpl


def panel_demo():
    demo = Demo()
    pane = demo.display()
    env = Environment(loader=FileSystemLoader(str(templates_path)))
    demo_template = env.get_template('demo_template.html')
    tmpl = pn.Template(demo_template)
    tmpl.add_panel(name='demo', panel=pane)
    return tmpl


def run():
    logging.getLogger("bokeh").setLevel(logging.WARNING)
    logging.getLogger("tornado").setLevel(logging.WARNING)
    logging.getLogger("pyhard").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    port = 5001
    pn.serve({'demo': panel_demo, 'viz': panel_viz,
              'log': panel_log, 'main': panel_main, '/-': flask_app},
             port=port, show=True, **{'index': str(index)}, title='Graphene',
             websocket_origin=[f'127.0.0.1:{port}', f'localhost:{port}'])


if __name__ == "__main__":
    run()
