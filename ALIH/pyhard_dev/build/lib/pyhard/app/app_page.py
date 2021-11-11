import io
import logging
import kthread
import pandas as pd
import panel as pn
from pandas.errors import EmptyDataError
from pyhard.integrator import run_pipeline, clear_cache

logger = logging.getLogger(__name__)

pyhard_thread = None
tabs = None
button1 = pn.widgets.Button(name='Run analysis', button_type='primary', sizing_mode='stretch_both')
button2 = pn.widgets.Button(name='\u2716', disabled=True, button_type='danger',
                            sizing_mode='stretch_both', max_width=40)

measures = ['kDN', 'DS', 'DCP', 'TD_P', 'TD_U', 'CL', 'CLD', 'MV', 'CB', 'N1', 'N2', 'LSC', 'LSR', 'Harmfulness',
            'Usefulness', 'F1', 'F2', 'F3', 'F4']

algorithms = {
    'Linear SVM': 'svc_linear',
    'RBF SVM': 'svc_rbf',
    'Random Forest': 'random_forest',
    'Gradient Boosting': 'gradient_boosting',
    # 'Multilayer Perceptron': 'mlp',
    'Bagging': 'bagging',
    'Naive Bayes': 'gaussian_nb',
    'Logistic Regression': 'logistic_regression'
}

w_data = pn.widgets.FileInput(accept='.csv', mime_type='text/csv')
w_measures = pn.widgets.CheckBoxGroup(options=measures, value=measures, inline=False)
w_algo = pn.widgets.CheckBoxGroup(options=list(algorithms.keys()), value=list(algorithms.keys()), inline=False)

w_folds = pn.widgets.IntSlider(name='Folds', start=2, end=11, step=1, value=5)
w_iter = pn.widgets.IntSlider(name='Iterations', start=1, end=11, step=1, value=1)
w_metric = pn.widgets.RadioButtonGroup(name='metric', options=['logloss', 'brier'],
                                       button_type='default', value='logloss')

w_fs = pn.widgets.RadioBoxGroup(name='Feature selection', options=['on', 'off'], inline=True)
w_n_feat = pn.widgets.IntSlider(name='Max number of selected features', start=1, end=len(measures), step=1, value=10)
w_method = pn.widgets.Select(name='Method', options=['icap', 'mifs', 'mrmr'], value='icap')

w_hpo = pn.widgets.RadioBoxGroup(name='Hyper-parameter optimization', options=['on', 'off'], inline=True)
w_evals = pn.widgets.IntSlider(name='Number of evaluations', start=50, end=500, step=50, value=100)

info_msg = pn.Row("You can start the analysis by choosing a data set and clicking the button above", min_height=100)


def load_configurations():
    configurations = {
        'measures_list': w_measures.value,
        'algo_list': list(map(algorithms.get, w_algo.value)),

        'n_folds': w_folds.value,
        'n_iter': w_iter.value,
        'metric': w_metric.value,

        'feat_select': True if w_fs.value.lower() == 'on' else False,
        'max_n_features': w_n_feat.value,
        'method': w_method.value,
        # 'eta': w_eta.value,

        'hyper_param_optm': True if w_hpo.value.lower() == 'on' else False,
        'hpo_evals': w_evals.value
    }
    return configurations


def get_input_data():
    try:
        return pd.read_csv(io.BytesIO(w_data.value))
    except EmptyDataError:
        logger.exception("NO INPUT DATA PROVIDED! Select first a data set to analyse.")
        raise


def run_chain():
    global pyhard_thread, tabs, button1
    try:
        button2.disabled = False
        info_msg.clear()
        info_msg.append("Process started, please wait until finished...")
        data = get_input_data()
        config = load_configurations()
        run_pipeline(data, config)

        info_msg.clear()
        info_msg.append('Process successfully completed! Access the '
                        '<a href="./viz" target="_blank">visualization page</a> to see the results.')
    except Exception:
        info_msg.clear()
        info_msg.append('An ERROR occurred during execution. See the '
                        '<a href="./log" target="_blank">LOG page</a> for more details.')
    finally:
        logger.info("Thread finished")
        button1.disabled = False
        button2.disabled = True


def button1_callback(event):
    global pyhard_thread, button1
    button1.disabled = True
    pyhard_thread = kthread.KThread(target=run_chain, name="PyhardKillableThread")
    pyhard_thread.start()


def button2_callback(event):
    global pyhard_thread
    try:
        if pyhard_thread.isAlive():
            pyhard_thread.terminate()
            clear_cache()
            info_msg.clear()
            info_msg.append("Analysis interrupted by user.")
            logger.warning("Instance hardness analysis interrupted.")
    except AttributeError:
        pass
