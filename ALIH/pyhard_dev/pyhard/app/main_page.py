import io
import logging
import os
import queue
import sys
import traceback
import kthread
import pandas as pd
import panel as pn
from multiprocessing import Process, Event, Queue
from pandas.errors import EmptyDataError
from pyhard.integrator import run_pipeline

logger = logging.getLogger(__name__)
_configurations = None
pyhard_thread = None
pyhard_process = None
watcher = None

spinner = """
<div class="spinner-border text-secondary" role="status">
<span class="sr-only">Loading...</span>
</div>
"""

button_run = pn.widgets.Button(name='Run', button_type='primary')
button_stop = pn.widgets.Button(name='Cancel', button_type='danger', disabled=True)
w_file = pn.widgets.FileInput(accept='.csv', mime_type='text/csv')
w_spinner = pn.pane.HTML("")
w_hidden = pn.widgets.Button(name='hidden')
info = pn.widgets.StaticText(name='INFO', value='not started.')

w_hidden.jscallback(args={'a': w_spinner},
                    **{'disabled': """window.open("./viz")"""})


def process_task(data: pd.DataFrame, config: dict, status: Queue, event: Event):
    sys.stdout = open(os.devnull, 'w')
    try:
        run_pipeline(data, config.copy())
        status.put((0, ))
    except Exception:
        status.put((1, traceback.format_exc()))
    finally:
        status.close()
        event.set()


def thread_task():
    try:
        data = get_input_data()
        run_pipeline(data, _configurations.copy())

        info.value = 'success! Access the <a href="./viz" target="_blank">visualization page</a> ' \
                     'to see the results.'
        w_hidden.disabled = not w_hidden.disabled
        logger.info("Instance hardness analysis successfully finished.")
    except pd.errors.EmptyDataError:
        info.value = "choose a dataset before running."
        logger.exception("NO INPUT DATA PROVIDED! Select first a dataset to analyse.")
    except Exception:
        info.value = 'An ERROR occurred during execution. See the ' \
                     '<a href="./log" target="_blank">log page</a> for more details.'
        logger.exception("An error occurred during execution.")
    finally:
        button_run.disabled = False
        button_stop.disabled = True
        toggle_spinner(False)


def watch(event: Event, status_queue: Queue):
    event.wait()
    pyhard_process.join()

    try:
        status = status_queue.get()

        if status[0] == 0:
            info.value = 'success! Access the <a href="./viz" target="_blank">visualization page</a> ' \
                         'to see the results.'
            w_hidden.disabled = not w_hidden.disabled
            logger.info("Instance hardness analysis successfully finished.")
        elif status[0] == 1:
            info.value = 'An ERROR occurred during execution. See the ' \
                         '<a href="./log" target="_blank">log page</a> for more details.'
            logger.exception(status[1])
    except queue.Empty:
        logger.exception("Status queue empty.")
        info.value = 'An ERROR occurred during execution. See the ' \
                     '<a href="./log" target="_blank">log page</a> for more details.'
    finally:
        event.clear()
        button_run.disabled = False
        button_stop.disabled = True
        toggle_spinner(False)


def callback_run(event=None):
    global pyhard_thread
    try:
        if pyhard_thread.isAlive():
            pyhard_thread.terminate()
            info.value = "Previous task aborted."
            logger.warning("Previous analysis running interrupted.")
    except AttributeError:
        pass

    button_run.disabled = True
    button_stop.disabled = False
    info.value = "waiting for the analysis to finish..."
    toggle_spinner(True)

    pyhard_thread = kthread.KThread(target=thread_task, name="PyHardProcess", daemon=True)
    pyhard_thread.start()


def callback_run_process(event=None):
    global pyhard_process, watcher
    completion_event = Event()
    status_queue = Queue()

    try:
        if pyhard_process.is_alive():
            pyhard_process.terminate()
            info.value = "Previous task aborted."
            logger.warning("Previous analysis running interrupted.")
    except AttributeError:
        pass

    try:
        data = get_input_data()
        watcher = kthread.KThread(target=watch, args=(completion_event, status_queue), daemon=True)
        watcher.start()

        pyhard_process = Process(target=process_task, args=(data, _configurations.copy(),
                                                            status_queue, completion_event), daemon=True)
        pyhard_process.start()

        button_run.disabled = True
        button_stop.disabled = False
        info.value = "waiting for the analysis to finish..."
        toggle_spinner(True)
    except pd.errors.EmptyDataError:
        info.value = "choose a dataset before running."
        logger.exception("NO INPUT DATA PROVIDED! Select first a dataset to analyse.")


def callback_stop(event=None):
    try:
        if pyhard_thread.isAlive():
            info.value = "stopping..."
            pyhard_thread.terminate()
            info.value = "analysis interrupted by user."
            logger.warning("Instance hardness analysis interrupted by user.")
    except AttributeError:
        pass
    finally:
        toggle_spinner(False)
        button_run.disabled = False
        button_stop.disabled = True


def callback_stop_process(event=None):
    global pyhard_process, watcher

    try:
        if watcher.isAlive():
            watcher.terminate()
    except AttributeError:
        pass

    try:
        if pyhard_process.is_alive():
            info.value = "stopping..."
            pyhard_process.terminate()
            while pyhard_process.is_alive():
                pass
            info.value = "analysis interrupted by user."
            logger.warning("Instance hardness analysis interrupted by user.")
    except Exception:
        logger.exception("Exception while stopping process.")
    finally:
        toggle_spinner(False)
        button_run.disabled = False
        button_stop.disabled = True


def toggle_spinner(state):
    if state:
        w_spinner.object = spinner
    else:
        w_spinner.object = ""


def get_input_data():
    return pd.read_csv(io.BytesIO(w_file.value))


def set_parameters(config: dict):
    global _configurations
    _configurations = config.copy()


def reset_state():
    try:
        if pyhard_thread.isAlive():
            pyhard_thread.terminate()
            info.value = "not started."
            logger.warning("Resetting page state. Previous analysis running interrupted.")
    except AttributeError:
        pass

    info.value = "not started."
    toggle_spinner(False)
    button_run.disabled = False
    button_stop.disabled = True


button_run.on_click(callback_run_process)
button_stop.on_click(callback_stop_process)
