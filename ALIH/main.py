from run_modal import run_modal
from run_pyhard import run_pyhard
from run_baseline import run_baseline
from threading import Thread
from environment.config import *

if __name__ == '__main__':

    baseline_thread = Thread(target=run_baseline, args=(datasets,))
    modal_thread = Thread(target=run_modal, args=(datasets,))
    pyhard_thread = Thread(target=run_pyhard, args=(datasets,))

    modal_thread.start()
    pyhard_thread.start()
    baseline_thread.start()
