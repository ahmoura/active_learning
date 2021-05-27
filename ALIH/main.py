from run_modal import run_modal
from run_pyhard import run_pyhard
from threading import Thread
from environment.config import *

if __name__ == '__main__':

    modal_thread = Thread(target=run_modal, args=(datasets,))
    pyhard_thread = Thread(target=run_pyhard, args=(datasets,))

    modal_thread.start()
    pyhard_thread.start()
