#!/usr/bin/env python
__doc__ = """

Asynchronous Sampler

Nicholas Turner <nturner@cs.princeton.edu>, 2017
Kisuk Lee <kisuklee@mit.edu>, 2015-2017
"""

import os
import h5py

from Queue import Queue
import threading


def sampler_daemon(sampler, q):
    " Function run by the thread "
    while True:
        q.put(sampler(imgs=['input']), block=True, timeout=None)
        # if not q.full():
        #     q.put(sampler(imgs=["input"]))
        # else:
        #     q.join()

class AsyncSampler(object):
    " Wrapper class for asynchronous sampling functions "

    def __init__(self, sampler, queue_size=10):

        self.q = Queue(queue_size)
        self.t = threading.Thread(target=sampler_daemon, args=(sampler, self.q))
        self.t.daemon = True
        self.t.start()

    def get(self):
        " Pulls a sample from the queue "
        res = self.q.get()
        self.q.task_done()
        return res
