#!/usr/bin/env python
__doc__ = """

Asynchronous Sampler

Nicholas Turner <nturner@cs.princeton.edu>, 2017-8
Kisuk Lee <kisuklee@mit.edu>, 2015-2017
"""

import os
import h5py

from queue import Queue
import threading


def sampler_daemon(sampler, q):
    " Function run by the thread "
    while True:
        q.put(sampler(), block=True, timeout=None)


class AsyncSampler(object):
    " Wrapper class for asynchronous sampling functions "

    def __init__(self, sampler, queue_size=10):

        self.q = Queue(queue_size)
        self.t = threading.Thread(target=sampler_daemon, args=(sampler, self.q))
        self.t.daemon = True
        self.t.start()

    def __call__(self):
        " Pulls a sample from the queue "
        res = self.q.get()
        self.q.task_done()
        return res
