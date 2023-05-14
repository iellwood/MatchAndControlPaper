"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Code for the class ParallelRun which was used to parallelize many of the computations
"""

import time
import datetime
import multiprocessing
from multiprocessing import Queue

class ParallelRun:

    def __init__(self, target_function, argument_function, max_number_of_processes, iterations, main_loop_sleep_time=0.5):
        '''
        This is a  simple implementation of a parallel compution.
        The function target_function is run on multiple processes with arguments given by argument_function.
        Heuristically, it computes::

            def run():
                outputs = []
                for i in range(iterations):
                    outputs.append(target_function(argument_function(i)))
                return outputs

        WARNING: The call to run must be enclosed in::

            if __name__ == '__main__':
                # your code here

        :param target_function: Function to be computed
        :param argument_function: A function that returns the appropriate arguments of target_function. Should take exactly one integer argument
        :param max_number_of_processes: Max number of processes
        :param iterations: Number of iterations
        :param main_loop_sleep_time: How long the main loop sleeps for between testing if processes have finished
        '''
        self._iterations = iterations
        self._max_number_of_processes = max_number_of_processes
        self._target_function = target_function
        self._argument_function = argument_function
        self._main_loop_sleep_time = main_loop_sleep_time

        self._outputs = []
        self._queue = Queue()
        self._t_0 = time.time()

    @staticmethod
    def _target_function(queue, target_function, arguments):
        queue.put(target_function(*arguments))

    def run(self):
        processes = []
        iteration_index = 0

        self._t_0 = time.time()

        # Until all iterations are complete, make sure that there are _max_number_of_processes running
        while iteration_index < self._iterations:

            processes = [process for process in processes if process.is_alive()]

            if len(processes) < self._max_number_of_processes:
                process = multiprocessing.Process(
                    target=ParallelRun._target_function,
                    args=(self._queue, self._target_function, self._argument_function(iteration_index),)
                )
                process.start()
                processes.append(process)
                iteration_index += 1

            self.get_output_from_queue()

            time.sleep(self._main_loop_sleep_time)

        # Wait for the final processes to finish up
        while len(processes) > 0:
            processes = [process for process in processes if process.is_alive()]
            self.get_output_from_queue()
            time.sleep(self._main_loop_sleep_time)

        # Grab any remaining items from the queue
        while not self._queue.empty():
            self.get_output_from_queue()

        return self._outputs


    def get_output_from_queue(self):
        if not self._queue.empty():
            self._outputs.append(self._queue.get())
            seconds_elapsed_since_run_start = time.time() - self._t_0
            time_per_trial = seconds_elapsed_since_run_start / len(self._outputs)
            time_remaining = (self._iterations - len(self._outputs)) * time_per_trial
            print('Computed', len(self._outputs), ' of', self._iterations, '--',
                  'Time elapsed =', str(datetime.timedelta(seconds=seconds_elapsed_since_run_start)),
                  'time_remaining (est.) =', str(datetime.timedelta(seconds=time_remaining))
                  )

