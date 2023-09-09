import multiprocessing
import os

class TaskProcessor:
    def __init__(self, gpu_list):
        self.gpu_list = gpu_list
        self.manager = multiprocessing.Manager()
        self.gpu_dict = self.manager.dict()

    def execute_tasks(self, tasks):
        process_args = [(task, self.gpu_dict) for task in tasks]
        with multiprocessing.Pool(processes=len(tasks), initializer=self._initializer,
                                  initargs=(self.gpu_dict, self.gpu_list)) as pool:
            results = pool.map(self._execute_task, process_args)
        return results

    @staticmethod
    def _initializer(gpu_dict, gpu_list):
        process_id = os.getpid()
        gpu_dict[process_id] = gpu_list[process_id]

    @staticmethod
    def _execute_task(task_args):
        task, gpu_dict = task_args
        process_id = os.getpid()
        gpu = gpu_dict[process_id]
        result = task(gpu)  # Passing GPU as an argument
        return result
