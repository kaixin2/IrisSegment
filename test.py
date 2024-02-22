import asyncio

from PyQt5.QtCore import QRunnable
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QPushButton
from predict import *
def test01():
   file_directory = 'E:/py_projects/deep_learning/IrisSegment/myTest/source'
   store_path = 'E:/py_projects/deep_learning/IrisSegment/myTest/sinal'
   batch_predict_file(file_directory=file_directory,store_path=store_path)


class AsyncTaskForBatch:
   def __init__(self, file_directory='', store_path=''):
      self.result = None
      self.callback = None
      self.file_directory = file_directory
      self.store_path = store_path

   async def my_async_task(self):
      self.result = await batch_predict_file(file_directory=self.file_directory, store_path=self.store_path)
      if self.callback:
         self.callback(self.result)


class WorkerRunnable(QRunnable):
    def __init__(self, async_task):
        super().__init__()
        self.async_task = async_task

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_task.my_async_task())


class A:

   def test02(self):
      read_path = 'E:/py_projects/deep_learning/IrisSegment/myTest/source'
      store_path = 'E:/py_projects/deep_learning/IrisSegment/myTest/sinal'
      async_task = AsyncTaskForBatch(file_directory=read_path, store_path=store_path)

      # 创建工作线程并连接信号与槽
      runnable = WorkerRunnable(async_task)
      async_task.callback = self.callback
      self.thread_pool.start(runnable)

   def callback(self, result):
      print("finish")
if __name__ == '__main__':
   a = A()
   a.test02()
