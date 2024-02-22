import asyncio
import sys

from PyQt5.QtCore import Qt, QCoreApplication, QThreadPool, QRunnable, pyqtSlot
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QDialog, \
    QLineEdit, QMessageBox
import os
from predict import *

"""
    为批量分割图片创建的异步任务类
"""


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


"""
    为单独分割图片创建的异步任务类
"""


class AsyncTaskForSingle:
    def __init__(self, file_path=''):
        self.result = None
        self.callback = None
        self.file_path = file_path

    async def my_async_task(self):
        self.result = await predict_file(file_path=self.file_path)
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


class MainWindow(QWidget):
    def __init__(self, width=300, height=300):
        super().__init__()
        self.myWidth = width
        self.myHeight = height
        self.init_ui()

    def init_ui(self):
        # 创建布局
        main_layout = QVBoxLayout()

        # 顶部按钮
        top_button_layout = QHBoxLayout()
        btn1 = QPushButton('分割虹膜图片', self)
        btn2 = QPushButton('批量分割虹膜图片', self)
        btn1.clicked.connect(self.show_window1)
        btn2.clicked.connect(self.show_window2)
        top_button_layout.addWidget(btn1)
        top_button_layout.addWidget(btn2)

        # 底部图片展示
        self.image_label = QLabel(self)
        main_layout.addLayout(top_button_layout)
        main_layout.addWidget(self.image_label)

        # 设置主窗口布局
        self.setLayout(main_layout)

        # 获取屏幕的宽度和高度
        screen_width = QApplication.desktop().screenGeometry().width()
        screen_height = QApplication.desktop().screenGeometry().height()

        # 计算窗口的左上角位置，使其居中
        x = (screen_width - self.myWidth) // 2
        y = (screen_height - self.myHeight) // 2
        # 设置窗口属性
        self.setGeometry(x, y, self.myWidth, self.myHeight)
        self.setWindowTitle('虹膜分割')

    def show_window1(self):
        window1 = Window1(self, self.myWidth, self.myHeight)
        self.hide()
        window1.exec_()

    def show_window2(self):
        window2 = Window2(self, self.myWidth, self.myHeight)
        self.hide()
        window2.exec_()


class Window1(QDialog):
    def __init__(self, parent=None, width=None, height=None):
        super().__init__(parent)
        self.myWidth = width
        self.myHeight = height
        self.init_ui()

    def init_ui(self):
        # 创建布局
        layout = QVBoxLayout()

        # 顶部按钮
        btn_layout = QHBoxLayout()
        btn_select_image = QPushButton('选择图片', self)
        btn_back = QPushButton('返回', self)
        btn_select_image.clicked.connect(self.select_image)
        btn_back.clicked.connect(self.return_to_parent)
        btn_layout.addWidget(btn_select_image)
        btn_layout.addWidget(btn_back)

        # 图片展示
        image_layout = QHBoxLayout()
        self.image_label = QLabel(self)
        self.image_label_right = QLabel(self)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_label_right)

        layout.addLayout(btn_layout)
        layout.addLayout(image_layout)
        # 获取屏幕的宽度和高度
        screen_width = QApplication.desktop().screenGeometry().width()
        screen_height = QApplication.desktop().screenGeometry().height()

        # 计算窗口的左上角位置，使其居中
        x = (screen_width - self.myWidth) // 2
        y = (screen_height - self.myHeight) // 2

        # 设置窗口布局
        self.setLayout(layout)
        # 设置窗口属性
        self.setGeometry(x, y, self.myWidth, self.myHeight)
        self.setWindowTitle('虹膜图片分割')

        self.thread_pool = QThreadPool()

    def return_to_parent(self):
        self.parent().show()
        self.close()

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            # 创建异步任务对象
            async_task = AsyncTaskForSingle(file_path=file_path)

            # 创建工作线程并连接信号与槽
            runnable = WorkerRunnable(async_task)
            async_task.callback = self.callback
            self.thread_pool.start(runnable)

            # 读取图片并调整大小以适应窗口
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(self.width(), self.height())
            # 在标签中设置调整大小后的图片
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

            self.movie = QMovie("resource/loading.gif")  # 替换为你的动画文件路径
            self.image_label_right.setMovie(self.movie)
            self.movie.start()

    def callback(self, result):
        self.movie.stop()
        if os.path.exists(result):
            # 读取图片并调整大小以适应窗口
            pixmap = QPixmap(result)
            pixmap = pixmap.scaled(self.myWidth, self.myHeight)
            self.image_label_right.setPixmap(pixmap)
            self.image_label_right.setScaledContents(True)


class Window2(QDialog):
    def __init__(self, parent=None, width=None, height=None):
        super().__init__(parent)
        self.myWidth = width
        self.myHeight = height
        self.init_ui()

    def init_ui(self):
        # 创建水平布局
        hbox = QHBoxLayout()

        # 创建两个按钮
        btn1 = QPushButton('开始批量分割', self)
        btn2 = QPushButton('返回上一级', self)
        btn1.clicked.connect(self.start_segmentation)
        btn2.clicked.connect(self.return_to_parent)

        # 将按钮添加到水平布局中
        hbox.addWidget(btn1)
        hbox.addWidget(btn2)

        # 创建垂直布局
        vbox = QVBoxLayout()

        # 将水平布局添加到垂直布局中
        vbox.addLayout(hbox)

        # 创建标签和只读输入框
        label = QLabel('分割图片路径:', self)
        self.read_path_edit = QLineEdit(self)
        self.read_path_edit.setReadOnly(True)

        # 创建文件夹按钮
        model_btn = QPushButton('选择训练图片文件夹', self)
        model_btn.clicked.connect(lambda: self.show_directory_dialog(self.read_path_edit))

        first_layout = QHBoxLayout()
        first_layout.addWidget(label)
        first_layout.addWidget(self.read_path_edit)
        first_layout.addWidget(model_btn)

        # 创建标签和只读输入框
        second_label = QLabel('结果存储路径:', self)
        self.store_path_edit = QLineEdit(self)
        self.store_path_edit.setReadOnly(True)

        # 创建文件夹按钮
        folder_btn2 = QPushButton('选择存储文件夹', self)
        folder_btn2.clicked.connect(lambda: self.show_directory_dialog(self.store_path_edit))

        second_layout = QHBoxLayout()
        second_layout.addWidget(second_label)
        second_layout.addWidget(self.store_path_edit)
        second_layout.addWidget(folder_btn2)

        # 将标签、输入框和文件夹按钮添加到垂直布局中
        vbox.addLayout(first_layout)
        vbox.addLayout(second_layout)

        # 获取屏幕的宽度和高度
        screen_width = QApplication.desktop().screenGeometry().width()
        screen_height = QApplication.desktop().screenGeometry().height()
        # 计算窗口的左上角位置，使其居中
        x = (screen_width - self.myWidth) // 2
        y = (screen_height - self.myHeight) // 2
        # 设置窗口属性
        self.setGeometry(x, y, self.myWidth, self.myHeight)
        self.setWindowTitle('虹膜图片分割')

        self.thread_pool = QThreadPool()

        # 设置窗口布局
        self.setLayout(vbox)
        self.setWindowTitle('批量分割虹膜图片')

    def return_to_parent(self):
        self.parent().show()
        self.close()

    def start_segmentation(self):
        read_path = self.read_path_edit.text()
        store_path = self.store_path_edit.text()
        self.msg_box = QMessageBox(self)
        self.msg_box.setIcon(QMessageBox.Information)
        self.msg_box.setWindowTitle('提示')
        if not read_path:
            self.msg_box.setText("图片路径不能为空")
            self.msg_box.exec_()

        elif not store_path:
            self.msg_box.setText("存储路径不能为空")
            self.msg_box.exec_()
        else:
            # 创建异步任务对象
            # read_path = 'E:/py_projects/deep_learning/IrisSegment/myTest/source'
            # store_path = 'E:/py_projects/deep_learning/IrisSegment/myTest/sinal'
            file_path = 'E:/py_projects/deep_learning/IrisSegment/myTest/source/S1001L01.jpg'
            # 创建工作线程并连接信号与槽
            async_task = AsyncTaskForBatch(file_directory=read_path, store_path=store_path)
            runnable = WorkerRunnable(async_task)
            async_task.callback = self.callback
            self.thread_pool.start(runnable)

            self.myDialog = QDialog(self)
            self.myDialog.setWindowTitle('图片处理中')
            self.myDialog.setFixedSize(self.myWidth, self.myHeight)

            self.label = QLabel(self)
            # 创建 QMovie 对象并设置动态图
            self.movie = QMovie("resource/loading.gif")
            self.label.setMovie(self.movie)
            layout = QVBoxLayout()
            layout.addWidget(self.label)
            self.movie.start()

            # 获取屏幕的宽度和高度
            screen_width = QApplication.desktop().screenGeometry().width()
            screen_height = QApplication.desktop().screenGeometry().height()
            # 计算窗口的左上角位置，使其居中
            x = (screen_width - self.myWidth) // 2
            y = (screen_height - self.myHeight) // 2
            self.myDialog.setGeometry(x, y, self.myWidth // 2, self.myHeight // 2)
            self.myDialog.setLayout(layout)

            self.myDialog.exec_()

    @pyqtSlot(str)
    def callback(self, result):
        self.movie.stop()
        text = ''
        if result == 'success':
            text = '图片处理完成'
        else:
            text = '图片处理失败'
        self.label.setText(text)
        self.label.setAlignment(Qt.AlignCenter)

        # self.myDialog.close()
        # msg_box = QMessageBox(self)
        # msg_box.setIcon(QMessageBox.Information)
        # msg_box.setWindowTitle('提示')
        # # msg_box.setStandardButtons(QMessageBox.Ok)
        # msg_box.setWindowModality(0)  # 设置为非模态
        # if result == 'success':
        #     msg_box.setText('图片分割成功')
        # else:
        #     msg_box.setText('图片分割失败')
        # msg_box.exec_()

    def show_directory_dialog(self, edit):
        # 打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            # 将选择的文件夹路径设置为只读输入框的文本
            edit.setText(folder_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
