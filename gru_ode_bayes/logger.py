# 
# # Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# import tensorflow as tf
# import numpy as np
# import scipy.misc
# try:
#     from StringIO import StringIO  # Python 2.7
# except ImportError:
#     from io import BytesIO         # Python 3.x

# # === Enable TF1 compatibility ===
# tf.compat.v1.disable_v2_behavior()


# class Logger(object):

#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer = tf.compat.v1.summary.FileWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         # 如果是 numpy 类型，先转成 Python float
#         if isinstance(value, np.ndarray):
#             # 处理 0 维或 1 维数组的情况
#             if value.size == 1:
#                 value = float(value.reshape(()))  # 或者 value.item()
#             else:
#                 # 如果是多维数组，这里你可以选择取均值、最大值等
#                 value = float(value.mean())
#         elif isinstance(value, (np.generic,)):  # np.float32, np.float64 等
#             value = float(value)

#         summary = tf.compat.v1.Summary(
#             value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
#         )
#         self.writer.add_summary(summary, step)

#     def save_dict(self, info, epoch):
#         """
#         info is a dictionary of scalars to log. Epoch is the epoch number.
#         """
#         for tag, value in info.items():
#             self.scalar_summary(tag, value, epoch)

#     def image_summary(self, tag, images, step):
#         """Log a list of images."""
#         img_summaries = []
#         for i, img in enumerate(images):
#             try:
#                 s = StringIO()
#             except:
#                 s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")

#             img_sum = tf.compat.v1.Summary.Image(
#                 encoded_image_string=s.getvalue(),
#                 height=img.shape[0],
#                 width=img.shape[1]
#             )

#             img_summaries.append(
#                 tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum)
#             )

#         summary = tf.compat.v1.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)

#     def histo_summary(self, tag, values, step, bins=1000):
#         """Log a histogram of the tensor of values."""
#         counts, bin_edges = np.histogram(values, bins=bins)

#         hist = tf.compat.v1.HistogramProto()
#         hist.min = float(np.min(values))
#         hist.max = float(np.max(values))
#         hist.num = int(np.prod(values.shape))
#         hist.sum = float(np.sum(values))
#         hist.sum_squares = float(np.sum(values ** 2))

#         bin_edges = bin_edges[1:]

#         for edge in bin_edges:
#             hist.bucket_limit.append(edge)
#         for c in counts:
#             hist.bucket.append(c)

#         summary = tf.compat.v1.Summary(
#             value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)]
#         )
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

import os
import numpy as np

class Logger(object):
    """
    Minimal logger that mimics the original TensorFlow-based API
    but does not depend on TensorFlow. It simply does nothing
    (or you can extend it to write to txt/csv if you like).
    """

    def __init__(self, log_dir):
        # 创建日志目录以防后面代码依赖这个路径存在
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable (no-op or simple print)."""
        # 如果你想看数值，也可以改成：
        # print(f"[LOG][{step}] {tag}: {value}")
        pass

    def save_dict(self, info, epoch):
        """
        info 是一个 dict：{tag: value}
        epoch 是当前 epoch 编号。
        这里我们直接什么也不做。
        """
        for tag, value in info.items():
            self.scalar_summary(tag, value, epoch)

    def image_summary(self, tag, images, step):
        """原来用来写图片到 TensorBoard，这里直接 no-op。"""
        pass

    def histo_summary(self, tag, values, step, bins=1000):
        """原来用来写直方图到 TensorBoard，这里直接 no-op。"""
        pass
