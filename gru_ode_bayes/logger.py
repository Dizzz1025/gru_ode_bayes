from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime, timedelta, timezone
import os

class Logger(object):

    def __init__(self, log_dir):
        bj_time = datetime.now(timezone(timedelta(hours=8)))
        timestamp = bj_time.strftime("%Y%m%d-%H%M")
        log_dir = os.path.join(log_dir, f'run_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """ Log a scalar variable """
        self.writer.add_scalar(tag, value, step)