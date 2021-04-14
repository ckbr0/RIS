import os
from typing import TYPE_CHECKING, Callable, DefaultDict, List, Optional

import numpy as np
import monai.handlers
from ignite.engine import Engine, Events

class ValidationHandler(monai.handlers.ValidationHandler):

    def __init__(self, validator, start, interval, epoch_level=True) -> None:
        super().__init__(interval, validator, epoch_level)
        self.start = start

    def attach(self, engine) -> None:
        
        event_filter = lambda engine, event: True if (event >= (self.start-1) and (event-self.start) % self.interval == 0) else False
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(event_filter=event_filter), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(event_filter=event_filter), self)

class MetricLogger(monai.handlers.MetricLogger):

    def __init__(self, log_dir: str, validator: Optional[Engine]) -> None:
        super().__init__(evaluator=validator)
        self.log_dir = log_dir

    def attach(self, engine: Engine) -> None:
        super().attach(engine)
        engine.add_event_handler(Events.COMPLETED, self.write_log)

    def __call__(self, engine: Engine) -> None:
        super().__call__(engine)        
        loss_file = os.path.join(self.log_dir, "log_loss.txt")
        with self.lock:
            with open(loss_file, "a") as f:
                np.savetxt(f, self.loss)
    
    def log_metrics(self, engine: Engine) -> None:
        super().log_metrics(engine)
        with self.lock:
            for m, v in self.metrics:
                m_file = os.path.join(self.log_dir, f"log_{m}.txt")
                with open(m_file, "a") as f:
                    np.savetxt(f, v)

    def write_log(self, engine: Engine):
        loss_file = os.path.join(self.log_dir, "log_loss.txt")
        np.savetxt(loss_file, self.loss)
        
        for m, v in sefl.metrics:
            m_file = os.path.join(self.log_dir, f"log_{m}.txt")
            np.savetxt(m_file, v)
