from monai.handlers import ValidationHandler
from ignite.engine import Events

class ValidationHandlerCT(ValidationHandler):

    def __init__(self, validator, start, interval, epoch_level) -> None:
        super().__init__(interval, validator, epoch_level)
        self.start = start

    def attach(self, engine) -> None:
        
        event_filter = lambda engine, event: True if (event >= self.start and (event-self.start) % self.interval == 0) else False

        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(event_filter=event_filter), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(event_filter=event_filter), self)
        
