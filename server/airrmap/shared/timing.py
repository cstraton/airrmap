"""Simple class to measure code performance"""

# tests -> test_timing.py

import time
import json
from typing import List, Any


class TimeEvent():
    def __init__(self,
                 name: str,
                 process_time: float,
                 cumulative_time: float,
                 meta: Any = None):
        """
        A single time event.

        Parameters
        ----------
        name : str
            Name of the event.

        process_time : float
            Amount of time the process took in fractional seconds.

        cumulative_time : float
            Cumulative time in fractional seconds.

        meta : Any (optional)
            Additional data to store alongside the event.
        """

        self.name = name
        self.process_time = process_time
        self.cumulative_time = cumulative_time
        self.meta = meta


class Timing():
    """A simple class to measure code performance

    NOTE: __dict__ may be dumped to json for debugging,
    and may be sent to the user, e.g. json.dumps(o, default=vars). 
    Ensure all properties remain relevant.

    Example:
    t = Timer()  # starts timer going.
    t.add('Start of process.')
    t.add('Step 1 finished.')
    t.add('Step 2 finished.')
    t.add('End of process.')

    t.time_events  # Get list of time events.

    """

    def __init__(self):
        self.time_events: List[TimeEvent] = []
        self.init_time = time.perf_counter()

    def add(self, name: str, meta: Any = None) -> TimeEvent:
        """Add a new time event"""

        # Get previous cumulative time (or 0 if first event)
        previous_cumulative_time = 0. if len(self.time_events) == 0 \
            else self.time_events[-1].cumulative_time

        # Get time differences
        cumulative_time = time.perf_counter() - self.init_time
        process_time = cumulative_time - previous_cumulative_time

        # Create new event
        time_event = TimeEvent(
            name=name,
            process_time=process_time,
            cumulative_time=cumulative_time,
            meta=meta
        )

        # Add to list
        self.time_events.append(time_event)

        # Return
        return time_event
