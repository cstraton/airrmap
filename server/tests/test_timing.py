
# Imports
import unittest
import time
import json
from airrmap.shared.timing import TimeEvent, Timing


class TestTiming(unittest.TestCase):

    def testTiming(self):

        # Init and set timer going
        FIRST_DELAY = 0.1  # seconds
        SECOND_DELAY = 0.2
        PLACES = 1
        t = Timing() 

        # Wait 100ms
        time.sleep(FIRST_DELAY)

        # Add a new event
        result = t.add("Test1")

        self.assertIsInstance(
            result,
            TimeEvent,
            'TimeEvent instance should be returned.'
        )

        self.assertAlmostEqual(
            first=result.process_time,
            second=FIRST_DELAY,
            places=PLACES,
            msg='Process time should be reported correctly.'
        )

        self.assertAlmostEqual(
            first=result.process_time,
            second=result.cumulative_time,
            places=PLACES,
            msg='Cumulative time and process time should be the same for the first event.'
        )

        # Wait 200ms
        time.sleep(SECOND_DELAY)
        result = t.add("Test2")

        self.assertAlmostEqual(
            first=result.process_time,
            second=SECOND_DELAY,
            places=PLACES,
            msg='Process time should be reported correctly for second event.'
        )

        self.assertAlmostEqual(
            first=result.cumulative_time,
            second=FIRST_DELAY + SECOND_DELAY,
            places=PLACES,
            msg='Cumulative time should be correct after second event.'
        )

        self.assertIs(
            result,
            t.time_events[1],
            'Second event should be added to the list (in addition to first).'
        )



if __name__ == '__main__':
    unittest.main()
