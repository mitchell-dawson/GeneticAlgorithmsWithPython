import statistics
import sys
import time


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(100):
            sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            if i < 10 or i % 10 == 9:
                print(
                    "{} {:3.2f} {:3.2f}".format(
                        1 + i, mean, statistics.stdev(timings, mean) if i > 1 else 0
                    )
                )
