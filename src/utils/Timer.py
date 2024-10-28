import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.durations = []

    def add_time(self):
        current_time = time.time()
        duration = current_time - self.start_time
        self.durations.append(duration)
        self.start_time = current_time

    def print_current_duration(self):
        if self.durations:
            print(f"Current iteration duration: {self.durations[-1]:.2f} seconds")

    def print_overview(self):
        total_time = sum(self.durations)
        average_time = total_time / len(self.durations) if self.durations else 0
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per iteration: {average_time:.2f} seconds")