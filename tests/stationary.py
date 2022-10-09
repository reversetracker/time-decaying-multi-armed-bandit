import random
import time

from arm import Arm

ts = int(time.time())
arm = Arm(created_at=ts)


for i in range(10000):
    arm.update(reward=random.choice([False, True, True, False]))

expectation = arm.pull()
print(expectation)
assert 0.48 < expectation < 0.52
