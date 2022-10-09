import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

from arm import Arm

histories = pd.read_csv("histories.csv")


arm = Arm(created_at=1663589575)

result = []
for history in histories.values:
    time, reward = history[1], history[4] > 0
    current_timestamp = int(parser.parse(time).timestamp())
    arm.update(reward=reward, current_timestamp=current_timestamp)
    result.append(arm.pull(current_timestamp=current_timestamp))


print(f"기사 시작 시간: {histories.values[0][1]}")
print(f"기사 마지막 시간: {histories.values[-1][1]}")
plt.plot(range(len(result)), result)
plt.show()
