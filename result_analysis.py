import numpy as np
import matplotlib.pyplot as plt
import operator
faster_filenames = [
  "0.25SupRateV2.npy",
  "0.5SupRateV3.npy",
  "0.75SupRateV2.npy",
  "1SupRateV2.npy",
  ]

slower_filenames = [
  "0.01SupRateV2.npy",
  "0.08SupRateV2.npy",
  "0.1SupRateV2.npy",
  ]

all_filenames = [
  "0.25SupRateV2.npy",
  "0.5SupRateV3.npy",
  "0.75SupRateV2.npy",
  "1SupRateV2.npy",
  "0.01SupRateV2.npy",
  "0.08SupRateV2.npy",
  "0.1SupRateV2.npy"
]
dict = {}

x = []
y = []

for file in all_filenames:
  dict_name = file.replace("SupRateV2.npy","")
  dict_name = dict_name.replace("SupRateV3.npy", "")
  sup_rate = float(dict_name)
  dict_name += "_Supervision_Rate"
  array = np.load("results/{}".format(file))
  number_runs = np.mean(array)
  x.append(sup_rate)
  y.append(number_runs)

L = sorted(zip(x, y), key=operator.itemgetter(0))
x, y = zip(*L)

print(x)
print(y)
x = np.array(x)
y = np.array(y)
plt.xlim(1, 0)
plt.plot(x,y)
plt.xlabel('decreasing  supervised rate')
plt.ylabel('mean score')
plt.show()

# for file in faster_filenames:
#   dict_name = file.replace("SupRateV2.npy","")
#   dict_name = dict_name.replace("SupRateV3.npy", "")
#   dict_name += "_Supervision_Rate"
#   dict[dict_name] = np.load("results/{}".format(file))

# for key in dict:
#   plt.plot(dict[key], label=key.replace("_"," "), alpha=0.8)

# plt.xlabel("runs")
# plt.ylabel("score")
# plt.legend(loc='best')
# plt.show()
