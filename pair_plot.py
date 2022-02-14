import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(style="ticks", color_codes=True)
iris = pd.read_csv('dataset_train.csv')

iris = iris.dropna()
iris.pop('Index')
g = sns.pairplot(iris, hue="Hogwarts House", height=1.0)

plt.xlabel('Hogwarts House')

plt.show()