import matplotlib.pyplot as plt
import numpy as np

routit_f1s = [[1, 3, 5, 7, 9, 11],[8.37, 8.33, 8.33, 8.37, 8.33, 8.17]]

plt.plot(routit_f1s[0], routit_f1s[1])
plt.xlabel('Amount of routing iterations')
plt.ylabel('F1 score')
plt.savefig('images/routit.pdf', bbox_inches='tight')