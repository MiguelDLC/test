#%%
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


# %%

exam, projet = np.meshgrid(np.linspace(0, 20, 101), np.linspace(0, 20, 101))
rexam = exam/20
rprojet = projet/20

weight = interp1d([0, 5/20, 9/20, 1], [0.25, 0.25, 0.55, 0.55])

x = np.linspace(0, 1, 101)
y = weight(x)

exam_w = weight(rexam)
# %%

def not_moronically_stupid_locator(cs):
    for collection in cs.collections:
        for path in collection.get_paths():
            yield path.vertices[len(path.vertices)//2]

score = exam_w*projet + (1-exam_w)*exam

fig = plt.figure(figsize=(10, 10))
plt.contourf(exam, projet, exam_w*20, shading='gouraud', levels=np.arange(0, 21), cmap='turbo')
cs = plt.contour(exam, projet, exam_w*20, levels=np.arange(0, 21), colors='black', linewidths=0.5)
plt.gca().clabel(cs, inline=True, fontsize=10, manual=not_moronically_stupid_locator(cs))
plt.xlabel('Note examen')
plt.ylabel('Note projet')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xticks(np.arange(0, 21, 1))
plt.yticks(np.arange(0, 21, 1))
plt.grid(color="k", alpha=0.2)
plt.show()
# %%
