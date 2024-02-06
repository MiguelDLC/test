#%%
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import slim4
from slim4 import slim2d
from scipy.spatial import ConvexHull

mesh = slim2d.Mesh("square.msh")
# %%


tri = mesh.triangles
xn = mesh.xnodes[0]
yn = mesh.xnodes[1]
nt = mesh.n_triangles
nn = len(xn)
# %%

row = np.zeros(6*nt, dtype=int)
col = np.zeros(6*nt, dtype=int)
data = np.ones(6*nt)
for i in range(3):
    row[i::6] = tri[:, i]
    col[i::6] = tri[:, (i+1)%3]
    col[i+3::6] = tri[:, i]
    row[i+3::6] = tri[:, (i+1)%3]

graph = csr_matrix((data, (row, col)), shape=(nn, nn))
graph.data[:] = 1
row, col = graph.nonzero()
#%%
def getrow(graph, i):
    return graph[i].indices

def plotgraph(graph, xn, yn, **kwargs):
    nn = graph.shape[0]
    for i in range(nn):
        neighbours = getrow(graph, i)
        for j in neighbours:
            plt.plot([xn[i], xn[j]], [yn[i], yn[j]], '-k', **kwargs)

#%%

def coarsegraph(graph):
    nn = graph.shape[0]
    nfree = graph.indptr[1:] - graph.indptr[0:-1]
    subgroup = -np.ones(nn, dtype=int)
    curr_set = 0

    stack = [0]
    while(len(stack) > 0):
        # print("stack = ", stack)
        # plotgraph(graph, xn, yn)
        # for ig in range(curr_set):
        #     xx = xn[subgroup == ig]
        #     yy = yn[subgroup == ig]
        #     bb = plt.fill(xx, yy, linewidth=2)
        #     color = bb[0].get_facecolor()
        #     plt.plot([xx[-1], *xx], [yy[-1], *yy], '-o', lw=2, color=color)
        # for i in range(nn):
        #     plt.text(xn[i], yn[i], str(i), fontsize=8)
        # plt.plot(xn[stack], yn[stack], 'sb', ms=10)
        # plt.show()


        stack.sort(key=lambda x: -nfree[x])
        start = stack[-1]
        if(subgroup[start] != -1):
            stack.pop()
            continue
        
        neighbours = getrow(graph, start)
        # print(start, subgroup[start], neighbours, nfree[neighbours])
        neighbours = neighbours[subgroup[neighbours] == -1]
        if len(neighbours) <= 1:
            # print("No neighbours %d, stacklen %d" % (start, len(stack)))
            neighbours = getrow(graph, start)
            group = {}
            for n in neighbours:
                group[subgroup[n]] = group.get(subgroup[n], 0) + 1
            
            newset = max(group, key=group.get)
            
            subgroup[start] = newset
            # curr_set += 1
            stack = list(filter(lambda a: a != start, stack))
            if (len(stack) == 0):
                stack.extend(np.nonzero(subgroup == -1)[0])

            continue
            # break
        
        if (len(neighbours) == 1):
            n1 = neighbours[0]
            subgroup[start] = curr_set
            subgroup[n1] = curr_set
            curr_set += 1
            nfree[neighbours] -= 1
            neighbours1 = getrow(graph, n1)
            neighbours1 = neighbours1[subgroup[neighbours1] == -1]
            stack.extend(neighbours )
            stack.extend(neighbours1)
            nfree[neighbours1] -= 1
            # stack.pop()
            stack = list(filter(lambda a: a != start, stack))
            stack = list(filter(lambda a: a != n1, stack))
            continue

        if (len(neighbours) == 2):
            n1 = neighbours[0]
            n2 = neighbours[1]
            subgroup[start] = curr_set
            subgroup[n1] = curr_set
            subgroup[n2] = curr_set
            curr_set += 1
            neighbours1 = getrow(graph, n1)
            neighbours1 = neighbours1[subgroup[neighbours1] == -1]
            neighbours2 = getrow(graph, n2)
            neighbours2 = neighbours2[subgroup[neighbours2] == -1]
            nfree[neighbours ] -= 1
            nfree[neighbours1] -= 1
            nfree[neighbours2] -= 1
            stack.extend(neighbours )
            stack.extend(neighbours1)
            stack.extend(neighbours2)
            # stack.pop()
            stack = list(filter(lambda a: a != start, stack))
            stack = list(filter(lambda a: a != n1, stack))
            stack = list(filter(lambda a: a != n2, stack))
            continue

        for n1 in neighbours:
            if(n1 == start):
                continue
            neighbours1 = getrow(graph, n1)
            neighbours1 = neighbours1[subgroup[neighbours1] == -1]

            for(i2, n2) in enumerate(neighbours1):
                if(n2 == start):
                    continue
                neighbours2 = getrow(graph, n2)
                neighbours2 = neighbours2[subgroup[neighbours2] == -1]
                
                for(i3, n3) in enumerate(neighbours2):
                    if(n3 == start):
                        subgroup[n1] = curr_set
                        subgroup[n2] = curr_set
                        subgroup[n3] = curr_set
                        curr_set += 1
                        nfree[neighbours ] -= 1
                        nfree[neighbours1] -= 1
                        nfree[neighbours2] -= 1
                        stack.extend(neighbours )
                        stack.extend(neighbours1)
                        stack.extend(neighbours2)
                        
                        # stack.pop()
                        stack = list(filter(lambda a: a != n1, stack))
                        stack = list(filter(lambda a: a != n2, stack))
                        stack = list(filter(lambda a: a != n3, stack))
                        break
        if (len(stack) == 0):
            stack.extend(np.nonzero(subgroup == -1)[0])
    return subgroup

#     print(ixdx)

#     break
# plotgraph(graph, xn, yn)

def plot_set(xn, yn, subgroup, n, **kwargs):
    for ig in range(n):
        xx = xn[subgroup == ig]
        yy = yn[subgroup == ig]
        if(len (xx) > 3):
            HULL = ConvexHull(np.array([xx, yy]).T)
            xx = xx[HULL.vertices]
            yy = yy[HULL.vertices]
            
        bb = plt.fill(xx, yy, **kwargs)
        # color = bb[0].get_facecolor()
        # plt.plot([xx[-1], *xx], [yy[-1], *yy], '-o', lw=2, color=color)


def subgraph(graph, subgroup, xn, yn):
    nnz = graph.nnz
    row, col = graph.nonzero()
    print(nnz, row.shape, col.shape, subgroup.shape)
    n = graph.shape[0]
    row2 = np.zeros(nnz, dtype=int)
    col2 = np.zeros(nnz, dtype=int)

    row2[0:nnz] = subgroup[row]
    col2[0:nnz] = subgroup[col]
    data2 = np.ones(nnz)
    data2[row2 == col2] = 0
    graph2 = csr_matrix((data2, (row2, col2)))
    graph2.eliminate_zeros()
    
    nnew = graph2.shape[0]
    xn2 = np.zeros(nnew)
    yn2 = np.zeros(nnew)
    for(i, ig) in enumerate(range(nnew)):
        xx = xn[subgroup == ig]
        yy = yn[subgroup == ig]
        xn2[i] = np.mean(xx)
        yn2[i] = np.mean(yy)

    return graph2, xn2, yn2

plt.triplot(xn, yn, tri, "-k", alpha=0.05, lw=2, zorder=-1000)

subgroup = coarsegraph(graph)
graph2, xn2, yn2 = subgraph(graph, subgroup, xn, yn)
# plot_set(xn, yn, subgroup, graph2.shape[0], alpha=0.5)
plotgraph(graph2, xn2, yn2, alpha=0.1, lw=1)


subgroup2 = coarsegraph(graph2)
graph3, xn3, yn3 = subgraph(graph2, subgroup2, xn2, yn2)
# plot_set(xn2, yn2, subgroup2, graph3.shape[0], alpha=0.5)
plotgraph(graph3, xn3, yn3, alpha=0.2, lw=0.5)

subgroup2 = coarsegraph(graph3)
graph4, xn4, yn4 = subgraph(graph3, subgroup2, xn3, yn3)
plot_set(xn3, yn3, subgroup2, graph4.shape[0], alpha=0.5)
plotgraph(graph4, xn4, yn4, alpha=1, lw=0.5)




print("compression ratio 1-2 = ", graph.shape[0]/graph2.shape[0])
print("compression ratio 2-3 = ", graph2.shape[0]/graph3.shape[0])
print("compression ratio 1-3 = ", graph.shape[0]/graph3.shape[0])
print("compression ratio 1-4 = ", graph.shape[0]/graph4.shape[0])

# plt.show()

plt.figure()
plt.spy(graph, markersize=1)
plt.figure()
plt.spy(graph2, markersize=1)
plt.figure()
plt.spy(graph3, markersize=1)
plt.figure()
plt.spy(graph4, markersize=1)
plt.show()
# %%
