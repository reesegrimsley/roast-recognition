from  matplotlib import pyplot as plt
import pickle
import numpy as np

clusters=  pickle.load(open('class_clusters-innerbox.pickle','rb'))


def gen_hist_plot(hist_list, plotname):
    print('generate plot of histograms for "%s"' % plotname)
    # plt.clf()
    fig = plt.figure()

    colors = ("b", "g", "r")

    plt.title(plotname)
    for (hist, color) in zip(hist_list, colors):
        # print(hist)
        # print(color)
        plt.plot(hist, color=color)
        # print('plotted')


def classify_histogram(hist, clusters):
    distances = {c:np.inf for c in list(clusters.keys())}
    # clusters = {c:{'count':0, 'sum_hist':None} for c in classes}

    for c in list(clusters.keys()):
        cluster_avg_hist = clusters[c]['average_histogram']
        dist = ((hist - cluster_avg_hist) ** 2).sum()
        distances[c] = dist
    
    print(distances)
    # blue_sorted = np.argsortq
    ind = np.argmin([distances[k] for k in clusters.keys()])
    roast_level = list(clusters.keys())[ind]
    # print('classed as %s ' % roast_level)
    return roast_level


for k in list(clusters.keys()):
    cluster = clusters[k]
    gen_hist_plot(cluster['average_histogram'], k)

fig = plt.figure()

new_hist = pickle.load(open('hist.pickle', 'rb'))
print(new_hist.shape)
new_hist[[0,1,2],:] = new_hist[[2,1,0],:]

roast = classify_histogram(new_hist, clusters)
print(roast)

colors = ("b", "g", "r")

plt.title('new (underroasted)')
for (hist, color) in zip(new_hist, colors):
    plt.plot(hist, color=color)
plt.show()
