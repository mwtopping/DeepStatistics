import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch



# Generate a sample composed of randon N-dimensional gaussians
def generate_fake_data(Nsamples, Ncategories, Ndim=2, ratios=None):

    np.random.seed(123456789)

    # determine number of samples per category
    if ratios is not None:
        ratios = torch.Tensor(ratios) / torch.Sum(ratios)
    else:
        ratios = torch.ones(Ncategories) / Ncategories


    total_data = None
    total_classes = None

    for ii in range(Ncategories):
        Nsamp = Nsamples * ratios[ii]
        means = np.random.uniform(-10, 10, size=Ndim)
        stds = np.random.uniform(0.2, 2, size=Ndim)
        data = None
        for jj in range(Ndim):
            thisdim = torch.Tensor(np.random.normal(loc=means[jj], scale=stds[jj], size=int(Nsamp))).reshape(int(Nsamp),1)
            if data is None:
                data = torch.Tensor(thisdim)
            else:
                data = torch.hstack((data, thisdim))

        classes = ii * torch.ones(int(Nsamp))

        if total_classes == None:
            total_classes = classes
            total_data = data
        else:
            total_classes = torch.hstack((total_classes, classes))
            total_data = torch.vstack((total_data, data))

    total_classes = total_classes.reshape(len(total_classes), 1)

    return total_data, total_classes

def one_hot(arr):
    Nunique = len(torch.unique(arr))

    onehot = None

    for row in arr:
        newrow = torch.zeros(1, Nunique)
        newrow[0, int(row)] = 1

        if onehot is None:
            onehot = newrow
        else:
            onehot = torch.vstack((onehot, newrow))

    return onehot


def renormalize_data():
    pass


def show_data(data, classes, ax=None):

    colors = mcolors.TABLEAU_COLORS
    colors = [colors[k] for k in colors]

    if ax is not None:
        for ii in range(len(classes)):
            ax.scatter(data[ii,0], data[ii,1], c=colors[int(classes[ii,0])%len(colors)], linewidth=0.5, edgecolor='black')


def knn(data, classes, point, K):
    Ndim = data.shape[1]
    class_arrs = one_hot(classes)

    dist_sqr = torch.zeros(data.shape[0])

    for ii in range(Ndim):
        dist_sqr += (data[:,ii]-point[ii])**2 

    closest_points = [int(x) for x in list(torch.argsort(dist_sqr)[:K])]

    neighbor_classes = torch.sum(class_arrs[closest_points], dim=0)
    return int(neighbor_classes.argmax())



def test_knn_2D(data, classes, K, ax=None):
    Nunique = len(torch.unique(classes))

    xtest = np.linspace(-10, 10, 30)
    ytest = np.linspace(-10, 10, 30)

    colors = mcolors.TABLEAU_COLORS
    colors = [colors[k] for k in colors]
    cmap = plt.get_cmap("tab10")


    img = np.zeros((len(xtest), len(ytest)))

    for ix, x in enumerate(xtest):
        for iy, y in enumerate(ytest):
            c = knn(data, classes, [x, y], K)
            img[iy, ix] = c
    if ax is not None:
        ax.imshow(img, extent=(-10, 10, -10, 10), origin='lower', cmap="tab10")

if __name__ == "__main__":
    fig, ax = plt.subplots()
    data, classes = generate_fake_data(100, 5)
#    knn(data, classes, torch.Tensor([0,0]), 5)
    test_knn_2D(data, classes, 5, ax=ax)
    show_data(data, classes, ax=ax)

    plt.show()
