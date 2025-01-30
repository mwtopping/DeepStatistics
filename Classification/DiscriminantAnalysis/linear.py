import matplotlib.pyplot as plt
import numpy as np



# come up with some fake data of 2D gaussians

def gauss_2d(xs, mu, cov):

    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    return 1/(np.sqrt(6.28*det)) * np.exp( -0.5 * np.dot(np.dot((xs - mu).T , cov_inv) , (xs - mu)))


def generate_data():
    xs = np.array([])
    ys = np.array([])
    outs = np.array([])

    xmeans = [-2, 0, 2.5]
    ymeans = [-2, 3, 0]
    Nsamps = [50, 50, 50]

    for ii, (xmean, ymean, N) in enumerate(zip(xmeans, ymeans, Nsamps)):
        xs = np.append(xs, np.random.normal(loc=xmean, scale=1, size=N) )
        ys = np.append(ys, np.random.normal(loc=ymean, scale=1, size=N))

        outs = np.append(outs, ii+np.zeros(N))
    
    return xs, ys, outs


# testing
def plot_2dgauss():
    xx = np.linspace(-5, 5, 30)
    yy = np.linspace(-5, 5, 30)
    XX, YY = np.meshgrid(xx, yy)

    res = np.zeros_like(XX)

    mu = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])

    for ix, x in enumerate(xx):
        for iy, y in enumerate(yy):
            val = gauss_2d(np.array([x, y]), mu, cov)

            res[iy,ix] = val


    fig, ax = plt.subplots()
    ax.contourf(XX, YY, res)
    plt.show()




def plot_points(xs, ys,outs):
    fig, ax = plt.subplots()
    colors = ["blue", "orange", "pink"]
    for x, y, o in zip(xs, ys, outs):

        ax.scatter(x, y, color=colors[int(o)])


    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
#    plt.show()





def LDA(xs, ys, outs, px, py):

    categories = sorted(list(set(outs)))
    mus = []
    covs = []
    fracs = []



    # calculate
    for c in categories:
        # get data for category c
        inds = np.where(outs == c)[0]
        if len(inds) == 0:
            raise Exception(f"No data for category {c}")
        _xs = xs[inds]
        _ys = ys[inds]
        mus.append([float(np.mean(_xs)), 
                    float(np.mean(_ys))])

        covs.append([[float(np.std(_xs)), 0], 
                     [0, float(np.std(_ys))]])
        fracs.append(float(len(inds)) / len(outs))


    deltas = []
    for ii in range(len(categories)):
        delta = np.dot( np.dot(np.array([px, py]).T, np.linalg.inv(covs[ii])), 
                        np.array(mus[ii])) -  \
                        0.5*np.dot( np.dot(np.array(mus[ii]).T, np.linalg.inv(covs[ii])), 
                        np.array(mus[ii])) + np.log(fracs[ii])
        deltas.append(delta)
    

    return categories[np.argmax(deltas)]


def test_lda(xs, ys, covs):
    xx = np.linspace(-5, 5, 30)
    yy = np.linspace(-5, 5, 30)
    XX, YY = np.meshgrid(xx, yy)

    res = np.zeros_like(XX)

    for ix, x in enumerate(xx):
        for iy, y in enumerate(yy):
            cat = LDA(xs, ys, covs, x, y)
            res[iy,ix] = cat


    fig, ax = plt.subplots()
    ax.contourf(XX, YY, res)




if __name__ == "__main__":

    xs, ys, outs = generate_data()
    plot_points(xs, ys, outs)
    test_lda(xs, ys, outs)


    plt.show()



