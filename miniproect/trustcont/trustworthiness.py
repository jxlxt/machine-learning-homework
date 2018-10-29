#
#    Dimensionality Reduction Tools
#    Copyright (C) 2010 Ilja Sidoroff
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    trustcont.py
#
#    Trustworthiness and Continuity error measures for reduced datasets
#
#    See e. g. Jarkko Venna, `Dimensionality Reduction for Visual
#    Exploration of Similarity Structures', Helsinki University of
#    Technology, Dissertations in Computer and Information Science, Espoo
#    2007

import distance
import numpy as np
import matplotlib.pyplot as plt

def trustworthiness(orig, proj, ks):
    """Calculate a trustworthiness values for dataset.
    orig
      matrix containing the data in the original space
    proj
      matrix containing the data in the projected space
    ks range indicating neighbourhood(s) for which
      trustworthiness is calculated.
    Return list of trustworthiness values
    """

    dd_orig = distance.distance_matrix(orig)
    dd_proj = distance.distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_orig = distance.rank_matrix(dd_orig)

    trust = []
    for k in ks:
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_in(nn_orig, nn_proj, i, k))

        trust.append(trustcont_sum(moved, ranks_orig, k))

    return trust

def continuity(orig, proj, ks):
    """Calculate a continuity values for dataset
    orig
      matrix containing the data in the original space
    proj
      matrix containing the data in the projected space
    ks range indicating neighbourhood(s) for which continuity
      is calculated.
    Return a list of continuity values
    """

    dd_orig = distance.distance_matrix(orig)
    dd_proj = distance.distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_proj = distance.rank_matrix(dd_proj)

    cont = []
    for k in ks:
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_out(nn_orig, nn_proj, i, k))

        cont.append(trustcont_sum(moved, ranks_proj, k))

    return cont

def moved_out(nn_orig, nn_proj, i, k):
    """Determine points that were neighbours in the original space,
    but are not neighbours in the projection space.
    nn_orig
      neighbourhood matrix for original data
    nn_proj
      neighbourhood matrix for projection data
    i
      index of the point considered
    k
      size of the neighbourhood considered
    Return a list of indices for 'moved out' values
    """

    oo = list(nn_orig[i, 1:k+1])
    pp = list(nn_proj[i, 1:k+1])

    for j in pp:
        if (j in pp) and (j in oo):
            oo.remove(j)

    return oo

def moved_in(nn_orig, nn_proj, i, k):
    """Determine points that are neighbours in the projection space,
    but were not neighbours in the original space.
    nn_orig
      neighbourhood matrix for original data
    nn_proj
      neighbourhood matrix for projection data
    i
      index of the point considered
    k
      size of the neighbourhood considered
    Return a list of indices for points which are 'moved in' to point i
    """

    pp = list(nn_proj[i, 1:k+1])
    oo = list(nn_orig[i, 1:k+1])

    for j in oo:
        if (j in oo) and (j in pp):
            pp.remove(j)

    return pp


def scaling_term(k, n):
    """Term that scales measure between zero and one
    k  size of the neighbourhood
    n  number of datapoints
    """
    k = k + 1
    if k < (n / 2.0):
        return 2.0 / ((n*k)*(2*n - 3*k - 1))
    else:
        return 2.0 / (n * (n - k) * (n - k - 1))


def trustcont_sum(moved, ranks, k):
    """Calculate sum used in trustworthiness or continuity calculation.
    moved
       List of lists of indices for those datapoints that have either
       moved away in (Continuity) or moved in (Trustworthiness)
       projection
    ranks
       Rank matrix of data set. For trustworthiness, ranking is in the
       original space, for continuity, ranking is in the projected
       space.
    k
       size of the neighbournood
    """

    n = ranks.shape[0]
    s = 0

    # todo: weavefy this for speed
    for i in range(n):
        for j in moved[i]:
            s = s + (ranks[i, j] - k)

    a = scaling_term(k, n)

    return 1 - a * s



x = np.loadtxt('mnist2500_X.txt')
y = np.loadtxt('mnist2500_Result_tsne.txt')
z = np.loadtxt('mnist2500_Result_trimap.txt')
m = np.loadtxt('mnist_Result_largevis.txt')
t_trimap = trustworthiness(x, y, ks=range(50))
t_tsne = trustworthiness(x, z, ks=range(50))
t_largeVis = trustworthiness(x, m, ks=range(50))



# x = [1.0, 0.99577400360504709, 0.99306126920507687, 0.99133739723280534, 0.98988218298555375, 0.98863312587833763, 0.98743527521092811, 0.98645133668341711, 0.98546341288996153, 0.9848244556248742, 0.98373099989016222, 0.98287249647390695, 0.98198805210918116, 0.98118020692238972, 0.9805142995559144, 0.97974256715814989, 0.97898120690475054, 0.97826724188293446, 0.97757379283903811, 0.97697928325571981]
# y = [1.0, 0.99398966553174439, 0.99001213092852369, 0.98579354321235213, 0.98242401284109149, 0.97983700729438539, 0.97797451644378119, 0.976061648241206, 0.97423192991865559, 0.97274178305494063, 0.97142777431992089, 0.97014307206662642, 0.96878704714640196, 0.9677849621026543, 0.96657793567487549, 0.96542328822460111, 0.96431894051072331, 0.9632109470845972, 0.96209640247928607, 0.96116002429641623]
k = range(50)
plt.plot([i for i in k], t_trimap, label = 'TriMap')
plt.plot([i for i in k], t_tsne, label = 't-SNE')
plt.plot([i for i in k], t_largeVis, label = 'LargeVis')
plt.legend()
plt.xlabel('K nearest neighbourhood')
plt.ylabel('Trustworthiness')
plt.title('Compared with t-SNE, LargeVis and TriMap')
plt.show()
