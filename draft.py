from scipy.sparse.csgraph import minimum_spanning_tree


# assume N cameras


# Ec = 

def CycleCost(G, T, e):
    '''
    Given adj matrix T representing a spanning Tree and edge,
    find the cycle

    Alternatively i think we can compute the weights along the cycle directly without returning the ccle
    '''

    # should we precompute the unique path between every 2 vertices in T first?
    # that will make computing the weight of C = e Union T easier 

eps = 1

def removeOutliers(G):
    # G = np.zeros(N)
    # as we want maximum spanning tree, multiply by -1 inside and outside
    T = -minimum_spanning_tree(-G)
    Ec = T
    # how to get elements of G \ T?
    for e in G - T:
        if cycleCost(G, T, E) < sqrt(|C|) * eps:
            # Ec = T union e
            # Ec[e] = 1
    return Ec



# basically we just want to get rid of outlier views
# (is this even necessary, does our data set contain a lot of outliers?)

# Compute a maximum spanning tree, T , with weights wij.
# Set Ec = T .
# for each e 2 E n T
# Let C be the cycle formed by e and T .
# if the error in C is less than pjCj
# Ec = Ec [ e
# Estimate absolute orientations from Ec (see Section 4.1).
# Apply additional search heuristics (see text)

