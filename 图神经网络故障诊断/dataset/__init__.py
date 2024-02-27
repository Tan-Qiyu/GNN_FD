from itertools import combinations
import networkx as nx

def visibility_graph(series):

    g = nx.Graph()
    
    # convert list of magnitudes into list of tuples that hold the index
    tseries = []
    n = 0
    for magnitude in series:
        tseries.append( (n, magnitude ) )
        n += 1

    # contiguous time points always have visibility
    for n in range(0,len(tseries)-1):
        (ta, ya) = tseries[n]
        (tb, yb) = tseries[n+1]
        g.add_node(ta, mag=ya)
        g.add_node(tb, mag=yb)
        g.add_edge(ta, tb)

    for a,b in combinations(tseries, 2):
        # two points, maybe connect
        (ta, ya) = a
        (tb, yb) = b

        connect = True
        
        # let's see all other points in the series
        for tc, yc in tseries[ta:tb]:
            # other points, not a or b
            if tc != ta and tc != tb:
                # does c obstruct?
                if yc > yb + (ya - yb) * ( (tb - tc) / (tb - ta) ):
                    connect = False
                    
        if connect:
            g.add_edge(ta, tb)


    return g
