from sksfa import SFA

def run_sfa_on_edges(edges, n_components=2):
    """
    edges: array (T, E). Each column is an edge-time-series feature.
    n_components: how many slow features to extract.
    """
    sfa = SFA(n_components=n_components)
    Y = sfa.fit_transform(edges)        # (T, n_components), slowest first
    deltas = sfa.delta_values_          # mean squared one-step diffs per component
    return Y, deltas, sfa