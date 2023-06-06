from oneshot import MLPoly, get_term_table, rosenberg_criterion, Rosenberg, full_Rosenberg
from ising import IMul
from polyfit import fit


cache_miss = 0
cache_hit = 0

def relevant_info(poly):
    cur = poly.num_variables()
    return (cur,frozenset([(key,value) for key, value in poly.coeffs.items() if len(key) > 2]))

def get_cache(poly, cache):
    global cache_miss, cache_hit

    keyset = relevant_info(poly)
    cur = keyset[0]
    dat = keyset[1]
    for i in range(12, cur+1):
        prev = (i, dat) 
        if prev in cache:
            cache_hit += 1
            return cache[prev]

    cache_miss += 1
    return None

def set_cache(poly, val, cache):
    keyset = relevant_info(poly)
    #print(f'{val - poly.num_variables()} {keyset}')
    cache[keyset] = val - poly.num_variables()


def tree_search(poly: MLPoly, ub, cache) -> int:

    cached_result = get_cache(poly, cache)
    if cached_result is not None:
        #print('cached')
        return poly.num_variables() + cached_result, cache

    if poly.degree() == 2:
        print(poly.num_variables())
        return poly.num_variables(), cache

    if poly.num_variables() >= ub - 1:
        return ub, cache


    candidates = get_term_table(poly, criterion = rosenberg_criterion, size = 2).items()
    candidates = sorted(candidates, key = lambda pair: -len(pair[1]))

    for C, H in candidates[:5]:
        if not len(H):
            continue

        new_poly = Rosenberg(poly, C, H)
        val, cache = tree_search(new_poly, ub, cache)
        ub = min(ub, val)

    set_cache(poly, ub, cache)
    
    return ub, cache

def experiment():
    circuit = IMul(3,3)
    poly = fit(circuit, 3)
    print(poly)

    baseline = full_Rosenberg(poly).num_variables()
    print(f'baseline is {baseline}')
    
    optimal, cache = tree_search(poly, baseline, {})
    print(f'optimal is {optimal}')

    print(f'cache stats: hit = {cache_hit} miss = {cache_miss}')

experiment()
