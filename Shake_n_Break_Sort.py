import random
import math
from collections import deque

#Breakpoint scoring utilities

def is_break(a, i):
    """
    Return 1 if the adjacent pair (a[i], a[i+1]) is inverted, else 0.
    This is the smallest scoring primitive.
    """
    return 1 if a[i] > a[i + 1] else 0

def count_breaks_in_range(a, left_i, right_i):
    """
    Count breakpoints for indices i in [left_i, right_i] inclusive.
    Important:
    - Breakpoints are defined on pairs (i, i+1), so the last valid i is n-2.
    - We clamp the requested range so we never go out of bounds.
    - Time complexity is O(right-left+1), which is why we try to keep
      left/right small (local updates only).
    """
    if not a or len(a) < 2:
        #With <2 elements you cannot have adjacent inversions
        return 0

    n = len(a)
    
    #Clamp to valid breakpoint index range
    left_i = max(0, left_i)
    right_i = min(n - 2, right_i)
    
    #If the requested range is empty after clamping answer is 0
    if left_i > right_i:
        return 0

    total = 0
    for i in range(left_i, right_i + 1):
        total += is_break(a, i)
    return total



def breaks_for_chunk(left_val, chunk, right_val):
    """
    Count breakpoints for a candidate window without writing into the main array.

    This matches count_breaks_in_range(a, start-1, end-1) where `chunk` == a[start:end],
    with optional boundaries:
      - left boundary compares a[start-1] vs chunk[0]
      - right boundary compares chunk[-1] vs a[end]
    """
    m = len(chunk)
    if m < 2:
        #Only possible breaks are boundaries
        b = 0
        if m == 1:
            if left_val is not None and left_val > chunk[0]:
                b += 1
            if right_val is not None and chunk[0] > right_val:
                b += 1
        return b

    b = 0
    #internal breaks
    prev = chunk[0]
    for j in range(1, m):
        cur = chunk[j]
        if prev > cur:
            b += 1
        prev = cur

    #boundaries
    if left_val is not None and left_val > chunk[0]:
        b += 1
    if right_val is not None and chunk[-1] > right_val:
        b += 1
    return b

def breakpoint_indices(a):
    """
    Return a list of all breakpoint indices.

    This is more expensive than sampling, but useful:
    - For fallback when sampling doesn't find a breakpoint
    - For endgame deterministic cleanup
    - For debugging / inspection
    Return all indices i where a[i] > a[i+1].
    """
    return [i for i in range(len(a) - 1) if a[i] > a[i + 1]]


def finish_insertion(a):
    """
    Endgame finisher: insertion sort.
    Very fast when the list is nearly sorted (small breakpoint score).
    """
    for i in range(1, len(a)):
        if a[i - 1] <= a[i]:
            continue
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def total_breakpoints(a):
    """
    Count total breakpoints across the entire array.

    We avoid calling this every step because it's O(n).
    Instead, we mostly update the score using local deltas.
    """
    return count_breaks_in_range(a, 0, len(a) - 2)


def dynamic_batch_bogo_sort(
    arr,
    max_steps=300_000,
    on_step=None,      #visualization/log callback: on_step(a, score, start, end, step, accepted, shook)
    viz_every=1
):
    """
    Sort arr using breakpoint-guided stochastic optimization.

    The algorithm minimizes "breakpoints" (adjacent inversions where a[i] > a[i+1]).
    A fully sorted array has zero breakpoints.

    Strategy:
    1. Find a breakpoint to target (via sampling)
    2. Build a window around it
    3. Try deterministic improvements (sort window, swap pair, rotate triple)
    4. Try random shuffles of the window
    5. Accept improvements; occasionally accept worse moves (simulated annealing)
    6. When nearly sorted, finish with insertion sort

    Args:
        arr: Input sequence to sort
        max_steps: Maximum optimization steps before returning
        on_step: Optional callback for visualization (a, score, start, end, step, accepted, shook)
        viz_every: Call on_step every N steps

    Returns:
        Sorted list (or best-effort if max_steps exceeded)
    """

    a = list(arr) #work on a copy so caller isn't mutated
    n = len(a)

    #Trivial cases
    if n < 2:
        return a

    #Initial global score (O(n) but only done once here)
    score = total_breakpoints(a)
    if score == 0:
        return a

    #Track how long weve gone without improvement (for shake logic)
    no_progress = 0

    #Track score history for trend 
    score_hist = deque(maxlen=300)  #recent scores for trend
    score_hist.append(score)    
    best_ever = score #best score seen overall 

    step = 0
    while step < max_steps:
        step += 1

        #------------------------------------------------------------
        #1) Pick a breakpoint index to target
        #------------------------------------------------------------
        #First try cheap sampling: fallback to full scan if needed.
        best_k = None
        best_gap = -1

        #Sample random indices to find breakpoints - cheaper than full O(n) scan
        #Scale samples with n: small arrays need fewer, large arrays benefit from more
        samples = max(20, min(120, n // 25))
        for _ in range(samples):
            k = random.randrange(n - 1)
            if a[k] > a[k + 1]:
                gap = a[k] - a[k + 1]
                if gap > best_gap:
                    best_gap = gap
                    best_k = k

        if best_k is None:
            bad = breakpoint_indices(a)
            if not bad:
                return a
            i = random.choice(bad)
        else:
            i = best_k

        #------------------------------------------------------------
        #2) Dynamic controller - adapts parameters based on current state
        #------------------------------------------------------------
        #score_ratio is 0..1: how unsorted we are, relative to max possible (n-1)
        score_ratio = score / max(1, (n - 1))

        #stuck_ratio is 0..1-ish: how long weve stagnated compared to n
        stuck_ratio = min(1.0, no_progress / max(1, n))

        # --- Window size (fraction of n) ---
        #If very unsorted => bigger windows
        #If stuck => bigger windows
        #If nearly sorted => *sometimes* bigger windows again to fix long range misplacements
        base_frac   = 0.02 + 0.25 * score_ratio
        stuck_boost = 0.30 * stuck_ratio
        late_boost  = 0.20 * (1.0 - score_ratio) if score_ratio < 0.05 else 0.0
        window_frac = min(0.90, base_frac + stuck_boost + late_boost)

        batch_size = max(2, int(window_frac * n))

        # --- Tries per step ---
        #More tries when very unsorted or stuck
        #Scales with batch_size so big windows get more effort
        base_budget = int(40 + 500 * score_ratio + 400 * stuck_ratio)
        tries = max(10, int(base_budget * (batch_size / n)))

        #Cap tries so insane sizes dont freeze your CPU
        tries = min(tries, 500)

        # --- Annealing temperature ---
        #Higher when unsorted or stuck => accepts "worse" moves more often
        T = 1e-6 + 0.05 * score_ratio + 0.25 * stuck_ratio

        # --- Shake threshold ---
        #When very unsorted shake sooner
        #When nearly sorted allow more tries before shaking
        shake_after_dyn = int(0.5 * n + 5.0 * n * (1.0 - score_ratio))

        #------------------------------------------------------------
        #3) Build window bounds [start:end)
        #------------------------------------------------------------
        start = max(0, min(i, n - batch_size))
        end = start + batch_size

        #Key optimization: only the windows internal pairs + two boundary pairs affect the score
        #This allows O(window_size) scoring instead of O(n) per candidate
        
        original_chunk = a[start:end]
        left_val  = a[start - 1] if start > 0 else None
        right_val = a[end]       if end < n else None
        local_old = breaks_for_chunk(left_val, original_chunk, right_val)

        best_chunk = original_chunk[:]  #best window arrangement found this step
        best_score = score              #best global score found this step

        #------------------------------------------------------------
        #4) Deterministic candidate attempts (cheap "smart moves")
        #------------------------------------------------------------

        #Candidate A: fully sort the window (often helps)
        cand = sorted(original_chunk)
        if cand != original_chunk:
            local_new = breaks_for_chunk(left_val, cand, right_val)
            cand_global = score - local_old + local_new
            if cand_global < best_score:
                best_score = cand_global
                best_chunk = cand

        #Candidate B: swap the actual breakpoint pair
        li = i - start
        if 0 <= li < len(original_chunk) - 1:
            cand2 = original_chunk[:]
            cand2[li], cand2[li + 1] = cand2[li + 1], cand2[li]
            local_new = breaks_for_chunk(left_val, cand2, right_val)
            cand_global = score - local_old + local_new
            if cand_global < best_score:
                best_score = cand_global
                best_chunk = cand2

        #Candidate C: rotate triples around breakpoint
        for off in (-1, 0):
            j = li + off
            if 0 <= j and j + 2 < len(original_chunk):
                cand3 = original_chunk[:]
                x, y, z = cand3[j], cand3[j + 1], cand3[j + 2]
                cand3[j], cand3[j + 1], cand3[j + 2] = y, z, x
                local_new = breaks_for_chunk(left_val, cand3, right_val)
                cand_global = score - local_old + local_new
                if cand_global < best_score:
                    best_score = cand_global
                    best_chunk = cand3

        #------------------------------------------------------------
        #5) Stochastic search: try random permutations of the window
        #------------------------------------------------------------
        for _ in range(tries):
            cand = original_chunk[:]
            random.shuffle(cand)
            if cand == original_chunk:
                continue

            #Score candidate in-place without modifying the main array
            local_new = breaks_for_chunk(left_val, cand, right_val)
            cand_global = score - local_old + local_new

            if cand_global < best_score:
                best_score = cand_global
                best_chunk = cand

        #------------------------------------------------------------
        #6) Acceptance decision (dynamic annealing)
        #------------------------------------------------------------
        delta = best_score - score  # <0 better, 0 equal, >0 worse

        if delta < 0:
            accept = True
        elif delta == 0:
            #accept some equals accept more if stuck
            accept = (random.random() < (0.02 + 0.10 * stuck_ratio))
        else:
            #accept worse moves with annealing probability
            accept = (random.random() < math.exp(-delta / T))

        accepted = False
        shook = False

        if accept:
            #Commit best chunk found (only write once)
            a[start:end] = best_chunk

            #best_score already accounts for the local delta
            score = max(0, best_score)
            accepted = True

            #Endgame: when <1% of pairs are inverted switch to insertion sort
            #(insertion sort is O(n) on nearly-sorted data)
            if (score / max(1, n - 1)) < 0.01:
                finish_insertion(a)
                score = total_breakpoints(a)

                #insertion touched the whole list; force full redraw if visualizing
                if on_step is not None:
                    on_step(a, score, 0, n, step, True, False)

                if score == 0:
                    return a

            #stagnation tracking
            if delta < 0:
                no_progress = 0
            else:
                no_progress += 1

            if score == 0:
                if on_step is not None:
                    on_step(a, score, start, end, step, True, False)
                return a
        else:
            no_progress += 1

        #------------------------------------------------------------
        #7) Dynamic shake if stuck (perturb inside window)
        #------------------------------------------------------------
        if no_progress >= shake_after_dyn:
            chunk_before = a[start:end]
            local_before = breaks_for_chunk(left_val, chunk_before, right_val)

            #number of random swaps grows with stuck_ratio
            swaps = int(5 + 60 * stuck_ratio)
            for _ in range(swaps):
                x = random.randrange(start, end)
                y = random.randrange(start, end)
                a[x], a[y] = a[y], a[x]

            chunk_after = a[start:end]
            local_after = breaks_for_chunk(left_val, chunk_after, right_val)
            score = max(0, score - local_before + local_after)

            no_progress = 0
            shook = True

        #------------------------------------------------------------
        #8) Bookkeeping + visualization callback
        #------------------------------------------------------------
        score_hist.append(score)
        best_ever = min(best_ever, score)

        if on_step is not None and (step % viz_every == 0):
            on_step(a, score, start, end, step, accepted, shook)

    #If we hit max_steps return the current state (not guaranteed best-ever)
    return a

    
__all__ = [
    'dynamic_batch_bogo_sort',
    'total_breakpoints',
    'count_breaks_in_range',
    'breaks_for_chunk',
    'breakpoint_indices',
    'finish_insertion',
    'is_break',
]
