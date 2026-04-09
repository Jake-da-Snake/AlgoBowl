"""
AlgoBOWL: Enclose Horse — Optimal Solver
=========================================

Approach: region-expansion search.
Instead of "where do I place walls?", ask "which region should the horse be in?"

A region is valid when:
  - It contains the horse
  - Every passable tile on its boundary is a grass/pre-placed-wall tile (wallable)
  - The boundary size <= wall budget
  - No region tile sits on the grid perimeter

Non-wallable tiles (apple, bee, cherry, portal) adjacent to the region flood in
automatically — you can't wall them, so the horse can always reach them.

Two search strategies are combined:
  BEAM SEARCH   — keeps top-K states ranked by score. Reliable on constrained grids.
  PARETO SEARCH — keeps the Pareto frontier of (score, boundary_size). Much better
                  on large open grids where many diverse region shapes are valid.

Each strategy runs with two tile-ordering heuristics:
  SCORE-GREEDY  — expand the highest-value boundary tile first.
  EFFICIENCY    — prefer tiles that add score without expanding the boundary much.

Usage:
    python optimal_solver.py input.txt              # solve one file
    python optimal_solver.py inputs.zip             # solve all .txt in zip
    python optimal_solver.py input.txt output.txt   # verify an existing output
    python optimal_solver.py --verify inputs.zip outputs/  # batch verify
    python optimal_solver.py --check input.txt      # check input validity only

Outputs go to ./outputs/ (or ~/algobowl/outputs/ if cwd is read-only).
"""

import sys, time, zipfile, os
from collections import deque

# ── Constants ─────────────────────────────────────────────────────────────────
WATER  = '#'
GRASS  = '.'
HORSE  = 'H'
WALL   = 'W'
APPLE  = 'a'
BEES   = 'b'
CHERRY = 'c'
PORTAL = 'p'

MUST_INCLUDE   = {HORSE, APPLE, BEES, CHERRY, PORTAL}  # cannot be walled
BOUNDARY_TYPES = {GRASS, WALL}                          # can form the enclosure wall
PASSABLE_FINAL = {GRASS, HORSE, APPLE, BEES, CHERRY, PORTAL}  # no WALL in final grid

TILE_SCORE = {GRASS: 1, HORSE: 1, APPLE: 11, BEES: -4, CHERRY: 4, PORTAL: 1}
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_input(text):
    lines = [l.rstrip('\n') for l in text.strip().splitlines()]
    idx = 0
    budget = int(lines[idx]); idx += 1
    R, C = map(int, lines[idx].split()); idx += 1
    grid = []
    for r in range(R):
        grid.append(list(lines[idx])); idx += 1
    P = int(lines[idx]); idx += 1
    portals = {}
    for _ in range(P):
        r1, c1, r2, c2 = map(int, lines[idx].split()); idx += 1
        portals[(r1,c1)] = (r2,c2)
        portals[(r2,c2)] = (r1,c1)
    return budget, R, C, grid, portals

def find_horse(grid, R, C):
    for r in range(R):
        for c in range(C):
            if grid[r][c] == HORSE:
                return r, c
    raise ValueError("No horse (H) in grid")

def score_region(region, grid):
    # Pre-placed walls (W) in the region become '.' in output — score as 1
    return sum(
        TILE_SCORE.get(GRASS if grid[r][c] == WALL else grid[r][c], 0)
        for r, c in region
    )


# ── Region primitives ─────────────────────────────────────────────────────────

def flood_must_include(seeds, region, boundary, grid, R, C, portals):
    """
    BFS from seeds: MUST_INCLUDE neighbours join region automatically (can't wall them).
    GRASS/WALL neighbours go to boundary (need explicit walls or expansion).
    Modifies region and boundary in place.
    """
    q = deque(seeds)
    while q:
        r, c = q.popleft()
        if grid[r][c] == PORTAL and (r,c) in portals:
            pr, pc = portals[(r,c)]
            if (pr,pc) not in region:
                region.add((pr,pc)); boundary.discard((pr,pc)); q.append((pr,pc))
        for dr,dc in DIRS:
            nr,nc = r+dr,c+dc
            if not (0<=nr<R and 0<=nc<C): continue
            if (nr,nc) in region or (nr,nc) in boundary: continue
            t = grid[nr][nc]
            if t in MUST_INCLUDE:
                region.add((nr,nc)); q.append((nr,nc))
            elif t in BOUNDARY_TYPES:
                boundary.add((nr,nc))

def initial_state(hr, hc, grid, R, C, portals):
    region = {(hr,hc)}; boundary = set()
    flood_must_include([(hr,hc)], region, boundary, grid, R, C, portals)
    return frozenset(region), frozenset(boundary)

def expand(region, boundary, tile, grid, R, C, portals):
    """Open a boundary tile — include it in region, expose new neighbours."""
    nr = set(region); nb = set(boundary)
    nr.add(tile); nb.discard(tile)
    flood_must_include([tile], nr, nb, grid, R, C, portals)
    return frozenset(nr), frozenset(nb)


# ── Validity ──────────────────────────────────────────────────────────────────

def on_perimeter(region, R, C):
    return any(r==0 or r==R-1 or c==0 or c==C-1 for r,c in region)

def is_valid(region, boundary, budget, R, C):
    return len(boundary) <= budget and not on_perimeter(region, R, C)

def expandable(boundary, R, C):
    """Boundary tiles that aren't on the perimeter (those must stay as walls)."""
    return [t for t in boundary if not(t[0]==0 or t[0]==R-1 or t[1]==0 or t[1]==C-1)]


# ── Tile sort heuristics ──────────────────────────────────────────────────────

def tile_val(r, c, grid):
    """Score contribution of this tile when opened."""
    t = grid[r][c]
    return TILE_SCORE.get(GRASS if t == WALL else t, 0)

def score_greedy_key(t, grid, region, boundary, budget, R, C):
    """Expand highest-value tile first."""
    return tile_val(t[0], t[1], grid) * 10

def efficiency_key(t, grid, region, boundary, budget, R, C):
    """
    Prefer tiles that gain score without growing boundary.
    in_neighbors: count of region tiles adjacent to t — these reduce net boundary growth.
    new_exposed: count of new boundary tiles this expansion would add.
    """
    r, c = t
    in_n  = sum(1 for dr,dc in DIRS if (r+dr,c+dc) in region)
    new_e = sum(1 for dr,dc in DIRS
                if 0<=r+dr<R and 0<=c+dc<C
                and (r+dr,c+dc) not in region
                and (r+dr,c+dc) not in boundary
                and grid[r+dr][c+dc] in BOUNDARY_TYPES)
    return tile_val(r, c, grid) * 3 + in_n * 4 - new_e * 2


# ── Beam search (good for constrained grids) ──────────────────────────────────

def beam_search(budget, R, C, grid, portals,
                beam_width=500, tile_sort_key=None,
                time_limit=5.0, start_time=None,
                record_fn=None):
    """
    Keep top `beam_width` states ranked by score.
    Best for small/constrained grids where the best region is not far from the start.
    """
    if start_time is None:
        start_time = time.time()

    hr, hc = find_horse(grid, R, C)
    init_r, init_b = initial_state(hr, hc, grid, R, C, portals)

    if record_fn and is_valid(init_r, init_b, budget, R, C):
        record_fn(init_r, score_region(init_r, grid))

    beam = [(init_r, init_b)]
    seen = {init_r}

    while beam:
        if time.time() - start_time > time_limit:
            break
        candidates = []

        for region, boundary in beam:
            tiles = expandable(boundary, R, C)
            if tile_sort_key:
                tiles.sort(
                    key=lambda t: tile_sort_key(t, grid, region, boundary, budget, R, C),
                    reverse=True
                )
            for tile in tiles:
                nr, nb = expand(region, boundary, tile, grid, R, C, portals)
                if nr in seen: continue
                seen.add(nr)

                if record_fn and is_valid(nr, nb, budget, R, C):
                    record_fn(nr, score_region(nr, grid))

                b = len(nb)
                on_edge = on_perimeter(nr, R, C)
                s = score_region(nr, grid)

                if b <= budget and not on_edge:
                    pri = (2, s, budget - b)
                else:
                    excess = max(0, b - budget) + (20 if on_edge else 0)
                    pri = (1, s - excess * 5, -b)
                candidates.append((pri, nr, nb))

        if not candidates: break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(r, b) for _, r, b in candidates[:beam_width]]


# ── Pareto search (better for large open grids) ───────────────────────────────

def pareto_search(budget, R, C, grid, portals,
                  max_states=800, tile_sort_key=None,
                  time_limit=5.0, start_time=None,
                  record_fn=None):
    """
    Keep the Pareto frontier of (score, boundary_size) — non-dominated states.
    A state A dominates B if A.score >= B.score AND A.boundary <= B.boundary.
    This naturally preserves both high-score and tight-enclosure solutions.
    Far superior on large grids where many region shapes are budget-feasible.
    """
    if start_time is None:
        start_time = time.time()

    hr, hc = find_horse(grid, R, C)
    init_r, init_b = initial_state(hr, hc, grid, R, C, portals)

    if record_fn and is_valid(init_r, init_b, budget, R, C):
        record_fn(init_r, score_region(init_r, grid))

    beam = [(init_r, init_b)]
    seen = {init_r}

    while beam:
        if time.time() - start_time > time_limit:
            break
        candidates = []  # (score, boundary_size, region, boundary)

        for region, boundary in beam:
            tiles = expandable(boundary, R, C)
            if tile_sort_key:
                tiles.sort(
                    key=lambda t: tile_sort_key(t, grid, region, boundary, budget, R, C),
                    reverse=True
                )
            for tile in tiles:
                nr, nb = expand(region, boundary, tile, grid, R, C, portals)
                if nr in seen: continue
                seen.add(nr)

                on_edge = on_perimeter(nr, R, C)
                if on_edge and len(nb) > budget:
                    continue  # prune: over budget AND touches perimeter — hopeless

                if record_fn and is_valid(nr, nb, budget, R, C):
                    record_fn(nr, score_region(nr, grid))

                s = score_region(nr, grid)
                candidates.append((s, len(nb), nr, nb))

        if not candidates: break

        # Build Pareto frontier: sort by score desc, keep only those with
        # strictly decreasing boundary (each step trades score for tighter enclosure)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        pareto = []; min_b = float('inf')
        for s, b, r, bnd in candidates:
            if b <= min_b:
                pareto.append((r, bnd)); min_b = b

        # Cap to avoid state explosion
        if len(pareto) > max_states:
            # Weighted sample: half by score, half by boundary tightness
            by_s = sorted(candidates, key=lambda x: -x[0])
            by_b = sorted(candidates, key=lambda x:  x[1])
            seen_id = set(); out = []
            for _, _, r, b in by_s[:max_states//2] + by_b[:max_states//2]:
                if id(r) not in seen_id:
                    seen_id.add(id(r)); out.append((r,b))
            pareto = out

        beam = pareto


# ── Multi-strategy solver ─────────────────────────────────────────────────────

def solve(budget, R, C, grid, portals, time_limit=20.0):
    """
    Run 4 strategies and return the best (region, score) found.

    Strategy 1 — Score-greedy beam:
        Expand highest-value tile first. Fast and reliable on constrained grids.
    Strategy 2 — Efficiency beam:
        Prefer tiles that add score without growing the boundary. Better when the
        budget is tight relative to the reachable area.
    Strategy 3 — Score-greedy Pareto:
        Same tile ordering as S1 but keeps the Pareto frontier of states.
        Finds much better solutions on large, open grids.
    Strategy 4 — Efficiency Pareto:
        Combines efficiency tile ordering with Pareto state selection.
        Complementary to S3.
    """
    start = time.time()
    per   = time_limit / 4

    best_score  = None
    best_region = None
    winner      = None

    def record(label):
        def fn(region, score):
            nonlocal best_score, best_region, winner
            if best_score is None or score > best_score:
                best_score = score; best_region = region; winner = label
                print(f"    [{label}] score: {score}")
        return fn

    print("  Strategy 1: score-greedy beam...")
    beam_search(budget, R, C, grid, portals,
                beam_width=500, tile_sort_key=score_greedy_key,
                time_limit=per, start_time=start,
                record_fn=record("score-greedy beam"))

    print("  Strategy 2: efficiency beam...")
    beam_search(budget, R, C, grid, portals,
                beam_width=500, tile_sort_key=efficiency_key,
                time_limit=per, start_time=time.time(),
                record_fn=record("efficiency beam"))

    print("  Strategy 3: score-greedy Pareto...")
    pareto_search(budget, R, C, grid, portals,
                  max_states=800, tile_sort_key=score_greedy_key,
                  time_limit=per, start_time=time.time(),
                  record_fn=record("score-greedy Pareto"))

    print("  Strategy 4: efficiency Pareto...")
    pareto_search(budget, R, C, grid, portals,
                  max_states=800, tile_sort_key=efficiency_key,
                  time_limit=per, start_time=time.time(),
                  record_fn=record("efficiency Pareto"))

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s | best: {best_score} (winner: {winner})")
    return best_region, best_score


# ── Grid reconstruction ───────────────────────────────────────────────────────

def region_to_grid(best_region, orig_grid, R, C):
    sol = [list(row) for row in orig_grid]
    rs = set(best_region)

    # Tiles inside region: open pre-placed walls back to grass
    for r,c in rs:
        if orig_grid[r][c] == WALL:
            sol[r][c] = GRASS

    # Grass tiles adjacent to region: become walls
    for r in range(R):
        for c in range(C):
            if (r,c) not in rs and orig_grid[r][c] == GRASS:
                if any(0<=r+dr<R and 0<=c+dc<C and (r+dr,c+dc) in rs
                       for dr,dc in DIRS):
                    sol[r][c] = WALL

    # Pre-placed walls not adjacent to region: remove (they waste budget)
    for r in range(R):
        for c in range(C):
            if (r,c) not in rs and orig_grid[r][c] == WALL:
                if not any(0<=r+dr<R and 0<=c+dc<C and (r+dr,c+dc) in rs
                           for dr,dc in DIRS):
                    sol[r][c] = GRASS

    return sol


# ── Verifier ──────────────────────────────────────────────────────────────────

def full_reachable(grid, R, C, portals, hr, hc):
    visited = set(); q = deque([(hr,hc)]); visited.add((hr,hc)); perim = False
    while q:
        r,c = q.popleft()
        if r==0 or r==R-1 or c==0 or c==C-1: perim = True
        if grid[r][c]==PORTAL and (r,c) in portals:
            pr,pc = portals[(r,c)]
            if (pr,pc) not in visited and grid[pr][pc] in PASSABLE_FINAL:
                visited.add((pr,pc)); q.append((pr,pc))
        for dr,dc in DIRS:
            nr,nc = r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc] in PASSABLE_FINAL:
                visited.add((nr,nc)); q.append((nr,nc))
    return visited, perim


def verify(budget, R, C, orig_grid, portals, sol_grid, claimed_score, verbose=True):
    """Returns list of error strings. Empty = valid."""
    errors = []
    total_walls = sum(sol_grid[r][c]==WALL for r in range(R) for c in range(C))
    if total_walls > budget:
        errors.append(f"FAIL: {total_walls} walls in solution, budget is {budget}")

    illegal = {APPLE, BEES, CHERRY, PORTAL}
    for r in range(R):
        for c in range(C):
            if sol_grid[r][c]==WALL and orig_grid[r][c] in illegal:
                errors.append(f"FAIL: wall on '{orig_grid[r][c]}' at ({r},{c})")

    try: hr,hc = find_horse(sol_grid, R, C)
    except ValueError:
        errors.append("FAIL: horse missing"); return errors

    region, perim = full_reachable(sol_grid, R, C, portals, hr, hc)
    if perim: errors.append("FAIL: horse can reach the perimeter")

    actual = sum(TILE_SCORE.get(sol_grid[r][c],0) for r,c in region)
    if actual != claimed_score:
        errors.append(f"FAIL: claimed {claimed_score}, actual is {actual}")

    if not errors and verbose:
        new_w = sum(sol_grid[r][c]==WALL and orig_grid[r][c]!=WALL
                    for r in range(R) for c in range(C))
        print(f"  ✓ VALID | score: {actual} | "
              f"walls: {total_walls}/{budget} ({new_w} new + {total_walls-new_w} kept)")
    return errors


# ── Per-input pipeline ────────────────────────────────────────────────────────

def get_out_dir(hint_dir):
    d = os.path.join(hint_dir, 'outputs')
    try:
        os.makedirs(d, exist_ok=True)
        # Test write
        test = os.path.join(d, '.write_test')
        open(test,'w').close(); os.remove(test)
        return d
    except OSError:
        fallback = os.path.join(os.path.expanduser('~'), 'algobowl', 'outputs')
        os.makedirs(fallback, exist_ok=True)
        return fallback


def solve_input(name, text, out_dir, time_limit=20.0):
    """Parse, solve, verify, write. Returns (score, valid, elapsed)."""
    t0 = time.time()
    try:
        budget, R, C, grid, portals = parse_input(text)
    except Exception as e:
        print(f"  [PARSE ERROR] {e}"); return None, False, 0

    hr, hc = find_horse(grid, R, C)
    _, horse_free = full_reachable(grid, R, C, portals, hr, hc)

    print(f"\n{'─'*56}")
    print(f"  {name}  [{R}×{C}  budget={budget}  portals={len(portals)//2}]")
    if not horse_free:
        print("  NOTE: horse already enclosed before any walls")

    best_region, best_score = solve(budget, R, C, grid, portals, time_limit=time_limit)

    if best_region is None:
        print("  ERROR: no valid enclosure found"); return None, False, time.time()-t0

    sol_grid = region_to_grid(best_region, grid, R, C)
    errors   = verify(budget, R, C, grid, portals, sol_grid, best_score)
    for e in errors: print(f"  {e}")

    if not errors:
        out_text = str(best_score) + '\n' + '\n'.join(''.join(row) for row in sol_grid)
        out_path = os.path.join(out_dir, name.replace('.txt', '_solution.txt'))
        with open(out_path, 'w') as f:
            f.write(out_text + '\n')
        print(f"  → {out_path}")

    return best_score, not errors, time.time()-t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args: print(__doc__); sys.exit(0)

    # ── Batch verify mode ─────────────────────────────────────────────────────
    if '--verify' in args:
        args.remove('--verify')
        if len(args) < 2:
            print("Usage: optimal_solver.py --verify inputs.zip outputs/"); sys.exit(1)
        zip_path, sol_dir = args[0], args[1]
        print(f"\nBatch verification: {zip_path} vs {sol_dir}/")
        print(f"{'─'*56}")
        results = []
        with zipfile.ZipFile(zip_path) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith('.txt')):
                base      = os.path.basename(name)
                sol_name  = base.replace('.txt', '_solution.txt')
                sol_path  = os.path.join(sol_dir, sol_name)
                if not os.path.exists(sol_path):
                    print(f"  ✗ {base:<35} MISSING solution"); results.append((base,None,False)); continue
                input_text = zf.read(name).decode('utf-8')
                budget,R,C,grid,portals = parse_input(input_text)
                lines = open(sol_path).read().strip().splitlines()
                claimed = int(lines[0]); sol_grid = [list(l) for l in lines[1:]]
                errors = verify(budget,R,C,grid,portals,sol_grid,claimed,verbose=False)
                if errors:
                    score = int(lines[0])
                    print(f"  ✗ {base:<35} score={score} — {errors[0]}")
                    results.append((base,score,False))
                else:
                    score = int(lines[0])
                    total_w = sum(sol_grid[r][c]==WALL for r in range(R) for c in range(C))
                    print(f"  ✓ {base:<35} score={score}  walls={total_w}/{budget}")
                    results.append((base,score,True))
        print(f"{'─'*56}")
        valid = sum(1 for _,_,v in results if v)
        total = sum(s for _,s,v in results if v and s)
        print(f"  Valid: {valid}/{len(results)}   Total score: {total}")
        sys.exit(0)

    check_only  = '--check' in args
    args        = [a for a in args if a != '--check']
    source      = args[0]
    verify_file = args[1] if len(args) > 1 else None

    # ── ZIP solve mode ────────────────────────────────────────────────────────
    if source.endswith('.zip'):
        print(f"Opening {source}")
        out_dir = get_out_dir(os.path.dirname(os.path.abspath(source)))
        results = []
        with zipfile.ZipFile(source) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith('.txt'))
            print(f"Found {len(names)} inputs → outputs to {out_dir}/")
            for name in names:
                text = zf.read(name).decode('utf-8')
                base = os.path.basename(name)
                score, valid, elapsed = solve_input(base, text, out_dir)
                results.append((base, score, valid, elapsed))
        print(f"\n{'='*56}")
        print(f"  SUMMARY")
        print(f"{'─'*56}")
        total_score = 0
        for name, score, valid, elapsed in results:
            s = score if score is not None else 'ERR'
            ok = '✓' if valid else '✗'
            print(f"  {ok}  {name:<35}  score={s!s:<8}  {elapsed:.0f}s")
            if score and valid: total_score += score
        print(f"{'─'*56}")
        print(f"  Total score: {total_score}   "
              f"Valid: {sum(1 for _,_,v,_ in results if v)}/{len(results)}")
        sys.exit(0)

    # ── Single file ───────────────────────────────────────────────────────────
    with open(source) as f:
        text = f.read()
    budget, R, C, grid, portals = parse_input(text)

    if verify_file:
        lines = open(verify_file).read().strip().splitlines()
        claimed = int(lines[0]); sol_grid = [list(l) for l in lines[1:]]
        print(f"\nVerifying {verify_file}...")
        for e in verify(budget,R,C,grid,portals,sol_grid,claimed):
            print(f"  {e}")
        sys.exit(0)

    if check_only:
        hr,hc = find_horse(grid,R,C)
        _,free = full_reachable(grid,R,C,portals,hr,hc)
        print(f"Horse {'CAN' if free else 'CANNOT'} reach perimeter"); sys.exit(0)

    out_dir = get_out_dir(os.path.dirname(os.path.abspath(source)))
    solve_input(os.path.basename(source), text, out_dir)


if __name__ == '__main__':
    main()