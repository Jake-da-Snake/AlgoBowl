"""
AlgoBOWL: Enclose Horse — Optimal Solver
=========================================

Usage:
    python optimal_solver.py input.txt              # solve one file
    python optimal_solver.py inputs.zip             # solve all .txt files in zip
    python optimal_solver.py input.txt output.txt   # verify an output
    python optimal_solver.py --check input.txt      # check input validity only

Output files are written alongside inputs as <name>_solution.txt
"""

import sys, time, zipfile, os
from collections import deque

# ── Tile constants ────────────────────────────────────────────────────────────
WATER  = '#'
GRASS  = '.'
HORSE  = 'H'
WALL   = 'W'
APPLE  = 'a'
BEES   = 'b'
CHERRY = 'c'
PORTAL = 'p'

# Tiles the horse can move through
PASSABLE = {GRASS, HORSE, APPLE, BEES, CHERRY, PORTAL, WALL}

# These cannot have new walls placed on them → flood into region automatically
# (WALL is excluded: pre-placed walls are passable but go to boundary, not auto-flood)
MUST_INCLUDE = {HORSE, APPLE, BEES, CHERRY, PORTAL}

# Tiles that form the enclosure boundary (can be walled or opened)
BOUNDARY_TYPES = {GRASS, WALL}

TILE_SCORE = {
    GRASS: 1, HORSE: 1, APPLE: 11, BEES: -4, CHERRY: 4, PORTAL: 1, WALL: 0,
}

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
        portals[(r1, c1)] = (r2, c2)
        portals[(r2, c2)] = (r1, c1)
    return budget, R, C, grid, portals

def find_horse(grid, R, C):
    for r in range(R):
        for c in range(C):
            if grid[r][c] == HORSE:
                return r, c
    raise ValueError("No horse (H) found in grid")

def score_region(region, grid):
    # Pre-placed walls in the region become grass in the solution — score as 1
    return sum(
        TILE_SCORE.get(GRASS if grid[r][c]==WALL else grid[r][c], 0)
        for r, c in region
    )


# ── Region primitives ─────────────────────────────────────────────────────────

def flood_must_include(seeds, region, boundary, grid, R, C, portals):
    """
    From newly added seeds, flood MUST_INCLUDE neighbours automatically.
    GRASS and WALL neighbours go to boundary (they need explicit walls or expansion).
    Modifies region and boundary in place.
    """
    queue = deque(seeds)
    while queue:
        r, c = queue.popleft()
        # Portal: paired destination floods in automatically
        if grid[r][c] == PORTAL and (r, c) in portals:
            pr, pc = portals[(r, c)]
            if (pr, pc) not in region:
                region.add((pr, pc))
                boundary.discard((pr, pc))
                queue.append((pr, pc))
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if (nr, nc) in region or (nr, nc) in boundary:
                continue
            tile = grid[nr][nc]
            if tile in MUST_INCLUDE:
                # Non-wallable: must join region
                region.add((nr, nc))
                queue.append((nr, nc))
            elif tile in BOUNDARY_TYPES:
                # Grass or pre-placed wall: goes to boundary (costs 1 wall each)
                boundary.add((nr, nc))
            # Water / other: impassable, ignored


def initial_state(hr, hc, grid, R, C, portals):
    """Build starting (region, boundary) from horse position."""
    region = {(hr, hc)}
    boundary = set()
    flood_must_include([(hr, hc)], region, boundary, grid, R, C, portals)
    return frozenset(region), frozenset(boundary)


def expand(region, boundary, tile, grid, R, C, portals):
    """
    Open boundary tile `tile` (include it in region instead of walling it).
    MUST_INCLUDE neighbours of the new tile flood in automatically.
    New grass/wall neighbours go to boundary.
    Returns (new_region, new_boundary) as frozensets.
    """
    new_region = set(region)
    new_boundary = set(boundary)
    new_region.add(tile)
    new_boundary.discard(tile)
    flood_must_include([tile], new_region, new_boundary, grid, R, C, portals)
    return frozenset(new_region), frozenset(new_boundary)


# ── Validity and scoring ──────────────────────────────────────────────────────

def is_valid(region, boundary, budget, R, C):
    """
    Valid enclosure:
    1. Total boundary cost ≤ budget
       (each boundary tile, whether grass or pre-placed wall, needs 1 wall)
    2. No region tile on the perimeter (horse can't reach grid edge)
    """
    if len(boundary) > budget:
        return False
    return not any(r == 0 or r == R-1 or c == 0 or c == C-1 for r, c in region)


# ── Beam search ───────────────────────────────────────────────────────────────

def beam_search(budget, R, C, grid, portals,
                beam_width=500, tile_sort_key=None,
                time_limit=8.0, start_time=None):
    """
    Grow enclosed regions one boundary tile at a time.
    Never expand perimeter grass/wall tiles (they must remain as walls).
    Track and return the best valid enclosure found.
    """
    if start_time is None:
        start_time = time.time()

    hr, hc = find_horse(grid, R, C)
    init_region, init_boundary = initial_state(hr, hc, grid, R, C, portals)

    best_score = None
    best_region = None

    def record(region, boundary):
        nonlocal best_score, best_region
        if is_valid(region, boundary, budget, R, C):
            s = score_region(region, grid)
            if best_score is None or s > best_score:
                best_score = s
                best_region = region

    record(init_region, init_boundary)

    beam = [(init_region, init_boundary)]
    seen = {init_region}

    while beam:
        if time.time() - start_time > time_limit:
            break

        next_candidates = []

        for region, boundary in beam:
            # Only expand non-perimeter boundary tiles
            # (perimeter tiles must stay as walls)
            tiles = [
                t for t in boundary
                if not (t[0] == 0 or t[0] == R-1 or t[1] == 0 or t[1] == C-1)
            ]

            if tile_sort_key is not None:
                tiles.sort(
                    key=lambda t: tile_sort_key(t[0], t[1], grid, region, boundary, budget, R, C),
                    reverse=True
                )

            for tile in tiles:
                new_region, new_boundary = expand(region, boundary, tile, grid, R, C, portals)
                if new_region in seen:
                    continue
                seen.add(new_region)
                record(new_region, new_boundary)

                b = len(new_boundary)
                on_edge = any(r2==0 or r2==R-1 or c2==0 or c2==C-1 for r2,c2 in new_region)
                s = score_region(new_region, grid)

                if b <= budget and not on_edge:
                    pri = (2, s, budget - b)  # valid: rank by score
                else:
                    excess = max(0, b - budget) + (20 if on_edge else 0)
                    pri = (1, s - excess * 5, -b)  # invalid: penalize

                next_candidates.append((pri, new_region, new_boundary))

        if not next_candidates:
            break

        next_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(r, b) for _, r, b in next_candidates[:beam_width]]

    return best_region, best_score


# ── Multi-strategy solver ─────────────────────────────────────────────────────

def solve(budget, R, C, grid, portals, time_limit=20.0, verbose=True):
    """
    Run 4 beam search strategies. Return best region and score found.
    """
    start = time.time()
    per   = time_limit / 4
    best_score = None
    best_region = None

    def update(region, score, label):
        nonlocal best_score, best_region
        if region is not None and (best_score is None or score > best_score):
            best_score = score
            best_region = region
            if verbose:
                print(f"    [{label}] score: {score}")

    # Strategy 1: Score-greedy — chase apples/cherries first
    if verbose: print("  Strategy 1: score-greedy...")
    def score_key(r, c, grid, region, boundary, budget, R, C):
        return TILE_SCORE.get(grid[r][c], 0) * 10
    update(*beam_search(budget, R, C, grid, portals, beam_width=400,
                        tile_sort_key=score_key, time_limit=per, start_time=start),
           "score-greedy")

    # Strategy 2: Efficiency — prefer tiles that don't grow boundary much
    if verbose: print("  Strategy 2: efficiency-greedy...")
    def eff_key(r, c, grid, region, boundary, budget, R, C):
        in_nbrs = sum(1 for dr,dc in DIRS if (r+dr,c+dc) in region)
        new_exp = sum(
            1 for dr,dc in DIRS
            if 0<=r+dr<R and 0<=c+dc<C
            and (r+dr,c+dc) not in region and (r+dr,c+dc) not in boundary
            and grid[r+dr][c+dc] in BOUNDARY_TYPES
        )
        return TILE_SCORE.get(grid[r][c], 0) * 3 + in_nbrs * 4 - new_exp * 2
    update(*beam_search(budget, R, C, grid, portals, beam_width=400,
                        tile_sort_key=eff_key, time_limit=per, start_time=time.time()),
           "efficiency-greedy")

    # Strategy 3: Compact — minimise boundary growth (tight enclosures)
    if verbose: print("  Strategy 3: compact...")
    def compact_key(r, c, grid, region, boundary, budget, R, C):
        in_nbrs = sum(1 for dr,dc in DIRS if (r+dr,c+dc) in region)
        return in_nbrs * 5 + TILE_SCORE.get(grid[r][c], 0)
    update(*beam_search(budget, R, C, grid, portals, beam_width=400,
                        tile_sort_key=compact_key, time_limit=per, start_time=time.time()),
           "compact")

    # Strategy 4: Wide beam — broad unordered exploration
    if verbose: print("  Strategy 4: wide-beam...")
    update(*beam_search(budget, R, C, grid, portals, beam_width=800,
                        tile_sort_key=None, time_limit=per, start_time=time.time()),
           "wide-beam")

    if verbose:
        print(f"  Done in {time.time()-start:.1f}s | best score: {best_score}")

    return best_region, best_score


# ── Grid reconstruction ───────────────────────────────────────────────────────

def region_to_grid(best_region, orig_grid, R, C):
    """
    Build the solution grid from an enclosed region.

    Rules:
      - Tiles in region that were W → become '.' (pre-placed wall removed/opened)
      - Grass tiles adjacent to region → become 'W' (new wall placed)
      - Pre-placed walls adjacent to region → stay 'W' (natural wall kept)
      - Pre-placed walls NOT adjacent to region → become '.' (removed to free budget)
      - Everything else: unchanged
    """
    sol = [list(row) for row in orig_grid]
    region_set = set(best_region)

    # 1. Open pre-placed walls that ended up inside the region
    for (r, c) in region_set:
        if orig_grid[r][c] == WALL:
            sol[r][c] = GRASS

    # 2. Wall grass tiles adjacent to region
    for r in range(R):
        for c in range(C):
            if (r, c) not in region_set and orig_grid[r][c] == GRASS:
                if any(0<=r+dr<R and 0<=c+dc<C and (r+dr,c+dc) in region_set
                       for dr,dc in DIRS):
                    sol[r][c] = WALL

    # 3. Remove pre-placed walls not adjacent to region
    #    (they consume budget without serving the enclosure)
    for r in range(R):
        for c in range(C):
            if (r, c) not in region_set and orig_grid[r][c] == WALL:
                adj = any(0<=r+dr<R and 0<=c+dc<C and (r+dr,c+dc) in region_set
                          for dr,dc in DIRS)
                if not adj:
                    sol[r][c] = GRASS  # remove unnecessary pre-placed wall

    return sol


# ── Full reachability check (for verify) ─────────────────────────────────────

def full_reachable(grid, R, C, portals, hr, hc):
    """BFS from horse through passable tiles. Returns (reachable_set, hits_perimeter)."""
    passable = {GRASS, HORSE, APPLE, BEES, CHERRY, PORTAL}  # W is a wall in final grid
    visited = set()
    queue = deque([(hr, hc)])
    visited.add((hr, hc))
    perimeter = False
    while queue:
        r, c = queue.popleft()
        if r==0 or r==R-1 or c==0 or c==C-1:
            perimeter = True
        if grid[r][c] == PORTAL and (r,c) in portals:
            pr, pc = portals[(r,c)]
            if (pr,pc) not in visited and grid[pr][pc] in passable:
                visited.add((pr,pc)); queue.append((pr,pc))
        for dr,dc in DIRS:
            nr,nc = r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc] in passable:
                visited.add((nr,nc)); queue.append((nr,nc))
    return visited, perimeter


# ── Verifier ──────────────────────────────────────────────────────────────────

def verify(budget, R, C, orig_grid, portals, sol_grid, claimed_score, verbose=True):
    """
    Verify solution grid against spec. Returns list of error strings.
    Valid solutions: empty list.

    Wall count = ALL 'W' tiles in the solution (pre-placed kept + new).
    """
    errors = []

    # Total walls in solution (spec: total W tiles ≤ budget)
    total_walls = sum(sol_grid[r][c] == WALL for r in range(R) for c in range(C))
    if total_walls > budget:
        errors.append(f"FAIL: {total_walls} total walls in solution, budget is {budget}")

    # No wall placed on illegal tile (apple/bee/cherry/portal in ORIGINAL)
    illegal = {APPLE, BEES, CHERRY, PORTAL}
    for r in range(R):
        for c in range(C):
            if sol_grid[r][c] == WALL and orig_grid[r][c] in illegal:
                errors.append(f"FAIL: wall on '{orig_grid[r][c]}' at ({r},{c})")

    # Horse still present
    try:
        hr, hc = find_horse(sol_grid, R, C)
    except ValueError:
        errors.append("FAIL: horse missing from solution")
        return errors

    # Horse enclosed (cannot reach perimeter)
    region, perimeter = full_reachable(sol_grid, R, C, portals, hr, hc)
    if perimeter:
        errors.append("FAIL: horse can reach the perimeter")

    # Score matches
    actual = sum(TILE_SCORE.get(sol_grid[r][c], 0) for r, c in region)
    if actual != claimed_score:
        errors.append(f"FAIL: claimed {claimed_score}, actual score is {actual}")

    if not errors and verbose:
        new_walls = sum(sol_grid[r][c]==WALL and orig_grid[r][c]!=WALL
                        for r in range(R) for c in range(C))
        kept_walls = total_walls - new_walls
        print(f"  ✓ VALID | score: {actual} | walls: {total_walls}/{budget} "
              f"({new_walls} new + {kept_walls} kept)")

    return errors


# ── Per-input solver ──────────────────────────────────────────────────────────

def solve_input(name, text, out_dir=None, time_limit=20.0):
    """
    Parse, solve, verify, and write solution for one input.
    Returns (score, valid, elapsed).
    """
    t0 = time.time()
    try:
        budget, R, C, grid, portals = parse_input(text)
    except Exception as e:
        print(f"  [PARSE ERROR] {e}")
        return None, False, 0

    hr, hc = find_horse(grid, R, C)
    _, horse_free = full_reachable(grid, R, C, portals, hr, hc)

    print(f"\n{'─'*55}")
    print(f"  {name}  [{R}×{C}  budget={budget}  portals={len(portals)//2}]")

    if not horse_free:
        print("  NOTE: horse already enclosed before any walls — trivial input")

    # Solve
    best_region, best_score = solve(budget, R, C, grid, portals,
                                    time_limit=time_limit, verbose=True)

    if best_region is None:
        print("  ERROR: no valid enclosure found")
        return None, False, time.time()-t0

    # Build solution grid
    sol_grid = region_to_grid(best_region, grid, R, C)

    # Verify
    errors = verify(budget, R, C, grid, portals, sol_grid, best_score, verbose=True)
    for e in errors:
        print(f"  {e}")

    # Write output
    if not errors:
        out_text = str(best_score) + '\n' + '\n'.join(''.join(row) for row in sol_grid)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            out_dir = os.path.join(os.path.expanduser('~'), 'algobowl', 'outputs')
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name.replace('.txt', '_solution.txt'))
        with open(out_path, 'w') as f:
            f.write(out_text + '\n')
        print(f"  → {out_path}")

    return best_score, not errors, time.time()-t0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)

    check_only  = '--check' in args
    args        = [a for a in args if a != '--check']
    source      = args[0]
    verify_file = args[1] if len(args) > 1 and not args[1].endswith('.zip') else None

    # ── ZIP mode ─────────────────────────────────────────────────────────────
    if source.endswith('.zip'):
        print(f"Opening zip: {source}")
        out_dir = os.path.join(os.getcwd(), 'outputs')
        results = []
        with zipfile.ZipFile(source) as zf:
            txt_files = sorted(n for n in zf.namelist() if n.endswith('.txt'))
            print(f"Found {len(txt_files)} input files")
            for name in txt_files:
                text = zf.read(name).decode('utf-8')
                base = os.path.basename(name)
                score, valid, elapsed = solve_input(base, text, out_dir=out_dir)
                results.append((base, score, valid, elapsed))

        print(f"\n{'='*55}")
        print(f"  SUMMARY ({len(results)} inputs)")
        print(f"{'─'*55}")
        total_score = 0
        for name, score, valid, elapsed in results:
            status = '✓' if valid else '✗'
            s = score if score is not None else 'ERR'
            print(f"  {status}  {name:<30}  score={s:<8}  {elapsed:.1f}s")
            if score: total_score += score
        print(f"{'─'*55}")
        print(f"  Total score: {total_score}")
        sys.exit(0)

    # ── Single file mode ─────────────────────────────────────────────────────
    with open(source) as f:
        text = f.read()

    budget, R, C, grid, portals = parse_input(text)

    # Verify mode
    if verify_file:
        with open(verify_file) as f:
            lines = f.read().strip().splitlines()
        claimed = int(lines[0])
        sol_grid = [list(l) for l in lines[1:]]
        print(f"\nVerifying {verify_file}...")
        for e in verify(budget, R, C, grid, portals, sol_grid, claimed):
            print(f"  {e}")
        sys.exit(0)

    # Check mode
    if check_only:
        hr, hc = find_horse(grid, R, C)
        _, free = full_reachable(grid, R, C, portals, hr, hc)
        print(f"Horse {'CAN' if free else 'CANNOT'} reach perimeter")
        sys.exit(0)

    # Solve mode
    out_dir = os.path.join(os.getcwd(), 'outputs')
    solve_input(os.path.basename(source), text, out_dir=out_dir)


if __name__ == '__main__':
    main()