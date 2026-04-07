"""
AlgoBOWL: Enclose Horse — Optimal Solver
=========================================

Core insight: reframe the problem as "which region should the horse be enclosed in?"

A region R is valid when:
  - It contains the horse
  - Its boundary (passable tiles adjacent to R but not in R) contains ONLY grass tiles
    (since only grass tiles can have walls placed on them)
  - len(boundary) <= wall budget
  - No tile in R is a perimeter tile (horse can't reach the edge)

KEY RULE FROM SPEC: Walls can only go on grass (.) tiles.
  - Apple, bee, cherry, portal tiles CANNOT have walls.
  - Therefore: if a non-wallable passable tile is adjacent to the region,
    it MUST be included in the region (horse can reach it — nothing can block it).
  - This means non-wallable tiles flood into the region automatically.

We grow candidate regions using beam search, returning the best valid enclosure.

Usage:
    python optimal_solver.py input.txt           # solve
    python optimal_solver.py input.txt out.txt   # verify existing output
    python optimal_solver.py --check input.txt   # check input validity only
"""

import os
import sys
import time
import zipfile
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

PASSABLE     = {GRASS, HORSE, APPLE, BEES, CHERRY, PORTAL}
MUST_INCLUDE = {HORSE, APPLE, BEES, CHERRY, PORTAL}  # can't place walls on these

TILE_SCORE = {
    GRASS: 1, HORSE: 1, APPLE: 11, BEES: -4, CHERRY: 4, PORTAL: 1,
}

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# ── Input / output ────────────────────────────────────────────────────────────

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
    raise ValueError("No horse (H) in grid")

def score_region(region, grid):
    return sum(TILE_SCORE.get(grid[r][c], 0) for r, c in region)

# ── Region primitives ─────────────────────────────────────────────────────────

def flood_non_wallable(seeds, region, boundary, grid, R, C, portals):
    """
    From newly added tiles in `seeds`, flood all MUST_INCLUDE neighbors
    into region automatically (they can't be walled), and add GRASS
    neighbors to boundary.
    Modifies region and boundary in place.
    """
    queue = deque(seeds)
    while queue:
        r, c = queue.popleft()
        # Portal: destination also floods in
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
                region.add((nr, nc))
                queue.append((nr, nc))
            elif tile == GRASS:
                boundary.add((nr, nc))
            # WATER / WALL: impassable, skip


def initial_state(hr, hc, grid, R, C, portals):
    """Build starting state from horse position."""
    region = {(hr, hc)}
    boundary = set()
    flood_non_wallable([(hr, hc)], region, boundary, grid, R, C, portals)
    return frozenset(region), frozenset(boundary)


def expand(region, boundary, grass_tile, grid, R, C, portals):
    """
    Open up a GRASS boundary tile (don't wall it — include it in the region).
    Non-wallable tiles adjacent to it flood in automatically.
    Returns (new_region, new_boundary) as frozensets.
    """
    new_region = set(region)
    new_boundary = set(boundary)
    new_region.add(grass_tile)
    new_boundary.discard(grass_tile)
    flood_non_wallable([grass_tile], new_region, new_boundary, grid, R, C, portals)
    return frozenset(new_region), frozenset(new_boundary)


# ── Validity ──────────────────────────────────────────────────────────────────

def is_valid(region, boundary, budget, R, C):
    if len(boundary) > budget:
        return False
    # Horse must not be able to reach any perimeter tile
    return not any(r == 0 or r == R-1 or c == 0 or c == C-1 for r, c in region)


# ── Beam search ───────────────────────────────────────────────────────────────

def beam_search(budget, R, C, grid, portals,
                beam_width=400,
                tile_sort_key=None,
                time_limit=10.0,
                start_time=None):
    """
    Beam search over enclosed regions.
    At each step, expand one boundary grass tile into the region.
    Track and return the best valid enclosure found at any point.
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
            tiles = list(boundary)
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

                b  = len(new_boundary)
                on_edge = any(r2==0 or r2==R-1 or c2==0 or c2==C-1 for r2,c2 in new_region)
                s  = score_region(new_region, grid)

                if b <= budget and not on_edge:
                    pri = (2, s, budget - b)
                else:
                    excess = max(0, b - budget) + (20 if on_edge else 0)
                    pri = (1, s - excess * 5, -b)

                next_candidates.append((pri, new_region, new_boundary))

        if not next_candidates:
            break

        next_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(r, b) for _, r, b in next_candidates[:beam_width]]

    return best_region, best_score


# ── Multi-strategy solver ─────────────────────────────────────────────────────

def solve(budget, R, C, grid, portals, time_limit=20.0, verbose=True):
    """Run 4 beam search strategies, return best result."""
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

    # 1. Score-greedy: chase highest-value tiles first
    if verbose: print("  Strategy 1: score-greedy...")
    def score_key(r, c, grid, region, boundary, budget, R, C):
        return TILE_SCORE.get(grid[r][c], 0) * 10
    update(*beam_search(budget, R, C, grid, portals, beam_width=300,
                        tile_sort_key=score_key, time_limit=per, start_time=start),
           "score-greedy")

    # 2. Efficiency: score per unit boundary
    if verbose: print("  Strategy 2: efficiency-greedy...")
    def eff_key(r, c, grid, region, boundary, budget, R, C):
        in_nbrs = sum(1 for dr,dc in DIRS if (r+dr,c+dc) in region)
        return TILE_SCORE.get(grid[r][c], 0) + in_nbrs * 4
    update(*beam_search(budget, R, C, grid, portals, beam_width=300,
                        tile_sort_key=eff_key, time_limit=per, start_time=time.time()),
           "efficiency-greedy")

    # 3. Compact: minimize net boundary growth
    if verbose: print("  Strategy 3: boundary-minimizer...")
    def compact_key(r, c, grid, region, boundary, budget, R, C):
        in_nbrs = sum(1 for dr,dc in DIRS if (r+dr,c+dc) in region)
        new_exp  = sum(
            1 for dr,dc in DIRS
            if 0<=r+dr<R and 0<=c+dc<C
            and (r+dr,c+dc) not in region
            and (r+dr,c+dc) not in boundary
            and grid[r+dr][c+dc] == GRASS
        )
        return in_nbrs * 3 - new_exp * 2
    update(*beam_search(budget, R, C, grid, portals, beam_width=300,
                        tile_sort_key=compact_key, time_limit=per, start_time=time.time()),
           "boundary-minimizer")

    # 4. Wide unordered beam
    if verbose: print("  Strategy 4: wide-beam...")
    update(*beam_search(budget, R, C, grid, portals, beam_width=700,
                        tile_sort_key=None, time_limit=per, start_time=time.time()),
           "wide-beam")

    if verbose:
        print(f"  Done in {time.time()-start:.2f}s. Best score: {best_score}")

    return best_region, best_score


# ── Grid reconstruction ───────────────────────────────────────────────────────

def region_to_grid(best_region, orig_grid, R, C):
    """
    Build solution grid: start from the original grid, keep all existing walls
    unless they lie inside the chosen region, and wall every GRASS tile adjacent
    to the region but not in it.
    """
    sol = [list(row) for row in orig_grid]
    region_set = set(best_region)

    # Any existing wall inside the reachable region must be removed.
    for (r, c) in region_set:
        if orig_grid[r][c] == WALL:
            sol[r][c] = GRASS

    # Any grass tile adjacent to the region but not in it becomes a wall.
    for r in range(R):
        for c in range(C):
            if (r, c) not in region_set and sol[r][c] == GRASS:
                if any((r + dr, c + dc) in region_set for dr, dc in DIRS
                       if 0 <= r + dr < R and 0 <= c + dc < C):
                    sol[r][c] = WALL

    return sol


# ── Verifier ──────────────────────────────────────────────────────────────────

def full_reachable(grid, R, C, portals, hr, hc):
    visited = set()
    queue   = deque([(hr, hc)])
    visited.add((hr, hc))
    perimeter = False
    while queue:
        r, c = queue.popleft()
        if r==0 or r==R-1 or c==0 or c==C-1:
            perimeter = True
        if grid[r][c] == PORTAL and (r,c) in portals:
            pr, pc = portals[(r,c)]
            if (pr,pc) not in visited and grid[pr][pc] in PASSABLE:
                visited.add((pr,pc)); queue.append((pr,pc))
        for dr,dc in DIRS:
            nr,nc = r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc] in PASSABLE:
                visited.add((nr,nc)); queue.append((nr,nc))
    return visited, perimeter


def verify(budget, R, C, orig_grid, portals, sol_grid, claimed_score):
    errors = []
    hr, hc = find_horse(sol_grid, R, C)
    walls_used = sum(
        1 for r in range(R) for c in range(C)
        if sol_grid[r][c] == WALL
    )
    if walls_used > budget:
        errors.append(f"FAIL: used {walls_used} walls, budget is {budget}")
    for r in range(R):
        for c in range(C):
            if sol_grid[r][c]==WALL and orig_grid[r][c] not in {GRASS, WALL}:
                errors.append(f"FAIL: wall on non-grass '{orig_grid[r][c]}' at ({r},{c})")
    region, perimeter = full_reachable(sol_grid, R, C, portals, hr, hc)
    if perimeter:
        errors.append("FAIL: horse can still reach the perimeter")
    actual = sum(TILE_SCORE.get(sol_grid[r][c], 0) for r,c in region)
    if actual != claimed_score:
        errors.append(f"FAIL: claimed {claimed_score}, actual is {actual}")
    if not errors:
        print(f"  ✓ VALID  |  walls: {walls_used}/{budget}  |  score: {actual}")
    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def print_grid(grid, label=""):
    if label: print(f"\n  {label}")
    for row in grid:
        print("  " + ''.join(row))


# ── Helper functions for batch and verification ───────────────────────────────

def solve_from_text(text, source_name, output_dir=None):
    budget, R, C, grid, portals = parse_input(text)
    print(f"\n{'='*60}")
    print(f"  Source: {source_name}")
    print(f"  Grid: {R}×{C}   Budget: {budget}   Portals: {len(portals)//2}")
    print(f"{'='*60}")

    hr, hc = find_horse(grid, R, C)
    _, horse_free = full_reachable(grid, R, C, portals, hr, hc)
    print(f"  INPUT: horse {'can' if horse_free else 'CANNOT'} reach perimeter")

    print("\n  Solving...")
    best_region, best_score = solve(budget, R, C, grid, portals, time_limit=20.0)

    if best_region is None:
        print("  ERROR: No valid enclosure found.")
        return 1

    sol_grid = region_to_grid(best_region, grid, R, C)
    walls_used = sum(1 for r in range(R) for c in range(C)
                     if sol_grid[r][c] == WALL)
    print(f"\n  Score: {best_score}   Walls: {walls_used}/{budget}")
    print("  Verifying...")
    for e in verify(budget, R, C, grid, portals, sol_grid, best_score):
        print(f"  {e}")

    out_text = str(best_score) + '\n' + '\n'.join(''.join(row) for row in sol_grid)

    if output_dir is None:
        if source_name.endswith('.txt'):
            out_path = source_name.replace('.txt', '_optimal.txt')
        else:
            out_path = source_name + '_optimal.txt'
    else:
        base_name = os.path.basename(source_name)
        if base_name.endswith('.txt'):
            base_name = base_name[:-4]
        out_path = os.path.join(output_dir, base_name + '_optimal.txt')

    with open(out_path, 'w') as f:
        f.write(out_text + '\n')
    print(f"  Written: {out_path}")
    print(f"{'='*60}\n")
    return 0


def verify_from_text(input_text, output_text, source_name, output_name):
    budget, R, C, grid, portals = parse_input(input_text)
    print(f"\n{'='*60}")
    print(f"  Input:  {source_name}")
    print(f"  Output: {output_name}")
    print(f"  Grid: {R}×{C}   Budget: {budget}   Portals: {len(portals)//2}")
    print(f"{'='*60}")

    lines = output_text.strip().splitlines()
    claimed = int(lines[0])
    sol_grid = [list(l) for l in lines[1:]]

    print(f"\n  Verifying {output_name}...")
    errors = verify(budget, R, C, grid, portals, sol_grid, claimed)
    for e in errors:
        print(f"  {e}")
    if not errors:
        print(f"{'='*60}\n")
        return 0
    print(f"{'='*60}\n")
    return 1

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    check_only = '--check' in args
    args = [a for a in args if a != '--check']
    input_file = args[0]
    output_file = args[1] if len(args) > 1 else None

    if input_file.endswith('.zip'):
        if output_file:
            print("ZIP input does not support passing a single output file for verification.")
            sys.exit(1)

        zip_path = input_file
        output_dir = os.path.splitext(zip_path)[0] + '_optimal_outputs'
        os.makedirs(output_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path) as zf:
            txt_names = sorted(
                name for name in zf.namelist()
                if not name.endswith('/') and name.lower().endswith('.txt')
            )

            if not txt_names:
                print(f"No .txt files found in {zip_path}")
                sys.exit(1)

            exit_code = 0
            for name in txt_names:
                text = zf.read(name).decode('utf-8')
                if check_only:
                    budget, R, C, grid, portals = parse_input(text)
                    print(f"\n{'='*60}")
                    print(f"  Source: {name}")
                    print(f"  Grid: {R}×{C}   Budget: {budget}   Portals: {len(portals)//2}")
                    hr, hc = find_horse(grid, R, C)
                    _, horse_free = full_reachable(grid, R, C, portals, hr, hc)
                    print(f"  INPUT: horse {'can' if horse_free else 'CANNOT'} reach perimeter")
                    print(f"{'='*60}\n")
                else:
                    result = solve_from_text(text, name, output_dir=output_dir)
                    if result != 0:
                        exit_code = result

        if not check_only:
            print(f"All outputs written to: {output_dir}")
        sys.exit(exit_code)

    with open(input_file) as f:
        text = f.read()

    budget, R, C, grid, portals = parse_input(text)
    print(f"\n{'='*60}")
    print(f"  Grid: {R}×{C}   Budget: {budget}   Portals: {len(portals)//2}")
    print(f"{'='*60}")

    hr, hc = find_horse(grid, R, C)
    _, horse_free = full_reachable(grid, R, C, portals, hr, hc)
    print(f"  INPUT: horse {'can' if horse_free else 'CANNOT'} reach perimeter")

    if check_only:
        sys.exit(0)

    if output_file:
        with open(output_file) as f:
            out_text = f.read()
        exit_code = verify_from_text(text, out_text, input_file, output_file)
        sys.exit(exit_code)

    exit_code = solve_from_text(text, input_file)
    sys.exit(exit_code)

if __name__ == '__main__':
    main()