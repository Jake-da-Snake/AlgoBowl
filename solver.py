"""
AlgoBOWL: Enclose Horse — Verifier, Scorer, and Solver
=======================================================
Usage:
    python solver.py input.txt          # solve and print best output
    python solver.py input.txt out.txt  # verify an existing output file
    python solver.py --check input.txt  # just check if input is valid (horse can reach perimeter)
"""

import sys
import copy
from collections import deque
from itertools import combinations

# ── Tile constants ──────────────────────────────────────────────────────────
WATER   = '#'
GRASS   = '.'
HORSE   = 'H'
WALL    = 'W'
APPLE   = 'a'
BEES    = 'b'
CHERRY  = 'c'
PORTAL  = 'p'

TILE_SCORE = {
    GRASS:  1,
    HORSE:  1,
    APPLE:  11,   # 1 (grass) + 10 (apple)
    BEES:   -4,   # 1 (grass) - 5 (bees)
    CHERRY: 4,    # 1 (grass) + 3 (cherry)
    PORTAL: 1,    # counts as grass for scoring
}

PLACEABLE = {GRASS, HORSE, WALL}   # tiles where walls CAN be placed (horse tile too, though silly)
# Actually spec says: no wall on water, apple, bees, cherries, portal
# So walls go only on grass (.) or pre-placed wall (W) spots → only '.' is replaceable
WALL_OK = {GRASS}   # can place a NEW wall here (not on H, not on special tiles)


# ── Input parsing ────────────────────────────────────────────────────────────

def parse_input(text):
    lines = [l.rstrip('\n') for l in text.strip().splitlines()]
    idx = 0
    budget = int(lines[idx]); idx += 1
    R, C = map(int, lines[idx].split()); idx += 1
    grid = []
    for r in range(R):
        grid.append(list(lines[idx])); idx += 1
    P = int(lines[idx]); idx += 1
    portals = {}  # (r1,c1) -> (r2,c2) and vice versa
    for _ in range(P):
        r1, c1, r2, c2 = map(int, lines[idx].split()); idx += 1
        portals[(r1, c1)] = (r2, c2)
        portals[(r2, c2)] = (r1, c1)
    return budget, R, C, grid, portals


def parse_output(text):
    lines = [l.rstrip('\n') for l in text.strip().splitlines()]
    score = int(lines[0])
    grid = [list(l) for l in lines[1:]]
    return score, grid


# ── Core flood-fill with portal support ─────────────────────────────────────

def reachable(grid, R, C, portals, start_r, start_c):
    """
    BFS from (start_r, start_c) over grass-like tiles (not water, not wall).
    Returns set of (r,c) reachable, and whether any perimeter tile is reached.
    """
    passable = {GRASS, HORSE, APPLE, BEES, CHERRY, PORTAL}
    visited = set()
    queue = deque()

    if grid[start_r][start_c] not in passable:
        return visited, False

    queue.append((start_r, start_c))
    visited.add((start_r, start_c))
    hits_perimeter = False

    while queue:
        r, c = queue.popleft()

        if r == 0 or r == R-1 or c == 0 or c == C-1:
            hits_perimeter = True

        # orthogonal neighbours
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C:
                if (nr, nc) not in visited and grid[nr][nc] in passable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        # portal teleport
        if grid[r][c] == PORTAL and (r, c) in portals:
            tr, tc = portals[(r, c)]
            if (tr, tc) not in visited and grid[tr][tc] in passable:
                visited.add((tr, tc))
                queue.append((tr, tc))

    return visited, hits_perimeter


def score_region(region, grid):
    total = 0
    for (r, c) in region:
        tile = grid[r][c]
        total += TILE_SCORE.get(tile, 0)
    return total


def find_horse(grid, R, C):
    for r in range(R):
        for c in range(C):
            if grid[r][c] == HORSE:
                return r, c
    raise ValueError("No horse (H) found in grid!")


def count_placed_walls(original_grid, solution_grid, R, C):
    """Count walls in solution that weren't in the original."""
    count = 0
    for r in range(R):
        for c in range(C):
            if solution_grid[r][c] == WALL and original_grid[r][c] != WALL:
                count += 1
    return count


# ── Verifier ────────────────────────────────────────────────────────────────

def verify(budget, R, C, orig_grid, portals, sol_grid, claimed_score):
    errors = []

    # 1. Wall count
    nwalls = count_placed_walls(orig_grid, sol_grid, R, C)
    if nwalls > budget:
        errors.append(f"FAIL: Used {nwalls} walls but budget is {budget}")

    # 2. No wall on illegal tile
    for r in range(R):
        for c in range(C):
            if sol_grid[r][c] == WALL and orig_grid[r][c] not in {GRASS, WALL}:
                errors.append(f"FAIL: Wall placed on illegal tile '{orig_grid[r][c]}' at ({r},{c})")

    # 3. Horse still present
    try:
        hr, hc = find_horse(sol_grid, R, C)
    except ValueError:
        errors.append("FAIL: Horse missing from solution grid")
        return errors

    # 4. Horse cannot reach perimeter
    region, hits_perimeter = reachable(sol_grid, R, C, portals, hr, hc)
    if hits_perimeter:
        errors.append("FAIL: Horse can still reach the perimeter — not enclosed!")

    # 5. Score matches
    actual_score = score_region(region, sol_grid)
    if actual_score != claimed_score:
        errors.append(f"FAIL: Claimed score {claimed_score} but actual score is {actual_score}")

    if not errors:
        print(f"  ✓ Valid enclosure  |  Walls used: {nwalls}/{budget}  |  Score: {actual_score}")
    return errors


# ── Input validator ──────────────────────────────────────────────────────────

def check_input(budget, R, C, grid, portals):
    """
    A valid INPUT must have the horse unable to reach the perimeter
    given only the pre-placed walls. Also checks horse exists.
    """
    hr, hc = find_horse(grid, R, C)
    region, hits_perimeter = reachable(grid, R, C, portals, hr, hc)
    if hits_perimeter:
        print(f"  INPUT CHECK: Horse CAN reach the perimeter ✓ (good — needs walls to enclose)")
        print(f"  Horse at ({hr},{hc}), can reach {len(region)} tiles before any walls added")
    else:
        print(f"  INPUT CHECK: Horse is ALREADY enclosed by pre-placed walls/water — trivial input!")
    return hits_perimeter  # True means valid (horse is not yet enclosed)


# ── Candidate wall positions ─────────────────────────────────────────────────

def get_wall_candidates(grid, R, C, portals):
    """
    Return all (r,c) where a wall COULD be placed (grass tiles only, not horse tile).
    Prioritize tiles adjacent to the horse's reachable region — placing walls elsewhere
    is always useless.
    """
    hr, hc = find_horse(grid, R, C)
    region, _ = reachable(grid, R, C, portals, hr, hc)

    candidates = set()
    for (r, c) in region:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C:
                if grid[nr][nc] in WALL_OK and (nr, nc) not in region:
                    # boundary grass just outside the reachable region
                    candidates.add((nr, nc))
        # also the reachable cells themselves (inner walls)
        if grid[r][c] in WALL_OK:
            candidates.add((r, c))

    return sorted(candidates)


# ── Greedy solver ────────────────────────────────────────────────────────────

def greedy_solve(budget, R, C, grid, portals):
    """
    Greedy: repeatedly find the single wall placement that most improves score
    (or reduces escape routes). Falls back to any wall that helps enclose.
    """
    best_grid = copy.deepcopy(grid)
    walls_used = 0
    hr, hc = find_horse(grid, R, C)

    for _ in range(budget):
        region, hits_perimeter = reachable(best_grid, R, C, portals, hr, hc)
        if not hits_perimeter:
            break  # already enclosed

        candidates = get_wall_candidates(best_grid, R, C, portals)
        if not candidates:
            break

        best_delta = None
        best_pos = None

        for (wr, wc) in candidates:
            if best_grid[wr][wc] not in WALL_OK:
                continue
            test = copy.deepcopy(best_grid)
            test[wr][wc] = WALL
            new_region, new_hits = reachable(test, R, C, portals, hr, hc)
            new_score = score_region(new_region, test)
            old_score = score_region(region, best_grid)

            # Prefer: enclosed > not enclosed, then higher score
            enclosed_gain = (not new_hits) and hits_perimeter
            delta = (enclosed_gain, new_score - old_score, -(wr*C+wc))

            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_pos = (wr, wc)

        if best_pos:
            best_grid[best_pos[0]][best_pos[1]] = WALL
            walls_used += 1

    return best_grid, walls_used


# ── Brute-force solver (small grids / budgets) ───────────────────────────────

def brute_force_solve(budget, R, C, grid, portals, max_candidates=20):
    """
    Try all combinations of up to `budget` walls from the candidate set.
    Only feasible when candidate set is small. Caps at max_candidates.
    """
    hr, hc = find_horse(grid, R, C)
    candidates = get_wall_candidates(grid, R, C, portals)

    # Filter to only placeable
    candidates = [p for p in candidates if grid[p[0]][p[1]] in WALL_OK]

    if len(candidates) > max_candidates:
        print(f"  [brute force] {len(candidates)} candidates > {max_candidates} cap, skipping")
        return None, None

    print(f"  [brute force] trying C({len(candidates)},{budget}) = up to "
          f"{sum(1 for _ in combinations(range(len(candidates)), min(budget, len(candidates))))} combos")

    best_score = None
    best_grid = None

    for k in range(1, min(budget, len(candidates)) + 1):
        for combo in combinations(range(len(candidates)), k):
            test = copy.deepcopy(grid)
            for idx in combo:
                r, c = candidates[idx]
                test[r][c] = WALL
            region, hits = reachable(test, R, C, portals, hr, hc)
            if not hits:
                s = score_region(region, test)
                if best_score is None or s > best_score:
                    best_score = s
                    best_grid = test

    return best_grid, best_score


# ── Output formatter ─────────────────────────────────────────────────────────

def format_output(sol_grid, R, C, portals):
    hr, hc = find_horse(sol_grid, R, C)
    region, hits = reachable(sol_grid, R, C, portals, hr, hc)
    score = score_region(region, sol_grid)
    lines = [str(score)]
    for row in sol_grid:
        lines.append(''.join(row))
    return '\n'.join(lines), score


def print_grid(grid, label=""):
    if label:
        print(f"\n  {label}")
    for row in grid:
        print("  " + ''.join(row))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    check_only = '--check' in args
    args = [a for a in args if a != '--check']

    input_file = args[0]
    output_file = args[1] if len(args) > 1 else None

    with open(input_file) as f:
        input_text = f.read()

    budget, R, C, grid, portals = parse_input(input_text)

    print(f"\n{'='*60}")
    print(f"  Grid: {R}×{C}   Budget: {budget}   Portals: {len(portals)//2}")
    print_grid(grid, "Input:")
    print(f"{'='*60}")

    # Input validity check
    is_open = check_input(budget, R, C, grid, portals)

    if check_only:
        sys.exit(0)

    # Verify mode
    if output_file:
        print(f"\n  Verifying {output_file}...")
        with open(output_file) as f:
            out_text = f.read()
        claimed_score, sol_grid = parse_output(out_text)
        errors = verify(budget, R, C, grid, portals, sol_grid, claimed_score)
        for e in errors:
            print(f"  {e}")
        sys.exit(0)

    # Solve mode
    print(f"\n  Solving...")

    # Try brute force first (best for small grids)
    bf_grid, bf_score = brute_force_solve(budget, R, C, grid, portals, max_candidates=22)

    if bf_grid is not None:
        print(f"  [brute force] Best score: {bf_score}")
        print_grid(bf_grid, "Brute-force solution:")
        out, score = format_output(bf_grid, R, C, portals)
        walls = count_placed_walls(grid, bf_grid, R, C)
        print(f"\n  Score: {score}   Walls used: {walls}/{budget}")
        out_path = input_file.replace('.txt', '_solution.txt')
        with open(out_path, 'w') as f:
            f.write(out)
        print(f"  Solution written to: {out_path}")

    else:
        # Fall back to greedy
        print("  [greedy] Running greedy solver...")
        g_grid, g_walls = greedy_solve(budget, R, C, grid, portals)
        hr, hc = find_horse(g_grid, R, C)
        region, hits = reachable(g_grid, R, C, portals, hr, hc)
        g_score = score_region(region, g_grid)

        if hits:
            print(f"  [greedy] WARNING: Could not fully enclose horse with budget {budget}!")
        else:
            print(f"  [greedy] Score: {g_score}   Walls used: {g_walls}/{budget}")

        print_grid(g_grid, "Greedy solution:")
        out, score = format_output(g_grid, R, C, portals)
        out_path = input_file.replace('.txt', '_solution.txt')
        with open(out_path, 'w') as f:
            f.write(out)
        print(f"  Solution written to: {out_path}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()