from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


Coord = Tuple[int, int]


SCORE_MAP = {
    ".": 1,
    "H": 1,
    "p": 1,
    "a": 11,
    "b": -4,
    "c": 4,
    "#": 0,
    "W": 0,
}

ALLOWED_CHARS = set(SCORE_MAP.keys())
BLOCKED_TILES = {"#", "W"}
FIXED_TILES = {"#", "H", "p", "a", "b", "c"}  # must stay at same coordinates


@dataclass
class InstanceData:
    wall_budget: int
    rows: int
    cols: int
    grid: List[List[str]]
    portal_pairs: Dict[Coord, Coord]
    counts: Counter


@dataclass
class SolutionData:
    claimed_score: int
    grid: List[List[str]]
    counts: Counter
    horse_pos: Coord | None
    wall_count: int


def read_nonempty_lines(path: str) -> List[str]:
    return [line.rstrip("\n") for line in Path(path).read_text().splitlines() if line.strip() != ""]


def parse_instance(path: str) -> InstanceData:
    lines = read_nonempty_lines(path)
    if len(lines) < 4:
        raise ValueError("Instance file is too short.")

    idx = 0

    try:
        wall_budget = int(lines[idx].strip())
    except ValueError as e:
        raise ValueError("First line of instance must be an integer wall budget.") from e
    idx += 1

    try:
        rows, cols = map(int, lines[idx].split())
    except ValueError as e:
        raise ValueError("Second line of instance must contain: R C") from e
    idx += 1

    if rows <= 0 or cols <= 0:
        raise ValueError("Grid dimensions must be positive.")

    if idx + rows > len(lines):
        raise ValueError("Instance file does not contain enough grid rows.")

    raw_grid = lines[idx:idx + rows]
    idx += rows

    grid: List[List[str]] = []
    counts: Counter = Counter()

    for r, row in enumerate(raw_grid):
        if len(row) != cols:
            raise ValueError(
                f"Instance row {r} has length {len(row)}, expected {cols}."
            )

        row_list = list(row)
        for c, ch in enumerate(row_list):
            if ch not in ALLOWED_CHARS:
                raise ValueError(f"Invalid instance character {ch!r} at ({r}, {c}).")
            counts[ch] += 1
        grid.append(row_list)

    if idx >= len(lines):
        raise ValueError("Instance file missing portal pair count.")

    try:
        p = int(lines[idx].strip())
    except ValueError as e:
        raise ValueError("Portal pair count line must be an integer.") from e
    idx += 1

    portal_pairs: Dict[Coord, Coord] = {}
    used_portals = set()

    if idx + p > len(lines):
        raise ValueError("Instance file missing one or more portal pair lines.")

    for _ in range(p):
        try:
            r1, c1, r2, c2 = map(int, lines[idx].split())
        except ValueError as e:
            raise ValueError(f"Invalid portal pair line: {lines[idx]!r}") from e
        idx += 1

        a = (r1, c1)
        b = (r2, c2)

        for rr, cc in (a, b):
            if not (0 <= rr < rows and 0 <= cc < cols):
                raise ValueError(f"Portal coordinate ({rr}, {cc}) out of bounds.")

        if grid[r1][c1] != "p" or grid[r2][c2] != "p":
            raise ValueError(
                f"Portal pair {a} <-> {b} must point to 'p' tiles in the instance."
            )

        if a in used_portals or b in used_portals:
            raise ValueError("A portal coordinate appears in more than one pair.")

        used_portals.add(a)
        used_portals.add(b)
        portal_pairs[a] = b
        portal_pairs[b] = a

    if counts["p"] != 2 * p:
        raise ValueError(
            f"Instance contains {counts['p']} portal tiles, but portal pair data implies {2 * p}."
        )

    if counts["H"] != 1:
        raise ValueError(f"Instance must contain exactly one horse; found {counts['H']}.")

    return InstanceData(
        wall_budget=wall_budget,
        rows=rows,
        cols=cols,
        grid=grid,
        portal_pairs=portal_pairs,
        counts=counts,
    )


def parse_solution(path: str, expected_rows: int, expected_cols: int) -> SolutionData:
    lines = read_nonempty_lines(path)
    if len(lines) < 2:
        raise ValueError("Solution file is too short.")

    try:
        claimed_score = int(lines[0].strip())
    except ValueError as e:
        raise ValueError("First line of solution must be an integer claimed score.") from e

    raw_grid = lines[1:]
    if len(raw_grid) != expected_rows:
        raise ValueError(
            f"Solution has {len(raw_grid)} rows, expected {expected_rows}."
        )

    grid: List[List[str]] = []
    counts: Counter = Counter()
    horse_pos: Coord | None = None
    wall_count = 0

    for r, row in enumerate(raw_grid):
        if len(row) != expected_cols:
            raise ValueError(
                f"Solution row {r} has length {len(row)}, expected {expected_cols}."
            )

        row_list = list(row)
        for c, ch in enumerate(row_list):
            if ch not in ALLOWED_CHARS:
                raise ValueError(f"Invalid solution character {ch!r} at ({r}, {c}).")

            counts[ch] += 1

            if ch == "H":
                if horse_pos is not None:
                    raise ValueError("Solution contains more than one horse.")
                horse_pos = (r, c)
            elif ch == "W":
                wall_count += 1

        grid.append(row_list)

    if horse_pos is None:
        raise ValueError("Solution contains no horse.")

    return SolutionData(
        claimed_score=claimed_score,
        grid=grid,
        counts=counts,
        horse_pos=horse_pos,
        wall_count=wall_count,
    )


def validate_grid_legality(instance: InstanceData, solution: SolutionData) -> List[str]:
    errors: List[str] = []

    for ch in ("#", "a", "b", "c", "p"):
        if solution.counts[ch] != instance.counts[ch]:
            errors.append(
                f"Count mismatch for {ch!r}: solution has {solution.counts[ch]}, "
                f"instance has {instance.counts[ch]}."
            )

    if solution.counts["H"] != 1:
        errors.append(f"Solution must contain exactly 1 horse; found {solution.counts['H']}.")

    if solution.wall_count > instance.wall_budget:
        errors.append(
            f"Wall budget exceeded: solution has {solution.wall_count} wall(s), "
            f"budget is {instance.wall_budget}."
        )

    for r in range(instance.rows):
        base_row = instance.grid[r]
        sol_row = solution.grid[r]
        for c in range(instance.cols):
            base = base_row[c]
            sol = sol_row[c]

            if base in FIXED_TILES and sol != base:
                errors.append(
                    f"Illegal change at ({r}, {c}): fixed tile {base!r} changed to {sol!r}."
                )
                continue

            if base in {".", "W"} and sol not in {".", "W"}:
                errors.append(
                    f"Illegal change at ({r}, {c}): {base!r} may only become '.' or 'W', not {sol!r}."
                )

    for (r, c) in instance.portal_pairs:
        if solution.grid[r][c] != "p":
            errors.append(f"Portal missing at required coordinate ({r}, {c}).")

    return errors


def analyze_horse_region(
    solution_grid: List[List[str]],
    start: Coord,
    portal_pairs: Dict[Coord, Coord],
) -> Tuple[bool, int]:
    """
    Returns:
        escapes: True if the horse can reach the perimeter
        region_score: total score of the horse-reachable region
    """
    rows = len(solution_grid)
    cols = len(solution_grid[0])

    q = deque([start])
    visited = {start}
    region_score = 0
    escapes = False

    while q:
        r, c = q.popleft()
        ch = solution_grid[r][c]
        region_score += SCORE_MAP[ch]

        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            escapes = True

        if r > 0:
            nr, nc = r - 1, c
            if solution_grid[nr][nc] not in BLOCKED_TILES and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))

        if r + 1 < rows:
            nr, nc = r + 1, c
            if solution_grid[nr][nc] not in BLOCKED_TILES and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))

        if c > 0:
            nr, nc = r, c - 1
            if solution_grid[nr][nc] not in BLOCKED_TILES and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))

        if c + 1 < cols:
            nr, nc = r, c + 1
            if solution_grid[nr][nc] not in BLOCKED_TILES and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))

        if ch == "p":
            dest = portal_pairs.get((r, c))
            if dest is not None and dest not in visited:
                visited.add(dest)
                q.append(dest)

    return escapes, region_score


def evaluate_solution(instance: InstanceData, solution_path: str) -> Tuple[List[str], SolutionData, int]:
    solution = parse_solution(solution_path, instance.rows, instance.cols)
    errors = validate_grid_legality(instance, solution)

    escapes, region_score = analyze_horse_region(
        solution.grid,
        solution.horse_pos,
        instance.portal_pairs,
    )

    if region_score != solution.claimed_score:
        errors.append(
            f"Score mismatch: claimed {solution.claimed_score}, computed {region_score}."
        )

    if escapes:
        errors.append("Horse can reach the perimeter and escape.")

    return errors, solution, region_score


def validate_solution(instance_path: str, solution_path: str) -> bool:
    instance = parse_instance(instance_path)
    errors, solution, region_score = evaluate_solution(instance, solution_path)

    if errors:
        print("INVALID SOLUTION")
        for err in errors:
            print(f"- {err}")
        return False

    print("VALID SOLUTION")
    print(f"- Claimed score: {solution.claimed_score}")
    print(f"- Computed score: {region_score}")
    print(f"- Walls used: {solution.wall_count}/{instance.wall_budget}")
    print("- Horse is enclosed.")
    return True


def validate_solution_file(instance: InstanceData, solution_file: Path) -> str:
    """
    Returns:
        "valid", "invalid", or "error"
    """
    try:
        errors, solution, region_score = evaluate_solution(instance, str(solution_file))

        if errors:
            print(f"[INVALID] {solution_file.name}")
            for err in errors:
                print(f"  - {err}")
            return "invalid"

        print(
            f"[VALID]   {solution_file.name} | "
            f"score={region_score} | "
            f"walls={solution.wall_count}/{instance.wall_budget}"
        )
        return "valid"

    except Exception as e:
        print(f"[ERROR]   {solution_file.name} | {e}")
        return "error"


def validate_solution_folder(instance_path: str, folder_path: str) -> bool:
    instance = parse_instance(instance_path)
    folder = Path(folder_path)

    if not folder.is_dir():
        raise ValueError(f"{folder_path!r} is not a directory.")

    files = sorted(p for p in folder.iterdir() if p.is_file())

    if not files:
        print("No files found in folder.")
        return False

    valid_count = 0
    invalid_count = 0
    error_count = 0

    print(f"Checking {len(files)} file(s) in {folder}...\n")

    for file_path in files:
        result = validate_solution_file(instance, file_path)

        if result == "valid":
            valid_count += 1
        elif result == "invalid":
            invalid_count += 1
        else:
            error_count += 1

    print("\nSummary")
    print(f"- Valid:   {valid_count}")
    print(f"- Invalid: {invalid_count}")
    print(f"- Errors:  {error_count}")

    return invalid_count == 0 and error_count == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage:")
        print("  python validator.py <instance_file> <solution_file>")
        print("  python validator.py <instance_file> <solutions_folder>")
        raise SystemExit(1)

    instance_path = sys.argv[1]
    target_path = Path(sys.argv[2])

    if target_path.is_dir():
        ok = validate_solution_folder(instance_path, str(target_path))
    else:
        ok = validate_solution(instance_path, str(target_path))

    raise SystemExit(0 if ok else 2)