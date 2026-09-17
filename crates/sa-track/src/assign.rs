//! Minimum-cost assignment (Hungarian / Kuhn–Munkres, O(n³)) on a dense
//! rectangular cost matrix, plus the gating step that turns it into a
//! tracker association: pairs above the cost threshold are unmatched.

/// Solve min-cost assignment. `cost[r * cols + c]`. Returns `(r, c)` pairs;
/// every row of the smaller dimension is assigned.
pub fn hungarian(cost: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    // Work on the transposed problem if there are more rows than columns so
    // that n <= m, as the classic potentials formulation requires.
    let transposed = rows > cols;
    let (n, m) = if transposed { (cols, rows) } else { (rows, cols) };
    let at = |i: usize, j: usize| -> f32 {
        if transposed { cost[j * cols + i] } else { cost[i * cols + j] }
    };
    const INF: f32 = f32::INFINITY;
    let mut u = vec![0.0f32; n + 1];
    let mut v = vec![0.0f32; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![INF; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let cur = at(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            if !delta.is_finite() {
                // Remaining columns unreachable (all-inf row); leave unassigned.
                break;
            }
            // `j` indexes four parallel arrays (used/minv/p/u/v); iterating any
            // one of them would still need the index for the rest.
            #[allow(clippy::needless_range_loop)]
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        if j0 == 0 {
            continue;
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = Vec::with_capacity(n);
    // `j` is the COLUMN index of the assignment, not a position in `p`'s
    // iteration order — it is carried into the output pair.
    #[allow(clippy::needless_range_loop)]
    for j in 1..=m {
        if p[j] != 0 {
            let (r, c) = (p[j] - 1, j - 1);
            out.push(if transposed { (c, r) } else { (r, c) });
        }
    }
    out
}

/// Assignment with a gate: costs `> max_cost` (or non-finite) never match.
/// Returns (matches, unmatched_rows, unmatched_cols).
pub fn gated_assign(cost: &[f32], rows: usize, cols: usize, max_cost: f32) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    // Clamp gated-out entries to a large finite cost so the solver still
    // produces a complete assignment, then drop those pairs.
    const BIG: f32 = 1.0e6;
    let masked: Vec<f32> = cost.iter().map(|&c| if c.is_finite() && c <= max_cost { c } else { BIG }).collect();
    let raw = hungarian(&masked, rows, cols);
    let mut matches = Vec::with_capacity(raw.len());
    let mut row_used = vec![false; rows];
    let mut col_used = vec![false; cols];
    for (r, c) in raw {
        if masked[r * cols + c] < BIG {
            matches.push((r, c));
            row_used[r] = true;
            col_used[c] = true;
        }
    }
    let ur = (0..rows).filter(|&r| !row_used[r]).collect();
    let uc = (0..cols).filter(|&c| !col_used[c]).collect();
    (matches, ur, uc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square() {
        let c = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let mut m = hungarian(&c, 3, 3);
        m.sort();
        assert_eq!(m, vec![(0, 1), (1, 0), (2, 2)]);
    }

    #[test]
    fn rectangular_both_ways() {
        let c = [1.0, 9.0, 9.0, 9.0, 9.0, 1.0];
        let mut m = hungarian(&c, 2, 3);
        m.sort();
        assert_eq!(m, vec![(0, 0), (1, 2)]);
        let mut m2 = hungarian(&c, 3, 2); // interpreted as 3x2
        m2.sort();
        assert_eq!(m2.len(), 2);
    }

    #[test]
    fn gate_drops_bad_pairs() {
        let c = [0.1, 0.9, 0.9, 0.95];
        let (m, ur, uc) = gated_assign(&c, 2, 2, 0.5);
        assert_eq!(m, vec![(0, 0)]);
        assert_eq!(ur, vec![1]);
        assert_eq!(uc, vec![1]);
    }
}
