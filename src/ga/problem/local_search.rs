//! Critical-block local search on a chromosome's machine sequences.
//!
//! A schedule is no longer than the longest chain of activities that follow
//! one another, each starting the moment the one before it ends -- the
//! *critical path*. Only reordering activities on that path can shorten it, and
//! only at the edges of a *block* (a run of the path on one machine): swapping
//! two activities inside a block leaves the block as long as before
//! (Nowicki & Smutnicki 1996, "A fast taboo search algorithm for the job shop
//! problem", neighbourhood N5). Moving an activity inside a block to the
//! block's front or back -- past every other activity of the block at once --
//! is the extension of N5 to N6 (Zhang, Li, Guan & Rao 2007, "A tabu search
//! algorithm with a new neighborhood structure for the job shop scheduling
//! problem"); a move that would close a cycle with the tasks' order is found
//! when the sequences are timed, and skipped. A chromosome improved this way is written back
//! as the order the improved schedule was built in (Lamarckian), so the search
//! the GA runs next starts from it.
//!
//! Where activities choose among resources, a critical activity may also move
//! to another of its candidates -- the flexible job shop's assignment move
//! (Mastrolilli & Gambardella 2000, "Effective neighbourhood functions for the
//! flexible job shop problem") -- inserted where its start falls in that
//! resource's sequence; the chromosome then takes the new assignment too.

use super::*;

/// How many of the block moves estimated best are timed in full at each step.
const TIMED_SWAPS: usize = 3;

/// How many steps undoing a swap stays forbidden. Nowicki & Smutnicki use 8
/// over thousands of steps; with the 60 a chromosome gets here, 5-7 does best
/// (mean gap on Taillard's 40 instances, 200 generations: 3 -> 4.9, 5 -> 4.3,
/// 6 -> 4.5, 8 -> 4.7, 12 -> 4.9, 24 -> 5.2 %).
const TABOO_STEPS: usize = 6;

impl SchedulingGaProblem {
    /// Improves `chromosome` in place: repeatedly the best swap at the edge of a
    /// critical block that lowers the fitness, until none does or
    /// [`local_search_steps`](Self::local_search_steps) swaps have been made.
    /// The schedule the chromosome's order gives never gets worse, and the
    /// chromosome decodes to one no longer than before.
    ///
    /// A chromosome that leaves an activity out (no resource, or fewer
    /// occurrences than activities) is left as it is.
    pub(super) fn improve(&self, chromosome: &mut ScheduleChromosome) {
        if self.local_search_steps == 0 {
            return;
        }
        let view = self.view();
        let mut ix = self.indexed(chromosome);
        let n = self.activities.len();
        if ix.order.len() != n || ix.resource.iter().any(Option::is_none) {
            return;
        }

        // Each activity's neighbours in its task, and the activities that
        // wait for it in other tasks.
        let mut links = Links {
            task_pred: vec![None; n],
            task_succ: vec![None; n],
            waiting_on: vec![Vec::new(); n],
        };
        for ops in &self.task_activities {
            for pair in ops.windows(2) {
                links.task_pred[pair[1]] = Some(pair[0]);
                links.task_succ[pair[0]] = Some(pair[1]);
            }
        }
        for (a, predecessors) in self.predecessors.iter().enumerate() {
            for &p in predecessors {
                links.waiting_on[p].push(a);
            }
        }
        // The machine sequences the order gives.
        let mut seq: Vec<Vec<usize>> = vec![Vec::new(); self.resources.len() + ix.extra.len()];
        for &a in &ix.order {
            seq[ix.resource[a].expect("checked above")].push(a);
        }

        let mut scratch = Scratch::new(n);
        if !self.time_sequences(&view, &ix, &seq, &links, &mut scratch) {
            return;
        }
        let mut placed = std::mem::replace(&mut scratch.placed, Placed::new(n));
        let mut fitness = self.fitness_of(&view, &placed);

        // Taboo search over the block-edge swaps: the best swap is made even when
        // it lengthens the schedule, and undoing a swap is forbidden for a few
        // steps, so the search leaves the local optimum a strict descent stops
        // in. The best sequences seen are the result.
        let mut best_fitness = fitness;
        let mut best_order: Option<Vec<usize>> = None;
        let mut best_resource: Option<Vec<Option<usize>>> = None;
        let mut taboo: Vec<(usize, usize, usize)> = Vec::new();
        // (activity, the resource it left, until): moving it back is forbidden.
        let mut left: Vec<(usize, usize, usize)> = Vec::new();
        let flexible = self.activities.iter().any(|a| a.candidates.len() > 1);
        for step in 0..self.local_search_steps {
            taboo.retain(|&(_, _, until)| until > step);
            left.retain(|&(_, _, until)| until > step);
            let mut best: Option<(f64, Move)> = None;
            let path = self.critical_path(&view, &ix, &seq, &placed, &links);
            let mut moves = Self::block_moves(&ix, &seq, &path);
            self.keep_best_estimated(&view, &seq, &placed, &links, &mut moves);
            if flexible {
                moves.extend(self.reassignments(&view, &ix, &seq, &placed, &links, &path));
            }
            for m in moves {
                let forbidden = match m {
                    Move::Swap(..) | Move::Shift { .. } => {
                        let (u, v) = m.pair(&seq).expect("a block move has a pair");
                        taboo
                            .iter()
                            .any(|&(a, b, _)| (a, b) == (u, v) || (a, b) == (v, u))
                    }
                    Move::Reassign { activity, to, .. } => {
                        left.iter().any(|&(a, r, _)| (a, r) == (activity, to))
                    }
                };
                self.apply(&mut ix, &mut seq, m, false);
                if self.time_sequences(&view, &ix, &seq, &links, &mut scratch) {
                    let f = self.fitness_of(&view, &scratch.placed);
                    // A forbidden move is allowed when it beats the best so far.
                    if (!forbidden || f < best_fitness) && best.is_none_or(|(b, _)| f < b) {
                        best = Some((f, m));
                    }
                }
                self.apply(&mut ix, &mut seq, m, true);
            }
            let Some((f, m)) = best else {
                break;
            };
            match m {
                Move::Swap(..) | Move::Shift { .. } => {
                    let (u, v) = m.pair(&seq).expect("a block move has a pair");
                    taboo.push((u, v, step + TABOO_STEPS));
                }
                Move::Reassign { activity, from, .. } => {
                    left.push((activity, from, step + TABOO_STEPS));
                }
            }
            self.apply(&mut ix, &mut seq, m, false);
            // The best move's timing was in `scratch` only until the next
            // candidate overwrote it.
            let timed = self.time_sequences(&view, &ix, &seq, &links, &mut scratch);
            debug_assert!(timed, "a move that was timed once is timed again");
            fitness = f;
            std::mem::swap(&mut placed, &mut scratch.placed);
            if fitness < best_fitness {
                best_fitness = fitness;
                // `placed.order` is a topological order of the sequences just
                // timed, so placing in it reproduces them.
                best_order = Some(placed.order.clone());
                best_resource = Some(ix.resource.clone());
            }
        }

        if let Some(order) = best_order {
            chromosome.osv = order
                .iter()
                .map(|&a| self.task_ids[self.task_of[a]].clone())
                .collect();
        }
        if let Some(resource) = best_resource {
            for (a, r) in resource.into_iter().enumerate() {
                if let (Some(r), Some(gene)) = (r, chromosome.mav.get_mut(a)) {
                    *gene = match self.resources.get(r) {
                        Some(listed) => listed.id.clone(),
                        None => ix.extra[r - self.resources.len()].clone(),
                    };
                }
            }
        }
    }

    /// Makes `m` on the sequences and the assignment -- or, with `undo`,
    /// takes it back.
    fn apply(&self, ix: &mut Indexed, seq: &mut [Vec<usize>], m: Move, undo: bool) {
        match m {
            Move::Swap(r, i) => seq[r].swap(i, i + 1),
            Move::Shift { resource, from, to } => {
                let (a, b) = if undo { (to, from) } else { (from, to) };
                let activity = seq[resource].remove(a);
                seq[resource].insert(b, activity);
            }
            Move::Reassign {
                activity,
                from,
                at,
                to,
                pos,
            } => {
                let ((r1, i1), (r2, i2)) = if undo {
                    ((to, pos), (from, at))
                } else {
                    ((from, at), (to, pos))
                };
                seq[r1].remove(i1);
                seq[r2].insert(i2, activity);
                ix.resource[activity] = Some(r2);
                ix.duration[activity] = self.duration_on(activity, &self.resources[r2].id);
            }
        }
    }

    /// The moves of a critical activity to another of its candidates that
    /// look best, [`TIMED_SWAPS`] of them. On each candidate the activity goes
    /// where the longest path through it -- the later of its task
    /// predecessor's and its new machine predecessor's ends, its time there,
    /// and the longer of the tails after its task successor and its new
    /// machine successor -- is shortest, among the positions that cannot close
    /// a cycle: after nothing that starts once it ends, before nothing that
    /// ends before it starts (Mastrolilli & Gambardella 2000). The moves are
    /// ranked by that length, from the current heads and tails, so with
    /// setups or transport times it is a ranking, not a bound.
    fn reassignments(
        &self,
        view: &TaskView<'_>,
        ix: &Indexed,
        seq: &[Vec<usize>],
        placed: &Placed,
        links: &Links,
        path: &[usize],
    ) -> Vec<Move> {
        let tail = Self::tails(seq, placed, links);
        let span = |a: usize| placed.end[a] - placed.start[a];
        let after = |x: usize| span(x) + tail[x];
        let mut moves: Vec<(i64, i64, Move)> = Vec::new();
        for &a in path {
            let Some(from) = ix.resource[a] else {
                continue;
            };
            let Some(at) = seq[from].iter().position(|&x| x == a) else {
                continue;
            };
            let head = links.task_pred[a]
                .map_or(0, |t| placed.end[t])
                .max(self.release_of(view, a))
                .max(
                    self.predecessors[a]
                        .iter()
                        .map(|&p| placed.end[p])
                        .max()
                        .unwrap_or(0),
                );
            let rest = links.task_succ[a]
                .into_iter()
                .chain(links.waiting_on[a].iter().copied())
                .map(after)
                .max()
                .unwrap_or(0);
            for candidate in &self.activities[a].candidates {
                let Some(&to) = self.resource_index.get(candidate) else {
                    continue;
                };
                if to == from {
                    continue;
                }
                let time = self.duration_on(a, candidate).total_ms();
                let on = &seq[to];
                let first = on
                    .iter()
                    .take_while(|&&x| placed.end[x] <= placed.start[a])
                    .count();
                let last = on
                    .iter()
                    .take_while(|&&x| placed.start[x] < placed.end[a])
                    .count();
                let Some((length, pos)) = (first..=last)
                    .map(|pos| {
                        let before = pos.checked_sub(1).map_or(0, |k| placed.end[on[k]]);
                        let behind = on.get(pos).map_or(0, |&x| after(x));
                        (head.max(before) + time + rest.max(behind), pos)
                    })
                    .min()
                else {
                    continue;
                };
                moves.push((
                    length,
                    time,
                    Move::Reassign {
                        activity: a,
                        from,
                        at,
                        to,
                        pos,
                    },
                ));
            }
        }
        moves.sort_by_key(|&(length, time, _)| (length, time));
        moves
            .into_iter()
            .take(TIMED_SWAPS)
            .map(|(_, _, m)| m)
            .collect()
    }

    /// The longest path from the end of each activity of the timed sequences
    /// to the schedule's end, through machine and task successors.
    fn tails(seq: &[Vec<usize>], placed: &Placed, links: &Links) -> Vec<i64> {
        let n = placed.start.len();
        let mut machine_succ: Vec<Option<usize>> = vec![None; n];
        for s in seq {
            for pair in s.windows(2) {
                machine_succ[pair[0]] = Some(pair[1]);
            }
        }
        let mut tail = vec![0i64; n];
        for &a in placed.order.iter().rev() {
            tail[a] = [machine_succ[a], links.task_succ[a]]
                .into_iter()
                .flatten()
                .chain(links.waiting_on[a].iter().copied())
                .map(|succ| placed.end[succ] - placed.start[succ] + tail[succ])
                .max()
                .unwrap_or(0);
        }
        tail
    }

    /// A critical path of the timed sequences, first activity first: walked
    /// back from the one that ends last, preferring a machine predecessor that
    /// ends where an activity starts, so a block grows as far as it can.
    fn critical_path(
        &self,
        view: &TaskView<'_>,
        ix: &Indexed,
        seq: &[Vec<usize>],
        placed: &Placed,
        links: &Links,
    ) -> Vec<usize> {
        let Some(last) = placed.order.iter().copied().max_by_key(|&a| placed.end[a]) else {
            return Vec::new();
        };
        // Position of each activity in its machine's sequence.
        let mut position = vec![0usize; self.activities.len()];
        for s in seq {
            for (i, &a) in s.iter().enumerate() {
                position[a] = i;
            }
        }
        // The critical path, walked back from the last activity: a machine
        // predecessor that ends where this one starts is preferred, so a block
        // grows as far as it can.
        let mut path = vec![last];
        let mut a = last;
        loop {
            let r = ix.resource[a].expect("a placed activity has a resource");
            let machine_pred = position[a].checked_sub(1).map(|i| seq[r][i]);
            // A predecessor in another task that ends where this one starts.
            let waited_for = self.predecessors[a]
                .iter()
                .copied()
                .find(|&p| placed.end[p] == placed.start[a]);
            let next = match (machine_pred, links.task_pred[a]) {
                (Some(m), _) if placed.end[m] == placed.start[a] => m,
                (_, Some(t))
                    if self.arrival(
                        view,
                        ix,
                        placed,
                        Some(t),
                        a,
                        r,
                        placed.end[a] - placed.start[a],
                    ) == placed.start[a] =>
                {
                    t
                }
                _ => match waited_for {
                    Some(p) => p,
                    None => break,
                },
            };
            path.push(next);
            a = next;
        }
        path.reverse();
        path
    }

    /// The moves at the critical blocks of `path` (runs of it next to one
    /// another on one machine): swapping the first two or the last two
    /// activities of a block (N5), and moving an activity of a block of three
    /// or more to the block's front or back (N6) -- the moves that a swap of
    /// neighbours already gives are left out.
    fn block_moves(ix: &Indexed, seq: &[Vec<usize>], path: &[usize]) -> Vec<Move> {
        let mut position = vec![0usize; ix.resource.len()];
        for s in seq {
            for (i, &a) in s.iter().enumerate() {
                position[a] = i;
            }
        }
        let mut moves: Vec<Move> = Vec::new();
        let push = |m: Move, moves: &mut Vec<Move>| {
            if !moves.contains(&m) {
                moves.push(m);
            }
        };
        let mut i = 0;
        while i < path.len() {
            let r = ix.resource[path[i]].expect("a placed activity has a resource");
            let mut j = i;
            while j + 1 < path.len()
                && ix.resource[path[j + 1]] == Some(r)
                && position[path[j + 1]] == position[path[j]] + 1
            {
                j += 1;
            }
            if j > i {
                let (first, last) = (position[path[i]], position[path[j]]);
                push(Move::Swap(r, first), &mut moves);
                push(Move::Swap(r, last - 1), &mut moves);
                for at in first + 2..=last {
                    push(
                        Move::Shift {
                            resource: r,
                            from: at,
                            to: first,
                        },
                        &mut moves,
                    );
                }
                for at in first..last - 1 {
                    push(
                        Move::Shift {
                            resource: r,
                            from: at,
                            to: last,
                        },
                        &mut moves,
                    );
                }
            }
            i = j + 1;
        }
        moves
    }

    /// Cuts `moves` to the few that look best, so that only those are timed in
    /// full. A block move reorders one stretch of a machine sequence and
    /// changes the heads and tails of just the activities in it, which gives
    /// the longest path through the stretch from the current heads before it
    /// and tails after it without timing the sequences again -- for a swap of
    /// two neighbours, Taillard's estimate (1994, "Parallel taboo search
    /// techniques for the job shop scheduling problem"). The estimate follows
    /// the processing times as they are, so with sequence-dependent setups or
    /// transport times it is a ranking, not a bound.
    fn keep_best_estimated(
        &self,
        view: &TaskView<'_>,
        seq: &[Vec<usize>],
        placed: &Placed,
        links: &Links,
        moves: &mut Vec<Move>,
    ) {
        if moves.len() <= TIMED_SWAPS {
            return;
        }
        let span = |a: usize| placed.end[a] - placed.start[a];
        let tail = Self::tails(seq, placed, links);
        let ends = |x: Option<usize>| x.map_or(0, |x| placed.end[x]);
        let after = |x: Option<usize>| x.map_or(0, |x| span(x) + tail[x]);
        // The longest path through a stretch `lo..=hi` of machine `r` put in
        // `order`, from the heads before it and the tails after it.
        let through = |r: usize, lo: usize, hi: usize, order: &[usize]| {
            let mut head = ends(lo.checked_sub(1).map(|k| seq[r][k]));
            let mut heads = Vec::with_capacity(order.len());
            for &x in order {
                let h = ends(links.task_pred[x])
                    .max(head)
                    .max(self.release_of(view, x));
                heads.push(h);
                head = h + span(x);
            }
            let mut next = after(seq[r].get(hi + 1).copied());
            let mut longest = 0;
            for (&x, &h) in order.iter().zip(&heads).rev() {
                let t = after(links.task_succ[x]).max(next);
                longest = longest.max(h + span(x) + t);
                next = span(x) + t;
            }
            longest
        };
        moves.sort_by_cached_key(|&m| {
            let (r, lo, hi, order) = m.stretch(seq);
            through(r, lo, hi, &order)
        });
        moves.truncate(TIMED_SWAPS);
    }

    /// Times the machine sequences `seq` into `scratch.placed`: every activity
    /// starts as soon as its predecessor in its task, the one before it on its
    /// machine, its [`predecessors`](SchedulingGaProblem::predecessors) in
    /// other tasks and its release allow -- the bounds the decoders place by.
    /// `false` when the sequences contradict the tasks' order or those
    /// predecessors. The placement's order is a topological order of them all.
    fn time_sequences(
        &self,
        view: &TaskView<'_>,
        ix: &Indexed,
        seq: &[Vec<usize>],
        links: &Links,
        scratch: &mut Scratch,
    ) -> bool {
        let n = self.activities.len();
        let Scratch {
            machine_pred,
            machine_succ,
            waiting,
            ready,
            placed,
        } = scratch;
        machine_pred.fill(None);
        machine_succ.fill(None);
        for s in seq {
            for pair in s.windows(2) {
                machine_pred[pair[1]] = Some(pair[0]);
                machine_succ[pair[0]] = Some(pair[1]);
            }
        }
        ready.clear();
        for a in 0..n {
            waiting[a] = usize::from(machine_pred[a].is_some())
                + usize::from(links.task_pred[a].is_some())
                + self.predecessors[a].len();
            if waiting[a] == 0 {
                ready.push(a);
            }
        }
        placed.order.clear();
        while let Some(a) = ready.pop() {
            let r = ix.resource[a].expect("a sequenced activity has a resource");
            let setup = self.setup_of(view, ix, a, r, machine_pred[a].map(|m| self.task_of[m]));
            let duration = ix.duration[a];
            let span = setup + duration.process_ms + duration.teardown_ms;
            let start = machine_pred[a]
                .map_or(0, |m| placed.end[m])
                .max(self.arrival(view, ix, placed, links.task_pred[a], a, r, span))
                .max(self.release_of(view, a))
                .max(
                    self.predecessors[a]
                        .iter()
                        .map(|&p| placed.end[p])
                        .max()
                        .unwrap_or(0),
                );
            let end = start + span;
            placed.put(a, start, end, setup);
            let successors = [machine_succ[a], links.task_succ[a]]
                .into_iter()
                .flatten()
                .chain(links.waiting_on[a].iter().copied());
            for succ in successors {
                waiting[succ] -= 1;
                if waiting[succ] == 0 {
                    ready.push(succ);
                }
            }
        }
        placed.order.len() == n
    }
}

/// A step of the local search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Move {
    /// Swap the activities at `.1` and `.1 + 1` of machine `.0`'s sequence.
    Swap(usize, usize),
    /// Move the activity at position `from` of `resource`'s sequence to
    /// position `to` of it.
    Shift {
        resource: usize,
        from: usize,
        to: usize,
    },
    /// Move `activity` from position `at` of resource `from` to position
    /// `pos` of resource `to`.
    Reassign {
        activity: usize,
        from: usize,
        at: usize,
        to: usize,
        pos: usize,
    },
}

impl Move {
    /// The activity a block move moves and the one it ends up next to in its
    /// place -- what undoing it would swap back.
    fn pair(self, seq: &[Vec<usize>]) -> Option<(usize, usize)> {
        match self {
            Move::Swap(r, i) => Some((seq[r][i], seq[r][i + 1])),
            Move::Shift { resource, from, to } => Some((seq[resource][from], seq[resource][to])),
            Move::Reassign { .. } => None,
        }
    }

    /// The stretch of a machine sequence a block move reorders, `(resource,
    /// lo, hi, its activities afterwards)`.
    fn stretch(self, seq: &[Vec<usize>]) -> (usize, usize, usize, Vec<usize>) {
        match self {
            Move::Swap(r, i) => (r, i, i + 1, vec![seq[r][i + 1], seq[r][i]]),
            Move::Shift {
                resource: r,
                from,
                to,
            } if to < from => {
                let mut order = vec![seq[r][from]];
                order.extend_from_slice(&seq[r][to..from]);
                (r, to, from, order)
            }
            Move::Shift {
                resource: r,
                from,
                to,
            } => {
                let mut order = seq[r][from + 1..=to].to_vec();
                order.push(seq[r][from]);
                (r, from, to, order)
            }
            Move::Reassign { .. } => unreachable!("a reassignment is not estimated"),
        }
    }
}

/// The neighbours of each activity in its task, and the activities of other
/// tasks that wait for it.
struct Links {
    task_pred: Vec<Option<usize>>,
    task_succ: Vec<Option<usize>>,
    waiting_on: Vec<Vec<usize>>,
}

/// What timing a set of sequences needs, kept between timings.
struct Scratch {
    machine_pred: Vec<Option<usize>>,
    machine_succ: Vec<Option<usize>>,
    waiting: Vec<usize>,
    ready: Vec<usize>,
    placed: Placed,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            machine_pred: vec![None; n],
            machine_succ: vec![None; n],
            waiting: vec![0; n],
            ready: Vec::with_capacity(n),
            placed: Placed::new(n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indexed(resource: Vec<Option<usize>>) -> Indexed {
        Indexed {
            order: Vec::new(),
            resource,
            duration: Vec::new(),
            extra: Vec::new(),
        }
    }

    #[test]
    fn a_block_of_four_gives_its_edge_swaps_and_the_moves_to_its_ends() {
        // One machine runs 0..4 in that order, all on the critical path.
        let ix = indexed(vec![Some(0); 4]);
        let seq = vec![vec![0, 1, 2, 3]];
        let moves = SchedulingGaProblem::block_moves(&ix, &seq, &[0, 1, 2, 3]);
        let shift = |from, to| Move::Shift {
            resource: 0,
            from,
            to,
        };
        assert_eq!(
            moves,
            vec![
                Move::Swap(0, 0),
                Move::Swap(0, 2),
                shift(2, 0),
                shift(3, 0),
                shift(0, 3),
                shift(1, 3),
            ]
        );
    }

    #[test]
    fn a_block_of_two_gives_one_swap_and_no_shift() {
        let ix = indexed(vec![Some(0), Some(0), Some(1)]);
        let seq = vec![vec![0, 1], vec![2]];
        let moves = SchedulingGaProblem::block_moves(&ix, &seq, &[0, 1, 2]);
        assert_eq!(moves, vec![Move::Swap(0, 0)]);
    }

    #[test]
    fn a_shift_names_the_stretch_it_reorders_and_the_pair_it_makes() {
        let seq = vec![vec![5, 6, 7, 8]];
        let front = Move::Shift {
            resource: 0,
            from: 3,
            to: 1,
        };
        assert_eq!(front.stretch(&seq), (0, 1, 3, vec![8, 6, 7]));
        assert_eq!(front.pair(&seq), Some((8, 6)));
        let back = Move::Shift {
            resource: 0,
            from: 0,
            to: 2,
        };
        assert_eq!(back.stretch(&seq), (0, 0, 2, vec![6, 7, 5]));
        // A swap is the stretch of two put the other way round.
        assert_eq!(Move::Swap(0, 2).stretch(&seq), (0, 2, 3, vec![8, 7]));
    }
}
