//! Critical-block local search on a chromosome's machine sequences.
//!
//! A schedule is no longer than the longest chain of activities that follow
//! one another, each starting the moment the one before it ends -- the
//! *critical path*. Only reordering activities on that path can shorten it, and
//! only at the edges of a *block* (a run of the path on one machine): swapping
//! two activities inside a block leaves the block as long as before
//! (Nowicki & Smutnicki 1996, "A fast taboo search algorithm for the job shop
//! problem", neighbourhood N5). A chromosome improved this way is written back
//! as the order the improved schedule was built in (Lamarckian), so the search
//! the GA runs next starts from it.
//!
//! Where activities choose among resources, a critical activity may also move
//! to another of its candidates -- the flexible job shop's assignment move
//! (Mastrolilli & Gambardella 2000, "Effective neighbourhood functions for the
//! flexible job shop problem") -- inserted where its start falls in that
//! resource's sequence; the chromosome then takes the new assignment too.

use super::*;

/// How many of the swaps estimated best are timed in full at each step.
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
            let mut swaps = Self::block_edge_swaps(&ix, &seq, &path);
            self.keep_best_estimated(&view, &seq, &placed, &links, &mut swaps);
            let mut moves: Vec<Move> = swaps.into_iter().map(|(r, i)| Move::Swap(r, i)).collect();
            if flexible {
                moves.extend(self.reassignments(&ix, &seq, &placed, &path));
            }
            for m in moves {
                let forbidden = match m {
                    Move::Swap(r, i) => {
                        let (u, v) = (seq[r][i], seq[r][i + 1]);
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
                Move::Swap(r, i) => taboo.push((seq[r][i], seq[r][i + 1], step + TABOO_STEPS)),
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

    /// The moves of a critical activity to another of its candidates, each
    /// inserted where its present start falls in that resource's sequence:
    /// those that take least time there, [`TIMED_SWAPS`] of them.
    fn reassignments(
        &self,
        ix: &Indexed,
        seq: &[Vec<usize>],
        placed: &Placed,
        path: &[usize],
    ) -> Vec<Move> {
        let mut moves: Vec<(i64, Move)> = Vec::new();
        for &a in path {
            let Some(from) = ix.resource[a] else {
                continue;
            };
            let Some(at) = seq[from].iter().position(|&x| x == a) else {
                continue;
            };
            for candidate in &self.activities[a].candidates {
                let Some(&to) = self.resource_index.get(candidate) else {
                    continue;
                };
                if to == from {
                    continue;
                }
                let pos = seq[to]
                    .iter()
                    .take_while(|&&x| placed.start[x] < placed.start[a])
                    .count();
                let time = self.duration_on(a, candidate).total_ms();
                moves.push((
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
        moves.sort_by_key(|&(time, _)| time);
        moves
            .into_iter()
            .take(TIMED_SWAPS)
            .map(|(_, m)| m)
            .collect()
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

    /// The swaps of two neighbours in a machine sequence that can shorten the
    /// critical `path`: the first two and the last two activities of every
    /// block of at least two, as `(machine, position of the first)`.
    fn block_edge_swaps(ix: &Indexed, seq: &[Vec<usize>], path: &[usize]) -> Vec<(usize, usize)> {
        let mut position = vec![0usize; ix.resource.len()];
        for s in seq {
            for (i, &a) in s.iter().enumerate() {
                position[a] = i;
            }
        }
        let mut swaps: Vec<(usize, usize)> = Vec::new();
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
                for swap in [(r, position[path[i]]), (r, position[path[j]] - 1)] {
                    if !swaps.contains(&swap) {
                        swaps.push(swap);
                    }
                }
            }
            i = j + 1;
        }
        swaps
    }

    /// Cuts `swaps` to the few that look best, so that only those are timed in
    /// full. A swap of `u` and `v` (neighbours on a machine) changes the heads
    /// and tails of just those two, which gives the longest path through them
    /// from the current heads and tails without timing the sequences again
    /// (Taillard 1994, "Parallel taboo search techniques for the job shop
    /// scheduling problem"). The estimate follows the processing times as they
    /// are, so with sequence-dependent setups or transport times it is a
    /// ranking, not a bound.
    fn keep_best_estimated(
        &self,
        view: &TaskView<'_>,
        seq: &[Vec<usize>],
        placed: &Placed,
        links: &Links,
        swaps: &mut Vec<(usize, usize)>,
    ) {
        if swaps.len() <= TIMED_SWAPS {
            return;
        }
        let n = self.activities.len();
        let span = |a: usize| placed.end[a] - placed.start[a];
        let mut machine_succ: Vec<Option<usize>> = vec![None; n];
        for s in seq {
            for pair in s.windows(2) {
                machine_succ[pair[0]] = Some(pair[1]);
            }
        }
        // The longest path from the end of each activity to the schedule's end.
        let mut tail = vec![0i64; n];
        for &a in placed.order.iter().rev() {
            tail[a] = [machine_succ[a], links.task_succ[a]]
                .into_iter()
                .flatten()
                .map(|succ| span(succ) + tail[succ])
                .max()
                .unwrap_or(0);
        }
        let ends = |x: Option<usize>| x.map_or(0, |x| placed.end[x]);
        let after = |x: Option<usize>| x.map_or(0, |x| span(x) + tail[x]);
        swaps.sort_by_cached_key(|&(r, i)| {
            let (u, v) = (seq[r][i], seq[r][i + 1]);
            let before_u = i.checked_sub(1).map(|k| seq[r][k]);
            let after_v = seq[r].get(i + 2).copied();
            let head_v = ends(links.task_pred[v])
                .max(ends(before_u))
                .max(self.release_of(view, v));
            let head_u = ends(links.task_pred[u])
                .max(head_v + span(v))
                .max(self.release_of(view, u));
            let tail_u = after(links.task_succ[u]).max(after(after_v));
            let tail_v = after(links.task_succ[v]).max(span(u) + tail_u);
            (head_v + span(v) + tail_v).max(head_u + span(u) + tail_u)
        });
        swaps.truncate(TIMED_SWAPS);
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
#[derive(Debug, Clone, Copy)]
enum Move {
    /// Swap the activities at `.1` and `.1 + 1` of machine `.0`'s sequence.
    Swap(usize, usize),
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
