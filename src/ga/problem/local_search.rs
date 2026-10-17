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

use super::*;

/// How many of the swaps estimated best are timed in full at each step.
const TIMED_SWAPS: usize = 3;

/// How many steps undoing a swap stays forbidden (Nowicki & Smutnicki use 8).
const TABOO_STEPS: usize = 8;

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
        let ix = self.indexed(chromosome);
        let n = self.activities.len();
        if ix.order.len() != n || ix.resource.iter().any(Option::is_none) {
            return;
        }

        // Each activity's neighbours in its task.
        let mut links = Links {
            task_pred: vec![None; n],
            task_succ: vec![None; n],
        };
        for ops in &self.task_activities {
            for pair in ops.windows(2) {
                links.task_pred[pair[1]] = Some(pair[0]);
                links.task_succ[pair[0]] = Some(pair[1]);
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
        let mut taboo: Vec<(usize, usize, usize)> = Vec::new();
        for step in 0..self.local_search_steps {
            taboo.retain(|&(_, _, until)| until > step);
            let mut best: Option<(f64, usize, usize)> = None;
            let mut swaps = self.block_edge_swaps(&ix, &seq, &placed, &links);
            self.keep_best_estimated(&view, &seq, &placed, &links, &mut swaps);
            for (r, i) in swaps {
                let (u, v) = (seq[r][i], seq[r][i + 1]);
                let forbidden = taboo
                    .iter()
                    .any(|&(a, b, _)| (a, b) == (u, v) || (a, b) == (v, u));
                seq[r].swap(i, i + 1);
                if self.time_sequences(&view, &ix, &seq, &links, &mut scratch) {
                    let f = self.fitness_of(&view, &scratch.placed);
                    // A forbidden swap is allowed when it beats the best so far.
                    if (!forbidden || f < best_fitness) && best.is_none_or(|(b, ..)| f < b) {
                        best = Some((f, r, i));
                    }
                }
                seq[r].swap(i, i + 1);
            }
            let Some((f, r, i)) = best else {
                break;
            };
            let (u, v) = (seq[r][i], seq[r][i + 1]);
            seq[r].swap(i, i + 1);
            // The best swap's timing was in `scratch` only until the next
            // candidate overwrote it.
            let timed = self.time_sequences(&view, &ix, &seq, &links, &mut scratch);
            debug_assert!(timed, "a swap that was timed once is timed again");
            fitness = f;
            std::mem::swap(&mut placed, &mut scratch.placed);
            taboo.push((u, v, step + TABOO_STEPS));
            if fitness < best_fitness {
                best_fitness = fitness;
                // `placed.order` is a topological order of the sequences just
                // timed, so placing in it reproduces them.
                best_order = Some(placed.order.clone());
            }
        }

        if let Some(order) = best_order {
            chromosome.osv = order
                .iter()
                .map(|&a| self.task_ids[self.task_of[a]].clone())
                .collect();
        }
    }

    /// The swaps of two neighbours in a machine sequence that can shorten the
    /// critical path: the first two and the last two activities of every block
    /// of at least two, as `(machine, position of the first)`.
    fn block_edge_swaps(
        &self,
        ix: &Indexed,
        seq: &[Vec<usize>],
        placed: &Placed,
        links: &Links,
    ) -> Vec<(usize, usize)> {
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
            let next = match (machine_pred, links.task_pred[a]) {
                (Some(m), _) if placed.end[m] == placed.start[a] => m,
                (_, Some(t)) if placed.end[t] == placed.start[a] => t,
                _ => break,
            };
            path.push(next);
            a = next;
        }
        path.reverse();

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
    /// are, so with sequence-dependent setups it is a ranking, not a bound.
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
                .max(view.release[self.task_of[v]]);
            let head_u = ends(links.task_pred[u])
                .max(head_v + span(v))
                .max(view.release[self.task_of[u]]);
            let tail_u = after(links.task_succ[u]).max(after(after_v));
            let tail_v = after(links.task_succ[v]).max(span(u) + tail_u);
            (head_v + span(v) + tail_v).max(head_u + span(u) + tail_u)
        });
        swaps.truncate(TIMED_SWAPS);
    }

    /// Times the machine sequences `seq` into `scratch.placed`: every activity
    /// starts as soon as its predecessor in its task, the one before it on its
    /// machine and its task's release allow. `false` when the sequences
    /// contradict the tasks' order. The placement's order is a topological
    /// order of the sequences.
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
            waiting[a] =
                u8::from(machine_pred[a].is_some()) + u8::from(links.task_pred[a].is_some());
            if waiting[a] == 0 {
                ready.push(a);
            }
        }
        placed.order.clear();
        while let Some(a) = ready.pop() {
            let r = ix.resource[a].expect("a sequenced activity has a resource");
            let t = self.task_of[a];
            let start = machine_pred[a]
                .map_or(0, |m| placed.end[m])
                .max(links.task_pred[a].map_or(0, |p| placed.end[p]))
                .max(view.release[t]);
            let setup = self.setup_of(view, ix, a, r, machine_pred[a].map(|m| self.task_of[m]));
            let duration = &self.activities[a].duration;
            let end = start + setup + duration.process_ms + duration.teardown_ms;
            placed.put(a, start, end, setup);
            for succ in [machine_succ[a], links.task_succ[a]].into_iter().flatten() {
                waiting[succ] -= 1;
                if waiting[succ] == 0 {
                    ready.push(succ);
                }
            }
        }
        placed.order.len() == n
    }
}

/// The neighbours of each activity in its task.
struct Links {
    task_pred: Vec<Option<usize>>,
    task_succ: Vec<Option<usize>>,
}

/// What timing a set of sequences needs, kept between timings.
struct Scratch {
    machine_pred: Vec<Option<usize>>,
    machine_succ: Vec<Option<usize>>,
    waiting: Vec<u8>,
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
