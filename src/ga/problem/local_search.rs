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
        let start_fitness = fitness;

        for _ in 0..self.local_search_steps {
            let mut best: Option<(f64, usize, usize)> = None;
            for (r, i) in self.block_edge_swaps(&ix, &seq, &placed, &links) {
                seq[r].swap(i, i + 1);
                if self.time_sequences(&view, &ix, &seq, &links, &mut scratch) {
                    let f = self.fitness_of(&view, &scratch.placed);
                    if f < fitness && best.is_none_or(|(b, ..)| f < b) {
                        best = Some((f, r, i));
                    }
                }
                seq[r].swap(i, i + 1);
            }
            let Some((f, r, i)) = best else {
                break;
            };
            seq[r].swap(i, i + 1);
            // The best swap's timing was in `scratch` only until the next
            // candidate overwrote it.
            let timed = self.time_sequences(&view, &ix, &seq, &links, &mut scratch);
            debug_assert!(timed, "a swap that was timed once is timed again");
            fitness = f;
            std::mem::swap(&mut placed, &mut scratch.placed);
        }

        if fitness < start_fitness {
            // `placed.order` is a topological order of the improved sequences,
            // so placing in it reproduces them.
            chromosome.osv = placed
                .order
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
