//! OSV/MAV dual-vector chromosome for scheduling.
//!
//! # Encoding
//!
//! The chromosome consists of two vectors:
//! - **OSV**: Permutation of task IDs. The k-th occurrence of a task ID
//!   corresponds to the k-th activity of that task.
//! - **MAV**: Parallel to the activity list (sorted by task/sequence).
//!   Each element is a resource ID from the activity's candidate list.
//!
//! # Reference
//! Bierwirth (1995), "A generalized permutation approach to JSSP"

use std::collections::HashMap;

use rand::prelude::IndexedRandom;
use rand::{Rng, RngExt};
use u_metaheur::ga::Individual;

use super::ActivityInfo;

/// OSV/MAV dual-vector chromosome for scheduling GA.
///
/// Lower fitness = better schedule (minimization convention).
#[derive(Debug, Clone)]
pub struct ScheduleChromosome {
    /// Operation Sequence Vector: task IDs in execution order.
    pub osv: Vec<String>,
    /// Machine Assignment Vector: resource ID per activity.
    pub mav: Vec<String>,
    /// (task_id, sequence) → index in MAV.
    pub activity_index: HashMap<(String, i32), usize>,
    /// Fitness value (lower = better).
    pub fitness: f64,
}

impl Individual for ScheduleChromosome {
    type Fitness = f64;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }
}

impl ScheduleChromosome {
    /// Creates a random chromosome.
    pub fn random<R: Rng>(activities: &[ActivityInfo], rng: &mut R) -> Self {
        let (osv, activity_index) = Self::create_random_osv(activities, rng);
        let mav = Self::create_random_mav(activities, rng);
        Self {
            osv,
            mav,
            activity_index,
            fitness: f64::INFINITY,
        }
    }

    /// Creates a load-balanced chromosome: random OSV, and each activity on
    /// the candidate where it would end the least loaded -- the resource's
    /// load so far plus the activity's time on it (`time_on(activity index,
    /// resource)`), which is then added to that load. Tasks are taken in a
    /// random order, so the population does not share one assignment.
    ///
    /// # Reference
    /// Global selection -- Zhang, Gao & Shi (2011), "An effective genetic
    /// algorithm for the flexible job-shop scheduling problem", *Expert
    /// Systems with Applications* 38(4), 3563-3573.
    pub fn with_load_balancing<R: Rng>(
        activities: &[ActivityInfo],
        time_on: impl Fn(usize, &str) -> i64,
        rng: &mut R,
    ) -> Self {
        let (osv, activity_index) = Self::create_random_osv(activities, rng);
        let mut resource_load: HashMap<&str, i64> = HashMap::new();
        let mut mav = vec![String::new(); activities.len()];

        let mut tasks: Vec<&str> = Vec::new();
        for act in activities {
            if !tasks.contains(&act.task_id.as_str()) {
                tasks.push(&act.task_id);
            }
        }
        // With no choice anywhere the order changes nothing: draw nothing.
        if activities.iter().any(|a| a.candidates.len() > 1) {
            u_numflow::random::shuffle(&mut tasks, rng);
        }
        for task in tasks {
            for (i, act) in activities.iter().enumerate() {
                if act.task_id != task || act.candidates.is_empty() {
                    continue;
                }
                let (best, ends) = act
                    .candidates
                    .iter()
                    .map(|c| {
                        let load = resource_load.get(c.as_str()).copied().unwrap_or(0);
                        (c, load + time_on(i, c))
                    })
                    .min_by_key(|&(_, ends)| ends)
                    .expect("candidates checked non-empty above");
                resource_load.insert(best, ends);
                mav[i] = best.clone();
            }
        }

        Self {
            osv,
            mav,
            activity_index,
            fitness: f64::INFINITY,
        }
    }

    /// Creates a chromosome with shortest processing time assignment.
    ///
    /// For each activity, selects the candidate resource it processes fastest
    /// on — `process_on(activity index, resource)` — the first such in
    /// candidate order.
    ///
    /// # Reference
    /// SPT (Shortest Processing Time) heuristic — Conway et al. (1967)
    pub fn with_shortest_time<R: Rng>(
        activities: &[ActivityInfo],
        process_on: impl Fn(usize, &str) -> i64,
        rng: &mut R,
    ) -> Self {
        let (osv, activity_index) = Self::create_random_osv(activities, rng);
        let mav = Self::create_shortest_time_mav(activities, process_on);
        Self {
            osv,
            mav,
            activity_index,
            fitness: f64::INFINITY,
        }
    }

    /// Decodes the OSV into (task_id, sequence) pairs.
    pub fn decode_osv(&self) -> Vec<(String, i32)> {
        let mut task_counters: HashMap<&str, i32> = HashMap::new();
        self.osv
            .iter()
            .map(|task_id| {
                let seq = task_counters.entry(task_id.as_str()).or_insert(0);
                *seq += 1;
                (task_id.clone(), *seq)
            })
            .collect()
    }

    /// Gets the assigned resource for a (task_id, sequence) pair.
    pub fn resource_for(&self, task_id: &str, sequence: i32) -> Option<&str> {
        self.activity_index
            .get(&(task_id.to_string(), sequence))
            .and_then(|&idx| self.mav.get(idx))
            .map(|s| s.as_str())
    }

    /// Sets the assigned resource for a (task_id, sequence) pair.
    ///
    /// Does nothing if the activity is not found or the index is out of bounds.
    pub fn set_resource(&mut self, task_id: &str, sequence: i32, resource_id: String) {
        if let Some(&idx) = self.activity_index.get(&(task_id.to_string(), sequence)) {
            if idx < self.mav.len() {
                self.mav[idx] = resource_id;
            }
        }
    }

    /// Validates the chromosome against activity info.
    pub fn is_valid(&self, activities: &[ActivityInfo]) -> bool {
        if self.osv.len() != activities.len() || self.mav.len() != activities.len() {
            return false;
        }

        // Check task count conservation
        let mut osv_counts: HashMap<&str, i32> = HashMap::new();
        for task_id in &self.osv {
            *osv_counts.entry(task_id.as_str()).or_insert(0) += 1;
        }
        let mut expected_counts: HashMap<&str, i32> = HashMap::new();
        for act in activities {
            *expected_counts.entry(act.task_id.as_str()).or_insert(0) += 1;
        }
        if osv_counts != expected_counts {
            return false;
        }

        // Check resource feasibility
        for (idx, act) in activities.iter().enumerate() {
            if !act.candidates.is_empty() && !act.candidates.contains(&self.mav[idx]) {
                return false;
            }
        }

        true
    }

    fn create_random_osv<R: Rng>(
        activities: &[ActivityInfo],
        rng: &mut R,
    ) -> (Vec<String>, HashMap<(String, i32), usize>) {
        // Build OSV: list of task IDs (one per activity)
        let mut osv: Vec<String> = activities.iter().map(|a| a.task_id.clone()).collect();
        u_numflow::random::shuffle(&mut osv, rng);

        // Build activity index
        let mut activity_index = HashMap::new();
        for (idx, act) in activities.iter().enumerate() {
            activity_index.insert((act.task_id.clone(), act.sequence), idx);
        }

        (osv, activity_index)
    }

    fn create_random_mav<R: Rng>(activities: &[ActivityInfo], rng: &mut R) -> Vec<String> {
        activities
            .iter()
            .map(|act| {
                if act.candidates.is_empty() {
                    String::new()
                } else {
                    act.candidates
                        .choose(rng)
                        .expect("candidates checked non-empty")
                        .clone()
                }
            })
            .collect()
    }

    fn create_shortest_time_mav(
        activities: &[ActivityInfo],
        process_on: impl Fn(usize, &str) -> i64,
    ) -> Vec<String> {
        activities
            .iter()
            .enumerate()
            .map(|(i, act)| {
                if act.candidates.is_empty() {
                    return String::new();
                }
                act.candidates
                    .iter()
                    .min_by_key(|c| process_on(i, c))
                    .expect("candidates checked non-empty above")
                    .clone()
            })
            .collect()
    }
}

// ======================== Crossover operators ========================

/// Uniform crossover of the machine assignments: where the parents differ,
/// each child takes either parent's resource with even odds (the other child
/// the other one). An assignment is a choice per activity, independent of the
/// order the OSV crossover recombines, so it recombines gene by gene.
///
/// # Reference
/// Pezzella, Morganti & Ciaschetti (2008), "A genetic algorithm for the
/// flexible job-shop scheduling problem", *Computers & Operations Research*
/// 35(10); Zhang, Gao & Shi (2011), *Expert Systems with Applications* 38(4).
fn uniform_mav<R: Rng>(p1: &[String], p2: &[String], rng: &mut R) -> (Vec<String>, Vec<String>) {
    let (mut c1, mut c2) = (p1.to_vec(), p2.to_vec());
    for i in 0..c1.len().min(c2.len()) {
        if c1[i] != c2[i] && rng.random_bool(0.5) {
            std::mem::swap(&mut c1[i], &mut c2[i]);
        }
    }
    (c1, c2)
}

/// Performs POX (Precedence Operation Crossover).
///
/// Selects a random subset of tasks, preserves their positions from parent 1,
/// fills remaining from parent 2 in order.
///
/// # Reference
/// Bierwirth et al. (1996)
pub fn pox_crossover<R: Rng>(
    p1: &ScheduleChromosome,
    p2: &ScheduleChromosome,
    activities: &[ActivityInfo],
    rng: &mut R,
) -> (ScheduleChromosome, ScheduleChromosome) {
    // The unique task IDs, in the order the activities first name them -- a
    // hash map's order differs from run to run, and with it the sample drawn
    // below, so a seeded run would not repeat.
    let task_ids: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        activities
            .iter()
            .filter(|act| seen.insert(act.task_id.as_str()))
            .map(|act| act.task_id.clone())
            .collect()
    };

    if task_ids.is_empty() {
        return (p1.clone(), p2.clone());
    }

    let set_size = rng.random_range(1..=task_ids.len().max(1));
    let selected: Vec<String> = task_ids.sample(rng, set_size).cloned().collect();
    let selected_set: std::collections::HashSet<&str> =
        selected.iter().map(|s| s.as_str()).collect();

    let child1_osv = pox_build_child(&p1.osv, &p2.osv, &selected_set);
    let child2_osv = pox_build_child(&p2.osv, &p1.osv, &selected_set);

    let (mav1, mav2) = uniform_mav(&p1.mav, &p2.mav, rng);
    let child1 = ScheduleChromosome {
        osv: child1_osv,
        mav: mav1,
        activity_index: p1.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    let child2 = ScheduleChromosome {
        osv: child2_osv,
        mav: mav2,
        activity_index: p2.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    (child1, child2)
}

fn pox_build_child(
    template: &[String],
    donor: &[String],
    selected: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut child = vec![String::new(); template.len()];
    let mut donor_iter = donor.iter().filter(|t| !selected.contains(t.as_str()));

    for (i, task) in template.iter().enumerate() {
        if selected.contains(task.as_str()) {
            child[i] = task.clone();
        } else if let Some(t) = donor_iter.next() {
            child[i] = t.clone();
        }
    }
    child
}

/// Performs LOX (Linear Order Crossover).
///
/// 1. Selects a random contiguous segment `[start..=end]` from parent 1.
/// 2. Copies that segment to the same positions in the child.
/// 3. Fills remaining positions circularly from parent 2, preserving
///    parent 2's relative order.
///
/// # Reference
/// Falkenauer & Bouffouix (1991), "A genetic algorithm for job shop"
pub fn lox_crossover<R: Rng>(
    p1: &ScheduleChromosome,
    p2: &ScheduleChromosome,
    _activities: &[ActivityInfo],
    rng: &mut R,
) -> (ScheduleChromosome, ScheduleChromosome) {
    let len = p1.osv.len();
    if len < 2 {
        return (p1.clone(), p2.clone());
    }

    let start = rng.random_range(0..len);
    let end = rng.random_range(0..len);
    let (start, end) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };

    let child1_osv = lox_build_child(&p1.osv, &p2.osv, start, end);
    let child2_osv = lox_build_child(&p2.osv, &p1.osv, start, end);

    let (mav1, mav2) = uniform_mav(&p1.mav, &p2.mav, rng);
    let child1 = ScheduleChromosome {
        osv: child1_osv,
        mav: mav1,
        activity_index: p1.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    let child2 = ScheduleChromosome {
        osv: child2_osv,
        mav: mav2,
        activity_index: p2.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    (child1, child2)
}

fn lox_build_child(p1: &[String], p2: &[String], start: usize, end: usize) -> Vec<String> {
    let len = p1.len();
    let mut child = vec![String::new(); len];

    // Copy segment from P1
    let segment: Vec<String> = p1[start..=end].to_vec();
    for (i, item) in segment.iter().enumerate() {
        child[start + i] = item.clone();
    }

    // Count how many of each task are in the segment
    let mut seg_counts: HashMap<&str, usize> = HashMap::new();
    for item in &segment {
        *seg_counts.entry(item.as_str()).or_insert(0) += 1;
    }

    // Track how many of each task from P2 we've already skipped
    let mut p2_counts: HashMap<&str, usize> = HashMap::new();
    for item in p2 {
        *p2_counts.entry(item.as_str()).or_insert(0) += 1;
    }

    // Fill remaining positions circularly from P2
    let mut child_idx = (end + 1) % len;
    let mut skip_counts: HashMap<&str, usize> = HashMap::new();

    for i in 0..len {
        let p2_idx = (end + 1 + i) % len;
        let item = &p2[p2_idx];

        let seg_count = seg_counts.get(item.as_str()).copied().unwrap_or(0);
        let skipped = skip_counts.get(item.as_str()).copied().unwrap_or(0);

        if skipped < seg_count {
            // Skip this occurrence (already in segment)
            *skip_counts.entry(item.as_str()).or_insert(0) += 1;
            continue;
        }

        if child_idx == start {
            break;
        }

        child[child_idx] = item.clone();
        child_idx = (child_idx + 1) % len;
    }

    child
}

/// Performs JOX (Job-based Order Crossover).
///
/// 1. Randomly selects a subset of jobs (task IDs).
/// 2. Preserves selected jobs from parent 1 at their exact positions.
/// 3. Fills remaining positions from parent 2 in relative order.
///
/// Differs from POX: JOX preserves exact positions (absolute), while
/// POX preserves positions relative to the template layout.
///
/// # Reference
/// Yamada & Nakano (1997), "Job shop scheduling"
pub fn jox_crossover<R: Rng>(
    p1: &ScheduleChromosome,
    p2: &ScheduleChromosome,
    activities: &[ActivityInfo],
    rng: &mut R,
) -> (ScheduleChromosome, ScheduleChromosome) {
    let task_ids: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        for act in activities {
            seen.insert(act.task_id.clone());
        }
        seen.into_iter().collect()
    };

    if task_ids.is_empty() {
        return (p1.clone(), p2.clone());
    }

    let set_size = rng.random_range(1..=task_ids.len().max(1));
    let selected: std::collections::HashSet<String> =
        task_ids.sample(rng, set_size).cloned().collect();

    let child1_osv = jox_build_child(&p1.osv, &p2.osv, &selected);
    let child2_osv = jox_build_child(&p2.osv, &p1.osv, &selected);

    let (mav1, mav2) = uniform_mav(&p1.mav, &p2.mav, rng);
    let child1 = ScheduleChromosome {
        osv: child1_osv,
        mav: mav1,
        activity_index: p1.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    let child2 = ScheduleChromosome {
        osv: child2_osv,
        mav: mav2,
        activity_index: p2.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    (child1, child2)
}

fn jox_build_child(
    primary: &[String],
    donor: &[String],
    selected: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut child = vec![String::new(); primary.len()];

    // Place selected jobs from primary at their exact positions
    for (i, task) in primary.iter().enumerate() {
        if selected.contains(task) {
            child[i] = task.clone();
        }
    }

    // Fill remaining positions from donor in order
    let mut donor_iter = donor.iter().filter(|t| !selected.contains(t.as_str()));
    for slot in &mut child {
        if slot.is_empty() {
            if let Some(t) = donor_iter.next() {
                *slot = t.clone();
            }
        }
    }

    child
}

// ======================== Mutation operators ========================

/// Swap mutation: exchanges two random positions in the OSV.
pub fn swap_mutation<R: Rng>(chromosome: &mut ScheduleChromosome, rng: &mut R) {
    let len = chromosome.osv.len();
    if len < 2 {
        return;
    }
    let i = rng.random_range(0..len);
    let j = rng.random_range(0..len);
    chromosome.osv.swap(i, j);
}

/// Insert mutation: removes an element and reinserts at a random position.
pub fn insert_mutation<R: Rng>(chromosome: &mut ScheduleChromosome, rng: &mut R) {
    let len = chromosome.osv.len();
    if len < 2 {
        return;
    }
    let from = rng.random_range(0..len);
    let to = rng.random_range(0..len);
    let item = chromosome.osv.remove(from);
    chromosome.osv.insert(to, item);
}

/// Invert mutation: reverses a random segment of the OSV.
pub fn invert_mutation<R: Rng>(chromosome: &mut ScheduleChromosome, rng: &mut R) {
    let len = chromosome.osv.len();
    if len < 2 {
        return;
    }
    let mut i = rng.random_range(0..len);
    let mut j = rng.random_range(0..len);
    if i > j {
        std::mem::swap(&mut i, &mut j);
    }
    chromosome.osv[i..=j].reverse();
}

/// MAV mutation: reassigns one random activity to a different candidate resource.
pub fn mav_mutation<R: Rng>(
    chromosome: &mut ScheduleChromosome,
    activities: &[ActivityInfo],
    rng: &mut R,
) {
    if chromosome.mav.is_empty() || activities.is_empty() {
        return;
    }
    let idx = rng.random_range(0..chromosome.mav.len().min(activities.len()));
    if !activities[idx].candidates.is_empty() {
        chromosome.mav[idx] = activities[idx]
            .candidates
            .choose(rng)
            .expect("candidates checked non-empty")
            .clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn sample_activities() -> Vec<ActivityInfo> {
        vec![
            ActivityInfo {
                id: "T1_1".into(),
                task_id: "T1".into(),
                sequence: 1,
                duration: crate::models::ActivityDuration::fixed(1000),
                candidates: vec!["M1".into(), "M2".into()],
            },
            ActivityInfo {
                id: "T1_2".into(),
                task_id: "T1".into(),
                sequence: 2,
                duration: crate::models::ActivityDuration::fixed(2000),
                candidates: vec!["M2".into()],
            },
            ActivityInfo {
                id: "T2_1".into(),
                task_id: "T2".into(),
                sequence: 1,
                duration: crate::models::ActivityDuration::fixed(1500),
                candidates: vec!["M1".into(), "M3".into()],
            },
        ]
    }

    #[test]
    fn test_random_chromosome() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = ScheduleChromosome::random(&acts, &mut rng);

        assert_eq!(ch.osv.len(), 3);
        assert_eq!(ch.mav.len(), 3);
        assert!(ch.is_valid(&acts));
        assert_eq!(ch.fitness, f64::INFINITY);
    }

    #[test]
    fn test_decode_osv() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = ScheduleChromosome::random(&acts, &mut rng);

        let decoded = ch.decode_osv();
        assert_eq!(decoded.len(), 3);

        // Count: T1 appears 2 times, T2 appears 1 time
        let t1_count = decoded.iter().filter(|(t, _)| t == "T1").count();
        let t2_count = decoded.iter().filter(|(t, _)| t == "T2").count();
        assert_eq!(t1_count, 2);
        assert_eq!(t2_count, 1);
    }

    #[test]
    fn test_load_balanced() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = ScheduleChromosome::with_load_balancing(&acts, |_, _| 1000, &mut rng);
        assert!(ch.is_valid(&acts));

        // Three one-step tasks, 10 on M1 or 12 on M2: whatever the order, the
        // loads go 10 | 12 | 20 -- two on M1, one on M2.
        let acts: Vec<ActivityInfo> = (0..3)
            .map(|t| ActivityInfo {
                id: format!("T{t}_1"),
                task_id: format!("T{t}"),
                sequence: 1,
                duration: crate::models::ActivityDuration::fixed(12),
                candidates: vec!["M1".into(), "M2".into()],
            })
            .collect();
        for seed in 0..5 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let ch = ScheduleChromosome::with_load_balancing(
                &acts,
                |_, r| if r == "M1" { 10 } else { 12 },
                &mut rng,
            );
            assert_eq!(ch.mav.iter().filter(|m| *m == "M1").count(), 2);
        }
    }

    #[test]
    fn crossover_recombines_the_machine_assignments() {
        // Parents that differ in every assignment: a child with genes of both.
        let acts: Vec<ActivityInfo> = (0..20)
            .map(|t| ActivityInfo {
                id: format!("T{t}_1"),
                task_id: format!("T{t}"),
                sequence: 1,
                duration: crate::models::ActivityDuration::fixed(5),
                candidates: vec!["M1".into(), "M2".into()],
            })
            .collect();
        let mut rng = SmallRng::seed_from_u64(7);
        let mut p1 = ScheduleChromosome::random(&acts, &mut rng);
        let mut p2 = p1.clone();
        p1.mav = vec!["M1".into(); 20];
        p2.mav = vec!["M2".into(); 20];
        for (c1, c2) in [
            pox_crossover(&p1, &p2, &acts, &mut rng),
            lox_crossover(&p1, &p2, &acts, &mut rng),
            jox_crossover(&p1, &p2, &acts, &mut rng),
        ] {
            assert!(c1.mav.contains(&"M1".to_string()) && c1.mav.contains(&"M2".to_string()));
            // The children split each gene between them.
            for i in 0..20 {
                assert_ne!(c1.mav[i], c2.mav[i]);
            }
        }
    }

    #[test]
    fn test_pox_crossover() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let p1 = ScheduleChromosome::random(&acts, &mut rng);
        let p2 = ScheduleChromosome::random(&acts, &mut rng);

        let (c1, c2) = pox_crossover(&p1, &p2, &acts, &mut rng);
        assert_eq!(c1.osv.len(), 3);
        assert_eq!(c2.osv.len(), 3);
        // Children have reset fitness
        assert_eq!(c1.fitness, f64::INFINITY);
        assert_eq!(c2.fitness, f64::INFINITY);
    }

    #[test]
    fn test_lox_crossover() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let p1 = ScheduleChromosome::random(&acts, &mut rng);
        let p2 = ScheduleChromosome::random(&acts, &mut rng);

        let (c1, c2) = lox_crossover(&p1, &p2, &acts, &mut rng);
        assert_eq!(c1.osv.len(), 3);
        assert_eq!(c2.osv.len(), 3);
        assert_eq!(c1.fitness, f64::INFINITY);
        assert_eq!(c2.fitness, f64::INFINITY);

        // Task counts must be preserved
        let mut c1_sorted = c1.osv.clone();
        c1_sorted.sort();
        let mut p1_sorted = p1.osv.clone();
        p1_sorted.sort();
        assert_eq!(c1_sorted, p1_sorted);
    }

    #[test]
    fn test_lox_crossover_preserves_segment() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(99);

        // Run multiple times to exercise various segments
        for seed in 0..20 {
            let mut rng2 = SmallRng::seed_from_u64(seed);
            let p1 = ScheduleChromosome::random(&acts, &mut rng2);
            let p2 = ScheduleChromosome::random(&acts, &mut rng2);

            let (c1, _c2) = lox_crossover(&p1, &p2, &acts, &mut rng);

            // OSV length preserved
            assert_eq!(c1.osv.len(), p1.osv.len());

            // Task count conservation
            let mut c1_sorted = c1.osv.clone();
            c1_sorted.sort();
            let mut p1_sorted = p1.osv.clone();
            p1_sorted.sort();
            assert_eq!(c1_sorted, p1_sorted, "seed={seed}");
        }
    }

    #[test]
    fn test_jox_crossover() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let p1 = ScheduleChromosome::random(&acts, &mut rng);
        let p2 = ScheduleChromosome::random(&acts, &mut rng);

        let (c1, c2) = jox_crossover(&p1, &p2, &acts, &mut rng);
        assert_eq!(c1.osv.len(), 3);
        assert_eq!(c2.osv.len(), 3);
        assert_eq!(c1.fitness, f64::INFINITY);
        assert_eq!(c2.fitness, f64::INFINITY);

        // Task counts must be preserved
        let mut c1_sorted = c1.osv.clone();
        c1_sorted.sort();
        let mut p1_sorted = p1.osv.clone();
        p1_sorted.sort();
        assert_eq!(c1_sorted, p1_sorted);
    }

    #[test]
    fn test_jox_preserves_selected_positions() {
        let acts = sample_activities();

        // Run multiple times
        for seed in 0..20 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let p1 = ScheduleChromosome::random(&acts, &mut rng);
            let p2 = ScheduleChromosome::random(&acts, &mut rng);

            let (c1, _c2) = jox_crossover(&p1, &p2, &acts, &mut rng);

            // Task count conservation
            let mut c1_sorted = c1.osv.clone();
            c1_sorted.sort();
            let mut p1_sorted = p1.osv.clone();
            p1_sorted.sort();
            assert_eq!(c1_sorted, p1_sorted, "seed={seed}");
        }
    }

    #[test]
    fn test_swap_mutation() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);
        let original = ch.osv.clone();

        // Run enough times to get a different OSV
        for _ in 0..100 {
            swap_mutation(&mut ch, &mut rng);
        }
        // OSV should still have same elements
        let mut sorted_orig = original.clone();
        sorted_orig.sort();
        let mut sorted_new = ch.osv.clone();
        sorted_new.sort();
        assert_eq!(sorted_orig, sorted_new);
    }

    #[test]
    fn test_insert_mutation() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        insert_mutation(&mut ch, &mut rng);
        assert_eq!(ch.osv.len(), 3);
    }

    #[test]
    fn test_invert_mutation() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        invert_mutation(&mut ch, &mut rng);
        assert_eq!(ch.osv.len(), 3);
    }

    #[test]
    fn test_mav_mutation() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        mav_mutation(&mut ch, &acts, &mut rng);
        assert!(ch.is_valid(&acts));
    }

    #[test]
    fn test_resource_for() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = ScheduleChromosome::random(&acts, &mut rng);

        // T1 seq 1 should have a valid resource
        let r = ch.resource_for("T1", 1);
        assert!(r.is_some());
        assert!(acts[0].candidates.contains(&r.unwrap().to_string()));
    }

    #[test]
    fn test_invalid_chromosome() {
        let acts = sample_activities();
        let ch = ScheduleChromosome {
            osv: vec!["T1".into(), "T1".into()], // Wrong length
            mav: vec!["M1".into(), "M2".into(), "M1".into()],
            activity_index: HashMap::new(),
            fitness: 0.0,
        };
        assert!(!ch.is_valid(&acts));
    }

    #[test]
    fn test_with_shortest_time() {
        let acts = sample_activities();
        // M1 is faster for T1/seq1, M3 for T2/seq1
        let times: HashMap<(usize, &str), i64> = [
            ((0, "M1"), 500),
            ((0, "M2"), 900),
            ((2, "M1"), 1500),
            ((2, "M3"), 800),
        ]
        .into_iter()
        .collect();
        let mut rng = SmallRng::seed_from_u64(42);

        let ch = ScheduleChromosome::with_shortest_time(
            &acts,
            |a, r| {
                times
                    .get(&(a, r))
                    .copied()
                    .unwrap_or(acts[a].duration.process_ms)
            },
            &mut rng,
        );

        assert!(ch.is_valid(&acts));
        assert_eq!(ch.resource_for("T1", 1), Some("M1"));
        assert_eq!(ch.resource_for("T1", 2), Some("M2")); // Only candidate
        assert_eq!(ch.resource_for("T2", 1), Some("M3"));
    }

    #[test]
    fn test_with_shortest_time_on_equal_times_takes_the_first_candidate() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);

        let ch = ScheduleChromosome::with_shortest_time(
            &acts,
            |a, _| acts[a].duration.process_ms,
            &mut rng,
        );

        assert!(ch.is_valid(&acts));
        assert_eq!(ch.resource_for("T1", 1), Some("M1"));
        assert_eq!(ch.resource_for("T2", 1), Some("M1"));
    }

    #[test]
    fn test_set_resource() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        // Change T1/seq1 resource to M2
        ch.set_resource("T1", 1, "M2".into());
        assert_eq!(ch.resource_for("T1", 1), Some("M2"));

        // Set unknown activity — should do nothing
        ch.set_resource("T99", 1, "X".into());
        assert!(ch.resource_for("T99", 1).is_none());
    }

    #[test]
    fn test_set_resource_preserves_validity() {
        let acts = sample_activities();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        // Set T2/seq1 to M1 (valid candidate)
        ch.set_resource("T2", 1, "M1".into());
        assert_eq!(ch.resource_for("T2", 1), Some("M1"));
        assert!(ch.is_valid(&acts));
    }
}
