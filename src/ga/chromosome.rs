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

use rand::Rng;
use rand::prelude::IndexedRandom;
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

    /// Creates a load-balanced chromosome (random OSV, least-loaded MAV).
    pub fn with_load_balancing<R: Rng>(
        activities: &[ActivityInfo],
        _resource_capacity: &HashMap<String, i64>,
        rng: &mut R,
    ) -> Self {
        let (osv, activity_index) = Self::create_random_osv(activities, rng);
        let mut resource_load: HashMap<String, i64> = HashMap::new();
        let mut mav = Vec::with_capacity(activities.len());

        for act in activities {
            if act.candidates.is_empty() {
                mav.push(String::new());
                continue;
            }
            // Select least-loaded candidate
            let best = act
                .candidates
                .iter()
                .min_by_key(|c| resource_load.get(*c).copied().unwrap_or(0))
                .unwrap()
                .clone();
            *resource_load.entry(best.clone()).or_insert(0) += act.process_ms;
            mav.push(best);
        }

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
        u_optim::random::shuffle(&mut osv, rng);

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
                    act.candidates.choose(rng).unwrap().clone()
                }
            })
            .collect()
    }
}

// ======================== Crossover operators ========================

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
    // Collect unique task IDs
    let task_ids: Vec<String> = {
        let mut seen = HashMap::new();
        for act in activities {
            seen.entry(act.task_id.clone()).or_insert(());
        }
        seen.into_keys().collect()
    };

    if task_ids.is_empty() {
        return (p1.clone(), p2.clone());
    }

    let set_size = rng.random_range(1..=task_ids.len().max(1));
    let selected: Vec<String> = task_ids.choose_multiple(rng, set_size).cloned().collect();
    let selected_set: std::collections::HashSet<&str> =
        selected.iter().map(|s| s.as_str()).collect();

    let child1_osv = pox_build_child(&p1.osv, &p2.osv, &selected_set);
    let child2_osv = pox_build_child(&p2.osv, &p1.osv, &selected_set);

    let child1 = ScheduleChromosome {
        osv: child1_osv,
        mav: p1.mav.clone(),
        activity_index: p1.activity_index.clone(),
        fitness: f64::INFINITY,
    };
    let child2 = ScheduleChromosome {
        osv: child2_osv,
        mav: p2.mav.clone(),
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
        chromosome.mav[idx] = activities[idx].candidates.choose(rng).unwrap().clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn sample_activities() -> Vec<ActivityInfo> {
        vec![
            ActivityInfo {
                task_id: "T1".into(),
                sequence: 1,
                process_ms: 1000,
                candidates: vec!["M1".into(), "M2".into()],
            },
            ActivityInfo {
                task_id: "T1".into(),
                sequence: 2,
                process_ms: 2000,
                candidates: vec!["M2".into()],
            },
            ActivityInfo {
                task_id: "T2".into(),
                sequence: 1,
                process_ms: 1500,
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
        let cap: HashMap<String, i64> = [("M1".into(), 1), ("M2".into(), 1), ("M3".into(), 1)]
            .into_iter()
            .collect();
        let ch = ScheduleChromosome::with_load_balancing(&acts, &cap, &mut rng);

        assert!(ch.is_valid(&acts));
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
}
