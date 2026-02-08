//! Configurable genetic operators for scheduling.
//!
//! Provides runtime-selectable crossover and mutation strategies
//! via [`GeneticOperators`].
//!
//! # Usage
//!
//! ```
//! use u_schedule::ga::operators::{GeneticOperators, CrossoverType, MutationType};
//!
//! let ops = GeneticOperators::default();
//! assert_eq!(ops.crossover_type, CrossoverType::POX);
//! assert_eq!(ops.mutation_type, MutationType::Swap);
//! ```

use rand::Rng;

use super::chromosome::{
    ScheduleChromosome, insert_mutation, invert_mutation, jox_crossover, lox_crossover,
    mav_mutation, pox_crossover, swap_mutation,
};
use super::problem::ActivityInfo;

/// Crossover strategy for scheduling chromosomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverType {
    /// Precedence Operation Crossover (Bierwirth et al., 1996).
    POX,
    /// Linear Order Crossover (Falkenauer & Bouffouix, 1991).
    LOX,
    /// Job-based Order Crossover (Yamada & Nakano, 1997).
    JOX,
}

/// Mutation strategy for scheduling chromosomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationType {
    /// Swap two random positions in the OSV.
    Swap,
    /// Remove and reinsert at a random position.
    Insert,
    /// Reverse a random segment of the OSV.
    Invert,
}

/// Runtime-selectable genetic operators for scheduling GA.
///
/// Wraps crossover and mutation strategy selection so that
/// u-aps can switch operators via configuration without
/// changing the GA problem definition.
///
/// # Example
///
/// ```
/// use u_schedule::ga::operators::{GeneticOperators, CrossoverType, MutationType};
///
/// let ops = GeneticOperators {
///     crossover_type: CrossoverType::LOX,
///     mutation_type: MutationType::Invert,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct GeneticOperators {
    /// Crossover strategy.
    pub crossover_type: CrossoverType,
    /// OSV mutation strategy.
    pub mutation_type: MutationType,
}

impl Default for GeneticOperators {
    fn default() -> Self {
        Self {
            crossover_type: CrossoverType::POX,
            mutation_type: MutationType::Swap,
        }
    }
}

impl GeneticOperators {
    /// Performs crossover using the configured strategy.
    pub fn crossover<R: Rng>(
        &self,
        p1: &ScheduleChromosome,
        p2: &ScheduleChromosome,
        activities: &[ActivityInfo],
        rng: &mut R,
    ) -> (ScheduleChromosome, ScheduleChromosome) {
        match self.crossover_type {
            CrossoverType::POX => pox_crossover(p1, p2, activities, rng),
            CrossoverType::LOX => lox_crossover(p1, p2, activities, rng),
            CrossoverType::JOX => jox_crossover(p1, p2, activities, rng),
        }
    }

    /// Performs mutation using the configured strategy.
    ///
    /// Always also applies MAV mutation to diversify resource assignments.
    pub fn mutate<R: Rng>(
        &self,
        chromosome: &mut ScheduleChromosome,
        activities: &[ActivityInfo],
        rng: &mut R,
    ) {
        match self.mutation_type {
            MutationType::Swap => swap_mutation(chromosome, rng),
            MutationType::Insert => insert_mutation(chromosome, rng),
            MutationType::Invert => invert_mutation(chromosome, rng),
        }
        mav_mutation(chromosome, activities, rng);
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
    fn test_default_operators() {
        let ops = GeneticOperators::default();
        assert_eq!(ops.crossover_type, CrossoverType::POX);
        assert_eq!(ops.mutation_type, MutationType::Swap);
    }

    #[test]
    fn test_crossover_pox() {
        let acts = sample_activities();
        let ops = GeneticOperators::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let p1 = ScheduleChromosome::random(&acts, &mut rng);
        let p2 = ScheduleChromosome::random(&acts, &mut rng);

        let (c1, c2) = ops.crossover(&p1, &p2, &acts, &mut rng);
        assert_eq!(c1.osv.len(), 3);
        assert_eq!(c2.osv.len(), 3);
    }

    #[test]
    fn test_crossover_lox() {
        let acts = sample_activities();
        let ops = GeneticOperators {
            crossover_type: CrossoverType::LOX,
            mutation_type: MutationType::Swap,
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let p1 = ScheduleChromosome::random(&acts, &mut rng);
        let p2 = ScheduleChromosome::random(&acts, &mut rng);

        let (c1, c2) = ops.crossover(&p1, &p2, &acts, &mut rng);
        assert_eq!(c1.osv.len(), 3);
        assert_eq!(c2.osv.len(), 3);
    }

    #[test]
    fn test_crossover_jox() {
        let acts = sample_activities();
        let ops = GeneticOperators {
            crossover_type: CrossoverType::JOX,
            mutation_type: MutationType::Swap,
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let p1 = ScheduleChromosome::random(&acts, &mut rng);
        let p2 = ScheduleChromosome::random(&acts, &mut rng);

        let (c1, c2) = ops.crossover(&p1, &p2, &acts, &mut rng);
        assert_eq!(c1.osv.len(), 3);
        assert_eq!(c2.osv.len(), 3);
    }

    #[test]
    fn test_mutation_swap() {
        let acts = sample_activities();
        let ops = GeneticOperators::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        ops.mutate(&mut ch, &acts, &mut rng);
        assert_eq!(ch.osv.len(), 3);
    }

    #[test]
    fn test_mutation_insert() {
        let acts = sample_activities();
        let ops = GeneticOperators {
            crossover_type: CrossoverType::POX,
            mutation_type: MutationType::Insert,
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        ops.mutate(&mut ch, &acts, &mut rng);
        assert_eq!(ch.osv.len(), 3);
    }

    #[test]
    fn test_mutation_invert() {
        let acts = sample_activities();
        let ops = GeneticOperators {
            crossover_type: CrossoverType::POX,
            mutation_type: MutationType::Invert,
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let mut ch = ScheduleChromosome::random(&acts, &mut rng);

        ops.mutate(&mut ch, &acts, &mut rng);
        assert_eq!(ch.osv.len(), 3);
    }

    #[test]
    fn test_mutate_always_applies_mav() {
        let acts = sample_activities();
        let ops = GeneticOperators::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = ScheduleChromosome::random(&acts, &mut rng);
        let original_mav = ch.mav.clone();

        // Run enough mutations that MAV changes at least once
        let mut mav_changed = false;
        for _ in 0..50 {
            let mut ch2 = ch.clone();
            ops.mutate(&mut ch2, &acts, &mut rng);
            if ch2.mav != original_mav {
                mav_changed = true;
                break;
            }
        }
        assert!(mav_changed, "MAV mutation should occur alongside OSV mutation");
    }
}
