//! Scheduling GA problem definition.
//!
//! Implements `u_metaheur::ga::GaProblem` for scheduling optimization.
//! Bridges domain models (Task, Resource) to the generic GA framework.
//!
//! # Reference
//! Cheng et al. (1996), "A Tutorial Survey of JSSP using GA"

use std::collections::HashMap;

use rand::Rng;
use u_metaheur::ga::GaProblem;

use super::chromosome::{
    ScheduleChromosome, insert_mutation, mav_mutation, pox_crossover, swap_mutation,
};
use crate::models::{
    Assignment, Resource, Schedule, Task, TransitionMatrixCollection,
};

/// Compact activity descriptor for GA encoding.
///
/// Extracted from `Task`/`Activity` to avoid cloning full domain objects.
#[derive(Debug, Clone)]
pub struct ActivityInfo {
    /// Parent task ID.
    pub task_id: String,
    /// Activity sequence within task (1-based).
    pub sequence: i32,
    /// Processing time (ms).
    pub process_ms: i64,
    /// Candidate resource IDs.
    pub candidates: Vec<String>,
}

impl ActivityInfo {
    /// Extracts activity info from domain tasks.
    pub fn from_tasks(tasks: &[Task]) -> Vec<Self> {
        let mut infos = Vec::new();
        for task in tasks {
            for (i, activity) in task.activities.iter().enumerate() {
                infos.push(ActivityInfo {
                    task_id: task.id.clone(),
                    sequence: (i + 1) as i32,
                    process_ms: activity.duration.process_ms,
                    candidates: activity.candidate_resources().into_iter().map(|s| s.to_string()).collect(),
                });
            }
        }
        infos
    }
}

/// GA problem definition for scheduling optimization.
///
/// Decodes chromosomes into schedules and evaluates fitness as makespan.
///
/// # Example
/// ```no_run
/// use u_schedule::ga::{SchedulingGaProblem, ActivityInfo};
/// use u_schedule::models::{Task, Resource, ResourceType};
/// use u_metaheur::ga::{GaConfig, GaRunner};
///
/// let tasks = vec![/* ... */];
/// let resources = vec![/* ... */];
/// let problem = SchedulingGaProblem::new(&tasks, &resources);
/// let config = GaConfig::default();
/// let result = GaRunner::run(&problem, &config);
/// ```
pub struct SchedulingGaProblem {
    /// Activity info (extracted from tasks).
    pub activities: Vec<ActivityInfo>,
    /// Available resources.
    pub resources: Vec<Resource>,
    /// Task categories (task_id → category).
    pub task_categories: HashMap<String, String>,
    /// Transition matrices for setup times.
    pub transition_matrices: TransitionMatrixCollection,
    /// Task deadlines (task_id → deadline_ms).
    pub deadlines: HashMap<String, i64>,
    /// Task release times (task_id → release_ms).
    pub release_times: HashMap<String, i64>,
    /// Weight for tardiness in fitness (default: 0.5).
    pub tardiness_weight: f64,
}

impl SchedulingGaProblem {
    /// Creates a problem from domain models.
    pub fn new(tasks: &[Task], resources: &[Resource]) -> Self {
        let activities = ActivityInfo::from_tasks(tasks);
        let mut task_categories = HashMap::new();
        let mut deadlines = HashMap::new();
        let mut release_times = HashMap::new();

        for task in tasks {
            task_categories.insert(task.id.clone(), task.category.clone());
            if let Some(dl) = task.deadline {
                deadlines.insert(task.id.clone(), dl);
            }
            if let Some(rt) = task.release_time {
                release_times.insert(task.id.clone(), rt);
            }
        }

        Self {
            activities,
            resources: resources.to_vec(),
            task_categories,
            transition_matrices: TransitionMatrixCollection::new(),
            deadlines,
            release_times,
            tardiness_weight: 0.5,
        }
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets tardiness weight (0.0 = pure makespan, 1.0 = pure tardiness).
    pub fn with_tardiness_weight(mut self, weight: f64) -> Self {
        self.tardiness_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Decodes a chromosome into a Schedule.
    pub fn decode(&self, chromosome: &ScheduleChromosome) -> Schedule {
        let mut schedule = Schedule::new();
        let mut resource_available: HashMap<&str, i64> = HashMap::new();
        let mut task_available: HashMap<&str, i64> = HashMap::new();
        let mut last_category: HashMap<&str, &str> = HashMap::new();

        // Initialize resource availability
        for resource in &self.resources {
            resource_available.insert(&resource.id, 0);
        }

        // Decode OSV to get operation order
        let operation_order = chromosome.decode_osv();

        for (task_id, seq) in &operation_order {
            // Find activity info
            let act_idx = match self
                .activities
                .iter()
                .position(|a| a.task_id == *task_id && a.sequence == *seq)
            {
                Some(idx) => idx,
                None => continue,
            };
            let act = &self.activities[act_idx];

            // Get assigned resource from MAV
            let resource_id = match chromosome.resource_for(task_id, *seq) {
                Some(r) if !r.is_empty() => r,
                _ => continue,
            };

            // Calculate start time
            let resource_ready = resource_available.get(resource_id).copied().unwrap_or(0);
            let task_ready = task_available.get(task_id.as_str()).copied().unwrap_or(0);
            let release = self.release_times.get(task_id).copied().unwrap_or(0);
            let earliest = resource_ready.max(task_ready).max(release);

            // Setup time
            let setup = if let Some(&prev_cat) = last_category.get(resource_id) {
                let task_cat = self
                    .task_categories
                    .get(task_id)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                self.transition_matrices
                    .get_transition_time(resource_id, prev_cat, task_cat)
            } else {
                0
            };

            let start = earliest;
            let end = start + setup + act.process_ms;

            schedule.add_assignment(
                Assignment::new(&act.task_id, task_id, resource_id, start, end)
                    .with_setup(setup),
            );

            // Update state
            resource_available.insert(resource_id, end);
            task_available.insert(task_id, end);
            if let Some(cat) = self.task_categories.get(task_id) {
                last_category.insert(resource_id, cat);
            }
        }

        schedule
    }

    /// Computes fitness: weighted combination of makespan and tardiness.
    fn compute_fitness(&self, schedule: &Schedule) -> f64 {
        let makespan = schedule.makespan_ms() as f64;

        let total_tardiness: f64 = self
            .deadlines
            .iter()
            .map(|(task_id, &deadline)| {
                let completion = schedule.task_completion_time(task_id).unwrap_or(0);
                (completion - deadline).max(0) as f64
            })
            .sum();

        // Weighted combination (both terms in ms, comparable scale)
        let makespan_weight = 1.0 - self.tardiness_weight;
        makespan_weight * makespan + self.tardiness_weight * total_tardiness
    }
}

impl GaProblem for SchedulingGaProblem {
    type Individual = ScheduleChromosome;

    fn create_individual<R: Rng>(&self, rng: &mut R) -> ScheduleChromosome {
        // 50% random, 50% load-balanced
        if rng.random_bool(0.5) {
            ScheduleChromosome::random(&self.activities, rng)
        } else {
            let cap: HashMap<String, i64> = self
                .resources
                .iter()
                .map(|r| (r.id.clone(), r.capacity as i64))
                .collect();
            ScheduleChromosome::with_load_balancing(&self.activities, &cap, rng)
        }
    }

    fn evaluate(&self, individual: &ScheduleChromosome) -> f64 {
        let schedule = self.decode(individual);
        self.compute_fitness(&schedule)
    }

    fn crossover<R: Rng>(
        &self,
        parent1: &ScheduleChromosome,
        parent2: &ScheduleChromosome,
        rng: &mut R,
    ) -> Vec<ScheduleChromosome> {
        let (c1, c2) = pox_crossover(parent1, parent2, &self.activities, rng);
        vec![c1, c2]
    }

    fn mutate<R: Rng>(&self, individual: &mut ScheduleChromosome, rng: &mut R) {
        // OSV mutation: 50% swap, 50% insert
        if rng.random_bool(0.5) {
            swap_mutation(individual, rng);
        } else {
            insert_mutation(individual, rng);
        }
        // Always mutate MAV as well
        mav_mutation(individual, &self.activities, rng);
    }
}

// Make SchedulingGaProblem Send + Sync (all fields are owned data)
unsafe impl Send for SchedulingGaProblem {}
unsafe impl Sync for SchedulingGaProblem {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Activity, ActivityDuration, ResourceRequirement, ResourceType};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use u_metaheur::ga::{GaConfig, GaRunner};

    fn make_test_problem() -> (Vec<Task>, Vec<Resource>) {
        let tasks = vec![
            Task::new("T1")
                .with_category("TypeA")
                .with_priority(5)
                .with_deadline(10_000)
                .with_activity(
                    Activity::new("T1_O1", "T1", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into(), "M2".into()]),
                        ),
                )
                .with_activity(
                    Activity::new("T1_O2", "T1", 1)
                        .with_duration(ActivityDuration::fixed(2000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M2".into()]),
                        ),
                ),
            Task::new("T2")
                .with_category("TypeB")
                .with_priority(3)
                .with_activity(
                    Activity::new("T2_O1", "T2", 0)
                        .with_duration(ActivityDuration::fixed(1500))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into(), "M3".into()]),
                        ),
                ),
        ];

        let resources = vec![
            Resource::new("M1", ResourceType::Primary),
            Resource::new("M2", ResourceType::Primary),
            Resource::new("M3", ResourceType::Primary),
        ];

        (tasks, resources)
    }

    #[test]
    fn test_activity_info_from_tasks() {
        let (tasks, _) = make_test_problem();
        let infos = ActivityInfo::from_tasks(&tasks);
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].task_id, "T1");
        assert_eq!(infos[0].sequence, 1);
        assert_eq!(infos[0].process_ms, 1000);
        assert_eq!(infos[2].task_id, "T2");
    }

    #[test]
    fn test_decode_chromosome() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&tasks, &resources);
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem.create_individual(&mut rng);

        let schedule = problem.decode(&ch);
        // Should have assignments for all 3 activities
        assert!(schedule.assignment_count() > 0);
        assert!(schedule.makespan_ms() > 0);
    }

    #[test]
    fn test_fitness_computation() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&tasks, &resources);
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem.create_individual(&mut rng);

        let fitness = problem.evaluate(&ch);
        assert!(fitness.is_finite());
        assert!(fitness > 0.0);
    }

    #[test]
    fn test_ga_runner_integration() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&tasks, &resources);
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config);
        assert!(result.best_fitness.is_finite());
        assert!(result.best_fitness < f64::INFINITY);
        assert!(result.generations > 0);
    }

    #[test]
    fn test_crossover_and_mutation() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&tasks, &resources);
        let mut rng = SmallRng::seed_from_u64(42);

        let p1 = problem.create_individual(&mut rng);
        let p2 = problem.create_individual(&mut rng);

        let children = problem.crossover(&p1, &p2, &mut rng);
        assert_eq!(children.len(), 2);

        let mut child = children[0].clone();
        problem.mutate(&mut child, &mut rng);
        assert_eq!(child.osv.len(), p1.osv.len());
    }

    #[test]
    fn test_tardiness_weight() {
        let (tasks, resources) = make_test_problem();
        let problem_makespan = SchedulingGaProblem::new(&tasks, &resources)
            .with_tardiness_weight(0.0);
        let problem_tardy = SchedulingGaProblem::new(&tasks, &resources)
            .with_tardiness_weight(1.0);

        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem_makespan.create_individual(&mut rng);

        let f1 = problem_makespan.evaluate(&ch);
        let f2 = problem_tardy.evaluate(&ch);
        // Different weights should give different fitness
        assert!(f1 != f2 || (f1 == 0.0 && f2 == 0.0));
    }
}
