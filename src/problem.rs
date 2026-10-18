//! A scheduling problem whose input has been checked.

use crate::models::{Resource, Task};
use crate::validation::{validate_input, ValidationError};

/// Tasks and the resources they run on, checked once so every solver can rely
/// on them.
///
/// A `Problem` exists only for input that passes
/// [`validate_input`]: every task, activity and resource has its own id, every
/// resource an activity names and every predecessor it waits on exists, the
/// precedence graph has no cycle, and every task has an activity, each with
/// its own `sequence`. A task's activities are kept in `sequence` order — the
/// order its steps run in, which every solver reads. The solvers
/// ([`SchedulingGaProblem`](crate::ga::SchedulingGaProblem),
/// [`ScheduleCpBuilder`](crate::cp::ScheduleCpBuilder),
/// [`SimpleScheduler`](crate::scheduler::SimpleScheduler)) take only this type.
///
/// They look entities up by id, so input that breaks these rules used to be
/// solved anyway into something else: two tasks sharing an id became one, and
/// the other was missing from the schedule with nothing saying so. The check
/// existed, but nothing made a caller run it.
///
/// # Example
///
/// ```
/// use u_schedule::Problem;
/// use u_schedule::models::{Activity, Resource, Task};
///
/// let tasks = vec![
///     Task::new("J1").with_activity(Activity::new("O1", "J1", 0).with_process_time(100)),
///     Task::new("J1").with_activity(Activity::new("O2", "J1", 0).with_process_time(100)),
/// ];
/// let errors = Problem::new(tasks, vec![Resource::new("M1", "Machine")]).unwrap_err();
/// assert_eq!(errors[0].kind.code(), "duplicate_id");
/// ```
#[derive(Debug, Clone)]
pub struct Problem {
    tasks: Vec<Task>,
    resources: Vec<Resource>,
}

impl Problem {
    /// Checks the input and wraps it.
    ///
    /// # Errors
    ///
    /// Every problem [`validate_input`] finds, not only the first.
    pub fn new(
        mut tasks: Vec<Task>,
        resources: Vec<Resource>,
    ) -> Result<Self, Vec<ValidationError>> {
        validate_input(&tasks, &resources)?;
        for task in &mut tasks {
            task.activities.sort_by_key(|a| a.sequence);
        }
        Ok(Self { tasks, resources })
    }

    /// The tasks, in the order given — each one's activities in `sequence`
    /// order.
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// The resources, in the order given.
    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }

    /// Takes the tasks and resources back out.
    pub fn into_parts(self) -> (Vec<Task>, Vec<Resource>) {
        (self.tasks, self.resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Activity;

    #[test]
    fn valid_input_keeps_its_order() {
        let tasks = vec![
            Task::new("B").with_activity(Activity::new("B1", "B", 0).with_process_time(1)),
            Task::new("A").with_activity(Activity::new("A1", "A", 0).with_process_time(1)),
        ];
        let problem = Problem::new(tasks, vec![]).expect("valid");
        let ids: Vec<&str> = problem.tasks().iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, ["B", "A"]);
    }

    #[test]
    fn a_tasks_steps_run_in_sequence_order_whatever_order_they_are_listed_in() {
        let tasks = vec![Task::new("J")
            .with_activity(Activity::new("second", "J", 2).with_process_time(1))
            .with_activity(Activity::new("first", "J", 1).with_process_time(1))];
        let problem = Problem::new(tasks, vec![]).expect("valid");
        let ids: Vec<&str> = problem.tasks()[0]
            .activities
            .iter()
            .map(|a| a.id.as_str())
            .collect();
        assert_eq!(ids, ["first", "second"]);
    }

    #[test]
    fn two_steps_of_a_task_with_one_sequence_are_refused() {
        let tasks = vec![Task::new("J")
            .with_activity(Activity::new("a", "J", 1).with_process_time(1))
            .with_activity(Activity::new("b", "J", 1).with_process_time(1))];
        let errors = Problem::new(tasks, vec![]).expect_err("ambiguous order");
        assert_eq!(
            errors[0].kind,
            crate::validation::ValidationErrorKind::DuplicateSequence {
                task: "J".into(),
                sequence: 1
            }
        );
        assert_eq!(errors[0].kind.code(), "duplicate_id");
    }

    #[test]
    fn an_overlap_needs_a_previous_step_a_share_in_0_1_and_a_lag_of_at_least_0() {
        use crate::models::Overlap;
        use crate::validation::ValidationErrorKind as K;
        let tasks = vec![Task::new("J")
            .with_activity(
                Activity::new("a", "J", 1)
                    .with_process_time(1)
                    .with_overlap(Overlap::new(0.5)),
            )
            .with_activity(
                Activity::new("b", "J", 2)
                    .with_process_time(1)
                    .with_overlap(Overlap::new(1.5).with_finish_after(-1)),
            )];
        let errors = Problem::new(tasks, vec![]).expect_err("three faults");
        let kinds: Vec<&K> = errors.iter().map(|e| &e.kind).collect();
        assert!(kinds.contains(&&K::OverlapWithoutPreviousStep {
            activity: "a".into()
        }));
        assert!(kinds.contains(&&K::OverlapShareOutOfRange {
            activity: "b".into(),
            start_after: 1.5
        }));
        assert!(kinds.contains(&&K::NegativeOverlapLag {
            activity: "b".into(),
            finish_after_ms: -1
        }));
        // A NaN share is out of range too.
        let nan = vec![Task::new("J")
            .with_activity(Activity::new("a", "J", 1).with_process_time(1))
            .with_activity(
                Activity::new("b", "J", 2)
                    .with_process_time(1)
                    .with_overlap(Overlap::new(f64::NAN)),
            )];
        assert_eq!(
            Problem::new(nan, vec![]).expect_err("NaN")[0].kind.code(),
            "parameter_out_of_range"
        );
    }

    #[test]
    fn a_split_takes_at_least_one_part_and_no_negative_sizes() {
        use crate::models::Split;
        use crate::validation::ValidationErrorKind as K;
        let tasks = vec![Task::new("J").with_activity(
            Activity::new("a", "J", 1)
                .with_process_time(10)
                .with_split(Split::new(0).with_min_part(-1).with_part_overhead(-2)),
        )];
        let errors = Problem::new(tasks, vec![]).expect_err("three faults");
        let got: Vec<(&str, i64)> = errors
            .iter()
            .filter_map(|e| match &e.kind {
                K::SplitOutOfRange { parameter, got, .. } => Some((*parameter, *got)),
                _ => None,
            })
            .collect();
        assert_eq!(
            got,
            [
                ("max_parts", 0),
                ("min_part_ms", -1),
                ("part_overhead_ms", -2)
            ]
        );
        assert_eq!(errors[0].kind.code(), "parameter_out_of_range");
    }

    #[test]
    fn every_error_is_reported_not_only_the_first() {
        let tasks = vec![Task::new("empty"), Task::new("empty")];
        let errors = Problem::new(tasks, vec![]).expect_err("invalid");
        let codes: Vec<&str> = errors.iter().map(|e| e.kind.code()).collect();
        assert!(codes.contains(&"duplicate_id"), "{codes:?}");
        assert!(codes.contains(&"empty_input"), "{codes:?}");
    }
}
