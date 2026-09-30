//! A scheduling problem whose input has been checked.

use crate::models::{Resource, Task};
use crate::validation::{validate_input, ValidationError};

/// Tasks and the resources they run on, checked once so every solver can rely
/// on them.
///
/// A `Problem` exists only for input that passes
/// [`validate_input`]: every task, activity and resource has its own id, every
/// resource an activity names and every predecessor it waits on exists, the
/// precedence graph has no cycle, and every task has an activity. The solvers
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
/// let errors = Problem::new(tasks, vec![Resource::primary("M1")]).unwrap_err();
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
    pub fn new(tasks: Vec<Task>, resources: Vec<Resource>) -> Result<Self, Vec<ValidationError>> {
        validate_input(&tasks, &resources)?;
        Ok(Self { tasks, resources })
    }

    /// The tasks, in the order given.
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
    fn every_error_is_reported_not_only_the_first() {
        let tasks = vec![Task::new("empty"), Task::new("empty")];
        let errors = Problem::new(tasks, vec![]).expect_err("invalid");
        let codes: Vec<&str> = errors.iter().map(|e| e.kind.code()).collect();
        assert!(codes.contains(&"duplicate_id"), "{codes:?}");
        assert!(codes.contains(&"empty_task"), "{codes:?}");
    }
}
