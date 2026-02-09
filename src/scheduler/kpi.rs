//! Schedule quality metrics (KPIs).
//!
//! Computes standard scheduling performance indicators from a
//! completed schedule and its input tasks.
//!
//! # Metrics
//!
//! | Metric | Definition |
//! |--------|-----------|
//! | Makespan (C_max) | Latest completion time |
//! | Total Tardiness | Sum of max(0, completion - deadline) |
//! | Maximum Tardiness | Largest single delay |
//! | On-Time Rate | Fraction meeting deadlines |
//! | Avg Utilization | Mean resource busyness |
//! | Avg Flow Time | Mean time from release to completion |
//!
//! # Reference
//! Pinedo (2016), "Scheduling", Ch. 1.2: Performance Measures

use std::collections::HashMap;

use crate::models::{Schedule, Task};

/// Schedule performance indicators.
///
/// All time values are in milliseconds.
#[derive(Debug, Clone)]
pub struct ScheduleKpi {
    /// Makespan: latest completion time (ms).
    pub makespan_ms: i64,
    /// Sum of tardiness across all tasks (ms).
    pub total_tardiness_ms: i64,
    /// Maximum tardiness of any single task (ms).
    pub max_tardiness_ms: i64,
    /// Fraction of tasks completing on time (0.0..1.0).
    pub on_time_rate: f64,
    /// Average resource utilization (0.0..1.0).
    pub avg_utilization: f64,
    /// Per-resource utilization.
    pub utilization_by_resource: HashMap<String, f64>,
    /// Average flow time: mean(completion - release) in ms.
    pub avg_flow_time_ms: f64,
}

impl ScheduleKpi {
    /// Computes KPIs from a schedule and its input tasks.
    ///
    /// # Arguments
    /// * `schedule` - The completed schedule with assignments.
    /// * `tasks` - The input tasks (for deadlines and release times).
    pub fn calculate(schedule: &Schedule, tasks: &[Task]) -> Self {
        let makespan = schedule.makespan_ms();
        let mut total_tardiness: i64 = 0;
        let mut max_tardiness: i64 = 0;
        let mut on_time_count: usize = 0;
        let mut total_flow_time: f64 = 0.0;
        let mut counted_tasks: usize = 0;

        for task in tasks {
            if let Some(completion) = schedule.task_completion_time(&task.id) {
                counted_tasks += 1;

                // Flow time
                let release = task.release_time.unwrap_or(0);
                total_flow_time += (completion - release) as f64;

                // Tardiness
                if let Some(deadline) = task.deadline {
                    if completion > deadline {
                        let tardiness = completion - deadline;
                        total_tardiness += tardiness;
                        max_tardiness = max_tardiness.max(tardiness);
                    } else {
                        on_time_count += 1;
                    }
                } else {
                    // No deadline → considered on-time
                    on_time_count += 1;
                }
            }
        }

        // Utilization
        let utilization_by_resource = schedule.all_utilizations();
        let avg_utilization = if utilization_by_resource.is_empty() {
            0.0
        } else {
            let sum: f64 = utilization_by_resource.values().sum();
            sum / utilization_by_resource.len() as f64
        };

        let on_time_rate = if counted_tasks == 0 {
            1.0
        } else {
            on_time_count as f64 / counted_tasks as f64
        };

        let avg_flow_time_ms = if counted_tasks == 0 {
            0.0
        } else {
            total_flow_time / counted_tasks as f64
        };

        Self {
            makespan_ms: makespan,
            total_tardiness_ms: total_tardiness,
            max_tardiness_ms: max_tardiness,
            on_time_rate,
            avg_utilization,
            utilization_by_resource,
            avg_flow_time_ms,
        }
    }

    /// Whether the schedule meets the given quality thresholds.
    pub fn meets_thresholds(&self, max_tardiness: i64, min_utilization: f64) -> bool {
        self.max_tardiness_ms <= max_tardiness && self.avg_utilization >= min_utilization
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Activity, ActivityDuration, Assignment, ResourceRequirement};

    fn make_task(id: &str, duration_ms: i64, deadline: Option<i64>, release: Option<i64>) -> Task {
        let mut task = Task::new(id).with_activity(
            Activity::new(format!("{id}_O1"), id, 0)
                .with_duration(ActivityDuration::fixed(duration_ms))
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                ),
        );
        task.deadline = deadline;
        task.release_time = release;
        task
    }

    #[test]
    fn test_kpi_basic() {
        let tasks = vec![
            make_task("J1", 1000, Some(5000), Some(0)),
            make_task("J2", 2000, Some(5000), Some(0)),
        ];
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("J1_O1", "J1", "M1", 0, 1000));
        schedule.add_assignment(Assignment::new("J2_O1", "J2", "M1", 1000, 3000));

        let kpi = ScheduleKpi::calculate(&schedule, &tasks);
        assert_eq!(kpi.makespan_ms, 3000);
        assert_eq!(kpi.total_tardiness_ms, 0);
        assert_eq!(kpi.max_tardiness_ms, 0);
        assert!((kpi.on_time_rate - 1.0).abs() < 1e-10);
        assert!((kpi.avg_flow_time_ms - 2000.0).abs() < 1e-10); // (1000+3000)/2
    }

    #[test]
    fn test_kpi_tardiness() {
        let tasks = vec![
            make_task("J1", 1000, Some(500), Some(0)), // Deadline 500, completes at 1000 → tardy 500
            make_task("J2", 1000, Some(5000), Some(0)), // On time
        ];
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("J1_O1", "J1", "M1", 0, 1000));
        schedule.add_assignment(Assignment::new("J2_O1", "J2", "M1", 1000, 2000));

        let kpi = ScheduleKpi::calculate(&schedule, &tasks);
        assert_eq!(kpi.total_tardiness_ms, 500);
        assert_eq!(kpi.max_tardiness_ms, 500);
        assert!((kpi.on_time_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_kpi_utilization() {
        let tasks = vec![
            make_task("J1", 2000, None, None),
            make_task("J2", 1000, None, None),
        ];
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("J1_O1", "J1", "M1", 0, 2000));
        schedule.add_assignment(Assignment::new("J2_O1", "J2", "M2", 0, 1000));

        let kpi = ScheduleKpi::calculate(&schedule, &tasks);
        assert_eq!(kpi.makespan_ms, 2000);
        // M1: 2000/2000 = 1.0, M2: 1000/2000 = 0.5
        assert!((kpi.utilization_by_resource["M1"] - 1.0).abs() < 1e-10);
        assert!((kpi.utilization_by_resource["M2"] - 0.5).abs() < 1e-10);
        assert!((kpi.avg_utilization - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_kpi_flow_time() {
        let tasks = vec![
            make_task("J1", 1000, None, Some(1000)), // Released at 1000, completes at 3000 → flow 2000
            make_task("J2", 1000, None, Some(0)),    // Released at 0, completes at 1000 → flow 1000
        ];
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("J1_O1", "J1", "M1", 2000, 3000));
        schedule.add_assignment(Assignment::new("J2_O1", "J2", "M1", 0, 1000));

        let kpi = ScheduleKpi::calculate(&schedule, &tasks);
        // avg = (2000 + 1000) / 2 = 1500
        assert!((kpi.avg_flow_time_ms - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_kpi_empty() {
        let kpi = ScheduleKpi::calculate(&Schedule::new(), &[]);
        assert_eq!(kpi.makespan_ms, 0);
        assert_eq!(kpi.total_tardiness_ms, 0);
        assert!((kpi.on_time_rate - 1.0).abs() < 1e-10);
        assert!((kpi.avg_utilization - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kpi_no_deadline_on_time() {
        let tasks = vec![make_task("J1", 1000, None, None)];
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("J1_O1", "J1", "M1", 0, 1000));

        let kpi = ScheduleKpi::calculate(&schedule, &tasks);
        assert!((kpi.on_time_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_meets_thresholds() {
        let tasks = vec![make_task("J1", 1000, Some(500), None)]; // Tardy by 500
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("J1_O1", "J1", "M1", 0, 1000));

        let kpi = ScheduleKpi::calculate(&schedule, &tasks);
        assert!(kpi.meets_thresholds(500, 0.0));
        assert!(!kpi.meets_thresholds(499, 0.0));
        assert!(!kpi.meets_thresholds(1000, 1.5)); // Utilization too high
    }
}
