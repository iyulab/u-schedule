//! Simple priority-driven greedy scheduler.
//!
//! # Algorithm
//!
//! 1. Sort tasks by dispatching rule (or priority if no rule engine).
//! 2. For each task, process activities sequentially.
//! 3. For each activity, select the earliest-available candidate resource.
//! 4. Apply sequence-dependent setup times from transition matrices.
//!
//! # Complexity
//! O(n * m * c) where n=tasks, m=activities/task, c=candidate resources.
//!
//! # Reference
//! Pinedo (2016), "Scheduling", Ch. 4: Priority Dispatching

use std::collections::HashMap;

use super::timeline::ResourceTimeline;
use crate::dispatching::{RuleEngine, SchedulingContext};
use crate::models::{
    Activity, Assignment, Resource, ResourceRequirement, Schedule, Task, TransitionMatrixCollection,
};

/// Safety bound for the serial-SGS fixed-point search.
const MAX_FIXED_POINT_ITERS: usize = 10_000;

/// Input container for scheduling.
#[derive(Debug, Clone)]
pub struct ScheduleRequest {
    /// Tasks to schedule.
    pub tasks: Vec<Task>,
    /// Available resources.
    pub resources: Vec<Resource>,
    /// Schedule start time (ms).
    pub start_time_ms: i64,
    /// Sequence-dependent setup time matrices.
    pub transition_matrices: TransitionMatrixCollection,
    /// Additional constraints. The greedy scheduler does not yet place
    /// against these — they are validated post-hoc, so any breach shows
    /// up honestly in `Schedule::violations`.
    pub constraints: Vec<crate::models::Constraint>,
}

impl ScheduleRequest {
    /// Creates a new schedule request.
    pub fn new(tasks: Vec<Task>, resources: Vec<Resource>) -> Self {
        Self {
            tasks,
            resources,
            start_time_ms: 0,
            transition_matrices: TransitionMatrixCollection::new(),
            constraints: Vec::new(),
        }
    }

    /// Sets the schedule start time.
    pub fn with_start_time(mut self, start_time_ms: i64) -> Self {
        self.start_time_ms = start_time_ms;
        self
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets additional constraints (validated post-hoc).
    pub fn with_constraints(mut self, constraints: Vec<crate::models::Constraint>) -> Self {
        self.constraints = constraints;
        self
    }
}

/// Simple priority-driven greedy scheduler.
///
/// Schedules tasks by priority (or dispatching rule), assigning each
/// activity to the earliest-available candidate resource. Supports
/// sequence-dependent setup times via transition matrices.
///
/// # Example
///
/// ```
/// use u_schedule::scheduler::{SimpleScheduler, ScheduleRequest};
/// use u_schedule::models::{Task, Resource, ResourceType, Activity, ActivityDuration, ResourceRequirement};
///
/// let tasks = vec![
///     Task::new("J1").with_activity(
///         Activity::new("O1", "J1", 0)
///             .with_duration(ActivityDuration::fixed(1000))
///             .with_requirement(
///                 ResourceRequirement::new("Machine")
///                     .with_candidates(vec!["M1".into()])
///             )
///     ),
/// ];
/// let resources = vec![Resource::new("M1", ResourceType::Primary)];
/// let request = ScheduleRequest::new(tasks, resources);
///
/// let scheduler = SimpleScheduler::new();
/// let schedule = scheduler.schedule_request(&request);
/// assert_eq!(schedule.assignment_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleScheduler {
    transition_matrices: TransitionMatrixCollection,
    rule_engine: Option<RuleEngine>,
}

impl SimpleScheduler {
    /// Creates a new scheduler.
    pub fn new() -> Self {
        Self {
            transition_matrices: TransitionMatrixCollection::new(),
            rule_engine: None,
        }
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets a rule engine for task ordering.
    ///
    /// When set, tasks are sorted by the rule engine instead of by priority.
    pub fn with_rule_engine(mut self, engine: RuleEngine) -> Self {
        self.rule_engine = Some(engine);
        self
    }

    /// Schedules tasks on resources.
    ///
    /// # Algorithm (serial SGS)
    /// 1. Sort tasks by rule engine or priority (descending).
    /// 2. For each task, schedule activities in sequence order.
    /// 3. For each activity, find via fixed-point search the earliest
    ///    common start where **every** resource requirement can hold
    ///    `quantity` resources simultaneously — honoring calendars and
    ///    capacities through [`ResourceTimeline`].
    /// 4. The occupied span is `max(setup) + process + teardown`, where a
    ///    resource's setup comes from its transition matrix if one is
    ///    defined, else from `ActivityDuration::setup_ms`.
    ///
    /// Activities whose requirements cannot be met (unknown candidates,
    /// calendar that can never contain the duration) are left unassigned;
    /// the self-annotating feasibility check reports them as
    /// `RequirementUnfilled` rather than silently shrinking the schedule.
    ///
    /// # Reference
    /// Kolisch & Hartmann (1999), serial schedule generation scheme.
    pub fn schedule(&self, tasks: &[Task], resources: &[Resource], start_time_ms: i64) -> Schedule {
        let mut schedule = Schedule::new();
        let mut timelines: HashMap<String, ResourceTimeline> = resources
            .iter()
            .map(|r| (r.id.clone(), ResourceTimeline::new(r)))
            .collect();
        let mut last_category: HashMap<String, String> = HashMap::new();

        let task_order = self.sort_tasks(tasks, start_time_ms);

        for &task_idx in &task_order {
            let task = &tasks[task_idx];
            let mut ready = task
                .release_time
                .unwrap_or(start_time_ms)
                .max(start_time_ms);

            for activity in &task.activities {
                if activity.resource_requirements.is_empty() {
                    continue; // 자원 무요구 activity는 배정하지 않음 (기존 의미 유지)
                }
                let Some(placement) = self.place_activity(
                    activity,
                    task,
                    &timelines,
                    &last_category,
                    resources,
                    ready,
                ) else {
                    continue; // 충족 불가 — feasibility 검사가 RequirementUnfilled로 보고
                };

                for (resource_id, setup_ms) in &placement.selections {
                    let tl = timelines
                        .get_mut(resource_id)
                        .expect("selection references known resource");
                    tl.book(placement.start_ms, placement.end_ms);
                    schedule.add_assignment(
                        Assignment::new(
                            &activity.id,
                            &task.id,
                            resource_id,
                            placement.start_ms,
                            placement.end_ms,
                        )
                        .with_setup(*setup_ms),
                    );
                    if self.transition_matrices.has_matrix(resource_id) {
                        last_category.insert(resource_id.clone(), task.category.clone());
                    }
                }
                ready = placement.end_ms;
            }
        }

        let input = super::feasibility::FeasibilityInput {
            tasks,
            resources,
            constraints: &[],
        };
        super::feasibility::annotate_schedule(&mut schedule, &input);
        schedule
    }

    /// Finds the earliest common start for an activity across all its
    /// resource requirements (serial-SGS fixed point).
    fn place_activity(
        &self,
        activity: &Activity,
        task: &Task,
        timelines: &HashMap<String, ResourceTimeline>,
        last_category: &HashMap<String, String>,
        resources: &[Resource],
        ready: i64,
    ) -> Option<Placement> {
        let process = activity.duration.process_ms;
        let teardown = activity.duration.teardown_ms;
        let setup_for = |resource_id: &str| -> i64 {
            if self.transition_matrices.has_matrix(resource_id) {
                match last_category.get(resource_id) {
                    Some(prev) => self.transition_matrices.get_transition_time(
                        resource_id,
                        prev,
                        &task.category,
                    ),
                    None => 0, // 첫 작업엔 changeover 없음 (기존 의미 유지)
                }
            } else {
                activity.duration.setup_ms
            }
        };

        // Phase 1: 고정점 — 모든 requirement가 t에서 시작 가능한 최소 t
        let mut t = ready;
        let mut selections: Vec<(String, i64)> = Vec::new();
        for _ in 0..MAX_FIXED_POINT_ITERS {
            selections.clear();
            let mut t_next = t;
            for req in &activity.resource_requirements {
                let mut fits: Vec<(i64, &str, i64)> = resolve_candidates(req, resources)
                    .into_iter()
                    .filter_map(|rid| {
                        let setup = setup_for(rid);
                        timelines
                            .get(rid)?
                            .earliest_fit(t, setup + process + teardown)
                            .map(|fit| (fit, rid, setup))
                    })
                    .collect();
                fits.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
                if (fits.len() as i32) < req.quantity {
                    return None; // 후보 부족 — 영원히 충족 불가
                }
                for &(fit, rid, setup) in fits.iter().take(req.quantity as usize) {
                    t_next = t_next.max(fit);
                    selections.push((rid.to_string(), setup));
                }
            }
            if t_next == t {
                break;
            }
            t = t_next;
        }

        // Phase 2: 공통 duration D(최대 setup 기준)로 전 자원 동시 fit 재검증
        let max_setup = selections.iter().map(|&(_, s)| s).max().unwrap_or(0);
        let d = max_setup + process + teardown;
        for _ in 0..MAX_FIXED_POINT_ITERS {
            let mut t_next = t;
            for (rid, _) in &selections {
                let tl = timelines.get(rid).expect("selected resource has timeline");
                // None: 캘린더가 D를 영원히 못 담음
                let fit = tl.earliest_fit(t, d)?;
                t_next = t_next.max(fit);
            }
            if t_next == t {
                return Some(Placement {
                    start_ms: t,
                    end_ms: t + d,
                    selections,
                });
            }
            t = t_next;
        }
        None // 고정점 미수렴 (방어적 상한 — property test가 실질 커버)
    }

    /// Schedules from a request.
    ///
    /// Re-annotates with the request's constraints so `TimeWindow`,
    /// `Synchronize`, `NoOverlap`, tightened `Capacity`, and cross-task
    /// `Precedence` breaches are reported in `Schedule::violations`.
    pub fn schedule_request(&self, request: &ScheduleRequest) -> Schedule {
        let scheduler = Self {
            transition_matrices: request.transition_matrices.clone(),
            rule_engine: self.rule_engine.clone(),
        };
        let mut schedule =
            scheduler.schedule(&request.tasks, &request.resources, request.start_time_ms);
        let input = super::feasibility::FeasibilityInput {
            tasks: &request.tasks,
            resources: &request.resources,
            constraints: &request.constraints,
        };
        super::feasibility::annotate_schedule(&mut schedule, &input);
        schedule
    }

    /// Returns task indices sorted by rule engine or priority.
    fn sort_tasks(&self, tasks: &[Task], start_time_ms: i64) -> Vec<usize> {
        if let Some(ref engine) = self.rule_engine {
            let ctx = SchedulingContext::at_time(start_time_ms);
            engine.sort_indices(tasks, &ctx)
        } else {
            // Default: sort by priority descending
            let mut indices: Vec<usize> = (0..tasks.len()).collect();
            indices.sort_by(|&a, &b| tasks[b].priority.cmp(&tasks[a].priority));
            indices
        }
    }
}

impl Default for SimpleScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// A resolved activity placement: common interval plus per-resource setup.
struct Placement {
    start_ms: i64,
    end_ms: i64,
    selections: Vec<(String, i64)>, // (resource_id, setup_ms)
}

/// Candidate resources for a requirement: explicit candidates (skill-filtered)
/// if any, otherwise every skill-satisfying resource.
fn resolve_candidates<'a>(req: &'a ResourceRequirement, resources: &'a [Resource]) -> Vec<&'a str> {
    let skill_ok = |r: &Resource| req.required_skills.iter().all(|s| r.has_skill(s));
    if req.candidates.is_empty() {
        resources
            .iter()
            .filter(|r| skill_ok(r))
            .map(|r| r.id.as_str())
            .collect()
    } else {
        req.candidates
            .iter()
            .filter(|id| resources.iter().any(|r| &r.id == *id && skill_ok(r)))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatching::rules;
    use crate::models::{
        Activity, ActivityDuration, Resource, ResourceRequirement, ResourceType, TransitionMatrix,
    };

    fn make_resource(id: &str) -> Resource {
        Resource::new(id, ResourceType::Primary)
    }

    fn make_task_with_resource(
        id: &str,
        duration_ms: i64,
        resource_id: &str,
        priority: i32,
    ) -> Task {
        Task::new(id)
            .with_priority(priority)
            .with_category("default")
            .with_activity(
                Activity::new(format!("{id}_O1"), id, 0)
                    .with_duration(ActivityDuration::fixed(duration_ms))
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(vec![resource_id.into()]),
                    ),
            )
    }

    #[test]
    fn test_multi_resource_simultaneous_hold() {
        // probe S1: 설비+금형 — 금형(TOOL1) 공유로 두 작업 직렬화
        let mk = |tid: &str, machine: &str| {
            Task::new(tid)
                .with_priority(1)
                .with_category("default")
                .with_activity(
                    Activity::new(format!("{tid}_O1"), tid, 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec![machine.into()]),
                        )
                        .with_requirement(
                            ResourceRequirement::new("Mold").with_candidates(vec!["TOOL1".into()]),
                        ),
                )
        };
        let tasks = vec![mk("J1", "M1"), mk("J2", "M2")];
        let resources = vec![
            Resource::new("M1", ResourceType::Primary),
            Resource::new("M2", ResourceType::Primary),
            Resource::new("TOOL1", ResourceType::Secondary),
        ];
        let s = SimpleScheduler::new().schedule(&tasks, &resources, 0);

        // 각 activity가 설비+금형 2개 배정
        assert_eq!(s.assignments_for_task("J1").len(), 2);
        assert_eq!(s.assignments_for_task("J2").len(), 2);
        // 금형 위에서 직렬화 (겹침 없음)
        let tool = s.assignments_for_resource("TOOL1");
        assert_eq!(tool.len(), 2);
        let (a, b) = (tool[0], tool[1]);
        assert!(
            a.end_ms <= b.start_ms || b.end_ms <= a.start_ms,
            "TOOL1 double-booked: {a:?} vs {b:?}"
        );
        // 동시 점유: 같은 activity의 두 배정 구간 동일
        let j1 = s.assignments_for_task("J1");
        assert_eq!(j1[0].start_ms, j1[1].start_ms);
        assert_eq!(j1[0].end_ms, j1[1].end_ms);
    }

    #[test]
    fn test_calendar_enforced() {
        // probe S3: 가용창 [0,5000)∪[10000,20000), 3000ms 작업 2건 → 둘째는 10000 시작
        let cal = crate::models::Calendar::new("shift")
            .with_window(0, 5000)
            .with_window(10_000, 20_000);
        let tasks = vec![
            make_task_with_resource("J1", 3000, "M1", 10),
            make_task_with_resource("J2", 3000, "M1", 5),
        ];
        let resources = vec![Resource::new("M1", ResourceType::Primary).with_calendar(cal)];
        let s = SimpleScheduler::new().schedule(&tasks, &resources, 0);
        let j2 = s.assignment_for_activity("J2_O1").unwrap();
        assert_eq!(j2.start_ms, 10_000);
        assert_eq!(j2.end_ms, 13_000);
    }

    #[test]
    fn test_capacity_two_parallel() {
        // probe S4: capacity=2 → 두 작업 병렬
        let tasks = vec![
            make_task_with_resource("J1", 1000, "M1", 10),
            make_task_with_resource("J2", 1000, "M1", 5),
        ];
        let resources = vec![Resource::new("M1", ResourceType::Primary).with_capacity(2)];
        let s = SimpleScheduler::new().schedule(&tasks, &resources, 0);
        assert_eq!(s.assignment_for_activity("J1_O1").unwrap().start_ms, 0);
        assert_eq!(s.assignment_for_activity("J2_O1").unwrap().start_ms, 0);
    }

    #[test]
    fn test_duration_components_enforced() {
        // probe S5: (500,1000,300) → 점유 1800ms, setup 기록 500
        let task = Task::new("J1")
            .with_priority(1)
            .with_category("default")
            .with_activity(
                Activity::new("J1_O1", "J1", 0)
                    .with_duration(ActivityDuration::new(500, 1000, 300))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            );
        let s = SimpleScheduler::new().schedule(
            &[task],
            &[Resource::new("M1", ResourceType::Primary)],
            0,
        );
        let a = s.assignment_for_activity("J1_O1").unwrap();
        assert_eq!(a.end_ms - a.start_ms, 1800);
        assert_eq!(a.setup_ms, 500);
    }

    #[test]
    fn test_simple_single_task() {
        let tasks = vec![make_task_with_resource("J1", 1000, "M1", 0)];
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&tasks, &resources, 0);
        assert_eq!(schedule.assignment_count(), 1);

        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        assert_eq!(a.start_ms, 0);
        assert_eq!(a.end_ms, 1000);
        assert_eq!(a.resource_id, "M1");
    }

    #[test]
    fn test_priority_ordering() {
        let tasks = vec![
            make_task_with_resource("low", 1000, "M1", 1),
            make_task_with_resource("high", 1000, "M1", 10),
        ];
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&tasks, &resources, 0);

        // High priority scheduled first
        let high_a = schedule.assignment_for_activity("high_O1").unwrap();
        let low_a = schedule.assignment_for_activity("low_O1").unwrap();
        assert!(high_a.start_ms < low_a.start_ms);
    }

    #[test]
    fn test_two_resources() {
        let tasks = vec![
            make_task_with_resource("J1", 2000, "M1", 10),
            make_task_with_resource("J2", 1000, "M1", 5),
        ];
        // Only M1 → J1 first (priority), then J2 at 2000
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&tasks, &resources, 0);
        let j1 = schedule.assignment_for_activity("J1_O1").unwrap();
        let j2 = schedule.assignment_for_activity("J2_O1").unwrap();
        assert_eq!(j1.start_ms, 0);
        assert_eq!(j1.end_ms, 2000);
        assert_eq!(j2.start_ms, 2000);
        assert_eq!(j2.end_ms, 3000);
    }

    #[test]
    fn test_parallel_resources() {
        // J1→M1, J2→M2 can run in parallel
        let tasks = vec![
            make_task_with_resource("J1", 2000, "M1", 10),
            make_task_with_resource("J2", 1000, "M2", 5),
        ];
        let resources = vec![make_resource("M1"), make_resource("M2")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&tasks, &resources, 0);
        let j1 = schedule.assignment_for_activity("J1_O1").unwrap();
        let j2 = schedule.assignment_for_activity("J2_O1").unwrap();
        // Both start at 0 since they use different resources
        assert_eq!(j1.start_ms, 0);
        assert_eq!(j2.start_ms, 0);
    }

    #[test]
    fn test_multi_activity_task() {
        let task = Task::new("J1")
            .with_priority(1)
            .with_category("TypeA")
            .with_activity(
                Activity::new("O1", "J1", 0)
                    .with_duration(ActivityDuration::fixed(1000))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            )
            .with_activity(
                Activity::new("O2", "J1", 1)
                    .with_duration(ActivityDuration::fixed(2000))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            );

        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&[task], &resources, 0);

        let o1 = schedule.assignment_for_activity("O1").unwrap();
        let o2 = schedule.assignment_for_activity("O2").unwrap();
        // O2 must start after O1 ends (intra-task precedence)
        assert_eq!(o1.end_ms, 1000);
        assert!(o2.start_ms >= o1.end_ms);
        assert_eq!(o2.end_ms, 3000);
    }

    #[test]
    fn test_transition_matrix_setup() {
        let mut tm = TransitionMatrix::new("changeover", "M1").with_default(500);
        tm.set_transition("TypeA", "TypeB", 1000);

        let matrices = TransitionMatrixCollection::new().with_matrix(tm);

        let tasks = vec![
            Task::new("J1")
                .with_priority(10)
                .with_category("TypeA")
                .with_activity(
                    Activity::new("O1", "J1", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                        ),
                ),
            Task::new("J2")
                .with_priority(5)
                .with_category("TypeB")
                .with_activity(
                    Activity::new("O2", "J2", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                        ),
                ),
        ];

        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new().with_transition_matrices(matrices);

        let schedule = scheduler.schedule(&tasks, &resources, 0);
        let o2 = schedule.assignment_for_activity("O2").unwrap();
        // J1 ends at 1000, setup A→B = 1000, J2 starts at 1000, ends at 1000+1000+1000 = 3000
        assert_eq!(o2.start_ms, 1000);
        assert_eq!(o2.setup_ms, 1000);
        assert_eq!(o2.end_ms, 3000);
    }

    #[test]
    fn test_with_rule_engine() {
        // Use SPT rule → shorter task first regardless of priority
        let tasks = vec![
            make_task_with_resource("long", 5000, "M1", 100), // High priority but long
            make_task_with_resource("short", 1000, "M1", 1),  // Low priority but short
        ];
        let resources = vec![make_resource("M1")];
        let engine = RuleEngine::new().with_rule(rules::Spt);
        let scheduler = SimpleScheduler::new().with_rule_engine(engine);

        let schedule = scheduler.schedule(&tasks, &resources, 0);
        let short_a = schedule.assignment_for_activity("short_O1").unwrap();
        let long_a = schedule.assignment_for_activity("long_O1").unwrap();
        // SPT orders short first despite lower priority
        assert_eq!(short_a.start_ms, 0);
        assert!(long_a.start_ms >= short_a.end_ms);
    }

    #[test]
    fn test_schedule_request() {
        let tasks = vec![make_task_with_resource("J1", 1000, "M1", 0)];
        let resources = vec![make_resource("M1")];
        let request = ScheduleRequest::new(tasks, resources).with_start_time(5000);

        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule_request(&request);

        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        assert_eq!(a.start_ms, 5000);
        assert_eq!(a.end_ms, 6000);
    }

    #[test]
    fn test_release_time_respected() {
        let mut task = make_task_with_resource("J1", 1000, "M1", 0);
        task.release_time = Some(5000);
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&[task], &resources, 0);
        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        // Must not start before release_time
        assert_eq!(a.start_ms, 5000);
    }

    #[test]
    fn test_empty_input() {
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&[], &[], 0);
        assert_eq!(schedule.assignment_count(), 0);
        assert_eq!(schedule.makespan_ms(), 0);
    }

    #[test]
    fn test_no_candidate_resources() {
        // Activity with no resource requirement → skipped
        let task = Task::new("J1").with_priority(1).with_activity(
            Activity::new("O1", "J1", 0).with_duration(ActivityDuration::fixed(1000)),
            // No resource requirement
        );
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&[task], &resources, 0);
        assert_eq!(schedule.assignment_count(), 0);
    }
}
