//! Post-hoc feasibility validation of produced schedules.
//!
//! Independent of any scheduler: compares a `Schedule` against the model
//! declarations (requirements, calendars, capacities, constraints) and
//! reports violations. Schedulers self-annotate their output with this
//! checker so `Schedule::is_valid()` is meaningful.
//!
//! Requirement attribution is greedy (requirements claim unclaimed
//! assigned resources in declaration order) — an approximation that is
//! exact when requirement candidate sets are disjoint, the common case.
//! A requirement claims only resources in its pool
//! ([`ResourceRequirement::is_candidate`](crate::models::ResourceRequirement::is_candidate): listed in `candidates`, or of its
//! `resource_type` when none are listed), so a resource of another type
//! leaves the requirement unfilled. A claimed resource that lacks a required
//! skill is reported as a skill mismatch -- together the two checks apply
//! [`ResourceRequirement::accepts`](crate::models::ResourceRequirement::accepts), the rule the schedulers place by.
//!
//! Every assignment under a model task must name one of its activities (an
//! assignment under another task is an external booking -- a pinned job --
//! that counts only against capacity and calendars), at most once per unit its
//! requirements take, last at least its setup (as the assignment records it -- a changeover or
//! the activity's own), processing and teardown on the resources it runs on
//! ([`Activity::duration_on`](crate::models::Activity::duration_on)), and start no earlier than its
//! task's release time.
//!
//! Not checked (documented gap): `Constraint::TransitionCost` (needs
//! sequence semantics), and so whether an assignment's recorded setup is the
//! one the resource's transitions call for.
//!
//! # Reference
//! Brucker (2007), "Scheduling Algorithms", Ch. 2 — constraint taxonomy.

use std::collections::{HashMap, HashSet};

use crate::models::{Activity, Constraint, Resource, Schedule, Task, Violation};

/// Model context a schedule is validated against.
pub struct FeasibilityInput<'a> {
    /// Tasks whose activities the schedule should cover.
    pub tasks: &'a [Task],
    /// Resources with calendars and capacities.
    pub resources: &'a [Resource],
    /// Additional scheduling constraints.
    pub constraints: &'a [Constraint],
}

/// Checks a schedule against the model; returns all detected violations.
pub fn check_schedule(schedule: &Schedule, input: &FeasibilityInput) -> Vec<Violation> {
    let mut v = Vec::new();
    let res_by_id: HashMap<&str, &Resource> =
        input.resources.iter().map(|r| (r.id.as_str(), r)).collect();
    check_assignments(schedule, input, &mut v);
    check_requirements(schedule, input, &res_by_id, &mut v);
    check_capacity(schedule, input, &res_by_id, &mut v);
    check_calendars(schedule, &res_by_id, &mut v);
    check_precedence(schedule, input, &mut v);
    check_deadlines(schedule, input, &mut v);
    check_constraints(schedule, input, &mut v);
    v
}

/// Runs `check_schedule` and stores the result in `schedule.violations`.
pub fn annotate_schedule(schedule: &mut Schedule, input: &FeasibilityInput) {
    let violations = check_schedule(schedule, input);
    schedule.violations = violations;
}

/// Each assignment under a model task names one of its activities, at most as
/// many times as the activity's requirements take units, lasts at least the
/// activity's span (its recorded setup + processing + teardown) and starts no
/// earlier than the task's release time. An assignment under a task the model
/// does not have is an external booking and is not judged here.
fn check_assignments(schedule: &Schedule, input: &FeasibilityInput, out: &mut Vec<Violation>) {
    let mut by_id: HashMap<&str, (&Task, &Activity)> = HashMap::new();
    for task in input.tasks {
        for activity in &task.activities {
            by_id.insert(activity.id.as_str(), (task, activity));
        }
    }
    let model_tasks: HashSet<&str> = input.tasks.iter().map(|t| t.id.as_str()).collect();
    // The resources each activity runs on: its time depends on them.
    let mut runs_on: HashMap<&str, Vec<&str>> = HashMap::new();
    for a in &schedule.assignments {
        runs_on
            .entry(a.activity_id.as_str())
            .or_default()
            .push(a.resource_id.as_str());
    }
    let mut uses: HashMap<&str, i32> = HashMap::new();
    for a in &schedule.assignments {
        let Some(&(task, activity)) = by_id.get(a.activity_id.as_str()) else {
            // An assignment under a task the model does not have is an
            // external booking (a pinned job): it counts against capacity and
            // calendars, and is not judged as an activity.
            if model_tasks.contains(a.task_id.as_str()) {
                out.push(Violation::unknown_activity(
                    &a.activity_id,
                    format!("task '{}' has no activity '{}'", a.task_id, a.activity_id),
                ));
            }
            continue;
        };
        if a.task_id != task.id {
            out.push(Violation::unknown_activity(
                &a.activity_id,
                format!(
                    "activity '{}' belongs to task '{}', not '{}'",
                    a.activity_id, task.id, a.task_id
                ),
            ));
            continue;
        }
        // More assignments than the activity's requirements take units: a
        // repeated placement, which the greedy attribution would let pass.
        let units: i32 = activity
            .resource_requirements
            .iter()
            .map(|r| r.quantity)
            .sum();
        let used = uses.entry(a.activity_id.as_str()).or_default();
        *used += 1;
        if *used > units.max(1) {
            out.push(Violation::unknown_activity(
                &a.activity_id,
                format!(
                    "activity '{}' is assigned {} times; its requirements take {} unit(s)",
                    a.activity_id,
                    used,
                    units.max(1)
                ),
            ));
        }
        let duration = activity.duration_on(runs_on[a.activity_id.as_str()].iter().copied());
        let needed = a.setup_ms + duration.process_ms + duration.teardown_ms;
        if a.end_ms - a.start_ms < needed {
            out.push(Violation::duration_too_short(
                &a.activity_id,
                format!(
                    "activity '{}' on '{}' lasts {} ms; setup {} + processing {} + teardown {} need {needed}",
                    a.activity_id,
                    a.resource_id,
                    a.end_ms - a.start_ms,
                    a.setup_ms,
                    duration.process_ms,
                    duration.teardown_ms
                ),
            ));
        }
        if let Some(release) = task.release_time {
            if a.start_ms < release {
                out.push(Violation::early_start(
                    &a.activity_id,
                    format!(
                        "activity '{}' starts at {} before its task's release {release}",
                        a.activity_id, a.start_ms
                    ),
                ));
            }
        }
    }
}

fn check_requirements(
    schedule: &Schedule,
    input: &FeasibilityInput,
    res_by_id: &HashMap<&str, &Resource>,
    out: &mut Vec<Violation>,
) {
    for task in input.tasks {
        for activity in &task.activities {
            if activity.resource_requirements.is_empty() {
                continue;
            }
            let assigns = schedule.assignments_for_activity_all(&activity.id);
            // 동시 점유: 모든 assignment 구간 동일
            if let Some(first) = assigns.first() {
                if assigns
                    .iter()
                    .any(|a| a.start_ms != first.start_ms || a.end_ms != first.end_ms)
                {
                    out.push(Violation::requirement_unfilled(
                        &activity.id,
                        format!(
                            "activity '{}' assignments are not simultaneous",
                            activity.id
                        ),
                    ));
                }
            }
            // greedy 귀속: 요구별로 미귀속 자원 claim
            let mut unclaimed: Vec<&str> = assigns.iter().map(|a| a.resource_id.as_str()).collect();
            for req in &activity.resource_requirements {
                let candidates: HashSet<&str> = input
                    .resources
                    .iter()
                    .filter(|r| req.is_candidate(r))
                    .map(|r| r.id.as_str())
                    .collect();
                let mut claimed = 0;
                let mut i = 0;
                while i < unclaimed.len() && claimed < req.quantity {
                    if candidates.contains(unclaimed[i]) {
                        let rid = unclaimed.remove(i);
                        claimed += 1;
                        if let Some(res) = res_by_id.get(rid) {
                            for skill in &req.required_skills {
                                if !res.has_skill(skill) {
                                    out.push(Violation::skill_mismatch(
                                        rid,
                                        format!(
                                            "resource '{rid}' lacks skill '{skill}' for activity '{}'",
                                            activity.id
                                        ),
                                    ));
                                }
                            }
                        }
                    } else {
                        i += 1;
                    }
                }
                if claimed < req.quantity {
                    out.push(Violation::requirement_unfilled(
                        &activity.id,
                        format!(
                            "activity '{}' requirement '{}' filled {claimed}/{} units",
                            activity.id, req.resource_type, req.quantity
                        ),
                    ));
                }
            }
        }
    }
}

fn check_capacity(
    schedule: &Schedule,
    input: &FeasibilityInput,
    res_by_id: &HashMap<&str, &Resource>,
    out: &mut Vec<Violation>,
) {
    // Constraint::Capacity로 조여진 한도 반영
    let mut cap_override: HashMap<&str, i32> = HashMap::new();
    for c in input.constraints {
        if let Constraint::Capacity {
            resource_id,
            max_capacity,
        } = c
        {
            cap_override.insert(resource_id.as_str(), *max_capacity);
        }
    }
    let mut by_resource: HashMap<&str, Vec<(i64, i64)>> = HashMap::new();
    for a in &schedule.assignments {
        by_resource
            .entry(a.resource_id.as_str())
            .or_default()
            .push((a.start_ms, a.end_ms));
    }
    for (rid, intervals) in by_resource {
        let base = res_by_id.get(rid).map_or(1, |r| r.capacity.max(1));
        let cap = cap_override.get(rid).map_or(base, |&c| base.min(c));
        // 동시성은 구간 시작점에서만 증가
        let max_concurrent = intervals
            .iter()
            .map(|&(s, _)| {
                intervals
                    .iter()
                    .filter(|&&(s2, e2)| s2 <= s && s < e2)
                    .count() as i32
            })
            .max()
            .unwrap_or(0);
        if max_concurrent > cap {
            out.push(Violation::capacity_exceeded(
                rid,
                format!(
                    "resource '{rid}' peak concurrency {max_concurrent} exceeds capacity {cap}"
                ),
            ));
        }
    }
}

fn check_calendars(
    schedule: &Schedule,
    res_by_id: &HashMap<&str, &Resource>,
    out: &mut Vec<Violation>,
) {
    for a in &schedule.assignments {
        if let Some(res) = res_by_id.get(a.resource_id.as_str()) {
            if let Some(cal) = &res.calendar {
                if !cal.interval_fits(a.start_ms, a.end_ms) {
                    out.push(Violation::resource_unavailable(
                        &a.resource_id,
                        format!(
                            "assignment '{}' [{}..{}) outside working time of '{}'",
                            a.activity_id, a.start_ms, a.end_ms, a.resource_id
                        ),
                    ));
                }
            }
        }
    }
}

/// Activity span in the schedule: (min start, max end) across its assignments.
fn activity_span(schedule: &Schedule, activity_id: &str) -> Option<(i64, i64)> {
    let assigns = schedule.assignments_for_activity_all(activity_id);
    let start = assigns.iter().map(|a| a.start_ms).min()?;
    let end = assigns
        .iter()
        .map(|a| a.end_ms)
        .max()
        .expect("non-empty when min exists");
    Some((start, end))
}

fn check_precedence(schedule: &Schedule, input: &FeasibilityInput, out: &mut Vec<Violation>) {
    let mut push = |before: &str, after: &str, delay: i64| {
        if let (Some((_, before_end)), Some((after_start, _))) = (
            activity_span(schedule, before),
            activity_span(schedule, after),
        ) {
            if after_start < before_end + delay {
                out.push(Violation::precedence_violation(
                    after,
                    format!(
                        "'{after}' starts at {after_start} before '{before}' end {before_end} + delay {delay}"
                    ),
                ));
            }
        }
    };
    for task in input.tasks {
        let mut ordered: Vec<&Activity> = task.activities.iter().collect();
        ordered.sort_by_key(|a| a.sequence);
        for pair in ordered.windows(2) {
            push(&pair[0].id, &pair[1].id, 0);
        }
        for activity in &task.activities {
            for pred in &activity.predecessors {
                push(pred, &activity.id, 0);
            }
        }
    }
    for c in input.constraints {
        if let Constraint::Precedence {
            before,
            after,
            min_delay_ms,
        } = c
        {
            push(before, after, *min_delay_ms);
        }
    }
}

fn check_deadlines(schedule: &Schedule, input: &FeasibilityInput, out: &mut Vec<Violation>) {
    for task in input.tasks {
        if let (Some(deadline), Some(completion)) =
            (task.deadline, schedule.task_completion_time(&task.id))
        {
            if task.tardiness_ms(completion) > 0 {
                out.push(Violation::deadline_miss(
                    &task.id,
                    format!(
                        "task '{}' completes at {completion}, deadline {deadline}",
                        task.id
                    ),
                ));
            }
        }
    }
}

fn check_constraints(schedule: &Schedule, input: &FeasibilityInput, out: &mut Vec<Violation>) {
    for c in input.constraints {
        match c {
            Constraint::TimeWindow {
                activity_id,
                start_ms,
                end_ms,
            } => {
                if let Some((s, e)) = activity_span(schedule, activity_id) {
                    if s < *start_ms || e > *end_ms {
                        out.push(Violation::time_window(
                            activity_id,
                            format!(
                                "'{activity_id}' [{s}..{e}) outside window [{start_ms}..{end_ms})"
                            ),
                        ));
                    }
                }
            }
            Constraint::NoOverlap {
                resource_id,
                activity_ids,
            } => {
                let spans: Vec<(&str, i64, i64)> = activity_ids
                    .iter()
                    .filter_map(|id| {
                        schedule
                            .assignments_for_activity_all(id)
                            .iter()
                            .find(|a| &a.resource_id == resource_id)
                            .map(|a| (id.as_str(), a.start_ms, a.end_ms))
                    })
                    .collect();
                for i in 0..spans.len() {
                    for j in (i + 1)..spans.len() {
                        let (ia, sa, ea) = spans[i];
                        let (ib, sb, eb) = spans[j];
                        if sa < eb && sb < ea {
                            out.push(Violation::capacity_exceeded(
                                resource_id,
                                format!("no-overlap: '{ia}' and '{ib}' overlap on '{resource_id}'"),
                            ));
                        }
                    }
                }
            }
            Constraint::Synchronize { activity_ids } => {
                let starts: Vec<i64> = activity_ids
                    .iter()
                    .filter_map(|id| activity_span(schedule, id).map(|(s, _)| s))
                    .collect();
                if starts.len() == activity_ids.len() && starts.windows(2).any(|w| w[0] != w[1]) {
                    out.push(Violation::synchronize(
                        activity_ids.join(","),
                        "synchronized activities have differing starts".to_string(),
                    ));
                }
            }
            // Precedence는 check_precedence, Capacity는 check_capacity에서 처리.
            // TransitionCost는 미지원 (모듈 doc 참조).
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        Activity, ActivityDuration, Assignment, Calendar, Resource, ResourceRequirement,
        ViolationType,
    };

    fn machine_activity(id: &str, task: &str, cands: Vec<String>) -> Activity {
        Activity::new(id, task, 0)
            .with_duration(ActivityDuration::fixed(1000))
            .with_requirement(ResourceRequirement::new("Machine").with_candidates(cands))
    }

    #[test]
    fn assignments_name_model_activities_last_their_span_and_respect_release() {
        let task = Task::new("J1").with_release_time(500).with_activity(
            Activity::new("O1", "J1", 0)
                .with_duration(ActivityDuration::new(100, 1000, 50))
                .with_requirement(ResourceRequirement::new("Machine")),
        );
        let resources = vec![Resource::new("M1", "Machine")];
        let input = FeasibilityInput {
            tasks: std::slice::from_ref(&task),
            resources: &resources,
            constraints: &[],
        };
        let kinds = |s: &Schedule| -> Vec<ViolationType> {
            check_schedule(s, &input)
                .into_iter()
                .map(|v| v.violation_type)
                .collect()
        };
        let mut ok = Schedule::new();
        ok.add_assignment(Assignment::new("O1", "J1", "M1", 500, 1650).with_setup(100));
        assert_eq!(kinds(&ok), []);

        let mut bad = Schedule::new();
        // Before release, and 1000 long where 100 + 1000 + 50 are needed.
        bad.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000).with_setup(100));
        bad.add_assignment(Assignment::new("X9", "J1", "M1", 2000, 3000));
        // An external booking under another task is not an activity.
        bad.add_assignment(Assignment::new("PIN", "PIN", "M1", 6000, 7000));
        // The same activity a second time, on a second resource.
        bad.add_assignment(Assignment::new("O1", "J1", "M1", 4000, 5150).with_setup(100));
        assert_eq!(
            kinds(&bad),
            [
                ViolationType::DurationTooShort,
                ViolationType::EarlyStart,
                ViolationType::UnknownActivity,
                ViolationType::UnknownActivity,
                // ... and the second O1 is not simultaneous with the first.
                ViolationType::RequirementUnfilled,
            ]
        );
    }

    /// Every scheduler's schedule keeps the model -- setup, teardown and
    /// release included, which the GA and CP schedulers used to drop.
    #[test]
    fn every_scheduler_s_schedule_passes_the_check() {
        use crate::cp::ScheduleCpBuilder;
        use crate::ga::SchedulingGaProblem;
        use crate::scheduler::SimpleScheduler;
        use crate::Problem;
        use u_metaheur::cp::{SimpleCpSolver, SolverConfig};
        use u_metaheur::ga::{GaConfig, GaRunner};

        let tasks: Vec<Task> = (0..4)
            .map(|j| {
                let id = format!("J{j}");
                let mut t = Task::new(&id).with_release_time(200 * j as i64);
                for k in 0..2 {
                    t = t.with_activity(
                        Activity::new(format!("{id}-O{k}"), &id, k)
                            .with_duration(ActivityDuration::new(100, 1000 + 100 * k as i64, 50))
                            .with_requirement(ResourceRequirement::new("Machine")),
                    );
                }
                t
            })
            .collect();
        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
        ];
        let problem = Problem::new(tasks.clone(), resources.clone()).expect("a valid problem");
        let input = FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let simple = SimpleScheduler::new().schedule(&problem, 0);
        let ga_problem = SchedulingGaProblem::new(&problem);
        let config = GaConfig::default()
            .with_population_size(10)
            .with_max_generations(5)
            .with_seed(3)
            .with_parallel(false);
        let best = GaRunner::run(&ga_problem, &config).expect("GA ran").best;
        let ga = ga_problem.decode(&best);
        let (cp, _) = ScheduleCpBuilder::new(&problem).solve(
            &SimpleCpSolver::new(),
            &SolverConfig::default(),
            60_000,
        );
        for (name, schedule) in [("simple", simple), ("ga", ga), ("cp", cp)] {
            let v = check_schedule(&schedule, &input);
            assert!(v.is_empty(), "{name}: {v:?}");
            for a in &schedule.assignments {
                assert!(a.end_ms - a.start_ms >= 1150, "{name}: {a:?}");
            }
        }
    }

    #[test]
    fn test_detects_unfilled_requirement() {
        // activity가 Machine+Mold를 요구하는데 Machine assignment만 있음
        let task = Task::new("J1").with_activity(
            machine_activity("O1", "J1", vec!["M1".into()]).with_requirement(
                ResourceRequirement::new("Mold").with_candidates(vec!["T1".into()]),
            ),
        );
        let resources = vec![Resource::new("M1", "Machine"), Resource::new("T1", "Mold")];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::RequirementUnfilled));
    }

    #[test]
    fn test_detects_capacity_exceeded() {
        let t1 = Task::new("J1").with_activity(machine_activity("O1", "J1", vec!["M1".into()]));
        let t2 = Task::new("J2").with_activity(machine_activity("O2", "J2", vec!["M1".into()]));
        let resources = vec![Resource::new("M1", "Machine")]; // capacity 1
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        s.add_assignment(Assignment::new("O2", "J2", "M1", 500, 1500)); // 겹침
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[t1, t2],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::CapacityExceeded));
    }

    #[test]
    fn test_capacity_two_allows_overlap() {
        let t1 = Task::new("J1").with_activity(machine_activity("O1", "J1", vec!["M1".into()]));
        let t2 = Task::new("J2").with_activity(machine_activity("O2", "J2", vec!["M1".into()]));
        let resources = vec![Resource::new("M1", "Machine").with_capacity(2)];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        s.add_assignment(Assignment::new("O2", "J2", "M1", 500, 1500));
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[t1, t2],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(!v
            .iter()
            .any(|x| x.violation_type == ViolationType::CapacityExceeded));
    }

    #[test]
    fn test_detects_calendar_violation() {
        let cal = Calendar::new("shift")
            .with_window(0, 5000)
            .with_window(10_000, 20_000);
        let task = Task::new("J1").with_activity(machine_activity("O1", "J1", vec!["M1".into()]));
        let resources = vec![Resource::new("M1", "Machine").with_calendar(cal)];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 3000, 6000)); // 창 침범
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::ResourceUnavailable));
    }

    #[test]
    fn without_candidates_a_resource_of_another_type_does_not_fill_the_requirement() {
        let task = Task::new("J1").with_activity(
            Activity::new("O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(ResourceRequirement::new("Machine").with_skill("milling")),
        );
        let resources = vec![
            Resource::new("W1", "Operator").with_skill("milling"),
            Resource::new("M1", "Machine").with_skill("milling"),
        ];
        let input = FeasibilityInput {
            tasks: std::slice::from_ref(&task),
            resources: &resources,
            constraints: &[],
        };

        // The operator has the skill but not the type.
        let mut wrong = Schedule::new();
        wrong.add_assignment(Assignment::new("O1", "J1", "W1", 0, 1000));
        let v = check_schedule(&wrong, &input);
        assert!(
            v.iter()
                .any(|x| x.violation_type == ViolationType::RequirementUnfilled),
            "{v:?}"
        );

        // The machine is accepted.
        let mut right = Schedule::new();
        right.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        assert!(check_schedule(&right, &input).is_empty());
    }

    #[test]
    fn without_candidates_a_resource_of_the_type_without_the_skill_is_a_skill_mismatch() {
        let task = Task::new("J1").with_activity(
            Activity::new("O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(ResourceRequirement::new("Machine").with_skill("milling")),
        );
        let resources = vec![Resource::new("M1", "Machine")];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::SkillMismatch));
    }

    #[test]
    fn test_detects_skill_mismatch() {
        let task = Task::new("J1").with_activity(
            Activity::new("O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(
                    ResourceRequirement::new("Machine")
                        .with_candidates(vec!["M1".into()])
                        .with_skill("welding"),
                ),
        );
        let resources = vec![Resource::new("M1", "Machine")]; // welding 없음
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::SkillMismatch));
    }

    #[test]
    fn test_clean_schedule_no_violations() {
        let task = Task::new("J1").with_activity(machine_activity("O1", "J1", vec!["M1".into()]));
        let resources = vec![Resource::new("M1", "Machine")];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v.is_empty(), "clean schedule flagged: {v:?}");
    }

    #[test]
    fn test_detects_precedence_violation() {
        let task = Task::new("J1")
            .with_activity(machine_activity("O1", "J1", vec!["M1".into()]))
            .with_activity({
                let mut a = machine_activity("O2", "J1", vec!["M2".into()]);
                a.sequence = 1;
                a
            });
        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
        ];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 500, 1500));
        s.add_assignment(Assignment::new("O2", "J1", "M2", 0, 1000)); // 선행보다 먼저
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::PrecedenceViolation));
    }

    #[test]
    fn test_detects_deadline_miss() {
        let task = Task::new("J1")
            .with_deadline(500)
            .with_activity(machine_activity("O1", "J1", vec!["M1".into()]));
        let resources = vec![Resource::new("M1", "Machine")];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[task],
                resources: &resources,
                constraints: &[],
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::DeadlineMiss));
    }

    #[test]
    fn test_detects_time_window_and_synchronize() {
        let t1 = Task::new("J1").with_activity(machine_activity("O1", "J1", vec!["M1".into()]));
        let t2 = Task::new("J2").with_activity(machine_activity("O2", "J2", vec!["M2".into()]));
        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
        ];
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        s.add_assignment(Assignment::new("O2", "J2", "M2", 500, 1500));
        let constraints = vec![
            Constraint::time_window("O1", 5000, 8000),
            Constraint::synchronize(vec!["O1".into(), "O2".into()]),
        ];
        let v = check_schedule(
            &s,
            &FeasibilityInput {
                tasks: &[t1, t2],
                resources: &resources,
                constraints: &constraints,
            },
        );
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::TimeWindowViolation));
        assert!(v
            .iter()
            .any(|x| x.violation_type == ViolationType::SynchronizeViolation));
    }
}
