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
//! task's release time. An activity starts no earlier than the one before it
//! in its task ends, plus the time the work takes to move between the
//! resources they run on ([`Resource::transport_ms`]).
//!
//! An activity with a calendar of its own
//! ([`Activity::calendar`](crate::models::Activity::calendar)) runs within it, on
//! top of its resources' calendars. An activity that holds no resource has one
//! [`TimeOnlySpan`](crate::models::TimeOnlySpan) in
//! [`Schedule::time_only`], at least its whole time long (resumable: that much
//! of its own working time), starting no earlier than its and its task's
//! release.
//!
//! Not checked (documented gap): `Constraint::TransitionCost` (needs
//! sequence semantics), and so whether an assignment's recorded setup is the
//! one the resource's transitions call for.
//!
//! # Reference
//! Brucker (2007), "Scheduling Algorithms", Ch. 2 — constraint taxonomy.

use std::collections::{HashMap, HashSet};

use crate::models::{transport_between, Activity, Constraint, Resource, Schedule, Task, Violation};

/// The resources among `assigned` (one activity's) held only as a companion
/// of another of them ([`Resource::companions`]).
fn companions_among<'a>(
    assigned: &[&'a str],
    res_by_id: &HashMap<&str, &Resource>,
) -> HashSet<&'a str> {
    assigned
        .iter()
        .filter(|&&r| {
            assigned.iter().any(|&p| {
                p != r
                    && res_by_id
                        .get(p)
                        .is_some_and(|res| res.companions.iter().any(|c| c == r))
            })
        })
        .copied()
        .collect()
}

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
    check_assignments(schedule, input, &res_by_id, &mut v);
    check_time_only(schedule, input, &mut v);
    check_requirements(schedule, input, &res_by_id, &mut v);
    check_capacity(schedule, input, &res_by_id, &mut v);
    check_calendars(schedule, input, &res_by_id, &mut v);
    check_precedence(schedule, input, &res_by_id, &mut v);
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
fn check_assignments(
    schedule: &Schedule,
    input: &FeasibilityInput,
    res_by_id: &HashMap<&str, &Resource>,
    out: &mut Vec<Violation>,
) {
    let mut by_id: HashMap<&str, (&Task, &Activity)> = HashMap::new();
    for task in input.tasks {
        for activity in &task.activities {
            by_id.insert(activity.id.as_str(), (task, activity));
        }
    }
    let model_tasks: HashSet<&str> = input.tasks.iter().map(|t| t.id.as_str()).collect();
    // The resources each activity runs on (its time depends on them), and
    // those it only holds as companions (they take no unit and no time).
    let mut runs_on: HashMap<&str, Vec<&str>> = HashMap::new();
    for a in &schedule.assignments {
        runs_on
            .entry(a.activity_id.as_str())
            .or_default()
            .push(a.resource_id.as_str());
    }
    let held: HashMap<&str, HashSet<&str>> = runs_on
        .iter()
        .map(|(&act, assigned)| (act, companions_among(assigned, res_by_id)))
        .collect();
    for (act, assigned) in runs_on.iter_mut() {
        assigned.retain(|r| !held[act].contains(r));
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
        if held[a.activity_id.as_str()].contains(a.resource_id.as_str()) {
            continue; // a companion: no unit, no time of its own
        }
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
        if let Some(release) = activity.release_time {
            if a.start_ms < release {
                out.push(Violation::early_start(
                    &a.activity_id,
                    format!(
                        "activity '{}' starts at {} before its release {release}",
                        a.activity_id, a.start_ms
                    ),
                ));
            }
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

/// Each activity that holds no resource has one time-only span, at least its
/// time long (in its own working time, if it has a calendar), from its
/// releases on; a span for any other activity is unknown.
fn check_time_only(schedule: &Schedule, input: &FeasibilityInput, out: &mut Vec<Violation>) {
    let mut by_id: HashMap<&str, (&Task, &Activity)> = HashMap::new();
    for task in input.tasks {
        for activity in &task.activities {
            by_id.insert(activity.id.as_str(), (task, activity));
        }
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for span in &schedule.time_only {
        let id = span.activity_id.as_str();
        let known = by_id
            .get(id)
            .filter(|(task, a)| a.resource_requirements.is_empty() && task.id == span.task_id);
        let Some(&(task, activity)) = known else {
            out.push(Violation::unknown_activity(
                id,
                format!(
                    "time-only span of '{id}' under task '{}' names no activity of that task that holds no resource",
                    span.task_id
                ),
            ));
            continue;
        };
        if !seen.insert(id) {
            out.push(Violation::unknown_activity(
                id,
                format!("activity '{id}' has more than one time-only span"),
            ));
            continue;
        }
        let work = activity.duration.total_ms();
        let calendars: Vec<&crate::models::Calendar> = activity.calendar.iter().collect();
        let worked = if activity.resumable {
            crate::models::joint_available_ms(&calendars, span.start_ms, span.end_ms)
        } else {
            span.end_ms - span.start_ms
        };
        if worked < work {
            out.push(Violation::duration_too_short(
                id,
                format!(
                    "time-only activity '{id}' [{}..{}) has {worked} ms of its time; it needs {work}",
                    span.start_ms, span.end_ms
                ),
            ));
        }
        let in_own = match &activity.calendar {
            None => true,
            Some(_) if activity.resumable => {
                crate::models::joint_work_end(&calendars, span.start_ms, 0).is_some()
            }
            Some(cal) => cal.interval_fits(span.start_ms, span.end_ms),
        };
        if !in_own {
            out.push(Violation::time_window(
                id,
                format!(
                    "time-only activity '{id}' [{}..{}) is outside its calendar",
                    span.start_ms, span.end_ms
                ),
            ));
        }
        for (release, whose) in [
            (activity.release_time, "its"),
            (task.release_time, "its task's"),
        ] {
            if let Some(release) = release.filter(|&r| span.start_ms < r) {
                out.push(Violation::early_start(
                    id,
                    format!(
                        "activity '{id}' starts at {} before {whose} release {release}",
                        span.start_ms
                    ),
                ));
            }
        }
    }
    for task in input.tasks {
        for activity in &task.activities {
            if activity.resource_requirements.is_empty() && !seen.contains(activity.id.as_str()) {
                out.push(Violation::time_window(
                    &activity.id,
                    format!(
                        "activity '{}' holds no resource and has no time in the schedule — its calendar never holds its time",
                        activity.id
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
            // greedy 귀속: 요구별로 미귀속 자원 claim — companions are held,
            // not claimed; each resource that is claimed holds its own.
            let assigned: Vec<&str> = assigns.iter().map(|a| a.resource_id.as_str()).collect();
            let held = companions_among(&assigned, res_by_id);
            for &r in assigned.iter().filter(|r| !held.contains(*r)) {
                if let Some(res) = res_by_id.get(r) {
                    for c in &res.companions {
                        if !assigned.contains(&c.as_str()) {
                            out.push(Violation::requirement_unfilled(
                                &activity.id,
                                format!(
                                    "activity '{}' runs on '{r}' without its companion '{c}'",
                                    activity.id
                                ),
                            ));
                        }
                    }
                }
            }
            let mut unclaimed: Vec<&str> = assigned
                .iter()
                .copied()
                .filter(|r| !held.contains(r))
                .collect();
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
    let activities: HashMap<&str, &Activity> = input
        .tasks
        .iter()
        .flat_map(|t| &t.activities)
        .map(|a| (a.id.as_str(), a))
        .collect();
    // Each assignment takes its activity's usage of the resource; an
    // external booking, one unit.
    let mut by_resource: HashMap<&str, Vec<(i64, i64, i32)>> = HashMap::new();
    for a in &schedule.assignments {
        let usage = match (
            activities.get(a.activity_id.as_str()),
            res_by_id.get(a.resource_id.as_str()),
        ) {
            (Some(activity), Some(resource)) => activity.usage_on(resource),
            _ => 1,
        };
        by_resource
            .entry(a.resource_id.as_str())
            .or_default()
            .push((a.start_ms, a.end_ms, usage));
    }
    for (rid, intervals) in by_resource {
        let base = res_by_id.get(rid).map_or(1, |r| r.capacity.max(1));
        let cap = cap_override.get(rid).map_or(base, |&c| base.min(c));
        // 부하는 구간 시작점에서만 증가
        let peak = intervals
            .iter()
            .map(|&(s, _, _)| {
                intervals
                    .iter()
                    .filter(|&&(s2, e2, _)| s2 <= s && s < e2)
                    .map(|&(_, _, u)| u)
                    .sum::<i32>()
            })
            .max()
            .unwrap_or(0);
        if peak > cap {
            out.push(Violation::capacity_exceeded(
                rid,
                format!("resource '{rid}' peak load {peak} exceeds capacity {cap}"),
            ));
        }
    }
}

fn check_calendars(
    schedule: &Schedule,
    input: &FeasibilityInput,
    res_by_id: &HashMap<&str, &Resource>,
    out: &mut Vec<Violation>,
) {
    let activities: HashMap<&str, &Activity> = input
        .tasks
        .iter()
        .flat_map(|t| &t.activities)
        .map(|a| (a.id.as_str(), a))
        .collect();
    for a in &schedule.assignments {
        let Some(res) = res_by_id.get(a.resource_id.as_str()) else {
            continue;
        };
        match activities.get(a.activity_id.as_str()) {
            // Resumable work may pause: it starts in the joint working time
            // of every resource it runs on, and that time within its span
            // covers the work.
            Some(activity) if activity.resumable => {
                let runs_on: Vec<&str> = schedule
                    .assignments_for_activity_all(&activity.id)
                    .iter()
                    .map(|x| x.resource_id.as_str())
                    .collect();
                let calendars: Vec<&crate::models::Calendar> = runs_on
                    .iter()
                    .filter_map(|r| res_by_id.get(r).and_then(|res| res.calendar.as_ref()))
                    .chain(&activity.calendar)
                    .collect();
                let held = companions_among(&runs_on, res_by_id);
                let d = activity.duration_on(runs_on.iter().copied().filter(|r| !held.contains(r)));
                let needed = a.setup_ms + d.process_ms + d.teardown_ms;
                let starts_in_work =
                    crate::models::joint_work_end(&calendars, a.start_ms, 0).is_some();
                let worked = crate::models::joint_available_ms(&calendars, a.start_ms, a.end_ms);
                if !starts_in_work || worked < needed {
                    out.push(Violation::resource_unavailable(
                        &a.resource_id,
                        format!(
                            "assignment '{}' [{}..{}) on '{}' has {worked} ms of joint working time{}; its work needs {needed}",
                            a.activity_id,
                            a.start_ms,
                            a.end_ms,
                            a.resource_id,
                            if starts_in_work { "" } else { " and starts outside it" }
                        ),
                    ));
                }
            }
            other => {
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
                if let Some(own) = other.and_then(|activity| activity.calendar.as_ref()) {
                    if !own.interval_fits(a.start_ms, a.end_ms) {
                        out.push(Violation::time_window(
                            &a.activity_id,
                            format!(
                                "assignment '{}' [{}..{}) on '{}' is outside the activity's calendar",
                                a.activity_id, a.start_ms, a.end_ms, a.resource_id
                            ),
                        ));
                    }
                }
            }
        }
    }
}

/// Activity span in the schedule: (min start, max end) across its
/// assignments, or its time-only span.
fn activity_span(schedule: &Schedule, activity_id: &str) -> Option<(i64, i64)> {
    if let Some(span) = schedule.time_only_for_activity(activity_id) {
        return Some((span.start_ms, span.end_ms));
    }
    let assigns = schedule.assignments_for_activity_all(activity_id);
    let start = assigns.iter().map(|a| a.start_ms).min()?;
    let end = assigns
        .iter()
        .map(|a| a.end_ms)
        .max()
        .expect("non-empty when min exists");
    Some((start, end))
}

fn check_precedence(
    schedule: &Schedule,
    input: &FeasibilityInput,
    res_by_id: &HashMap<&str, &Resource>,
    out: &mut Vec<Violation>,
) {
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
    let runs_on = |activity: &str| -> Vec<&str> {
        schedule
            .assignments_for_activity_all(activity)
            .iter()
            .map(|a| a.resource_id.as_str())
            .collect()
    };
    let mut overlapped = Vec::new();
    for task in input.tasks {
        let mut ordered: Vec<&Activity> = task.activities.iter().collect();
        ordered.sort_by_key(|a| a.sequence);
        for pair in ordered.windows(2) {
            // The work moves from where one ran to where the next runs.
            let transport = transport_between(
                |id| res_by_id.get(id).copied(),
                runs_on(&pair[0].id),
                &runs_on(&pair[1].id),
            );
            match &pair[1].overlap {
                None => push(&pair[0].id, &pair[1].id, transport),
                Some(overlap) => overlapped.push((pair[0], pair[1], overlap, transport)),
            }
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
    // A step that overlaps the one before: two bounds instead of end-to-start
    // (`Overlap`), each on top of the transport between them.
    for (before, after, overlap, transport) in overlapped {
        let (Some(previous), Some((start, end))) = (
            activity_span(schedule, &before.id),
            activity_span(schedule, &after.id),
        ) else {
            continue;
        };
        let earliest_start = overlap.earliest_start(previous) + transport;
        if start < earliest_start {
            out.push(Violation::precedence_violation(
                &after.id,
                format!(
                    "'{}' starts at {start}, before {earliest_start}: {} of '{}' ({}..{}) + transport {transport}",
                    after.id, overlap.start_after, before.id, previous.0, previous.1
                ),
            ));
        }
        let earliest_end = overlap.earliest_end(previous.1) + transport;
        if end < earliest_end {
            out.push(Violation::precedence_violation(
                &after.id,
                format!(
                    "'{}' ends at {end}, before {earliest_end}: '{}' end {} + lag {} + transport {transport}",
                    after.id, before.id, previous.1, overlap.finish_after_ms
                ),
            ));
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

        // Odd jobs overlap their second step with the first: from 0.4 of it
        // on, and ending 300 ms after it.
        let tasks: Vec<Task> = (0..4)
            .map(|j| {
                let id = format!("J{j}");
                let mut t = Task::new(&id).with_release_time(200 * j as i64);
                for k in 0..2 {
                    let mut step = Activity::new(format!("{id}-O{k}"), &id, k)
                        .with_duration(ActivityDuration::new(100, 1000 + 100 * k as i64, 50))
                        .with_requirement(ResourceRequirement::new("Machine"));
                    if k == 1 && j % 2 == 1 {
                        step = step
                            .with_overlap(crate::models::Overlap::new(0.4).with_finish_after(300));
                    }
                    t = t.with_activity(step);
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
    fn a_time_only_span_is_checked_like_an_assignment() {
        use crate::models::TimeOnlySpan;
        let task = Task::new("J1")
            .with_activity(machine_activity("O1", "J1", vec!["M1".into()]))
            .with_activity({
                let mut cure = Activity::new("CURE", "J1", 1).with_process_time(2000);
                cure.calendar = Some(crate::models::Calendar::new("c").with_window(0, 10_000));
                cure
            });
        let resources = vec![Resource::new("M1", "Machine")];
        let check = |s: &Schedule| {
            check_schedule(
                s,
                &FeasibilityInput {
                    tasks: std::slice::from_ref(&task),
                    resources: &resources,
                    constraints: &[],
                },
            )
        };
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 1000));
        // No span: the cure is not scheduled.
        assert!(check(&s).iter().any(
            |x| x.entity_id == "CURE" && x.violation_type == ViolationType::TimeWindowViolation
        ));
        // Before the machine work ends, too short and past its calendar.
        s.time_only.push(TimeOnlySpan::new("CURE", "J1", 500, 1000));
        let kinds: Vec<ViolationType> = check(&s).into_iter().map(|x| x.violation_type).collect();
        assert!(
            kinds.contains(&ViolationType::PrecedenceViolation),
            "{kinds:?}"
        );
        assert!(
            kinds.contains(&ViolationType::DurationTooShort),
            "{kinds:?}"
        );
        s.time_only[0] = TimeOnlySpan::new("CURE", "J1", 9000, 11_000);
        let kinds: Vec<ViolationType> = check(&s).into_iter().map(|x| x.violation_type).collect();
        assert_eq!(kinds, [ViolationType::TimeWindowViolation]);
        s.time_only[0] = TimeOnlySpan::new("CURE", "J1", 1000, 3000);
        assert!(check(&s).is_empty());
        // A span of an activity that holds a resource names nothing.
        s.time_only.push(TimeOnlySpan::new("O1", "J1", 0, 1000));
        assert!(check(&s)
            .iter()
            .any(|x| x.violation_type == ViolationType::UnknownActivity));
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

    /// A task of two 1 s activities: on `first`, then on any of `next`.
    fn moved(first: &str, next: &[&str]) -> Task {
        let on = |id: &str, seq: i32, machines: Vec<String>| {
            Activity::new(id, "J", seq)
                .with_process_time(1_000)
                .with_requirement(ResourceRequirement::new("Machine").with_candidates(machines))
        };
        Task::new("J")
            .with_activity(on("J-1", 0, vec![first.to_string()]))
            .with_activity(on("J-2", 1, next.iter().map(|m| m.to_string()).collect()))
    }

    fn overlapped(share: f64, lag: i64) -> Vec<Task> {
        vec![Task::new("J")
            .with_activity(
                Activity::new("cut", "J", 1)
                    .with_process_time(100)
                    .with_requirement(ResourceRequirement::new("M")),
            )
            .with_activity(
                Activity::new("bend", "J", 2)
                    .with_process_time(100)
                    .with_requirement(ResourceRequirement::new("M"))
                    .with_overlap(crate::models::Overlap::new(share).with_finish_after(lag)),
            )]
    }

    fn two_steps(bend_start: i64, bend_end: i64) -> Schedule {
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("cut", "J", "M1", 0, 100));
        s.add_assignment(Assignment::new("bend", "J", "M2", bend_start, bend_end));
        s
    }

    #[test]
    fn an_overlapping_step_may_start_once_its_share_is_done_but_not_end_first() {
        let tasks = overlapped(0.3, 10);
        let resources = vec![Resource::new("M1", "M"), Resource::new("M2", "M")];
        let input = FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        // From 30 (0.3 of 0..100) on, ending at 110 (100 + 10) or later: clean.
        assert!(check_schedule(&two_steps(30, 130), &input).is_empty());
        let early_start = check_schedule(&two_steps(10, 110), &input);
        assert_eq!(early_start.len(), 1, "{early_start:?}");
        assert!(early_start[0].message.contains("starts at 10"));
        // Ends at 105: the last batch is not through yet.
        let mut tasks = tasks.clone();
        tasks[0].activities[1].duration = crate::models::ActivityDuration::fixed(75);
        let input = FeasibilityInput {
            tasks: &tasks,
            ..input
        };
        let early_end = check_schedule(&two_steps(30, 105), &input);
        assert_eq!(early_end.len(), 1, "{early_end:?}");
        assert_eq!(
            early_end[0].violation_type,
            ViolationType::PrecedenceViolation
        );
        assert!(
            early_end[0].message.contains("ends at 105"),
            "{early_end:?}"
        );
    }

    #[test]
    fn starting_before_the_work_arrives_is_a_precedence_violation() {
        let tasks = [moved("A", &["B"])];
        let resources = [
            Resource::new("A", "Machine").with_transport_to("B", 500),
            Resource::new("B", "Machine"),
        ];
        let input = FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let at = |start: i64| {
            let mut s = Schedule::new();
            s.add_assignment(Assignment::new("J-1", "J", "A", 0, 1_000));
            s.add_assignment(Assignment::new("J-2", "J", "B", start, start + 1_000));
            check_schedule(&s, &input)
        };
        let early = at(1_200);
        assert_eq!(early.len(), 1, "{early:?}");
        assert_eq!(early[0].violation_type, ViolationType::PrecedenceViolation);
        assert!(at(1_500).is_empty());
    }

    /// A one-activity task on its own machine `M{id}` with operator `W1`,
    /// taking `usage` of the operator.
    fn shared(id: &str, usage: i32) -> Task {
        Task::new(id).with_activity(
            Activity::new(format!("{id}-1"), id, 0)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![format!("M{id}")]),
                )
                .with_requirement(
                    ResourceRequirement::new("Operator")
                        .with_candidates(vec!["W1".into()])
                        .with_usage(usage),
                ),
        )
    }

    fn operator_and_machines(ids: &[&str]) -> Vec<Resource> {
        let mut resources: Vec<Resource> = ids
            .iter()
            .map(|id| Resource::new(format!("M{id}"), "Machine"))
            .collect();
        resources.push(Resource::new("W1", "Operator").with_capacity(10));
        resources
    }

    #[test]
    fn the_usages_at_once_are_held_to_the_capacity() {
        let tasks = [shared("A", 5), shared("B", 6)];
        let resources = operator_and_machines(&["A", "B"]);
        let input = FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let at = |b_start: i64| {
            let mut s = Schedule::new();
            s.add_assignment(Assignment::new("A-1", "A", "MA", 0, 1_000));
            s.add_assignment(Assignment::new("A-1", "A", "W1", 0, 1_000));
            s.add_assignment(Assignment::new("B-1", "B", "MB", b_start, b_start + 1_000));
            s.add_assignment(Assignment::new("B-1", "B", "W1", b_start, b_start + 1_000));
            check_schedule(&s, &input)
        };
        let together = at(0);
        assert_eq!(together.len(), 1, "{together:?}");
        assert_eq!(together[0].violation_type, ViolationType::CapacityExceeded);
        assert!(at(1_000).is_empty());
    }

    /// J1: 1 s on A, then 1 s on B released at 5 s.
    fn released_second() -> Task {
        let on = |id: &str, seq: i32, machine: &str| {
            Activity::new(id, "J1", seq)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        Task::new("J1")
            .with_activity(on("J1-1", 0, "A"))
            .with_activity(on("J1-2", 1, "B").with_release_time(5_000))
    }

    #[test]
    fn starting_before_the_activity_release_is_an_early_start() {
        let tasks = [released_second()];
        let resources = [Resource::new("A", "Machine"), Resource::new("B", "Machine")];
        let input = FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("J1-1", "J1", "A", 0, 1_000));
        s.add_assignment(Assignment::new("J1-2", "J1", "B", 1_000, 2_000));
        let v = check_schedule(&s, &input);
        assert_eq!(v.len(), 1, "{v:?}");
        assert_eq!(v[0].violation_type, ViolationType::EarlyStart);
    }
}
