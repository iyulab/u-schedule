//! Activity (operation) model.
//!
//! An activity is the smallest schedulable unit of work. It belongs to a task,
//! requires resources, has a duration, and may have precedence constraints.
//!
//! # Duration Model
//!
//! Each activity has three time components:
//! - **Setup**: Preparation time (may depend on previous activity via TransitionMatrix)
//! - **Process**: Core work time
//! - **Teardown**: Cleanup/cooldown time
//!
//! The time may depend on the resources an activity runs on
//! ([`ResourceTiming`]): a resource that works at a speed (processing time
//! divided by it — Pinedo's uniform machines `Q`, and a worker's efficiency in
//! dual-resource-constrained scheduling), or one with a time of its own for this
//! activity (unrelated machines `R`, `p_ij`). [`Activity::duration_on`] is the
//! one rule every scheduler and the feasibility check read.
//!
//! # Reference
//! Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 2;
//! Treleven & Elvers (1985), dual-resource-constrained job shops.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use super::{Calendar, Resource};

/// An activity (operation) to be scheduled.
///
/// Represents a single processing step that requires one or more resources
/// for a specified duration. Activities within a task are linked by
/// precedence constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    /// Unique activity identifier.
    pub id: String,
    /// Parent task identifier.
    pub task_id: String,
    /// Position within the task (0-indexed).
    pub sequence: i32,
    /// Time required to complete this activity — on a resource without a
    /// [`ResourceTiming`] of its own.
    pub duration: ActivityDuration,
    /// How the time changes on particular resources, by resource id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub resource_timings: BTreeMap<String, ResourceTiming>,
    /// Whether the work may pause while its resources are not all in working
    /// time (a break, a night, a holiday) and resume after — the resources
    /// stay held across the pause. Otherwise the whole span must lie in one
    /// stretch of working time.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub resumable: bool,
    /// The earliest the activity may start (ms) — a release of its own, on
    /// top of its task's (an operation whose material arrives later than
    /// the job).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_time: Option<i64>,
    /// When the activity itself may run, on top of its resources' calendars
    /// — the working time of an activity that holds no resource (a cure, a
    /// wait for a carrier), or a window of its own. `None`: any time its
    /// resources work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calendar: Option<Calendar>,
    /// Resources needed (type + quantity + candidates).
    pub resource_requirements: Vec<ResourceRequirement>,
    /// IDs of activities that must complete before this one starts.
    pub predecessors: Vec<String>,
    /// Domain-specific metadata.
    pub attributes: HashMap<String, String>,
}

impl Activity {
    /// Creates a new activity.
    pub fn new(id: impl Into<String>, task_id: impl Into<String>, sequence: i32) -> Self {
        Self {
            id: id.into(),
            task_id: task_id.into(),
            sequence,
            duration: ActivityDuration::default(),
            resource_timings: BTreeMap::new(),
            resumable: false,
            release_time: None,
            calendar: None,
            resource_requirements: Vec::new(),
            predecessors: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Starts the activity no earlier than `release_ms`
    /// ([`release_time`](Self::release_time)).
    pub fn with_release_time(mut self, release_ms: i64) -> Self {
        self.release_time = Some(release_ms);
        self
    }

    /// Sets the duration.
    pub fn with_duration(mut self, duration: ActivityDuration) -> Self {
        self.duration = duration;
        self
    }

    /// Sets the processing time (setup=0, teardown=0).
    pub fn with_process_time(mut self, process_ms: i64) -> Self {
        self.duration = ActivityDuration::fixed(process_ms);
        self
    }

    /// On `resource_id` the processing runs at `speed` (processing time ÷
    /// `speed`; setup and teardown unchanged). `speed` must be finite and
    /// positive — [`Problem::new`](crate::Problem::new) refuses others.
    pub fn with_speed_on(mut self, resource_id: impl Into<String>, speed: f64) -> Self {
        self.resource_timings
            .insert(resource_id.into(), ResourceTiming::Speed(speed));
        self
    }

    /// On `resource_id` the activity takes `duration`, whatever the other
    /// resources it runs on.
    pub fn with_duration_on(
        mut self,
        resource_id: impl Into<String>,
        duration: ActivityDuration,
    ) -> Self {
        self.resource_timings
            .insert(resource_id.into(), ResourceTiming::Duration(duration));
        self
    }

    /// The activity's time on `resources` (every resource it runs on, one id
    /// per unit):
    ///
    /// - a resource with a [`ResourceTiming::Duration`] sets the whole time —
    ///   with several, the one of greatest total;
    /// - otherwise processing is [`duration`](Self::duration)'s divided by the
    ///   product of the resources' [`ResourceTiming::Speed`]s, rounded to the
    ///   nearest ms; setup and teardown are unchanged.
    ///
    /// Setup here is the activity's own — a resource with a transition
    /// matrix replaces it with its changeover.
    pub fn duration_on<'a>(
        &self,
        resources: impl IntoIterator<Item = &'a str>,
    ) -> ActivityDuration {
        timed_duration(&self.duration, &self.resource_timings, resources)
    }

    /// Lets the work pause outside its resources' joint working time and
    /// resume after ([`resumable`](Self::resumable)).
    pub fn resumable(mut self) -> Self {
        self.resumable = true;
        self
    }

    /// Runs the activity only within `calendar`'s working time
    /// ([`calendar`](Self::calendar)).
    pub fn with_calendar(mut self, calendar: Calendar) -> Self {
        self.calendar = Some(calendar);
        self
    }

    /// Adds a resource requirement.
    pub fn with_requirement(mut self, req: ResourceRequirement) -> Self {
        self.resource_requirements.push(req);
        self
    }

    /// Adds a predecessor activity ID.
    pub fn with_predecessor(mut self, predecessor_id: impl Into<String>) -> Self {
        self.predecessors.push(predecessor_id.into());
        self
    }

    /// How much of `resource`'s capacity this activity takes when it runs
    /// there: the [`usage`](ResourceRequirement::usage) of the first
    /// requirement `resource` is a candidate of, or 1 (a companion, or a
    /// resource no requirement names).
    pub fn usage_on(&self, resource: &Resource) -> i32 {
        self.resource_requirements
            .iter()
            .find(|r| r.is_candidate(resource))
            .map_or(1, |r| r.usage)
    }

    /// IDs of the resources in `resources` that some requirement of this
    /// activity [accepts](ResourceRequirement::accepts), in `resources`
    /// order, each once.
    pub fn eligible_resources<'a>(&self, resources: &'a [Resource]) -> Vec<&'a str> {
        resources
            .iter()
            .filter(|r| self.resource_requirements.iter().any(|req| req.accepts(r)))
            .map(|r| r.id.as_str())
            .collect()
    }
}

/// [`Activity::duration_on`]'s rule, for whatever holds an activity's base
/// duration and timings.
pub(crate) fn timed_duration<'a>(
    base: &ActivityDuration,
    timings: &BTreeMap<String, ResourceTiming>,
    resources: impl IntoIterator<Item = &'a str>,
) -> ActivityDuration {
    if timings.is_empty() {
        return *base;
    }
    let mut speed = 1.0;
    let mut fixed: Option<&ActivityDuration> = None;
    for id in resources {
        match timings.get(id) {
            Some(ResourceTiming::Speed(s)) => speed *= s,
            Some(ResourceTiming::Duration(d))
                if fixed.is_none_or(|f| d.total_ms() > f.total_ms()) =>
            {
                fixed = Some(d);
            }
            Some(ResourceTiming::Duration(_)) | None => {}
        }
    }
    if let Some(d) = fixed {
        return *d;
    }
    if speed == 1.0 {
        return *base;
    }
    ActivityDuration {
        process_ms: (base.process_ms as f64 / speed).round() as i64,
        ..*base
    }
}

/// How an activity's time changes on one resource.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceTiming {
    /// Processing runs at this speed: time ÷ speed (2.0 is twice as fast).
    Speed(f64),
    /// The activity takes this long here.
    Duration(ActivityDuration),
}

/// Time components of an activity.
///
/// # Components
/// - **Setup**: Preparation before processing (e.g., machine changeover).
///   May be overridden by `TransitionMatrix` for sequence-dependent setups.
/// - **Process**: Core work time (the actual operation).
/// - **Teardown**: Cleanup after processing (e.g., cooling, inspection).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActivityDuration {
    /// Setup/preparation time (ms).
    pub setup_ms: i64,
    /// Core processing time (ms).
    pub process_ms: i64,
    /// Teardown/cleanup time (ms).
    pub teardown_ms: i64,
}

impl ActivityDuration {
    /// Creates a duration with all three components.
    pub fn new(setup_ms: i64, process_ms: i64, teardown_ms: i64) -> Self {
        Self {
            setup_ms,
            process_ms,
            teardown_ms,
        }
    }

    /// Creates a fixed-duration activity (setup=0, teardown=0).
    pub fn fixed(process_ms: i64) -> Self {
        Self::new(0, process_ms, 0)
    }

    /// Total duration (setup + process + teardown).
    pub fn total_ms(&self) -> i64 {
        self.setup_ms + self.process_ms + self.teardown_ms
    }
}

impl Default for ActivityDuration {
    fn default() -> Self {
        Self::fixed(0)
    }
}

/// A resource requirement for an activity.
///
/// Specifies what type and quantity of resources are needed,
/// with optional candidate filtering and skill requirements.
/// [`accepts`](Self::accepts) is the one rule every solver and the
/// feasibility checker use to decide which resources may fill it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    /// Required resource type (e.g., "Machine", "Operator"), compared exactly
    /// with [`Resource::resource_type`] when `candidates` is empty.
    pub resource_type: String,
    /// Number of resources needed simultaneously, each a different one.
    pub quantity: i32,
    /// How much of each chosen resource's [`capacity`](Resource::capacity)
    /// the activity takes while it runs (default 1): a resource carries
    /// activities at once as long as their usages sum to at most its
    /// capacity -- an operator at capacity 10 can run two activities of
    /// usage 5 side by side (a renewable resource with demand `r_jk`,
    /// Kolisch 1996).
    #[serde(default = "one_unit")]
    pub usage: i32,
    /// Specific resource IDs that can fulfill this requirement.
    ///
    /// Empty = any resource whose `resource_type` equals this requirement's.
    /// Non-empty = exactly the listed resources: the explicit list wins, and
    /// a listed resource is accepted whatever its type.
    pub candidates: Vec<String>,
    /// Required skills: a resource fills the requirement only if
    /// [`Resource::skills`] contains every one (exact name match), whether
    /// it is listed in `candidates` or matched by type.
    pub required_skills: Vec<String>,
}

impl ResourceRequirement {
    /// Creates a new requirement for one unit of a resource type.
    pub fn new(resource_type: impl Into<String>) -> Self {
        Self {
            resource_type: resource_type.into(),
            quantity: 1,
            usage: 1,
            candidates: Vec::new(),
            required_skills: Vec::new(),
        }
    }

    /// Sets how much of each chosen resource's capacity the activity takes
    /// ([`usage`](Self::usage)).
    pub fn with_usage(mut self, usage: i32) -> Self {
        self.usage = usage;
        self
    }

    /// Sets the required quantity.
    pub fn with_quantity(mut self, quantity: i32) -> Self {
        self.quantity = quantity;
        self
    }

    /// Adds candidate resource IDs.
    pub fn with_candidates(mut self, candidates: Vec<String>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Adds a required skill.
    pub fn with_skill(mut self, skill: impl Into<String>) -> Self {
        self.required_skills.push(skill.into());
        self
    }

    /// Whether `resource` is in this requirement's pool, skills aside: listed
    /// in `candidates`, or -- when `candidates` is empty -- of this
    /// requirement's `resource_type`.
    pub fn is_candidate(&self, resource: &Resource) -> bool {
        if self.candidates.is_empty() {
            resource.resource_type == self.resource_type
        } else {
            self.candidates.contains(&resource.id)
        }
    }

    /// Whether `resource` may fill this requirement: it
    /// [is a candidate](Self::is_candidate) and has every required skill.
    pub fn accepts(&self, resource: &Resource) -> bool {
        self.is_candidate(resource) && self.required_skills.iter().all(|s| resource.has_skill(s))
    }

    /// The resources in `resources` this requirement
    /// [accepts](Self::accepts), in `resources` order.
    pub fn eligible<'a>(&'a self, resources: &'a [Resource]) -> impl Iterator<Item = &'a Resource> {
        resources.iter().filter(move |r| self.accepts(r))
    }
}

fn one_unit() -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_builder() {
        let act = Activity::new("O1", "J1", 0)
            .with_duration(ActivityDuration::new(100, 500, 50))
            .with_requirement(ResourceRequirement::new("Machine").with_quantity(1))
            .with_predecessor("O0");

        assert_eq!(act.id, "O1");
        assert_eq!(act.task_id, "J1");
        assert_eq!(act.sequence, 0);
        assert_eq!(act.duration.total_ms(), 650);
        assert_eq!(act.resource_requirements.len(), 1);
        assert_eq!(act.predecessors, vec!["O0"]);
    }

    #[test]
    fn test_activity_duration_fixed() {
        let d = ActivityDuration::fixed(1000);
        assert_eq!(d.setup_ms, 0);
        assert_eq!(d.process_ms, 1000);
        assert_eq!(d.teardown_ms, 0);
        assert_eq!(d.total_ms(), 1000);
    }

    #[test]
    fn test_activity_duration_components() {
        let d = ActivityDuration::new(100, 500, 50);
        assert_eq!(d.total_ms(), 650);
    }

    #[test]
    fn test_resource_requirement() {
        let req = ResourceRequirement::new("CNC")
            .with_quantity(2)
            .with_candidates(vec!["M1".into(), "M2".into(), "M3".into()])
            .with_skill("milling");

        assert_eq!(req.resource_type, "CNC");
        assert_eq!(req.quantity, 2);
        assert_eq!(req.candidates.len(), 3);
        assert_eq!(req.required_skills, vec!["milling"]);
    }

    #[test]
    fn test_eligible_resources() {
        let resources = [
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
            Resource::new("W1", "Operator"),
            Resource::new("W2", "Operator"),
        ];
        let act = Activity::new("O1", "J1", 0)
            .with_requirement(
                ResourceRequirement::new("Machine").with_candidates(vec!["M2".into(), "M1".into()]),
            )
            .with_requirement(
                ResourceRequirement::new("Operator").with_candidates(vec!["W1".into()]),
            );

        assert_eq!(act.eligible_resources(&resources), vec!["M1", "M2", "W1"]);
    }

    #[test]
    fn without_candidates_only_resources_of_the_requirement_type_with_the_skills_qualify() {
        let resources = [
            Resource::new("M1", "Machine").with_skill("milling"),
            Resource::new("M2", "Machine"),
            // Right skill, wrong type.
            Resource::new("W1", "Operator").with_skill("milling"),
        ];
        let req = ResourceRequirement::new("Machine").with_skill("milling");
        let ids: Vec<&str> = req.eligible(&resources).map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["M1"]);

        assert!(req.is_candidate(&resources[1]), "type matches");
        assert!(!req.accepts(&resources[1]), "but the skill is missing");
        assert!(!req.is_candidate(&resources[2]), "another type");

        // Without a skill requirement every resource of the type qualifies.
        let any_machine = ResourceRequirement::new("Machine");
        let ids: Vec<&str> = any_machine
            .eligible(&resources)
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(ids, vec!["M1", "M2"]);
    }

    #[test]
    fn a_listed_candidate_is_accepted_whatever_its_type_if_it_has_the_skills() {
        let resources = [
            Resource::new("M1", "Machine"),
            Resource::new("W1", "Operator").with_skill("milling"),
            Resource::new("W2", "Operator"),
        ];
        let req = ResourceRequirement::new("Machine")
            .with_candidates(vec!["W1".into(), "W2".into()])
            .with_skill("milling");
        let ids: Vec<&str> = req.eligible(&resources).map(|r| r.id.as_str()).collect();
        // M1 has the type but is not listed; W2 is listed but lacks the skill.
        assert_eq!(ids, vec!["W1"]);
    }
}
