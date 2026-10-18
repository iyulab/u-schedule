//! Resource model.
//!
//! Resources are the entities that perform activities: machines, workers,
//! tools, rooms, vehicles. Each resource has a type, a capacity, skills,
//! and an optional availability calendar.
//!
//! The type is a name from the same vocabulary as
//! [`ResourceRequirement::resource_type`](super::ResourceRequirement::resource_type)
//! ("Machine", "Operator", "Mold", ...): a requirement that lists no
//! candidates is filled only by resources of its type. The type has no
//! other scheduling meaning.
//!
//! # Reference
//! Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 1.2

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use super::Calendar;

/// A resource that can be assigned to activities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Unique resource identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Resource type (e.g., "Machine", "Operator"). A requirement with no
    /// explicit candidates is filled only by resources whose type equals its
    /// [`resource_type`](super::ResourceRequirement::resource_type) (exact,
    /// case-sensitive comparison).
    pub resource_type: String,
    /// Number of units available simultaneously (default: 1).
    pub capacity: i32,
    /// Availability schedule.
    pub calendar: Option<Calendar>,
    /// Skill names (e.g., "welding", "milling"), matched exactly against a
    /// requirement's [`required_skills`](super::ResourceRequirement::required_skills).
    pub skills: Vec<String>,
    /// Resources held together with this one whenever an activity runs on it
    /// (a robot that occupies its cell, a machine that needs its fixture) —
    /// for the activity's whole span, without changing its time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub companions: Vec<String>,
    /// Time to move work from this resource to another (ms), by the other's
    /// id: when a task's next activity runs there, it starts no earlier than
    /// this one's end plus the time. A pair not listed takes none.
    ///
    /// Transportation times between machines, `t_kl` -- Hurink & Knust
    /// (2001), "Makespan minimization for flow-shop problems with
    /// transportation times and a single robot".
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub transport_ms: BTreeMap<String, i64>,
    /// Domain-specific metadata.
    pub attributes: HashMap<String, String>,
}

impl Resource {
    /// Creates a resource of the given type, capacity 1, no skills, always
    /// available.
    pub fn new(id: impl Into<String>, resource_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            resource_type: resource_type.into(),
            capacity: 1,
            calendar: None,
            skills: Vec::new(),
            companions: Vec::new(),
            transport_ms: BTreeMap::new(),
            attributes: HashMap::new(),
        }
    }

    /// Holds `resource_id` too whenever an activity runs on this resource
    /// ([`companions`](Self::companions)).
    pub fn with_companion(mut self, resource_id: impl Into<String>) -> Self {
        self.companions.push(resource_id.into());
        self
    }

    /// Work moving from this resource to `resource_id` takes `time_ms`
    /// ([`transport_ms`](Self::transport_ms)).
    pub fn with_transport_to(mut self, resource_id: impl Into<String>, time_ms: i64) -> Self {
        self.transport_ms.insert(resource_id.into(), time_ms);
        self
    }

    /// The time to move work from this resource to `resource_id` (0 when
    /// none is given).
    pub fn transport_to(&self, resource_id: &str) -> i64 {
        self.transport_ms.get(resource_id).copied().unwrap_or(0)
    }

    /// Sets the resource name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the capacity.
    pub fn with_capacity(mut self, capacity: i32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets the availability calendar.
    pub fn with_calendar(mut self, calendar: Calendar) -> Self {
        self.calendar = Some(calendar);
        self
    }

    /// Adds a skill.
    pub fn with_skill(mut self, name: impl Into<String>) -> Self {
        self.skills.push(name.into());
        self
    }

    /// Adds a domain-specific attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Whether this resource has a given skill.
    pub fn has_skill(&self, name: &str) -> bool {
        self.skills.iter().any(|s| s == name)
    }

    /// Checks availability at a given time (ms).
    ///
    /// Returns `true` if no calendar is set (always available)
    /// or if the calendar indicates working time.
    pub fn is_available_at(&self, time_ms: i64) -> bool {
        match &self.calendar {
            None => true,
            Some(cal) => cal.is_working_time(time_ms),
        }
    }
}

/// The time to move a task's work from the resources one activity ran on
/// (`from`) to those its next activity runs on (`to`): the longest of the
/// pairs, since the work waits for the last of its moves. `resource` looks a
/// resource up by id; one it does not know moves nothing.
pub fn transport_between<'r, 'a>(
    resource: impl Fn(&str) -> Option<&'r Resource>,
    from: impl IntoIterator<Item = &'a str>,
    to: &[&str],
) -> i64 {
    from.into_iter()
        .filter_map(&resource)
        .flat_map(|r| to.iter().map(move |t| r.transport_to(t)))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_builder() {
        let r = Resource::new("M1", "Machine")
            .with_name("CNC Machine 1")
            .with_capacity(1)
            .with_skill("milling")
            .with_skill("drilling")
            .with_attribute("location", "Shop Floor A");

        assert_eq!(r.id, "M1");
        assert_eq!(r.name, "CNC Machine 1");
        assert_eq!(r.resource_type, "Machine");
        assert_eq!(r.capacity, 1);
        assert!(r.has_skill("milling"));
        assert!(r.has_skill("drilling"));
        assert!(!r.has_skill("welding"));
    }

    #[test]
    fn test_resource_availability_no_calendar() {
        let r = Resource::new("M1", "Machine");
        assert!(r.is_available_at(0));
        assert!(r.is_available_at(1_000_000));
    }
}
