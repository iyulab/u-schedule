//! Resource model.
//!
//! Resources are the entities that perform activities: machines, workers,
//! tools, rooms, vehicles. Each resource has a type, capacity, skills,
//! and an optional availability calendar.
//!
//! # Reference
//! Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 1.2

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::Calendar;

/// A resource that can be assigned to activities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Unique resource identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Resource classification.
    pub resource_type: ResourceType,
    /// Number of units available simultaneously (default: 1).
    pub capacity: i32,
    /// Work rate multiplier (1.0 = normal, <1.0 = slower, >1.0 = faster).
    pub efficiency: f64,
    /// Availability schedule.
    pub calendar: Option<Calendar>,
    /// Skills with proficiency levels.
    pub skills: Vec<Skill>,
    /// Economic cost per hour (optional, for cost optimization).
    pub cost_per_hour: Option<f64>,
    /// Domain-specific metadata.
    pub attributes: HashMap<String, String>,
}

/// Resource type classification.
///
/// Determines scheduling semantics (e.g., consumable resources deplete,
/// human resources have shift constraints).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// Main processing resource (e.g., machine, operating room).
    Primary,
    /// Support resource (e.g., tool, fixture, jig).
    Secondary,
    /// Human resource (e.g., operator, doctor, driver).
    Human,
    /// Depleting resource (e.g., raw material, energy budget).
    Consumable,
    /// Domain-specific type.
    Custom(String),
}

/// A skill with proficiency level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Skill name (e.g., "welding", "milling", "anesthesia").
    pub name: String,
    /// Proficiency level (0.0 to 1.0, where 1.0 = expert).
    pub level: f64,
}

impl Resource {
    /// Creates a new primary resource.
    pub fn new(id: impl Into<String>, resource_type: ResourceType) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            resource_type,
            capacity: 1,
            efficiency: 1.0,
            calendar: None,
            skills: Vec::new(),
            cost_per_hour: None,
            attributes: HashMap::new(),
        }
    }

    /// Creates a primary resource.
    pub fn primary(id: impl Into<String>) -> Self {
        Self::new(id, ResourceType::Primary)
    }

    /// Creates a human resource.
    pub fn human(id: impl Into<String>) -> Self {
        Self::new(id, ResourceType::Human)
    }

    /// Creates a secondary resource.
    pub fn secondary(id: impl Into<String>) -> Self {
        Self::new(id, ResourceType::Secondary)
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

    /// Sets the efficiency multiplier.
    pub fn with_efficiency(mut self, efficiency: f64) -> Self {
        self.efficiency = efficiency;
        self
    }

    /// Sets the availability calendar.
    pub fn with_calendar(mut self, calendar: Calendar) -> Self {
        self.calendar = Some(calendar);
        self
    }

    /// Adds a skill.
    pub fn with_skill(mut self, name: impl Into<String>, level: f64) -> Self {
        self.skills.push(Skill {
            name: name.into(),
            level: level.clamp(0.0, 1.0),
        });
        self
    }

    /// Sets the hourly cost.
    pub fn with_cost(mut self, cost_per_hour: f64) -> Self {
        self.cost_per_hour = Some(cost_per_hour);
        self
    }

    /// Adds a domain-specific attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Whether this resource has a given skill.
    pub fn has_skill(&self, name: &str) -> bool {
        self.skills.iter().any(|s| s.name == name)
    }

    /// Returns the proficiency level for a skill (0.0 if not found).
    pub fn skill_level(&self, name: &str) -> f64 {
        self.skills
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.level)
            .unwrap_or(0.0)
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

impl Skill {
    /// Creates a new skill.
    pub fn new(name: impl Into<String>, level: f64) -> Self {
        Self {
            name: name.into(),
            level: level.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_builder() {
        let r = Resource::primary("M1")
            .with_name("CNC Machine 1")
            .with_capacity(1)
            .with_efficiency(1.2)
            .with_skill("milling", 0.9)
            .with_skill("drilling", 0.7)
            .with_cost(50.0)
            .with_attribute("location", "Shop Floor A");

        assert_eq!(r.id, "M1");
        assert_eq!(r.name, "CNC Machine 1");
        assert_eq!(r.resource_type, ResourceType::Primary);
        assert_eq!(r.capacity, 1);
        assert!((r.efficiency - 1.2).abs() < 1e-10);
        assert!(r.has_skill("milling"));
        assert!(!r.has_skill("welding"));
        assert!((r.skill_level("milling") - 0.9).abs() < 1e-10);
        assert!((r.skill_level("unknown") - 0.0).abs() < 1e-10);
        assert_eq!(r.cost_per_hour, Some(50.0));
    }

    #[test]
    fn test_resource_types() {
        let m = Resource::primary("M1");
        assert_eq!(m.resource_type, ResourceType::Primary);

        let w = Resource::human("W1");
        assert_eq!(w.resource_type, ResourceType::Human);

        let t = Resource::secondary("T1");
        assert_eq!(t.resource_type, ResourceType::Secondary);
    }

    #[test]
    fn test_resource_availability_no_calendar() {
        let r = Resource::primary("M1");
        assert!(r.is_available_at(0));
        assert!(r.is_available_at(1_000_000));
    }

    #[test]
    fn test_skill_clamping() {
        let r = Resource::primary("M1")
            .with_skill("over", 1.5)
            .with_skill("under", -0.5);

        assert!((r.skill_level("over") - 1.0).abs() < 1e-10);
        assert!((r.skill_level("under") - 0.0).abs() < 1e-10);
    }
}
