//! Scheduling constraints and transition matrices.
//!
//! Defines the constraints that a valid schedule must satisfy:
//! precedence, capacity, time windows, no-overlap, and
//! sequence-dependent setup times.
//!
//! # Reference
//! Brucker (2007), "Scheduling Algorithms", Ch. 2

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A scheduling constraint.
///
/// Constraints define the rules that a valid schedule must satisfy.
/// The scheduler's job is to find an assignment that satisfies all
/// hard constraints while optimizing objectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Activity `after` cannot start until `before` finishes + `min_delay_ms`.
    ///
    /// # Reference
    /// Pinedo (2016), "Scheduling", precedence constraints (Ch. 2.1)
    Precedence {
        before: String,
        after: String,
        min_delay_ms: i64,
    },

    /// At most `max_capacity` activities may use `resource_id` simultaneously.
    Capacity {
        resource_id: String,
        max_capacity: i32,
    },

    /// Activity must be scheduled within [start_ms, end_ms).
    TimeWindow {
        activity_id: String,
        start_ms: i64,
        end_ms: i64,
    },

    /// Listed activities cannot overlap on the given resource.
    /// This is a mutual exclusion constraint (disjunctive resource).
    NoOverlap {
        resource_id: String,
        activity_ids: Vec<String>,
    },

    /// Sequence-dependent setup time between activity categories.
    /// When transitioning from `from_category` to `to_category`,
    /// `cost_ms` additional time is incurred.
    TransitionCost {
        from_category: String,
        to_category: String,
        cost_ms: i64,
    },

    /// Listed activities must start at the same time.
    Synchronize { activity_ids: Vec<String> },
}

impl Constraint {
    /// Creates a zero-delay precedence constraint.
    pub fn precedence(before: impl Into<String>, after: impl Into<String>) -> Self {
        Self::Precedence {
            before: before.into(),
            after: after.into(),
            min_delay_ms: 0,
        }
    }

    /// Creates a precedence constraint with a minimum delay.
    pub fn precedence_with_delay(
        before: impl Into<String>,
        after: impl Into<String>,
        delay_ms: i64,
    ) -> Self {
        Self::Precedence {
            before: before.into(),
            after: after.into(),
            min_delay_ms: delay_ms,
        }
    }

    /// Creates a capacity constraint.
    pub fn capacity(resource_id: impl Into<String>, max: i32) -> Self {
        Self::Capacity {
            resource_id: resource_id.into(),
            max_capacity: max,
        }
    }

    /// Creates a time window constraint.
    pub fn time_window(activity_id: impl Into<String>, start_ms: i64, end_ms: i64) -> Self {
        Self::TimeWindow {
            activity_id: activity_id.into(),
            start_ms,
            end_ms,
        }
    }

    /// Creates a no-overlap (disjunctive) constraint.
    pub fn no_overlap(resource_id: impl Into<String>, activity_ids: Vec<String>) -> Self {
        Self::NoOverlap {
            resource_id: resource_id.into(),
            activity_ids,
        }
    }

    /// Creates a synchronization constraint.
    pub fn synchronize(activity_ids: Vec<String>) -> Self {
        Self::Synchronize { activity_ids }
    }
}

/// Sequence-dependent setup time matrix.
///
/// Maps (from_category, to_category) → setup time in ms.
/// Used when the setup time on a resource depends on what was
/// processed previously (e.g., machine changeover, color change).
///
/// # Reference
/// Allahverdi et al. (2008), "A survey of scheduling problems with
/// setup times or costs"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionMatrix {
    /// Matrix identifier.
    pub name: String,
    /// Resource this matrix applies to.
    pub resource_id: String,
    /// Transition times: (from_category, to_category) → milliseconds.
    transitions: HashMap<(String, String), i64>,
    /// Default setup time when no explicit transition is defined.
    pub default_ms: i64,
}

impl TransitionMatrix {
    /// Creates a new transition matrix for a resource.
    pub fn new(name: impl Into<String>, resource_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            resource_id: resource_id.into(),
            transitions: HashMap::new(),
            default_ms: 0,
        }
    }

    /// Sets the default transition time.
    pub fn with_default(mut self, default_ms: i64) -> Self {
        self.default_ms = default_ms;
        self
    }

    /// Defines a transition time between two categories.
    pub fn set_transition(&mut self, from: impl Into<String>, to: impl Into<String>, time_ms: i64) {
        self.transitions.insert((from.into(), to.into()), time_ms);
    }

    /// Gets the transition time between two categories.
    ///
    /// Returns the explicit time if defined, otherwise the default.
    /// Same-category transitions return 0 unless explicitly set.
    pub fn get_transition(&self, from: &str, to: &str) -> i64 {
        if from == to {
            return *self
                .transitions
                .get(&(from.to_string(), to.to_string()))
                .unwrap_or(&0);
        }
        *self
            .transitions
            .get(&(from.to_string(), to.to_string()))
            .unwrap_or(&self.default_ms)
    }

    /// Number of explicitly defined transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}

/// A collection of transition matrices indexed by resource ID.
///
/// Provides unified lookup for sequence-dependent setup times
/// across multiple resources.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransitionMatrixCollection {
    matrices: HashMap<String, TransitionMatrix>,
}

impl TransitionMatrixCollection {
    /// Creates an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a transition matrix for a resource.
    pub fn add(&mut self, matrix: TransitionMatrix) {
        self.matrices.insert(matrix.resource_id.clone(), matrix);
    }

    /// Builder: adds a matrix and returns self.
    pub fn with_matrix(mut self, matrix: TransitionMatrix) -> Self {
        self.add(matrix);
        self
    }

    /// Gets the transition time for a resource between two categories.
    ///
    /// Returns 0 if no matrix exists for the resource.
    pub fn get_transition_time(&self, resource_id: &str, from: &str, to: &str) -> i64 {
        self.matrices
            .get(resource_id)
            .map(|m| m.get_transition(from, to))
            .unwrap_or(0)
    }

    /// Number of matrices in the collection.
    pub fn len(&self) -> usize {
        self.matrices.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precedence_constraint() {
        let c = Constraint::precedence("O1", "O2");
        match c {
            Constraint::Precedence {
                before,
                after,
                min_delay_ms,
            } => {
                assert_eq!(before, "O1");
                assert_eq!(after, "O2");
                assert_eq!(min_delay_ms, 0);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_precedence_with_delay() {
        let c = Constraint::precedence_with_delay("O1", "O2", 5000);
        match c {
            Constraint::Precedence { min_delay_ms, .. } => {
                assert_eq!(min_delay_ms, 5000);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_capacity_constraint() {
        let c = Constraint::capacity("M1", 2);
        match c {
            Constraint::Capacity {
                resource_id,
                max_capacity,
            } => {
                assert_eq!(resource_id, "M1");
                assert_eq!(max_capacity, 2);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_transition_matrix() {
        let mut tm = TransitionMatrix::new("changeover", "M1").with_default(500);

        tm.set_transition("TypeA", "TypeB", 1000);
        tm.set_transition("TypeB", "TypeA", 800);
        tm.set_transition("TypeA", "TypeA", 100); // Same-type changeover

        assert_eq!(tm.get_transition("TypeA", "TypeB"), 1000);
        assert_eq!(tm.get_transition("TypeB", "TypeA"), 800);
        assert_eq!(tm.get_transition("TypeA", "TypeA"), 100); // Explicitly set
        assert_eq!(tm.get_transition("TypeB", "TypeB"), 0); // Same-type default
        assert_eq!(tm.get_transition("TypeC", "TypeD"), 500); // Falls to default
        assert_eq!(tm.transition_count(), 3);
    }

    #[test]
    fn test_transition_matrix_same_category_default() {
        let tm = TransitionMatrix::new("tm", "M1").with_default(200);
        // Same category → 0 (not default) unless explicitly set
        assert_eq!(tm.get_transition("X", "X"), 0);
        // Different category → default
        assert_eq!(tm.get_transition("X", "Y"), 200);
    }

    #[test]
    fn test_no_overlap_constraint() {
        let c = Constraint::no_overlap("M1", vec!["O1".into(), "O2".into(), "O3".into()]);
        match c {
            Constraint::NoOverlap {
                resource_id,
                activity_ids,
            } => {
                assert_eq!(resource_id, "M1");
                assert_eq!(activity_ids.len(), 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_synchronize_constraint() {
        let c = Constraint::synchronize(vec!["O1".into(), "O2".into()]);
        match c {
            Constraint::Synchronize { activity_ids } => {
                assert_eq!(activity_ids.len(), 2);
            }
            _ => panic!("wrong variant"),
        }
    }
}
