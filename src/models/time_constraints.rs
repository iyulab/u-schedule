//! Time constraints and duration estimation for scheduling.
//!
//! Provides constraint-oriented time boundaries (hard/soft deadlines,
//! release times), violation tracking, PERT three-point estimation,
//! and probabilistic duration distributions.
//!
//! # Concepts
//!
//! - [`ActivityTimeConstraint`]: Scheduling-level time boundary (different
//!   from calendar [`TimeWindow`](super::TimeWindow) which models availability)
//! - [`TimeWindowViolation`]: Result of checking an activity against its constraint
//! - [`PertEstimate`]: PERT three-point duration estimation (O, M, P)
//! - [`DurationDistribution`]: Probabilistic duration model
//!
//! # References
//!
//! - Malcolm et al. (1959), "Application of a technique for R&D program evaluation" (PERT)
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems"

use serde::{Deserialize, Serialize};

// ================================
// Time Constraint (Hard/Soft)
// ================================

/// Time constraint type.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConstraintType {
    /// Must be satisfied (schedule invalid if violated).
    Hard,
    /// Should be satisfied (penalty if violated).
    #[default]
    Soft,
}

/// Time constraint for an activity.
///
/// Unlike calendar [`TimeWindow`](super::TimeWindow) (which models resource
/// availability), this struct represents scheduling-level boundaries:
/// earliest/latest start and end times, optionally with penalties.
///
/// # Examples
///
/// ```
/// use u_schedule::models::time_constraints::{ActivityTimeConstraint, ConstraintType};
///
/// // Hard deadline: must finish by 5000 ms
/// let c = ActivityTimeConstraint::deadline(5000);
/// assert!(c.check_violation(0, 4000).is_none());
/// assert!(c.check_violation(0, 6000).is_some());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityTimeConstraint {
    /// Earliest allowed start time (ms).
    pub earliest_start_ms: Option<i64>,
    /// Latest allowed start time (ms).
    pub latest_start_ms: Option<i64>,
    /// Earliest allowed end time (ms).
    pub earliest_end_ms: Option<i64>,
    /// Latest allowed end time (ms) — due date.
    pub latest_end_ms: Option<i64>,
    /// Hard or soft constraint.
    pub constraint_type: ConstraintType,
    /// Penalty per millisecond of violation (for soft constraints).
    pub penalty_per_ms: f64,
}

impl ActivityTimeConstraint {
    /// Creates an empty (unconstrained) time constraint.
    pub fn new() -> Self {
        Self {
            earliest_start_ms: None,
            latest_start_ms: None,
            earliest_end_ms: None,
            latest_end_ms: None,
            constraint_type: ConstraintType::Soft,
            penalty_per_ms: 1.0,
        }
    }

    /// Creates a constraint with start/end boundaries.
    pub fn bounded(start_ms: i64, end_ms: i64) -> Self {
        Self {
            earliest_start_ms: Some(start_ms),
            latest_end_ms: Some(end_ms),
            ..Self::new()
        }
    }

    /// Creates a hard deadline (must complete by).
    pub fn deadline(deadline_ms: i64) -> Self {
        Self {
            latest_end_ms: Some(deadline_ms),
            constraint_type: ConstraintType::Hard,
            penalty_per_ms: 0.0,
            ..Self::new()
        }
    }

    /// Creates a release time (cannot start before).
    pub fn release(release_ms: i64) -> Self {
        Self {
            earliest_start_ms: Some(release_ms),
            constraint_type: ConstraintType::Hard,
            penalty_per_ms: 0.0,
            ..Self::new()
        }
    }

    /// Sets as hard constraint.
    pub fn hard(mut self) -> Self {
        self.constraint_type = ConstraintType::Hard;
        self.penalty_per_ms = 0.0;
        self
    }

    /// Sets as soft constraint with penalty.
    pub fn soft(mut self, penalty_per_ms: f64) -> Self {
        self.constraint_type = ConstraintType::Soft;
        self.penalty_per_ms = penalty_per_ms;
        self
    }

    /// Sets earliest start.
    pub fn with_earliest_start(mut self, ms: i64) -> Self {
        self.earliest_start_ms = Some(ms);
        self
    }

    /// Sets latest start.
    pub fn with_latest_start(mut self, ms: i64) -> Self {
        self.latest_start_ms = Some(ms);
        self
    }

    /// Sets latest end (due date).
    pub fn with_due_date(mut self, ms: i64) -> Self {
        self.latest_end_ms = Some(ms);
        self
    }

    /// Checks if a scheduled time violates the constraint.
    ///
    /// Returns `None` if no violation, or a [`TimeWindowViolation`] with details.
    pub fn check_violation(&self, start_ms: i64, end_ms: i64) -> Option<TimeWindowViolation> {
        let mut total_early_ms = 0i64;
        let mut total_late_ms = 0i64;

        if let Some(earliest) = self.earliest_start_ms {
            if start_ms < earliest {
                total_early_ms += earliest - start_ms;
            }
        }
        if let Some(latest) = self.latest_start_ms {
            if start_ms > latest {
                total_late_ms += start_ms - latest;
            }
        }
        if let Some(earliest) = self.earliest_end_ms {
            if end_ms < earliest {
                total_early_ms += earliest - end_ms;
            }
        }
        if let Some(latest) = self.latest_end_ms {
            if end_ms > latest {
                total_late_ms += end_ms - latest;
            }
        }

        if total_early_ms == 0 && total_late_ms == 0 {
            return None;
        }

        Some(TimeWindowViolation {
            early_ms: total_early_ms,
            late_ms: total_late_ms,
            severity: if self.constraint_type == ConstraintType::Hard {
                ViolationSeverity::Critical
            } else {
                ViolationSeverity::Minor
            },
            penalty: (total_early_ms + total_late_ms) as f64 * self.penalty_per_ms,
        })
    }
}

impl Default for ActivityTimeConstraint {
    fn default() -> Self {
        Self::new()
    }
}

// ================================
// Violation Model
// ================================

/// Severity of a constraint violation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Informational — no impact.
    Info,
    /// Minor — small penalty.
    Minor,
    /// Major — significant penalty.
    Major,
    /// Critical — schedule may be invalid.
    Critical,
}

/// Time constraint violation details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindowViolation {
    /// Amount of time too early (ms).
    pub early_ms: i64,
    /// Amount of time too late (ms).
    pub late_ms: i64,
    /// Severity level.
    pub severity: ViolationSeverity,
    /// Calculated penalty value.
    pub penalty: f64,
}

impl TimeWindowViolation {
    /// Total violation time (absolute).
    pub fn total_violation_ms(&self) -> i64 {
        self.early_ms.abs() + self.late_ms.abs()
    }

    /// Whether this is a tardiness (late) violation.
    pub fn is_tardy(&self) -> bool {
        self.late_ms > 0
    }

    /// Whether this is an early start violation.
    pub fn is_early(&self) -> bool {
        self.early_ms > 0
    }
}

/// General constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Type of violation.
    pub violation_type: ConstraintViolationType,
    /// Related entity IDs.
    pub related_ids: Vec<String>,
    /// Severity level.
    pub severity: ViolationSeverity,
    /// Violation message.
    pub message: String,
    /// Penalty value.
    pub penalty: f64,
}

/// Types of constraint violations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConstraintViolationType {
    /// Time constraint violated.
    TimeWindow,
    /// Resource capacity exceeded.
    CapacityExceeded,
    /// Precedence constraint violated.
    PrecedenceViolated,
    /// Resource unavailable.
    ResourceUnavailable,
    /// Skill requirement not met.
    SkillMismatch,
    /// Other custom violation.
    Custom(String),
}

impl ConstraintViolation {
    /// Creates a time constraint violation.
    pub fn time_window(
        activity_id: &str,
        tardiness_ms: i64,
        severity: ViolationSeverity,
        penalty: f64,
    ) -> Self {
        Self {
            violation_type: ConstraintViolationType::TimeWindow,
            related_ids: vec![activity_id.to_string()],
            severity,
            message: format!("Activity {} is {} ms late", activity_id, tardiness_ms),
            penalty,
        }
    }

    /// Creates a capacity violation.
    pub fn capacity_exceeded(resource_id: &str, exceeded_by: i32) -> Self {
        Self {
            violation_type: ConstraintViolationType::CapacityExceeded,
            related_ids: vec![resource_id.to_string()],
            severity: ViolationSeverity::Critical,
            message: format!(
                "Resource {} capacity exceeded by {}",
                resource_id, exceeded_by
            ),
            penalty: exceeded_by as f64 * 1000.0,
        }
    }

    /// Creates a precedence violation.
    pub fn precedence_violated(before_id: &str, after_id: &str, overlap_ms: i64) -> Self {
        Self {
            violation_type: ConstraintViolationType::PrecedenceViolated,
            related_ids: vec![before_id.to_string(), after_id.to_string()],
            severity: ViolationSeverity::Critical,
            message: format!(
                "Activity {} must complete before {} (overlap: {} ms)",
                before_id, after_id, overlap_ms
            ),
            penalty: overlap_ms as f64 * 10.0,
        }
    }
}

// ================================
// PERT 3-Point Estimation
// ================================

/// PERT (Program Evaluation and Review Technique) duration estimation.
///
/// Uses three-point estimation:
/// - Optimistic (O): Best-case scenario
/// - Most Likely (M): Normal conditions
/// - Pessimistic (P): Worst-case scenario
///
/// Mean = (O + 4M + P) / 6, StdDev = (P - O) / 6
///
/// # References
///
/// Malcolm et al. (1959), Clark (1962)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PertEstimate {
    /// Optimistic duration (ms).
    pub optimistic_ms: i64,
    /// Most likely duration (ms).
    pub most_likely_ms: i64,
    /// Pessimistic duration (ms).
    pub pessimistic_ms: i64,
}

impl PertEstimate {
    /// Creates a new PERT estimate.
    pub fn new(optimistic_ms: i64, most_likely_ms: i64, pessimistic_ms: i64) -> Self {
        Self {
            optimistic_ms,
            most_likely_ms,
            pessimistic_ms,
        }
    }

    /// Creates from percentage variance.
    ///
    /// E.g., `from_variance(1000, 0.2)` creates O=800, M=1000, P=1200.
    pub fn from_variance(base_ms: i64, variance_ratio: f64) -> Self {
        let variance = (base_ms as f64 * variance_ratio) as i64;
        Self {
            optimistic_ms: base_ms - variance,
            most_likely_ms: base_ms,
            pessimistic_ms: base_ms + variance,
        }
    }

    /// Creates symmetric estimate.
    pub fn symmetric(most_likely_ms: i64, spread_ms: i64) -> Self {
        Self {
            optimistic_ms: most_likely_ms - spread_ms,
            most_likely_ms,
            pessimistic_ms: most_likely_ms + spread_ms,
        }
    }

    /// PERT mean (expected duration): `(O + 4M + P) / 6`.
    pub fn mean_ms(&self) -> f64 {
        (self.optimistic_ms as f64 + 4.0 * self.most_likely_ms as f64 + self.pessimistic_ms as f64)
            / 6.0
    }

    /// PERT standard deviation: `(P - O) / 6`.
    pub fn std_dev_ms(&self) -> f64 {
        (self.pessimistic_ms - self.optimistic_ms) as f64 / 6.0
    }

    /// Variance.
    pub fn variance_ms(&self) -> f64 {
        let sd = self.std_dev_ms();
        sd * sd
    }

    /// Duration at specified confidence level.
    ///
    /// Uses normal approximation via `u_optim::special::inverse_normal_cdf`.
    pub fn duration_at_confidence(&self, confidence: f64) -> i64 {
        let z = u_optim::special::inverse_normal_cdf(confidence);
        (self.mean_ms() + z * self.std_dev_ms()) as i64
    }

    /// Probability of completing within given duration.
    ///
    /// Uses `u_optim::special::standard_normal_cdf`.
    pub fn probability_of_completion(&self, duration_ms: i64) -> f64 {
        let z = (duration_ms as f64 - self.mean_ms()) / self.std_dev_ms();
        u_optim::special::standard_normal_cdf(z)
    }

    /// 50th percentile (median) duration.
    pub fn p50(&self) -> i64 {
        self.mean_ms() as i64
    }

    /// 85th percentile duration.
    pub fn p85(&self) -> i64 {
        self.duration_at_confidence(0.85)
    }

    /// 95th percentile duration.
    pub fn p95(&self) -> i64 {
        self.duration_at_confidence(0.95)
    }
}

// ================================
// Duration Distribution
// ================================

/// Duration distribution for probabilistic scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DurationDistribution {
    /// Fixed duration (deterministic).
    Fixed(i64),
    /// PERT-based distribution.
    Pert(PertEstimate),
    /// Uniform distribution between min and max.
    Uniform { min_ms: i64, max_ms: i64 },
    /// Triangular distribution.
    Triangular {
        min_ms: i64,
        mode_ms: i64,
        max_ms: i64,
    },
    /// Log-normal (common for task durations).
    LogNormal { mu: f64, sigma: f64 },
}

impl DurationDistribution {
    /// Expected (mean) duration.
    pub fn expected_duration_ms(&self) -> f64 {
        match self {
            Self::Fixed(d) => *d as f64,
            Self::Pert(p) => p.mean_ms(),
            Self::Uniform { min_ms, max_ms } => (*min_ms + *max_ms) as f64 / 2.0,
            Self::Triangular {
                min_ms,
                mode_ms,
                max_ms,
            } => (*min_ms + *mode_ms + *max_ms) as f64 / 3.0,
            Self::LogNormal { mu, sigma } => (mu + sigma.powi(2) / 2.0).exp(),
        }
    }

    /// Duration at confidence level.
    pub fn duration_at_confidence(&self, confidence: f64) -> i64 {
        match self {
            Self::Fixed(d) => *d,
            Self::Pert(p) => p.duration_at_confidence(confidence),
            Self::Uniform { min_ms, max_ms } => {
                let range = max_ms - min_ms;
                min_ms + (range as f64 * confidence) as i64
            }
            Self::Triangular {
                min_ms,
                mode_ms,
                max_ms,
            } => {
                let fc = (*mode_ms - *min_ms) as f64 / (*max_ms - *min_ms) as f64;
                if confidence < fc {
                    *min_ms
                        + ((*max_ms - *min_ms) as f64 * (*mode_ms - *min_ms) as f64 * confidence)
                            .sqrt() as i64
                } else {
                    *max_ms
                        - ((*max_ms - *min_ms) as f64
                            * (*max_ms - *mode_ms) as f64
                            * (1.0 - confidence))
                            .sqrt() as i64
                }
            }
            Self::LogNormal { mu, sigma } => {
                let z = u_optim::special::inverse_normal_cdf(confidence);
                (mu + z * sigma).exp() as i64
            }
        }
    }

    /// Creates from PERT estimates.
    pub fn from_pert(optimistic: i64, most_likely: i64, pessimistic: i64) -> Self {
        Self::Pert(PertEstimate::new(optimistic, most_likely, pessimistic))
    }
}

impl Default for DurationDistribution {
    fn default() -> Self {
        Self::Fixed(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_constraint_basic() {
        let c = ActivityTimeConstraint::bounded(1000, 5000);

        // Within bounds — no violation
        assert!(c.check_violation(1000, 4000).is_none());

        // Starts too early
        let v = c.check_violation(500, 4000).unwrap();
        assert_eq!(v.early_ms, 500);
        assert!(v.is_early());

        // Ends too late
        let v = c.check_violation(2000, 6000).unwrap();
        assert_eq!(v.late_ms, 1000);
        assert!(v.is_tardy());
    }

    #[test]
    fn test_time_constraint_hard_vs_soft() {
        let hard = ActivityTimeConstraint::deadline(5000).hard();
        let soft = ActivityTimeConstraint::deadline(5000).soft(2.0);

        let vh = hard.check_violation(0, 6000).unwrap();
        let vs = soft.check_violation(0, 6000).unwrap();

        assert_eq!(vh.severity, ViolationSeverity::Critical);
        assert_eq!(vs.severity, ViolationSeverity::Minor);
        assert!((vs.penalty - 2000.0).abs() < 0.01); // 1000ms * 2.0
    }

    #[test]
    fn test_pert_calculation() {
        let pert = PertEstimate::new(4000, 6000, 14000);

        // Mean = (4 + 4*6 + 14) / 6 = 42/6 = 7
        assert!((pert.mean_ms() - 7000.0).abs() < 0.01);

        // StdDev = (14 - 4) / 6 = 10/6 ≈ 1.667
        assert!((pert.std_dev_ms() - 1666.67).abs() < 1.0);
    }

    #[test]
    fn test_pert_from_variance() {
        let pert = PertEstimate::from_variance(10000, 0.2);
        assert_eq!(pert.optimistic_ms, 8000);
        assert_eq!(pert.most_likely_ms, 10000);
        assert_eq!(pert.pessimistic_ms, 12000);
    }

    #[test]
    fn test_pert_confidence_levels() {
        let pert = PertEstimate::new(6000, 10000, 14000);

        // P95 > P85 > P50
        assert!(pert.p95() > pert.p85());
        assert!(pert.p85() > pert.p50());
    }

    #[test]
    fn test_duration_distribution_expected() {
        let fixed = DurationDistribution::Fixed(5000);
        assert!((fixed.expected_duration_ms() - 5000.0).abs() < 0.01);

        let uniform = DurationDistribution::Uniform {
            min_ms: 4000,
            max_ms: 6000,
        };
        assert!((uniform.expected_duration_ms() - 5000.0).abs() < 0.01);

        let tri = DurationDistribution::Triangular {
            min_ms: 3000,
            mode_ms: 5000,
            max_ms: 7000,
        };
        assert!((tri.expected_duration_ms() - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_constraint_violation_creation() {
        let tw_v =
            ConstraintViolation::time_window("OP-001", 5000, ViolationSeverity::Minor, 500.0);
        assert_eq!(tw_v.violation_type, ConstraintViolationType::TimeWindow);
        assert!(tw_v.message.contains("OP-001"));

        let cap_v = ConstraintViolation::capacity_exceeded("M-001", 3);
        assert_eq!(
            cap_v.violation_type,
            ConstraintViolationType::CapacityExceeded
        );
        assert_eq!(cap_v.severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_violation_severity_ordering() {
        assert!(ViolationSeverity::Critical > ViolationSeverity::Major);
        assert!(ViolationSeverity::Major > ViolationSeverity::Minor);
        assert!(ViolationSeverity::Minor > ViolationSeverity::Info);
    }
}
