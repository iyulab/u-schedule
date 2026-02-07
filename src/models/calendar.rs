//! Calendar and time window models.
//!
//! Defines resource availability patterns: working hours, shifts,
//! and blocked periods (maintenance, holidays).
//!
//! # Time Model
//! All times are in milliseconds relative to a scheduling epoch.
//! The consumer defines what epoch means.
//!
//! # Precedence
//! Blocked periods override time windows. A timestamp is available iff:
//! - It falls within at least one `time_windows` entry, AND
//! - It does NOT fall within any `blocked_periods` entry.

use serde::{Deserialize, Serialize};

/// A time interval [start, end).
///
/// Half-open interval: includes start, excludes end.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeWindow {
    /// Interval start (ms, inclusive).
    pub start_ms: i64,
    /// Interval end (ms, exclusive).
    pub end_ms: i64,
}

impl TimeWindow {
    /// Creates a new time window.
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration of this window (ms).
    #[inline]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Whether a timestamp falls within this window.
    #[inline]
    pub fn contains(&self, time_ms: i64) -> bool {
        time_ms >= self.start_ms && time_ms < self.end_ms
    }

    /// Whether two windows overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

/// Resource availability calendar.
///
/// Combines positive availability windows with negative blocked periods.
/// If no time_windows are defined, the resource is always available
/// (subject to blocked periods).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calendar {
    /// Calendar identifier.
    pub id: String,
    /// Periods when the resource is available.
    /// Empty = always available.
    pub time_windows: Vec<TimeWindow>,
    /// Periods when the resource is unavailable (overrides time_windows).
    pub blocked_periods: Vec<TimeWindow>,
}

impl Calendar {
    /// Creates an empty calendar (no constraints = always available).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            time_windows: Vec::new(),
            blocked_periods: Vec::new(),
        }
    }

    /// Creates a calendar that is always available.
    pub fn always_available(id: impl Into<String>) -> Self {
        Self::new(id)
    }

    /// Adds an availability window.
    pub fn with_window(mut self, start_ms: i64, end_ms: i64) -> Self {
        self.time_windows.push(TimeWindow::new(start_ms, end_ms));
        self
    }

    /// Adds a blocked period.
    pub fn with_blocked(mut self, start_ms: i64, end_ms: i64) -> Self {
        self.blocked_periods
            .push(TimeWindow::new(start_ms, end_ms));
        self
    }

    /// Whether a timestamp is within working time.
    ///
    /// Returns `true` if the timestamp is in an availability window
    /// (or no windows are defined) AND not in any blocked period.
    pub fn is_working_time(&self, time_ms: i64) -> bool {
        // Check blocked periods first (they override)
        if self.blocked_periods.iter().any(|w| w.contains(time_ms)) {
            return false;
        }

        // If no windows defined, always available
        if self.time_windows.is_empty() {
            return true;
        }

        // Must be in at least one window
        self.time_windows.iter().any(|w| w.contains(time_ms))
    }

    /// Finds the next available time at or after `from_ms`.
    ///
    /// Returns `from_ms` if already available, or the start of the
    /// next availability window that isn't blocked.
    ///
    /// Returns `None` if no future availability exists.
    pub fn next_available_time(&self, from_ms: i64) -> Option<i64> {
        if self.is_working_time(from_ms) {
            return Some(from_ms);
        }

        // If no windows, we must be in a blocked period
        if self.time_windows.is_empty() {
            // Find end of current blocked period
            for bp in &self.blocked_periods {
                if bp.contains(from_ms) {
                    let candidate = bp.end_ms;
                    if self.is_working_time(candidate) {
                        return Some(candidate);
                    }
                }
            }
            return None;
        }

        // Search windows sorted by start time
        let mut candidates: Vec<i64> = self
            .time_windows
            .iter()
            .filter(|w| w.end_ms > from_ms)
            .map(|w| w.start_ms.max(from_ms))
            .collect();
        candidates.sort();

        for candidate in candidates {
            if self.is_working_time(candidate) {
                return Some(candidate);
            }
            // If candidate is blocked, try end of the blocking period
            for bp in &self.blocked_periods {
                if bp.contains(candidate)
                    && bp.end_ms < i64::MAX
                    && self.is_working_time(bp.end_ms)
                {
                    return Some(bp.end_ms);
                }
            }
        }

        None
    }

    /// Computes total available time within a range [start, end).
    pub fn available_time_in_range(&self, start_ms: i64, end_ms: i64) -> i64 {
        if end_ms <= start_ms {
            return 0;
        }

        let range = TimeWindow::new(start_ms, end_ms);

        // If no windows, total = range - blocked
        if self.time_windows.is_empty() {
            let blocked: i64 = self
                .blocked_periods
                .iter()
                .filter_map(|bp| overlap_duration(&range, bp))
                .sum();
            return range.duration_ms() - blocked;
        }

        // Sum window intersections with range, minus blocked intersections
        let mut available: i64 = 0;
        for w in &self.time_windows {
            if let Some(dur) = overlap_duration(&range, w) {
                available += dur;
            }
        }

        // Subtract blocked intersections
        let blocked: i64 = self
            .blocked_periods
            .iter()
            .filter_map(|bp| overlap_duration(&range, bp))
            .sum();

        (available - blocked).max(0)
    }
}

/// Computes overlap duration between two time windows.
fn overlap_duration(a: &TimeWindow, b: &TimeWindow) -> Option<i64> {
    let start = a.start_ms.max(b.start_ms);
    let end = a.end_ms.min(b.end_ms);
    if end > start {
        Some(end - start)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_window() {
        let w = TimeWindow::new(100, 200);
        assert_eq!(w.duration_ms(), 100);
        assert!(w.contains(100));
        assert!(w.contains(199));
        assert!(!w.contains(200)); // exclusive end
        assert!(!w.contains(50));
    }

    #[test]
    fn test_time_window_overlap() {
        let a = TimeWindow::new(0, 100);
        let b = TimeWindow::new(50, 150);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));

        let c = TimeWindow::new(100, 200); // touching but not overlapping
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_calendar_always_available() {
        let cal = Calendar::always_available("cal1");
        assert!(cal.is_working_time(0));
        assert!(cal.is_working_time(1_000_000));
    }

    #[test]
    fn test_calendar_with_windows() {
        let cal = Calendar::new("shifts")
            .with_window(0, 8_000)       // 0-8s: day shift
            .with_window(16_000, 24_000); // 16-24s: night shift

        assert!(cal.is_working_time(4_000));  // During day shift
        assert!(!cal.is_working_time(10_000)); // Between shifts
        assert!(cal.is_working_time(20_000)); // During night shift
    }

    #[test]
    fn test_calendar_blocked_overrides() {
        let cal = Calendar::new("cal")
            .with_window(0, 100_000)
            .with_blocked(50_000, 60_000); // Maintenance window

        assert!(cal.is_working_time(40_000));  // Before maintenance
        assert!(!cal.is_working_time(55_000)); // During maintenance
        assert!(cal.is_working_time(70_000));  // After maintenance
    }

    #[test]
    fn test_next_available_time() {
        let cal = Calendar::new("shifts")
            .with_window(0, 8_000)
            .with_window(16_000, 24_000);

        assert_eq!(cal.next_available_time(4_000), Some(4_000));   // Already available
        assert_eq!(cal.next_available_time(10_000), Some(16_000)); // Wait for next shift
    }

    #[test]
    fn test_next_available_blocked() {
        let cal = Calendar::always_available("cal")
            .with_blocked(50_000, 60_000);

        assert_eq!(cal.next_available_time(40_000), Some(40_000));
        assert_eq!(cal.next_available_time(55_000), Some(60_000));
    }

    #[test]
    fn test_available_time_in_range() {
        let cal = Calendar::new("cal")
            .with_window(0, 100_000)
            .with_blocked(40_000, 60_000); // 20s blocked

        let avail = cal.available_time_in_range(0, 100_000);
        assert_eq!(avail, 80_000); // 100k - 20k blocked

        let avail2 = cal.available_time_in_range(50_000, 70_000);
        assert_eq!(avail2, 10_000); // 60k-70k (50k-60k blocked)
    }

    #[test]
    fn test_available_time_no_windows() {
        let cal = Calendar::always_available("cal")
            .with_blocked(20_000, 30_000);

        let avail = cal.available_time_in_range(0, 50_000);
        assert_eq!(avail, 40_000); // 50k - 10k blocked
    }
}
