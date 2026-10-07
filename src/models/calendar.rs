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
use u_numflow::collections::IntervalSet;

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
        self.blocked_periods.push(TimeWindow::new(start_ms, end_ms));
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

    /// Whether the whole interval [start_ms, end_ms) is working time.
    ///
    /// The interval must fit inside a single availability window (no
    /// cross-window spans — window boundaries are treated as hard breaks)
    /// and must not overlap any blocked period. With no windows defined,
    /// only blocked periods constrain the interval.
    pub fn interval_fits(&self, start_ms: i64, end_ms: i64) -> bool {
        let in_window = self.time_windows.is_empty()
            || self
                .time_windows
                .iter()
                .any(|w| start_ms >= w.start_ms && end_ms <= w.end_ms);
        let hits_blocked = self
            .blocked_periods
            .iter()
            .any(|b| start_ms < b.end_ms && b.start_ms < end_ms);
        in_window && !hits_blocked
    }

    /// Finds the next available time at or after `from_ms`.
    ///
    /// Returns `from_ms` if already available, otherwise the first instant
    /// after it that lies in an availability window (or anywhere, with no
    /// windows) and outside every blocked period — overlapping or adjacent
    /// blocked periods are crossed together.
    ///
    /// Returns `None` if no future availability exists.
    pub fn next_available_time(&self, from_ms: i64) -> Option<i64> {
        self.available_set(from_ms, i64::MAX)
            .iter()
            .next()
            .map(|(start, _)| start)
    }

    /// Computes total available time within a range [start, end).
    ///
    /// Each instant counts once: overlapping windows are merged, overlapping
    /// blocked periods are merged, and blocked time outside every window is
    /// not subtracted (it was never available). Saturates at `i64::MAX`, which
    /// only a range longer than `i64::MAX` ms can reach.
    pub fn available_time_in_range(&self, start_ms: i64, end_ms: i64) -> i64 {
        i64::try_from(self.available_set(start_ms, end_ms).measure()).unwrap_or(i64::MAX)
    }

    /// The available instants in `[start_ms, end_ms)`:
    /// `(∪ windows, or everything) ∩ [start_ms, end_ms) − ∪ blocked`.
    ///
    /// A window or blocked period with `start_ms > end_ms` contains no instant
    /// (as [`TimeWindow::contains`] already says) and is skipped here;
    /// [`Problem::new`](crate::Problem::new) refuses such calendars.
    fn available_set(&self, start_ms: i64, end_ms: i64) -> IntervalSet<i64> {
        if end_ms <= start_ms {
            return IntervalSet::new();
        }
        let range = (start_ms, end_ms);
        let base = if self.time_windows.is_empty() {
            interval_set(std::iter::once(range))
        } else {
            interval_set(self.time_windows.iter().map(|w| (w.start_ms, w.end_ms)))
                .clip(start_ms, end_ms)
                .expect("range is ordered: checked above")
        };
        base.difference(&interval_set(
            self.blocked_periods.iter().map(|b| (b.start_ms, b.end_ms)),
        ))
    }
}

/// The union of the well-ordered `(start, end)` pairs; reversed pairs are empty.
fn interval_set(pairs: impl Iterator<Item = (i64, i64)>) -> IntervalSet<i64> {
    IntervalSet::from_intervals(pairs.filter(|(s, e)| s <= e))
        .expect("integer bounds are admissible and reversed pairs were filtered out")
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
    fn test_interval_fits() {
        let cal = Calendar::new("shift")
            .with_window(0, 5000)
            .with_window(10_000, 20_000);
        assert!(cal.interval_fits(0, 3000));
        assert!(cal.interval_fits(2000, 5000)); // 창 끝에 딱 맞음
        assert!(!cal.interval_fits(3000, 6000)); // 창 밖으로 삐져나감
        assert!(!cal.interval_fits(4000, 11_000)); // 두 창에 걸침 — 비분할 의미론
        assert!(cal.interval_fits(10_000, 13_000));

        let blocked = Calendar::new("mnt").with_blocked(5000, 6000); // 창 없음 = 상시 가용
        assert!(blocked.interval_fits(0, 5000));
        assert!(!blocked.interval_fits(4000, 5500));
        assert!(blocked.interval_fits(6000, 9000));
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
            .with_window(0, 8_000) // 0-8s: day shift
            .with_window(16_000, 24_000); // 16-24s: night shift

        assert!(cal.is_working_time(4_000)); // During day shift
        assert!(!cal.is_working_time(10_000)); // Between shifts
        assert!(cal.is_working_time(20_000)); // During night shift
    }

    #[test]
    fn test_calendar_blocked_overrides() {
        let cal = Calendar::new("cal")
            .with_window(0, 100_000)
            .with_blocked(50_000, 60_000); // Maintenance window

        assert!(cal.is_working_time(40_000)); // Before maintenance
        assert!(!cal.is_working_time(55_000)); // During maintenance
        assert!(cal.is_working_time(70_000)); // After maintenance
    }

    #[test]
    fn test_next_available_time() {
        let cal = Calendar::new("shifts")
            .with_window(0, 8_000)
            .with_window(16_000, 24_000);

        assert_eq!(cal.next_available_time(4_000), Some(4_000)); // Already available
        assert_eq!(cal.next_available_time(10_000), Some(16_000)); // Wait for next shift
    }

    #[test]
    fn test_next_available_blocked() {
        let cal = Calendar::always_available("cal").with_blocked(50_000, 60_000);

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
        let cal = Calendar::always_available("cal").with_blocked(20_000, 30_000);

        let avail = cal.available_time_in_range(0, 50_000);
        assert_eq!(avail, 40_000); // 50k - 10k blocked
    }

    /// Overlapping blocked periods (planned + unplanned stop sharing 14 h)
    /// used to be subtracted twice.
    #[test]
    fn overlapping_blocked_periods_count_once() {
        let h = 3_600_000;
        let cal = Calendar::always_available("asset")
            .with_blocked(0, 24 * h)
            .with_blocked(10 * h, 30 * h);
        assert_eq!(cal.available_time_in_range(0, 168 * h), 138 * h);
    }

    #[test]
    fn overlapping_windows_count_once() {
        let cal = Calendar::new("cal")
            .with_window(0, 100)
            .with_window(50, 150);
        assert_eq!(cal.available_time_in_range(0, 200), 150);
    }

    /// Blocked time outside every window was never available, so it must not
    /// be subtracted from the window time.
    #[test]
    fn blocked_time_outside_windows_is_not_subtracted() {
        let cal = Calendar::new("shift")
            .with_window(0, 8_000)
            .with_blocked(10_000, 12_000);
        assert_eq!(cal.available_time_in_range(0, 20_000), 8_000);
    }

    /// The end of the first blocked period lies inside the second; the next
    /// available instant is the end of the second (it used to be `None`).
    #[test]
    fn next_available_crosses_overlapping_blocked_periods() {
        let free = Calendar::always_available("cal")
            .with_blocked(10, 20)
            .with_blocked(15, 30);
        assert_eq!(free.next_available_time(12), Some(30));

        let shift = Calendar::new("shift")
            .with_window(0, 100)
            .with_blocked(10, 20)
            .with_blocked(15, 30);
        assert_eq!(shift.next_available_time(12), Some(30));
        assert_eq!(shift.next_available_time(100), None);
    }

    #[test]
    fn a_reversed_period_contains_no_time() {
        let cal = Calendar::new("cal").with_window(0, 10).with_window(50, 40);
        assert_eq!(cal.available_time_in_range(0, 100), 10);
        assert!(!cal.is_working_time(45));
    }

    proptest::proptest! {
        /// Available time equals the number of working milliseconds, counted
        /// one by one with `is_working_time`; the next available instant is the
        /// first such millisecond.
        #[test]
        fn availability_agrees_with_pointwise_model(
            windows in proptest::collection::vec((0i64..60, 0i64..20), 0..4),
            blocked in proptest::collection::vec((0i64..60, 0i64..20), 0..4),
            from in 0i64..80,
        ) {
            let mut cal = Calendar::new("p");
            for (s, l) in windows { cal = cal.with_window(s, s + l); }
            for (s, l) in blocked { cal = cal.with_blocked(s, s + l); }
            let working: Vec<i64> = (0..100).filter(|&t| cal.is_working_time(t)).collect();
            proptest::prop_assert_eq!(cal.available_time_in_range(0, 100), working.len() as i64);
            // Every period ends before 80 and `from` < 80, so the model's range
            // [0, 100) holds the answer even when the calendar is unbounded above.
            let expect = working.iter().copied().find(|&t| t >= from);
            proptest::prop_assert_eq!(cal.next_available_time(from), expect);
        }
    }
}
