//! Capacity- and calendar-aware booking ledger for a single resource.
//!
//! Backbone of the enforcing scheduler: answers "when can a duration-d
//! interval start on this resource, at or after t?" honoring the
//! resource's availability calendar and unit capacity.
//!
//! # Algorithm
//! Candidate-point search: a feasible earliest start is either `from`,
//! an availability-window start, a blocked-period end, or a booking end.
//! Each candidate is checked with `Calendar::interval_fits` and a
//! concurrent-booking count against capacity. O((W+B+K)²) per query
//! (W windows, B blocks, K bookings) — fine for the greedy scheduler's
//! workloads; verified against brute-force in property tests.
//!
//! # Reference
//! Kolisch & Hartmann (1999), "Heuristic algorithms for solving the
//! resource-constrained project scheduling problem" — serial schedule
//! generation scheme (SGS) resource feasibility check.

use crate::models::{Calendar, Resource, TimeWindow};

/// Booking ledger for one resource.
#[derive(Debug, Clone)]
pub struct ResourceTimeline {
    /// Resource this timeline tracks.
    pub resource_id: String,
    capacity: i32,
    calendar: Option<Calendar>,
    bookings: Vec<TimeWindow>,
}

impl ResourceTimeline {
    /// Creates a timeline from a resource (capacity floor of 1).
    pub fn new(resource: &Resource) -> Self {
        Self {
            resource_id: resource.id.clone(),
            capacity: resource.capacity.max(1),
            calendar: resource.calendar.clone(),
            bookings: Vec::new(),
        }
    }

    /// Whether [start, start+duration) fits: inside working time and
    /// with concurrent bookings below capacity.
    pub fn fits(&self, start_ms: i64, duration_ms: i64) -> bool {
        let end_ms = start_ms + duration_ms;
        let cal_ok = self
            .calendar
            .as_ref()
            .is_none_or(|c| c.interval_fits(start_ms, end_ms));
        cal_ok && self.capacity_ok(start_ms, end_ms)
    }

    /// Earliest start >= from_ms where duration_ms fits, or None if the
    /// calendar can never contain the duration.
    pub fn earliest_fit(&self, from_ms: i64, duration_ms: i64) -> Option<i64> {
        let mut candidates: Vec<i64> = vec![from_ms];
        if let Some(cal) = &self.calendar {
            candidates.extend(cal.time_windows.iter().map(|w| w.start_ms));
            candidates.extend(cal.blocked_periods.iter().map(|b| b.end_ms));
        }
        candidates.extend(self.bookings.iter().map(|b| b.end_ms));
        candidates.retain(|&t| t >= from_ms);
        candidates.sort_unstable();
        candidates.dedup();
        candidates.into_iter().find(|&t| self.fits(t, duration_ms))
    }

    /// Records a booking. Does not itself validate feasibility —
    /// callers query `earliest_fit`/`fits` first.
    pub fn book(&mut self, start_ms: i64, end_ms: i64) {
        self.bookings.push(TimeWindow::new(start_ms, end_ms));
    }

    /// Max concurrent bookings over [start, end) stays below capacity.
    fn capacity_ok(&self, start_ms: i64, end_ms: i64) -> bool {
        let overlapping: Vec<&TimeWindow> = self
            .bookings
            .iter()
            .filter(|b| b.start_ms < end_ms && start_ms < b.end_ms)
            .collect();
        if (overlapping.len() as i32) < self.capacity {
            return true;
        }
        // Concurrency only changes at booking starts: check each.
        overlapping.iter().all(|probe| {
            let p = probe.start_ms.max(start_ms);
            let concurrent = overlapping.iter().filter(|b| b.contains(p)).count() as i32;
            concurrent < self.capacity
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Calendar, Resource};

    #[test]
    fn test_earliest_fit_empty() {
        let tl = ResourceTimeline::new(&Resource::primary("M1"));
        assert_eq!(tl.earliest_fit(0, 1000), Some(0));
        assert_eq!(tl.earliest_fit(500, 1000), Some(500));
    }

    #[test]
    fn test_earliest_fit_after_booking() {
        let mut tl = ResourceTimeline::new(&Resource::primary("M1"));
        tl.book(0, 1000);
        assert_eq!(tl.earliest_fit(0, 500), Some(1000));
        tl.book(1500, 2000);
        // [1000,1500) 틈에 500 들어감
        assert_eq!(tl.earliest_fit(0, 500), Some(1000));
        // 600은 틈에 안 들어가고 2000 이후로
        assert_eq!(tl.earliest_fit(0, 600), Some(2000));
    }

    #[test]
    fn test_capacity_two_parallel() {
        let mut tl = ResourceTimeline::new(&Resource::primary("M1").with_capacity(2));
        tl.book(0, 1000);
        assert_eq!(tl.earliest_fit(0, 1000), Some(0)); // 두 번째 슬롯
        tl.book(0, 1000);
        assert_eq!(tl.earliest_fit(0, 1000), Some(1000)); // 꽉 참
    }

    #[test]
    fn test_calendar_respected() {
        let cal = Calendar::new("shift")
            .with_window(0, 5000)
            .with_window(10_000, 20_000);
        let mut tl = ResourceTimeline::new(&Resource::primary("M1").with_calendar(cal));
        tl.book(0, 3000);
        // 3000ms 작업: [3000,6000)은 창 밖 → 다음 창 10000
        assert_eq!(tl.earliest_fit(0, 3000), Some(10_000));
        // 2000ms 작업은 [3000,5000)에 들어감
        assert_eq!(tl.earliest_fit(0, 2000), Some(3000));
    }

    #[test]
    fn test_calendar_never_fits() {
        let cal = Calendar::new("short").with_window(0, 1000);
        let tl = ResourceTimeline::new(&Resource::primary("M1").with_calendar(cal));
        assert_eq!(tl.earliest_fit(0, 2000), None);
    }
}
