//! Capacity- and calendar-aware booking ledger for a single resource.
//!
//! Backbone of the enforcing scheduler: answers "when can a duration-d
//! interval start on this resource, at or after t?" honoring the
//! resource's availability calendar and unit capacity.
//!
//! # Algorithm
//! Candidate-point search: a feasible earliest start is either `from`,
//! an availability-window start, a blocked-period end, or a booking end.
//! Each candidate is checked with `Calendar::interval_fits` and the units
//! booked at once against capacity. O((W+B+K)²) per query
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
    /// Each booking and the units it takes.
    bookings: Vec<(TimeWindow, i32)>,
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

    /// Whether `units` more fit over [start, start+duration): inside working
    /// time, with the units booked at once staying within capacity.
    pub fn fits(&self, start_ms: i64, duration_ms: i64, units: i32) -> bool {
        let end_ms = start_ms + duration_ms;
        let cal_ok = self
            .calendar
            .as_ref()
            .is_none_or(|c| c.interval_fits(start_ms, end_ms));
        cal_ok && self.capacity_ok(start_ms, end_ms, units)
    }

    /// Earliest start >= from_ms where duration_ms of `units` fits, or None
    /// if the calendar can never contain the duration or `units` exceeds the
    /// capacity.
    pub fn earliest_fit(&self, from_ms: i64, duration_ms: i64, units: i32) -> Option<i64> {
        if units > self.capacity {
            return None; // never fits — no need to search
        }
        self.start_candidates(from_ms)
            .into_iter()
            .find(|&t| self.fits(t, duration_ms, units))
    }

    /// The earliest `[start, end)` from `from_ms` on in which `work_ms` of work
    /// fits when it pauses outside the calendar and the resource stays held
    /// across the pauses ([`Activity::resumable`](crate::models::Activity::resumable)),
    /// or `None` if the calendar never offers that much work.
    pub fn earliest_resumable(&self, from_ms: i64, work_ms: i64, units: i32) -> Option<(i64, i64)> {
        if units > self.capacity {
            return None;
        }
        let calendars: Vec<&Calendar> = self.calendar.iter().collect();
        self.start_candidates(from_ms).into_iter().find_map(|t| {
            let end = crate::models::joint_work_end(&calendars, t, work_ms)?;
            self.capacity_ok(t, end, units).then_some((t, end))
        })
    }

    /// The instants at or after `from_ms` at which an interval can first start
    /// here: `from_ms`, a window start, a blocked-period end, a booking end —
    /// in increasing order.
    pub(crate) fn start_candidates(&self, from_ms: i64) -> Vec<i64> {
        let mut candidates: Vec<i64> = vec![from_ms];
        if let Some(cal) = &self.calendar {
            candidates.extend(cal.time_windows.iter().map(|w| w.start_ms));
            candidates.extend(cal.blocked_periods.iter().map(|b| b.end_ms));
        }
        candidates.extend(self.bookings.iter().map(|(b, _)| b.end_ms));
        candidates.retain(|&t| t >= from_ms);
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// The resource's calendar, if it has one.
    pub fn calendar(&self) -> Option<&Calendar> {
        self.calendar.as_ref()
    }

    /// Whether `units` more fit over `[start_ms, end_ms)` (the units booked
    /// at once stay within capacity), working time or not.
    pub fn is_free(&self, start_ms: i64, end_ms: i64, units: i32) -> bool {
        self.capacity_ok(start_ms, end_ms, units)
    }

    /// Records a booking of `units`. Does not itself validate feasibility —
    /// callers query `earliest_fit`/`fits` first.
    pub fn book(&mut self, start_ms: i64, end_ms: i64, units: i32) {
        self.bookings
            .push((TimeWindow::new(start_ms, end_ms), units));
    }

    /// Booked unit-time: each booking's length times its units (a capacity-2
    /// resource with two concurrent one-unit bookings counts both).
    pub fn booked_ms(&self) -> i64 {
        self.bookings
            .iter()
            .map(|(b, units)| (b.end_ms - b.start_ms) * i64::from(*units))
            .sum()
    }

    /// End of the latest booking, if any.
    pub fn latest_end(&self) -> Option<i64> {
        self.bookings.iter().map(|(b, _)| b.end_ms).max()
    }

    /// Units that can be booked at once.
    pub fn capacity(&self) -> i32 {
        self.capacity
    }

    /// The units booked at once over [start, end), plus `units`, stay within
    /// capacity.
    fn capacity_ok(&self, start_ms: i64, end_ms: i64, units: i32) -> bool {
        let overlapping: Vec<&(TimeWindow, i32)> = self
            .bookings
            .iter()
            .filter(|(b, _)| b.start_ms < end_ms && start_ms < b.end_ms)
            .collect();
        let room = self.capacity - units;
        if room < 0 {
            return false;
        }
        if overlapping.iter().map(|(_, u)| u).sum::<i32>() <= room {
            return true;
        }
        // The load only rises at booking starts: check each.
        overlapping.iter().all(|(probe, _)| {
            let p = probe.start_ms.max(start_ms);
            let load: i32 = overlapping
                .iter()
                .filter(|(b, _)| b.contains(p))
                .map(|(_, u)| u)
                .sum();
            load <= room
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Calendar, Resource};

    #[test]
    fn test_earliest_fit_empty() {
        let tl = ResourceTimeline::new(&Resource::new("M1", "Machine"));
        assert_eq!(tl.earliest_fit(0, 1000, 1), Some(0));
        assert_eq!(tl.earliest_fit(500, 1000, 1), Some(500));
    }

    #[test]
    fn test_earliest_fit_after_booking() {
        let mut tl = ResourceTimeline::new(&Resource::new("M1", "Machine"));
        tl.book(0, 1000, 1);
        assert_eq!(tl.earliest_fit(0, 500, 1), Some(1000));
        tl.book(1500, 2000, 1);
        // [1000,1500) 틈에 500 들어감
        assert_eq!(tl.earliest_fit(0, 500, 1), Some(1000));
        // 600은 틈에 안 들어가고 2000 이후로
        assert_eq!(tl.earliest_fit(0, 600, 1), Some(2000));
    }

    #[test]
    fn test_capacity_two_parallel() {
        let mut tl = ResourceTimeline::new(&Resource::new("M1", "Machine").with_capacity(2));
        tl.book(0, 1000, 1);
        assert_eq!(tl.earliest_fit(0, 1000, 1), Some(0)); // 두 번째 슬롯
        tl.book(0, 1000, 1);
        assert_eq!(tl.earliest_fit(0, 1000, 1), Some(1000)); // 꽉 참
    }

    #[test]
    fn usages_share_the_capacity() {
        let mut tl = ResourceTimeline::new(&Resource::new("W1", "Operator").with_capacity(10));
        tl.book(0, 1000, 5);
        assert_eq!(tl.earliest_fit(0, 1000, 5), Some(0));
        assert_eq!(tl.earliest_fit(0, 1000, 6), Some(1000));
        tl.book(0, 1000, 5);
        assert_eq!(tl.earliest_fit(0, 500, 1), Some(1000));
        assert_eq!(tl.earliest_fit(0, 500, 11), None);
        assert_eq!(tl.booked_ms(), 10_000);
    }

    #[test]
    fn test_calendar_respected() {
        let cal = Calendar::new("shift")
            .with_window(0, 5000)
            .with_window(10_000, 20_000);
        let mut tl = ResourceTimeline::new(&Resource::new("M1", "Machine").with_calendar(cal));
        tl.book(0, 3000, 1);
        // 3000ms 작업: [3000,6000)은 창 밖 → 다음 창 10000
        assert_eq!(tl.earliest_fit(0, 3000, 1), Some(10_000));
        // 2000ms 작업은 [3000,5000)에 들어감
        assert_eq!(tl.earliest_fit(0, 2000, 1), Some(3000));
    }

    #[test]
    fn test_calendar_never_fits() {
        let cal = Calendar::new("short").with_window(0, 1000);
        let tl = ResourceTimeline::new(&Resource::new("M1", "Machine").with_calendar(cal));
        assert_eq!(tl.earliest_fit(0, 2000, 1), None);
    }
}
