//! When an activity's offer would start, without building the offer.
//!
//! The dispatcher decides among the activities that can start first, and for that it needs
//! only each one's start; the whole placement (the resources taken, the parts, the set-ups) is
//! needed for the one it picks. [`Cheap::start`] answers the start the way
//! [`SimpleScheduler::place_activity`] finds it -- the fixed point of "every requirement can
//! start at `t`" -- but only where that is the whole story:
//!
//! * no transition matrix, no companion and no transport anywhere (a booking would change
//!   what another resource charges, holds or takes to reach);
//! * an activity that is not resumable, not split, does not overlap its previous step, has no
//!   calendar of its own and takes as long on every resource.
//!
//! Anything else is [`Start::Declined`]: the caller places it in full. The equivalence with
//! `place_activity` is pinned by the tests below, which compare the two on random ledgers.

use std::collections::HashMap;

use super::{SimpleScheduler, MAX_FIXED_POINT_ITERS};
use crate::models::{Activity, Resource};
use crate::scheduler::timeline::ResourceTimeline;

/// What the cheap query knows about an offer's start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Start {
    /// The offer starts here: the start [`SimpleScheduler::place_activity`] gives.
    At(i64),
    /// There is no offer: `place_activity` gives `None` (too few resources can ever take it).
    Never,
    /// Not answered cheaply -- ask `place_activity`.
    Declined,
}

/// The cheap query, for one problem: what holds of every resource, found once.
pub(super) struct Cheap {
    /// No transition matrix, companion or transport anywhere.
    plain_world: bool,
}

impl Cheap {
    pub(super) fn new(scheduler: &SimpleScheduler, resources: &[Resource]) -> Self {
        Self {
            plain_world: scheduler.transition_matrices.is_empty()
                && resources
                    .iter()
                    .all(|r| r.companions.is_empty() && r.transport_ms.is_empty()),
        }
    }

    /// Whether `activity` is of the kind [`start`](Self::start) answers.
    fn plain(activity: &Activity) -> bool {
        !activity.resumable
            && activity.split.is_none()
            && activity.overlap.is_none()
            && activity.resource_timings.is_empty()
    }

    /// The start of `activity`'s offer from `ready` on (its task's own release, and the
    /// activity's, applied here as in `place_activity`).
    pub(super) fn start(
        &self,
        activity: &Activity,
        own: Option<&ResourceTimeline>,
        timelines: &HashMap<String, ResourceTimeline>,
        resources: &[Resource],
        ready: i64,
    ) -> Start {
        if !self.plain_world || own.is_some() || !Self::plain(activity) {
            return Start::Declined;
        }
        let ready = activity.release_time.map_or(ready, |r| ready.max(r));
        let d = activity.duration;
        let span = d.setup_ms + d.process_ms + d.teardown_ms;
        let mut fits: Vec<i64> = Vec::new();
        let mut t = ready;
        for _ in 0..MAX_FIXED_POINT_ITERS {
            let mut t_next = t;
            for req in &activity.resource_requirements {
                if req.quantity <= 0 {
                    continue;
                }
                let wanted = req.quantity as usize;
                fits.clear();
                fits.extend(req.eligible(resources).filter_map(|r| {
                    timelines
                        .get(r.id.as_str())?
                        .earliest_fit(t, span, req.usage)
                }));
                if fits.len() < wanted {
                    return Start::Never;
                }
                // The resources are taken in order of their fit, so the requirement is ready
                // when the `wanted`-th earliest is.
                let fit = if wanted == 1 {
                    *fits.iter().min().expect("at least one fit")
                } else {
                    let (_, nth, _) = fits.select_nth_unstable(wanted - 1);
                    *nth
                };
                t_next = t_next.max(fit);
            }
            if t_next == t {
                return Start::At(t);
            }
            t = t_next;
        }
        Start::Declined // not converged: leave it to the full placement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        ActivityDuration, Calendar, ResourceRequirement, ResourceTiming, Split, Task,
    };

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn chance(&mut self, in_n: u64) -> bool {
            self.below(in_n) == 0
        }
    }

    /// A pool of machines and one of operators (some with calendars or capacity above one),
    /// with bookings laid where each fits.
    fn world(rng: &mut Lcg) -> (Vec<Resource>, HashMap<String, ResourceTimeline>) {
        let mut resources: Vec<Resource> = Vec::new();
        for (kind, count) in [("Machine", 1 + rng.below(6)), ("Operator", rng.below(3))] {
            for k in 0..count {
                let mut r = Resource::new(format!("{kind}{k}"), kind);
                if rng.chance(4) {
                    r = r.with_capacity(2 + rng.below(2) as i32);
                }
                if rng.chance(4) {
                    let mut calendar = Calendar::new("shift");
                    let mut at = rng.below(500) as i64;
                    for _ in 0..(2 + rng.below(4)) {
                        let len = 800 + rng.below(4_000) as i64;
                        calendar = calendar.with_window(at, at + len);
                        at += len + 100 + rng.below(1_500) as i64;
                    }
                    r = r.with_calendar(calendar);
                }
                resources.push(r);
            }
        }
        let mut timelines: HashMap<String, ResourceTimeline> = resources
            .iter()
            .map(|r| (r.id.clone(), ResourceTimeline::new(r)))
            .collect();
        for r in &resources {
            for _ in 0..rng.below(8) {
                let tl = timelines.get_mut(&r.id).expect("a timeline per resource");
                let span = 200 + rng.below(2_500) as i64;
                let usage = 1 + rng.below(r.capacity.max(1) as u64) as i32;
                if let Some(start) = tl.earliest_fit(rng.below(6_000) as i64, span, usage) {
                    tl.book(start, start + span, usage);
                }
            }
        }
        (resources, timelines)
    }

    fn activity(rng: &mut Lcg) -> Activity {
        let mut a = Activity::new("A", "T", 0).with_duration(ActivityDuration {
            setup_ms: rng.below(3) as i64 * 50,
            process_ms: 100 + rng.below(3_000) as i64,
            teardown_ms: rng.below(3) as i64 * 40,
        });
        if rng.chance(3) {
            a.release_time = Some(rng.below(6_000) as i64);
        }
        for _ in 0..(1 + rng.below(3)) {
            let kind = if rng.chance(3) { "Operator" } else { "Machine" };
            let mut req = ResourceRequirement::new(kind)
                .with_quantity(1 + rng.below(2) as i32)
                .with_usage(1 + rng.below(2) as i32);
            if rng.chance(4) {
                req = req.with_candidates(vec![
                    format!("{kind}{}", rng.below(3)),
                    format!("{kind}{}", rng.below(3)),
                ]);
            }
            a = a.with_requirement(req);
        }
        a
    }

    fn full_start(
        scheduler: &SimpleScheduler,
        activity: &Activity,
        timelines: &HashMap<String, ResourceTimeline>,
        resources: &[Resource],
        ready: i64,
    ) -> Option<i64> {
        scheduler
            .place_activity(
                activity,
                &Task::new("T"),
                timelines,
                None,
                &HashMap::new(),
                resources,
                ready,
                &[],
            )
            .map(|p| p.start_ms)
    }

    #[test]
    fn the_cheap_start_is_the_start_of_the_full_placement() {
        let scheduler = SimpleScheduler::new();
        let (mut never, mut at) = (0, 0);
        for seed in 0..6_000u64 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let (resources, timelines) = world(&mut rng);
            let cheap = Cheap::new(&scheduler, &resources);
            for _ in 0..4 {
                let activity = activity(&mut rng);
                let ready = rng.below(7_000) as i64;
                let want = full_start(&scheduler, &activity, &timelines, &resources, ready);
                let got = cheap.start(&activity, None, &timelines, &resources, ready);
                match (got, want) {
                    (Start::At(g), Some(w)) => {
                        assert_eq!(g, w, "seed {seed}: {activity:?} ready {ready}");
                        at += 1;
                    }
                    (Start::Never, None) => never += 1,
                    other => panic!("seed {seed}: cheap {:?} but full {:?}", other.0, other.1),
                }
            }
        }
        // The comparison must have exercised both answers, or it proves nothing.
        assert!(at > 5_000, "only {at} placed");
        assert!(never > 100, "only {never} unplaceable");
    }

    #[test]
    fn what_it_cannot_answer_it_declines() {
        let r = vec![
            Resource::new("M0", "Machine"),
            Resource::new("M1", "Machine"),
        ];
        let timelines: HashMap<String, ResourceTimeline> = r
            .iter()
            .map(|r| (r.id.clone(), ResourceTimeline::new(r)))
            .collect();
        let base = || {
            Activity::new("A", "T", 0)
                .with_process_time(100)
                .with_requirement(ResourceRequirement::new("Machine"))
        };
        let scheduler = SimpleScheduler::new();
        let cheap = Cheap::new(&scheduler, &r);
        let ask = |a: &Activity| cheap.start(a, None, &timelines, &r, 0);
        assert_eq!(ask(&base()), Start::At(0));
        assert_eq!(ask(&base().resumable()), Start::Declined);
        assert_eq!(
            ask(&base().with_split(Split {
                max_parts: 2,
                min_part_ms: 0,
                part_overhead_ms: 0
            })),
            Start::Declined
        );
        let mut timed = base();
        timed
            .resource_timings
            .insert("M0".into(), ResourceTiming::Speed(2.0));
        assert_eq!(ask(&timed), Start::Declined);
        // An activity with a calendar of its own comes with its timeline.
        let own = ResourceTimeline::new(&Resource::new("own", "Machine"));
        assert_eq!(
            cheap.start(&base(), Some(&own), &timelines, &r, 0),
            Start::Declined
        );
        // A world with a companion, a transport or a transition matrix.
        let mut with_companion = r.clone();
        with_companion[0] = Resource::new("M0", "Machine").with_companion("M1");
        assert_eq!(
            Cheap::new(&scheduler, &with_companion).start(&base(), None, &timelines, &r, 0),
            Start::Declined
        );
        let mut with_transport = r.clone();
        with_transport[0] = Resource::new("M0", "Machine").with_transport_to("M1", 5);
        assert_eq!(
            Cheap::new(&scheduler, &with_transport).start(&base(), None, &timelines, &r, 0),
            Start::Declined
        );
        let matrices = crate::models::TransitionMatrixCollection::new()
            .with_matrix(crate::models::TransitionMatrix::new("m", "M0").with_default(50));
        let with_matrix = SimpleScheduler::new().with_transition_matrices(matrices);
        assert_eq!(
            Cheap::new(&with_matrix, &r).start(&base(), None, &timelines, &r, 0),
            Start::Declined
        );
        // A calendar that cannot hold the duration is no offer, as in the full placement.
        let short =
            Resource::new("S0", "Short").with_calendar(Calendar::new("c").with_window(0, 50));
        let rs = vec![short.clone()];
        let tls: HashMap<String, ResourceTimeline> =
            [(short.id.clone(), ResourceTimeline::new(&short))].into();
        let a = Activity::new("A", "T", 0)
            .with_process_time(100)
            .with_requirement(ResourceRequirement::new("Short"));
        assert_eq!(
            Cheap::new(&scheduler, &rs).start(&a, None, &tls, &rs, 0),
            Start::Never
        );
    }
}
