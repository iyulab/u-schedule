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
        // The fits of the candidates of one requirement, and what the requirements before it
        // in the same pass take of each resource (kept only when there is a later one to see it).
        let mut fits: Vec<(i64, &str)> = Vec::new();
        let mut taken: Vec<(&str, i32)> = Vec::new();
        let several = activity.resource_requirements.len() > 1;
        let mut t = ready;
        for _ in 0..MAX_FIXED_POINT_ITERS {
            let mut t_next = t;
            taken.clear();
            for req in &activity.resource_requirements {
                if req.quantity <= 0 {
                    continue;
                }
                let wanted = req.quantity as usize;
                let fit_of = req.eligible(resources).filter_map(|r| {
                    let rid = r.id.as_str();
                    // What an earlier requirement of this activity takes of the resource adds.
                    let usage = req.usage
                        + taken
                            .iter()
                            .filter(|(held, _)| *held == rid)
                            .map(|&(_, units)| units)
                            .sum::<i32>();
                    let fit = timelines.get(rid)?.earliest_fit(t, span, usage)?;
                    Some((fit, rid))
                });
                let fit = if wanted == 1 {
                    // The one resource that can start it first, the first declared on a tie.
                    let mut best: Option<(i64, &str)> = None;
                    for candidate in fit_of {
                        if best.is_none_or(|(fit, _)| candidate.0 < fit) {
                            best = Some(candidate);
                        }
                    }
                    let Some((fit, rid)) = best else {
                        return Start::Never;
                    };
                    if several {
                        taken.push((rid, req.usage));
                    }
                    fit
                } else {
                    fits.clear();
                    fits.extend(fit_of);
                    if fits.len() < wanted {
                        return Start::Never;
                    }
                    // The resources are taken in order of their fit (the first declared on a
                    // tie), so the requirement is ready when the `wanted`-th earliest is.
                    fits.sort_by_key(|&(fit, _)| fit);
                    if several {
                        taken.extend(fits[..wanted].iter().map(|&(_, rid)| (rid, req.usage)));
                    }
                    fits[wanted - 1].0
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
    use crate::dispatching::{rules, RuleEngine};
    use crate::models::{
        ActivityDuration, Calendar, Overlap, ResourceRequirement, ResourceTiming, Split, Task,
    };
    use crate::Problem;

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

    /// A problem of tasks of one to three activities on [`world`]'s resources, with now and
    /// then an activity of a kind the cheap query declines, a predecessor across tasks, an
    /// activity that holds no resource, or a world that declines everything.
    fn problem(rng: &mut Lcg) -> Option<(Problem, bool)> {
        let (mut resources, _) = world(rng);
        let kind = rng.below(10);
        if kind == 0 && resources.len() > 1 {
            let other = resources[1].id.clone();
            resources[0] = resources[0].clone().with_companion(other);
        }
        if kind == 1 && resources.len() > 1 {
            let other = resources[1].id.clone();
            resources[0] = resources[0].clone().with_transport_to(other, 20);
        }
        let matrix = kind == 2;
        let ids: Vec<String> = resources.iter().map(|r| r.id.clone()).collect();
        let tasks_n = 5 + rng.below(26) as usize;
        let tasks: Vec<Task> = (0..tasks_n)
            .map(|t| {
                let mut task = Task::new(format!("T{t}"))
                    .with_priority(rng.below(5) as i32)
                    .with_category(format!("c{}", rng.below(3)))
                    .with_release_time(rng.below(80) as i64 * 50)
                    .with_deadline(2_000 + rng.below(4_000) as i64 * 10);
                for s in 0..=rng.below(3) {
                    let mut a = activity(rng);
                    a.id = format!("T{t}-{s}");
                    a.task_id = format!("T{t}");
                    a.sequence = s as i32;
                    // Only the resources that exist.
                    for req in &mut a.resource_requirements {
                        req.candidates.retain(|c| ids.contains(c));
                    }
                    if rng.chance(8) {
                        a = a.resumable();
                    }
                    if rng.chance(8) {
                        a = a.with_split(Split {
                            max_parts: 2 + rng.below(2) as i32,
                            min_part_ms: 150,
                            part_overhead_ms: rng.below(3) as i64 * 20,
                        });
                    }
                    if s > 0 && rng.chance(8) {
                        a = a.with_overlap(Overlap::new(0.5));
                    }
                    if rng.chance(10) {
                        a = a.with_speed_on("Machine0", 2.0);
                    }
                    if rng.chance(12) {
                        // Working time of its own that binds: two windows, some work does not fit.
                        let at = rng.below(1_500) as i64;
                        let (first, gap) = (600 + rng.below(2_000) as i64, rng.below(1_500) as i64);
                        a = a.with_calendar(
                            Calendar::new("own")
                                .with_window(at, at + first)
                                .with_window(at + first + gap + 100, 100_000),
                        );
                    }
                    if s == 0 && t > 0 && rng.chance(6) {
                        a = a.with_predecessor(format!("T{}-0", rng.below(t as u64)));
                    }
                    if rng.chance(12) {
                        a.resource_requirements.clear();
                    }
                    task = task.with_activity(a);
                }
                task
            })
            .collect();
        Problem::new(tasks, resources).ok().map(|p| (p, matrix))
    }

    fn dispatcher(rule: usize, matrix: bool, reference: bool) -> SimpleScheduler {
        let mut scheduler = match rule {
            0 => SimpleScheduler::new(),
            1 => SimpleScheduler::new().with_rule_engine(RuleEngine::new().with_rule(rules::Spt)),
            _ => SimpleScheduler::new().with_rule_engine(
                RuleEngine::new()
                    .with_rule(rules::Winq)
                    .with_rule(rules::Spt),
            ),
        };
        if matrix {
            let matrices = crate::models::TransitionMatrixCollection::new().with_matrix(
                crate::models::TransitionMatrix::new("m", "Machine0").with_default(30),
            );
            scheduler = scheduler.with_transition_matrices(matrices);
        }
        if reference {
            scheduler = scheduler.without_cheap_offers();
        }
        scheduler
    }

    #[test]
    fn the_dispatcher_with_cheap_offers_schedules_what_the_reference_does() {
        let mut valid = 0;
        for seed in 0..260u64 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let Some((problem, matrix)) = problem(&mut rng) else {
                continue;
            };
            valid += 1;
            for rule in 0..3 {
                let now = dispatcher(rule, matrix, false).schedule(&problem, 0);
                let reference = dispatcher(rule, matrix, true).schedule(&problem, 0);
                assert_eq!(
                    format!("{now:?}"),
                    format!("{reference:?}"),
                    "seed {seed} rule {rule}"
                );
            }
        }
        assert!(valid > 200, "only {valid} of 260 problems were valid");
    }

    /// How long the dispatcher takes on `n` tasks of one machine activity each, all released
    /// at once on a pool of `m` machines (every offer reads the same machine), with and without
    /// the cheap start. Slow, so only on request, and meant for `--release`:
    /// `cargo test --release --lib dispatch_at_scale -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement: run it in release"]
    fn dispatch_at_scale() {
        for (n, m, with_reference) in [
            (1_000, 1, true),
            (2_000, 1, true),
            (2_000, 10, true),
            (3_000, 1, false),
            (3_000, 10, false),
        ] {
            let mut rng = Lcg(n as u64 * 31 + m as u64);
            let resources: Vec<Resource> = (0..m)
                .map(|k| Resource::new(format!("M{k}"), "Machine"))
                .collect();
            let tasks: Vec<Task> = (0..n)
                .map(|i| {
                    Task::new(format!("T{i}"))
                        .with_priority(rng.below(5) as i32)
                        .with_activity(
                            Activity::new(format!("T{i}-0"), format!("T{i}"), 0)
                                .with_process_time(500 + rng.below(30) as i64 * 100)
                                .with_requirement(ResourceRequirement::new("Machine")),
                        )
                })
                .collect();
            let problem = Problem::new(tasks, resources).expect("a valid problem");
            let timed = |reference: bool| {
                let started = std::time::Instant::now();
                let schedule = dispatcher(0, false, reference).schedule(&problem, 0);
                (started.elapsed().as_secs_f64(), format!("{schedule:?}"))
            };
            let (cheap, cheap_schedule) = timed(false);
            if with_reference {
                let (reference, reference_schedule) = timed(true);
                assert_eq!(cheap_schedule, reference_schedule, "n={n} m={m}");
                eprintln!("n={n:>5} m={m:>2}  cheap {cheap:>7.2} s  reference {reference:>7.2} s");
            } else {
                eprintln!("n={n:>5} m={m:>2}  cheap {cheap:>7.2} s");
            }
        }
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
