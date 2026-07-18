//! Probe-derived integration tests: scheduler output must self-validate.
//!
//! Scenarios originate from a consumer-side runtime probe that exposed the
//! model-solver enforcement gap fixed in 0.4.0 (multi-resource hold,
//! calendars, capacity, duration components, vacuous `is_valid()`).

use proptest::prelude::*;
use u_schedule::models::*;
use u_schedule::scheduler::{check_schedule, FeasibilityInput, ResourceTimeline, SimpleScheduler};

#[test]
fn scheduler_output_is_feasible_end_to_end() {
    // 설비 2 + 공유 금형 + 캘린더 + capacity 조합
    let cal = Calendar::new("shift")
        .with_window(0, 8000)
        .with_window(12_000, 30_000);
    let mk = |tid: &str, machine: &str| {
        Task::new(tid)
            .with_priority(1)
            .with_category("default")
            .with_activity(
                Activity::new(format!("{tid}_O1"), tid, 0)
                    .with_duration(ActivityDuration::new(200, 2000, 100))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                    )
                    .with_requirement(
                        ResourceRequirement::new("Mold").with_candidates(vec!["T1".into()]),
                    ),
            )
    };
    let tasks = vec![mk("J1", "M1"), mk("J2", "M2"), mk("J3", "M1")];
    let resources = vec![
        Resource::primary("M1").with_calendar(cal.clone()),
        Resource::primary("M2"),
        Resource::secondary("T1").with_capacity(1),
    ];
    let s = SimpleScheduler::new().schedule(&tasks, &resources, 0);
    assert!(
        s.violations.is_empty(),
        "self-reported violations: {:?}",
        s.violations
    );
    assert_eq!(s.assignments.len(), 6); // 3 activity × 2 자원
    let v = check_schedule(
        &s,
        &FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        },
    );
    assert!(v.is_empty(), "external check found: {v:?}");
}

#[test]
fn unfillable_requirement_reported_not_silent() {
    // probe S1의 정직성 요구: 충족 불가가 침묵하지 않는다
    let task = Task::new("J1").with_activity(
        Activity::new("O1", "J1", 0)
            .with_duration(ActivityDuration::fixed(1000))
            .with_requirement(
                ResourceRequirement::new("Machine").with_candidates(vec!["NOPE".into()]),
            ),
    );
    let s = SimpleScheduler::new().schedule(&[task], &[Resource::primary("M1")], 0);
    assert!(!s.is_valid());
    assert!(s
        .violations
        .iter()
        .any(|v| v.violation_type == ViolationType::RequirementUnfilled));
}

fn arb_instance() -> impl Strategy<Value = (Vec<Task>, Vec<Resource>)> {
    // 자원 2~4개 (첫 번째는 capacity 1~2 secondary 도구, 두 번째는 캘린더 보유)
    let res = prop::collection::vec(1..=2i32, 2..=4usize).prop_map(|caps| {
        caps.into_iter()
            .enumerate()
            .map(|(i, cap)| {
                if i == 0 {
                    Resource::secondary("T0").with_capacity(cap)
                } else if i == 1 {
                    Resource::primary(format!("M{i}")).with_calendar(
                        Calendar::new("c")
                            .with_window(0, 10_000)
                            .with_window(15_000, 60_000),
                    )
                } else {
                    Resource::primary(format!("M{i}"))
                }
            })
            .collect::<Vec<_>>()
    });
    let tasks = prop::collection::vec((100..3000i64, 0..2usize, any::<bool>()), 1..=5usize);
    (res, tasks).prop_map(|(resources, specs)| {
        let machine_ids: Vec<String> = resources
            .iter()
            .filter(|r| r.resource_type == ResourceType::Primary)
            .map(|r| r.id.clone())
            .collect();
        let tasks = specs
            .into_iter()
            .enumerate()
            .map(|(i, (dur, machine_idx, use_tool))| {
                let tid = format!("J{i}");
                let mid = machine_ids[machine_idx % machine_ids.len()].clone();
                let mut act = Activity::new(format!("{tid}_O1"), &tid, 0)
                    .with_duration(ActivityDuration::new(dur / 10, dur, dur / 20))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec![mid]),
                    );
                if use_tool {
                    act = act.with_requirement(
                        ResourceRequirement::new("Tool").with_candidates(vec!["T0".into()]),
                    );
                }
                Task::new(&tid)
                    .with_priority(i as i32)
                    .with_category("default")
                    .with_activity(act)
            })
            .collect::<Vec<_>>();
        (tasks, resources)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn scheduler_output_always_self_consistent((tasks, resources) in arb_instance()) {
        let s = SimpleScheduler::new().schedule(&tasks, &resources, 0);
        let non_unfilled: Vec<_> = s.violations.iter()
            .filter(|v| v.violation_type != ViolationType::RequirementUnfilled)
            .collect();
        prop_assert!(non_unfilled.is_empty(), "non-unfilled violations: {non_unfilled:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn earliest_fit_matches_brute_force(
        bookings in prop::collection::vec((0..150i64, 1..40i64), 0..6),
        cap in 1..3i32,
        dur in 1..50i64,
        from in 0..100i64,
    ) {
        let cal = Calendar::new("c")
            .with_window(0, 120)
            .with_window(160, 400)
            .with_blocked(60, 70);
        let mut tl = ResourceTimeline::new(
            &Resource::primary("M1").with_capacity(cap).with_calendar(cal),
        );
        for (s, d) in &bookings {
            tl.book(*s, s + d);
        }
        let fast = tl.earliest_fit(from, dur);
        let brute = (from..500).find(|&t| tl.fits(t, dur));
        prop_assert_eq!(fast, brute);
    }
}
