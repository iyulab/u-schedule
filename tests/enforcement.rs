//! Probe-derived integration tests: scheduler output must self-validate.
//!
//! Scenarios originate from a consumer-side runtime probe that exposed the
//! model-solver enforcement gap fixed in 0.4.0 (multi-resource hold,
//! calendars, capacity, duration components, vacuous `is_valid()`).

use proptest::prelude::*;
use u_schedule::models::*;
use u_schedule::scheduler::{
    check_schedule, FeasibilityInput, ResourceTimeline, ScheduleRequest, SimpleScheduler,
};

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

/// Two single-activity jobs contending for one machine (enforcement.rs style).
fn two_job_single_machine() -> (Vec<Task>, Vec<Resource>) {
    let mk = |tid: &str| {
        Task::new(tid)
            .with_priority(1)
            .with_category("default")
            .with_activity(
                Activity::new(format!("{tid}_O1"), tid, 0)
                    .with_duration(ActivityDuration::fixed(1000))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            )
    };
    let tasks = vec![mk("J1"), mk("J2")];
    let resources = vec![Resource::primary("M1")];
    (tasks, resources)
}

#[test]
fn fixed_assignment_is_seeded_and_serializes_others() {
    // M1 하나, J1(1000ms)·J2(1000ms). J2를 [2000,3000)에 pin.
    // 기대: J2 배정이 정확히 [2000,3000), J1은 pin 구간을 회피.
    let (tasks, resources) = two_job_single_machine();
    let pinned = Assignment::new("J2_O1", "J2", "M1", 2000, 3000);
    let s = SimpleScheduler::new()
        .with_fixed_assignments(vec![pinned])
        .schedule(&tasks, &resources, 0);
    let j2 = s
        .assignments
        .iter()
        .find(|a| a.activity_id == "J2_O1")
        .unwrap();
    assert_eq!((j2.start_ms, j2.end_ms), (2000, 3000));
    let j1 = s
        .assignments
        .iter()
        .find(|a| a.activity_id == "J1_O1")
        .unwrap();
    assert!(
        j1.end_ms <= 2000 || j1.start_ms >= 3000,
        "pin 구간 침범: {j1:?}"
    );
    // 선점 예약이 phantom violation을 만들지 않았음을 증명
    assert!(s.is_valid(), "seeded schedule flagged: {:?}", s.violations);
}

#[test]
fn conflicting_fixed_assignment_reports_violation_not_silent_move() {
    // M1 capacity 1. J1_O1·J2_O1을 둘 다 [2000,3000)에 pin → 자원 중복.
    // 기대: 무단 이동 없이 둘 다 그대로 배치되고 violation으로 정직 보고.
    let (tasks, resources) = two_job_single_machine();
    let p1 = Assignment::new("J1_O1", "J1", "M1", 2000, 3000);
    let p2 = Assignment::new("J2_O1", "J2", "M1", 2000, 3000);
    let s = SimpleScheduler::new()
        .with_fixed_assignments(vec![p1, p2])
        .schedule(&tasks, &resources, 0);
    assert!(!s.violations.is_empty(), "conflict not reported");
    assert!(!s.is_valid());
    assert!(s
        .violations
        .iter()
        .any(|v| v.violation_type == ViolationType::CapacityExceeded));
    // 무단 이동 금지: 두 pin 모두 명시 구간에 그대로
    let j1 = s
        .assignments
        .iter()
        .find(|a| a.activity_id == "J1_O1")
        .unwrap();
    assert_eq!((j1.start_ms, j1.end_ms), (2000, 3000));
    let j2 = s
        .assignments
        .iter()
        .find(|a| a.activity_id == "J2_O1")
        .unwrap();
    assert_eq!((j2.start_ms, j2.end_ms), (2000, 3000));
}

#[test]
fn multi_resource_pin_seeds_all_holds() {
    // 다중 requirement(Machine+Mold) activity를 두 자원 모두 pin.
    // 기대: 두 배정 모두 그대로 emit + 두 요구 모두 충족 → is_valid.
    let task = Task::new("J1")
        .with_priority(1)
        .with_category("default")
        .with_activity(
            Activity::new("J1_O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                )
                .with_requirement(
                    ResourceRequirement::new("Mold").with_candidates(vec!["T1".into()]),
                ),
        );
    let resources = vec![Resource::primary("M1"), Resource::secondary("T1")];
    let pins = vec![
        Assignment::new("J1_O1", "J1", "M1", 2000, 3000),
        Assignment::new("J1_O1", "J1", "T1", 2000, 3000),
    ];
    let s = SimpleScheduler::new()
        .with_fixed_assignments(pins)
        .schedule(&[task], &resources, 0);
    let holds = s.assignments_for_activity_all("J1_O1");
    assert_eq!(
        holds.len(),
        2,
        "both resource holds must be seeded: {holds:?}"
    );
    assert!(holds.iter().all(|a| (a.start_ms, a.end_ms) == (2000, 3000)));
    assert!(
        s.is_valid(),
        "fully-pinned multi-resource flagged: {:?}",
        s.violations
    );
}

#[test]
fn pinned_activity_pushes_successor_after_pin_end() {
    // 같은 task J1: O1 을 [2000,3000)에 pin, O2(sequence 1)는 normal — 같은 M1.
    // 기대(③): O2 는 pin end(3000) 이후 시작하고 스케줄은 valid.
    let task = Task::new("J1")
        .with_priority(1)
        .with_category("default")
        .with_activity(
            Activity::new("J1_O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                ),
        )
        .with_activity(
            Activity::new("J1_O2", "J1", 1)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                ),
        );
    let resources = vec![Resource::primary("M1")];
    let pin = Assignment::new("J1_O1", "J1", "M1", 2000, 3000);
    let s = SimpleScheduler::new()
        .with_fixed_assignments(vec![pin])
        .schedule(&[task], &resources, 0);
    let o2 = s.assignment_for_activity("J1_O2").unwrap();
    assert!(
        o2.start_ms >= 3000,
        "successor started before pin end: {o2:?}"
    );
    assert!(s.is_valid(), "flagged: {:?}", s.violations);
}

#[test]
fn pinned_successor_precedes_conflicting_unpinned_predecessor_reports_violation() {
    // 같은 task J1: O1(선행, unpinned, seq 0)·O2(후행, seq 1, [500,1500) pin) — 같은 M1(cap 1).
    // 기대: pin의 선점 예약이 O1을 pin 이후([1500,2500))로 밀어내고, annotate_schedule 후
    // O1(뒤로 밀림)이 O2(pin, 앞자리) 보다 늦게 끝나므로 PrecedenceViolation이 정직 보고되며,
    // pin 자체는 무단 이동 없이 정확히 [500,1500) 를 유지한다(회귀 가드 — Important #1).
    let task = Task::new("J1")
        .with_priority(1)
        .with_category("default")
        .with_activity(
            Activity::new("J1_O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                ),
        )
        .with_activity(
            Activity::new("J1_O2", "J1", 1)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                ),
        );
    let resources = vec![Resource::primary("M1")];
    let pin = Assignment::new("J1_O2", "J1", "M1", 500, 1500);
    let s = SimpleScheduler::new()
        .with_fixed_assignments(vec![pin])
        .schedule(&[task], &resources, 0);

    // ① 선점 예약: O1(선행)이 pin 구간 [500,1500)을 회피해 [1500,2500)으로 밀림.
    let o1 = s.assignment_for_activity("J1_O1").unwrap();
    assert_eq!(
        (o1.start_ms, o1.end_ms),
        (1500, 2500),
        "unpinned predecessor should be pushed past the pinned successor's reservation: {o1:?}"
    );

    // ④ 무단 이동 금지: pin(O2)은 정확히 명시 구간에 그대로.
    let o2 = s.assignment_for_activity("J1_O2").unwrap();
    assert_eq!(
        (o2.start_ms, o2.end_ms),
        (500, 1500),
        "pin must remain unmoved even though it now conflicts with precedence"
    );

    // ④ 정직 보고: O1이 O2보다 늦게 끝나므로 선후행 위반이 침묵 없이 발화한다.
    assert!(
        s.violations
            .iter()
            .any(|v| v.violation_type == ViolationType::PrecedenceViolation),
        "expected PrecedenceViolation for predecessor-after-pin ordering: {:?}",
        s.violations
    );
    assert!(!s.is_valid());
}

#[test]
fn pinned_activity_coexists_with_sgs_activity_under_capacity_two() {
    // M1 capacity=2. J1_O1을 [0,1000)에 pin, J2_O1(unpinned, 1000ms)은 같은 M1 사용.
    // 기대: pin이 1 유닛만 소비하므로 J2는 같은 [0,1000) 구간에 병렬 배정된다
    // (capacity=1 이었다면 J2는 pin 이후로 밀렸을 것) — 공존 가드 (Minor #2).
    let mk = |tid: &str| {
        Task::new(tid)
            .with_priority(1)
            .with_category("default")
            .with_activity(
                Activity::new(format!("{tid}_O1"), tid, 0)
                    .with_duration(ActivityDuration::fixed(1000))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            )
    };
    let tasks = vec![mk("J1"), mk("J2")];
    let resources = vec![Resource::primary("M1").with_capacity(2)];
    let pin = Assignment::new("J1_O1", "J1", "M1", 0, 1000);
    let s = SimpleScheduler::new()
        .with_fixed_assignments(vec![pin])
        .schedule(&tasks, &resources, 0);

    let j1 = s.assignment_for_activity("J1_O1").unwrap();
    assert_eq!((j1.start_ms, j1.end_ms), (0, 1000));

    let j2 = s.assignment_for_activity("J2_O1").unwrap();
    assert_eq!(
        (j2.start_ms, j2.end_ms),
        (0, 1000),
        "SGS activity should coexist with the pin under capacity 2, not be pushed out: {j2:?}"
    );

    assert!(
        s.is_valid(),
        "capacity-2 pin/SGS coexistence flagged: {:?}",
        s.violations
    );
}

#[test]
fn schedule_request_honors_fixed_assignments() {
    // schedule_request 진입점도 seed 를 존중해야 함 (Self 재구성 시 fixed 전파).
    let (tasks, resources) = two_job_single_machine();
    let pin = Assignment::new("J2_O1", "J2", "M1", 2000, 3000);
    let request = ScheduleRequest::new(tasks, resources);
    let s = SimpleScheduler::new()
        .with_fixed_assignments(vec![pin])
        .schedule_request(&request);
    let j2 = s.assignment_for_activity("J2_O1").unwrap();
    assert_eq!((j2.start_ms, j2.end_ms), (2000, 3000));
    assert!(s.is_valid(), "flagged: {:?}", s.violations);
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
