//! What the dispatcher keeps of an offer from one decision to the next.
//!
//! An offer's start is the fixed point of "every requirement can start at `t`": starting from
//! the ready time, `t` goes to the latest of the requirements' earliest fits until it stays.
//! A booking only takes time from a resource, so every resource's earliest fit can only move
//! later, and so can the fixed point -- the start found before is a floor under the one to be
//! found now, and the search may go on from it instead of from the ready time.
//!
//! The memory keeps, for each resource that can take a requirement, the earliest start it was
//! found to give, `at`. That is the answer for any time from the one it was asked for up to
//! `at` itself (nothing in between could start it), as long as the resource has not been
//! booked since: the resource's `version` says so. A decision books a few resources, so asking
//! again costs a comparison per resource and a fresh question for the few that changed -- not
//! a question per resource, and not the passes from the ready time again.
//!
//! This holds only where the requirements share no resource (what one takes of a resource
//! changes what the next sees of it, and the floor argument no longer holds): then there is no
//! memory, and [`Cheap::start`] answers. The answer is always the one [`Cheap::start`] gives,
//! the bound on the passes included: the passes made since the memory began are counted, and
//! when they could add up to more than [`Cheap::start`] would be allowed, the memory begins
//! afresh and the search is the one from the ready time.

use std::collections::HashMap;

use super::{Cheap, Start};
use crate::models::{Activity, Resource};
use crate::scheduler::timeline::ResourceTimeline;

/// One requirement as the memory holds it.
struct Want {
    /// How many resources it needs at once.
    quantity: usize,
    /// How much of each it takes.
    usage: i32,
    /// The resources that can take it and still might start it.
    fits: Vec<Fit>,
}

/// What a resource was last found to give a requirement.
struct Fit {
    /// The resource, by its position among the problem's.
    position: usize,
    /// The earliest start it gave.
    at: i64,
    /// Its version when it did.
    version: u32,
}

/// What is remembered of one offer: the activity it is for, how far its search got, and what
/// the resources gave.
pub(in crate::scheduler::simple) struct Memo {
    /// The place of the activity among its task's, which the memory is of.
    cursor: usize,
    /// Where the next search goes on from: the last start, the ready time before any.
    from: i64,
    /// The passes that moved the search on since the memory began.
    steps: usize,
    /// No offer: too few resources can take a requirement from `from` on, and so from any
    /// later time.
    never: bool,
    wants: Vec<Want>,
}

impl Memo {
    /// An empty memory of the offer of `activity` (the `cursor`-th of its task) from `ready`
    /// on; `None` when two requirements may take the same resource.
    fn begin(
        cursor: usize,
        activity: &Activity,
        resources: &[Resource],
        ready: i64,
    ) -> Option<Self> {
        let mut used = vec![false; resources.len()];
        let mut wants = Vec::new();
        for req in &activity.resource_requirements {
            if req.quantity <= 0 {
                continue;
            }
            let mut fits = Vec::new();
            for (position, resource) in resources.iter().enumerate() {
                if req.accepts(resource) {
                    if used[position] {
                        return None;
                    }
                    used[position] = true;
                    // Never asked: `at` and `version` say it has to be.
                    fits.push(Fit {
                        position,
                        at: i64::MIN,
                        version: u32::MAX,
                    });
                }
            }
            wants.push(Want {
                quantity: req.quantity as usize,
                usage: req.usage,
                fits,
            });
        }
        Some(Self {
            cursor,
            from: ready,
            steps: 0,
            never: false,
            wants,
        })
    }
}

impl Cheap {
    /// [`start`](Self::start), remembering in `slot` what the next ask can go on from.
    /// `versions[k]` counts the bookings of the `k`-th resource so far, and `cursor` is the
    /// place of `activity` among its task's.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::scheduler::simple) fn start_remembering(
        &self,
        slot: &mut Option<Memo>,
        cursor: usize,
        activity: &Activity,
        own: Option<&ResourceTimeline>,
        timelines: &HashMap<String, ResourceTimeline>,
        resources: &[Resource],
        versions: &[u32],
        ready: i64,
    ) -> Start {
        if !self.plain_world || own.is_some() || !Self::plain(activity) {
            return Start::Declined;
        }
        let from = activity.release_time.map_or(ready, |r| ready.max(r));
        let mut fresh = false;
        let mut memo = match slot.take() {
            Some(memo) if memo.cursor == cursor => memo,
            _ => match Memo::begin(cursor, activity, resources, from) {
                Some(memo) => {
                    fresh = true;
                    memo
                }
                None => return self.start(activity, own, timelines, resources, ready),
            },
        };
        loop {
            if let Some(start) = self.go_on(&mut memo, activity, timelines, resources, versions) {
                *slot = Some(memo);
                return start;
            }
            // The passes may have added up to more than the search from the ready time is
            // allowed: begin again, which is that search. A search that was already that
            // one has run out of passes.
            if fresh {
                return Start::Declined;
            }
            fresh = true;
            memo = Memo::begin(cursor, activity, resources, from)
                .expect("a memory that began once begins again");
        }
    }

    /// Goes on from where `memo` stands to the start of the offer; `None` when the passes
    /// would pass the bound.
    fn go_on(
        &self,
        memo: &mut Memo,
        activity: &Activity,
        timelines: &HashMap<String, ResourceTimeline>,
        resources: &[Resource],
        versions: &[u32],
    ) -> Option<Start> {
        if memo.never {
            return Some(Start::Never);
        }
        let d = activity.duration;
        let span = d.setup_ms + d.process_ms + d.teardown_ms;
        let mut t = memo.from;
        let mut earliest: Vec<i64> = Vec::new();
        loop {
            // The pass after `steps` moves is the `steps + 1`-th the search from the ready
            // time might make, and it is allowed `max_passes`.
            if memo.steps >= self.max_passes {
                return None;
            }
            let mut t_next = t;
            for want in &mut memo.wants {
                let usage = want.usage;
                want.fits.retain_mut(|fit| {
                    if fit.version == versions[fit.position] && fit.at >= t {
                        return true;
                    }
                    let found = timelines
                        .get(resources[fit.position].id.as_str())
                        .and_then(|timeline| timeline.earliest_fit(t, span, usage));
                    match found {
                        Some(at) => {
                            fit.at = at;
                            fit.version = versions[fit.position];
                            true
                        }
                        None => false,
                    }
                });
                if want.fits.len() < want.quantity {
                    memo.never = true;
                    return Some(Start::Never);
                }
                // The requirement is ready when the `quantity`-th earliest of them is.
                let ready_at = if want.quantity == 1 {
                    want.fits.iter().map(|fit| fit.at).min()
                } else {
                    earliest.clear();
                    earliest.extend(want.fits.iter().map(|fit| fit.at));
                    earliest.sort_unstable();
                    earliest.get(want.quantity - 1).copied()
                };
                t_next = t_next.max(ready_at.expect("a requirement has the resources it needs"));
            }
            if t_next == t {
                memo.from = t;
                return Some(Start::At(t));
            }
            t = t_next;
            memo.steps += 1;
        }
    }
}
