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
//!
//! Past the start of the last booking of every resource an offer may take, no hole is left to
//! look in: a resource of one unit round the clock is free from the end of its latest booking.
//! Then the start is a sum of the resources' tails, read from [`Books`] -- the same for every
//! offer of the same shape -- and neither the passes nor the versions are looked at
//! ([`Memo::past_the_tail`]).

use std::collections::HashMap;

use super::books::Books;
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
    /// Its shape in [`Books`], when the tail can answer for it: one unit of each resource.
    shape: Option<usize>,
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
        books: &mut Books,
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
            let quantity = req.quantity as usize;
            // A requirement of more than one unit is never a shape: a resource of one unit
            // cannot give it, and its fit is none.
            let shape = (req.usage <= 1 && fits.len() >= quantity)
                .then(|| books.shape_of(fits.iter().map(|fit| fit.position).collect(), quantity));
            wants.push(Want {
                quantity,
                usage: req.usage,
                fits,
                shape,
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

    /// The start of the offer when `from` is past the start of every last booking of the
    /// resources it may take and each is of one unit round the clock; `None` when that is not
    /// so, or when the search from the ready time might not have got there.
    ///
    /// Such a resource is free from `max(t, end of its latest booking)` on, for any `t` from
    /// `from` and any span of positive length -- the one booking that can cover `t` ends there
    /// and nothing starts after. So the earliest fit of the `quantity`-th resource of a
    /// requirement is `max(t, tail)`, and the fixed point from `from` is the latest of `from`
    /// and the tails: `from` is a floor under the start (the memory's argument above), so it is
    /// the start the search from the ready time would find.
    ///
    /// That search runs out of passes, and is [`Start::Declined`], where the tail would answer.
    /// Its passes are at most two when the offer asks one unit of one resource (the resource
    /// that gave the fit still fits there), and else one for each end of a booking it can step
    /// over, and one to see it stay. Beyond what the search is allowed this is not the
    /// answer, and the memory goes on as before. Having answered from the tail, the memory
    /// does not know how many passes lie behind its start, so `steps` is made to run out: the
    /// next ask that does not come from the tail begins afresh, which is the search from the
    /// ready time.
    fn past_the_tail(
        &mut self,
        books: &mut Books,
        timelines: &HashMap<String, ResourceTimeline>,
        resources: &[Resource],
        span: i64,
        max_passes: usize,
    ) -> Option<i64> {
        if self.never || span <= 0 || self.wants.is_empty() {
            return None;
        }
        let mut start = self.from;
        let mut bookings = 0;
        for want in &self.wants {
            let shape = books.read(want.shape?, timelines, resources);
            if !shape.plain || shape.last_start > self.from {
                return None;
            }
            start = start.max(shape.tail);
            bookings += shape.bookings;
        }
        let passes = match self.wants.as_slice() {
            [only] if only.quantity == 1 => 2,
            _ => bookings + 1,
        };
        if passes > max_passes {
            return None;
        }
        self.from = start;
        self.steps = usize::MAX;
        #[cfg(test)]
        {
            books.from_the_tail += 1;
        }
        Some(start)
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
        books: &mut Books,
        ready: i64,
    ) -> Start {
        if !self.plain_world || own.is_some() || !Self::plain(activity) {
            return Start::Declined;
        }
        let from = activity.release_time.map_or(ready, |r| ready.max(r));
        let d = activity.duration;
        let span = d.setup_ms + d.process_ms + d.teardown_ms;
        let mut fresh = false;
        let mut memo = match slot.take() {
            Some(memo) if memo.cursor == cursor => memo,
            _ => match Memo::begin(cursor, activity, resources, books, from) {
                Some(memo) => {
                    fresh = true;
                    memo
                }
                None => return self.start(activity, own, timelines, resources, ready),
            },
        };
        loop {
            if let Some(start) =
                memo.past_the_tail(books, timelines, resources, span, self.max_passes)
            {
                *slot = Some(memo);
                return Start::At(start);
            }
            if let Some(start) =
                self.go_on(&mut memo, activity, timelines, resources, books.versions())
            {
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
            memo = Memo::begin(cursor, activity, resources, books, from)
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
