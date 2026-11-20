//! What the dispatcher keeps of the bookings made so far, for the offers' memory to read.
//!
//! Two things are kept. How many times each resource has been booked (its `version`), which
//! says whether what an offer found of it still holds ([`Memo`](super::Memo)); and, for each
//! *shape* of requirement -- the resources that can take it and how many of them it needs --
//! where the resources stand at the end of their bookings: the time from which a requirement of
//! that shape finds every one of them free (`tail`), when the last of their bookings starts
//! (`last_start`) and how many bookings they hold.
//!
//! A requirement's shape is shared by every offer that asks the same of the same resources (a
//! pool of machines is one shape for all the activities that need one machine of it), so the
//! tail is found once per shape after a booking, not once per offer.

use std::collections::HashMap;

use crate::models::Resource;
use crate::scheduler::timeline::ResourceTimeline;

/// The resources that can take a requirement and how many it needs of them.
pub(super) struct Shape {
    /// The resources, by their position among the problem's, in that order.
    positions: Vec<usize>,
    /// How many of them one activity needs at once.
    quantity: usize,
    /// The booking generation this was read at.
    stamp: u64,
    /// Every one of them has one unit and works round the clock: a time past the start of
    /// their last booking finds each free from the end of its latest booking on.
    pub(super) plain: bool,
    /// The time from which `quantity` of them are free: the `quantity`-th earliest end of the
    /// resources' latest bookings (`i64::MIN` for a resource with none).
    pub(super) tail: i64,
    /// The start of the booking that starts last on any of them (`i64::MIN` if none).
    pub(super) last_start: i64,
    /// The bookings on all of them.
    pub(super) bookings: usize,
}

/// The bookings so far.
pub(in crate::scheduler::simple) struct Books {
    /// How many times each resource has been booked.
    versions: Vec<u32>,
    /// How many bookings in all: a shape read at this generation is as it is now.
    generation: u64,
    shapes: Vec<Shape>,
    by_shape: HashMap<(Vec<usize>, usize), usize>,
    ends: Vec<i64>,
    /// How many answers came from the tail, for the tests to see that it was used.
    #[cfg(test)]
    pub(super) from_the_tail: usize,
}

impl Books {
    /// No bookings yet, among `resources` resources.
    pub(in crate::scheduler::simple) fn new(resources: usize) -> Self {
        Self {
            versions: vec![0; resources],
            generation: 0,
            shapes: Vec::new(),
            by_shape: HashMap::new(),
            ends: Vec::new(),
            #[cfg(test)]
            from_the_tail: 0,
        }
    }

    /// A booking was made on the resource at `position`.
    pub(in crate::scheduler::simple) fn booked(&mut self, position: usize) {
        self.versions[position] += 1;
        self.generation += 1;
    }

    /// How many times each resource has been booked.
    pub(super) fn versions(&self) -> &[u32] {
        &self.versions
    }

    /// The shape of a requirement that needs `quantity` of `positions`, once for all.
    pub(super) fn shape_of(&mut self, positions: Vec<usize>, quantity: usize) -> usize {
        debug_assert!(quantity >= 1 && positions.len() >= quantity);
        if let Some(&shape) = self.by_shape.get(&(positions.clone(), quantity)) {
            return shape;
        }
        self.shapes.push(Shape {
            positions: positions.clone(),
            quantity,
            stamp: u64::MAX,
            plain: false,
            tail: i64::MIN,
            last_start: i64::MIN,
            bookings: 0,
        });
        self.by_shape
            .insert((positions, quantity), self.shapes.len() - 1);
        self.shapes.len() - 1
    }

    /// Where the resources of `shape` stand now.
    pub(super) fn read(
        &mut self,
        shape: usize,
        timelines: &HashMap<String, ResourceTimeline>,
        resources: &[Resource],
    ) -> &Shape {
        let shape = &mut self.shapes[shape];
        if shape.stamp == self.generation {
            return shape;
        }
        shape.stamp = self.generation;
        shape.plain = true;
        shape.last_start = i64::MIN;
        shape.bookings = 0;
        self.ends.clear();
        for &position in &shape.positions {
            match timelines.get(resources[position].id.as_str()) {
                Some(timeline) if timeline.is_plain_unit() => {
                    self.ends.push(timeline.latest_end().unwrap_or(i64::MIN));
                    shape.last_start = shape
                        .last_start
                        .max(timeline.latest_start().unwrap_or(i64::MIN));
                    shape.bookings += timeline.booking_count();
                }
                _ => {
                    shape.plain = false;
                    return shape;
                }
            }
        }
        self.ends.sort_unstable();
        shape.tail = self.ends[shape.quantity - 1];
        shape
    }
}
