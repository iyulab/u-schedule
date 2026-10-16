//! WASM bindings for u-schedule.
//!
//! # Functions
//!
//! - [`run_schedule`]: Priority dispatching on a flat job list (single or
//!   parallel machines).
//! - [`solve_jobshop`]: GA-based job-shop scheduling with multi-machine
//!   routing and precedence constraints.
//!
//! # Supported Dispatching Rules
//!
//! | Rule | Description |
//! |------|-------------|
//! | `"SPT"` | Shortest Processing Time |
//! | `"LPT"` | Longest Processing Time |
//! | `"EDD"` | Earliest Due Date |
//! | `"FCFS"` | First Come First Served |
//! | `"CR"` | Critical Ratio |
//! | `"WSPT"` | Weighted Shortest Processing Time |
//! | `"MST"` | Minimum Slack Time |
//! | `"S/RO"` | Slack per Remaining Operations |
//! | `"ATC"` | Apparent Tardiness Cost |
//! | `"LWKR"` | Least Work Remaining |
//! | `"MWKR"` | Most Work Remaining |
//! | `"PRIORITY"` | Task Priority |
//!
//! # Time Convention
//! JSON uses seconds (f64). Internally, the scheduling model uses
//! milliseconds (i64), converted by multiplying/dividing by 1000.

use std::ops::{Bound, RangeBounds};

use serde::{Deserialize, Serialize};
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::dispatching::rules;
use crate::dispatching::RuleEngine;
use crate::ga::operators::{CrossoverType, GeneticOperators, MutationType};
use crate::ga::SchedulingGaProblem;
use crate::models::{
    Activity, ActivityDuration, Assignment, Resource, ResourceRequirement, Schedule, Task,
    Violation, ViolationType,
};
use crate::scheduler::feasibility::{check_schedule, FeasibilityInput};
use crate::validation::{ValidationError, ValidationErrorKind};
use crate::Problem;
use u_metaheur::ga::{GaConfig, GaRunner};
use u_metaheur::{ConfigError, StopReason};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Every reason a function of this binding refuses with: the `code` an
/// `Error` carries, and the fields that can stand beside it. A refusal is
/// built from one of these ([`WireError::new`]), so [`codes`] -- the list a
/// caller reads -- is the list the binding refuses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Code {
    MalformedInput,
    UnknownOption,
    ParameterOutOfRange,
    InvalidOption,
    ValueNotFinite,
    EmptyInput,
    DuplicateId,
    MissingMachine,
    NoMachines,
    InvalidResourceReference,
    CyclicDependency,
    InvalidPredecessor,
    InvalidCalendarPeriod,
    Internal,
}

impl Code {
    const ALL: [Code; 14] = [
        Code::MalformedInput,
        Code::UnknownOption,
        Code::ParameterOutOfRange,
        Code::InvalidOption,
        Code::ValueNotFinite,
        Code::EmptyInput,
        Code::DuplicateId,
        Code::MissingMachine,
        Code::NoMachines,
        Code::InvalidResourceReference,
        Code::CyclicDependency,
        Code::InvalidPredecessor,
        Code::InvalidCalendarPeriod,
        Code::Internal,
    ];

    fn name(self) -> &'static str {
        match self {
            Code::MalformedInput => "malformed_input",
            Code::UnknownOption => "unknown_option",
            Code::ParameterOutOfRange => "parameter_out_of_range",
            Code::InvalidOption => "invalid_option",
            Code::ValueNotFinite => "value_not_finite",
            Code::EmptyInput => "empty_input",
            Code::DuplicateId => "duplicate_id",
            Code::MissingMachine => "missing_machine",
            Code::NoMachines => "no_machines",
            Code::InvalidResourceReference => "invalid_resource_reference",
            Code::CyclicDependency => "cyclic_dependency",
            Code::InvalidPredecessor => "invalid_predecessor",
            Code::InvalidCalendarPeriod => "invalid_calendar_period",
            Code::Internal => "internal",
        }
    }

    /// The fields a refusal with this code can carry besides `code`. A
    /// refusal from the problem check also carries `errors`, every finding.
    fn fields(self) -> &'static [&'static str] {
        match self {
            Code::MalformedInput | Code::ValueNotFinite => &["parameter", "index"],
            Code::UnknownOption => &["parameter", "got", "expected"],
            Code::ParameterOutOfRange => &[
                "parameter",
                "min",
                "max",
                "exclusive_min",
                "exclusive_max",
                "got",
                "index",
                "id",
                "task",
                "activity",
                "resource",
                "errors",
            ],
            Code::InvalidOption => &["parameter", "rule"],
            Code::EmptyInput => &["parameter", "task", "errors"],
            Code::DuplicateId => &["entity", "id", "first", "second", "errors"],
            Code::MissingMachine => &["job", "operation"],
            Code::NoMachines | Code::Internal => &[],
            Code::InvalidResourceReference => &["activity", "resource", "errors"],
            Code::CyclicDependency => &["activity", "errors"],
            Code::InvalidPredecessor => &["activity", "predecessor", "errors"],
            Code::InvalidCalendarPeriod => {
                &["parameter", "resource", "index", "start", "end", "errors"]
            }
        }
    }

    /// The code the problem check names a finding with.
    fn of(kind: &ValidationErrorKind) -> Self {
        let name = kind.code();
        Code::ALL
            .into_iter()
            .find(|c| c.name() == name)
            .expect("every problem-check reason is a code of this binding")
    }
}

/// A refusal on its way to JavaScript: the text for `Error.message`, and the
/// fields -- `code` first among them -- copied onto the `Error`.
#[derive(Debug)]
struct WireError {
    message: String,
    fields: serde_json::Value,
}

impl WireError {
    fn new(code: Code, message: String, mut extra: serde_json::Value) -> Self {
        if let Some(extra) = extra.as_object() {
            debug_assert!(
                extra.keys().all(|k| code.fields().contains(&k.as_str())),
                "{} refusal with undeclared fields: {:?}",
                code.name(),
                extra.keys().collect::<Vec<_>>()
            );
        }
        let mut fields = serde_json::Map::new();
        fields.insert("code".into(), json!(code.name()));
        if let Some(extra) = extra.as_object_mut() {
            fields.append(extra);
        }
        WireError {
            message,
            fields: serde_json::Value::Object(fields),
        }
    }

    /// The package failed, not the input: a result it could not hand back.
    fn internal(message: String) -> Self {
        Self::new(Code::Internal, message, json!({}))
    }

    /// An argument that is not the shape the function takes: a JSON string
    /// instead of a value, a wrong type, a missing or unknown key.
    fn malformed_input(parameter: &str, message: String) -> Self {
        Self::new(
            Code::MalformedInput,
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A string option that names none of the values the function knows.
    fn unknown_option(parameter: &str, got: &str, expected: &[&str]) -> Self {
        Self::new(
            Code::UnknownOption,
            format!(
                "Unknown {parameter} '{got}'. Supported: {}",
                expected.join(", ")
            ),
            json!({ "parameter": parameter, "got": got, "expected": expected }),
        )
    }

    /// A numeric option outside the range the solver accepts, each end open or
    /// closed as it is (`1.0..`, `0.0..=1.0`, `(Excluded(0.0), Unbounded)`).
    fn out_of_range(parameter: &str, accepted: impl RangeBounds<f64>, got: f64) -> Self {
        let bounds = Bounds::of(accepted);
        let mut fields = json!({ "parameter": parameter, "got": got });
        bounds.insert_into(&mut fields);
        Self::new(
            Code::ParameterOutOfRange,
            format!("{parameter} must be {}, got {got}", bounds.describe()),
            fields,
        )
    }

    /// The stable reason, as the `code` field carries it.
    #[cfg(test)]
    fn code(&self) -> &str {
        self.fields["code"]
            .as_str()
            .expect("every refusal carries a code")
    }
}

/// The bounds a `parameter_out_of_range` refusal reports. A closed (included)
/// end is `min` / `max`, an open (excluded) end `exclusive_min` /
/// `exclusive_max`; a side holds at most one of its two.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bounds {
    min: Option<f64>,
    max: Option<f64>,
    exclusive_min: Option<f64>,
    exclusive_max: Option<f64>,
}

impl Bounds {
    /// Reads a range's ends: `1.0..` is ">= 1", `(Excluded(0.0), Unbounded)`
    /// is "> 0".
    fn of(range: impl RangeBounds<f64>) -> Self {
        let split = |bound: Bound<&f64>| match bound {
            Bound::Included(&v) => (Some(v), None),
            Bound::Excluded(&v) => (None, Some(v)),
            Bound::Unbounded => (None, None),
        };
        let (min, exclusive_min) = split(range.start_bound());
        let (max, exclusive_max) = split(range.end_bound());
        Self {
            min,
            max,
            exclusive_min,
            exclusive_max,
        }
    }

    /// Writes the bounds into a refusal's fields: `min` and `max` always
    /// (`null` on a side that is open or unbounded), `exclusive_min` and
    /// `exclusive_max` only when that end is open.
    fn insert_into(self, fields: &mut serde_json::Value) {
        fields["min"] = json!(self.min);
        fields["max"] = json!(self.max);
        for (key, open) in [
            ("exclusive_min", self.exclusive_min),
            ("exclusive_max", self.exclusive_max),
        ] {
            if let Some(value) = open {
                fields[key] = json!(value);
            }
        }
    }

    /// The range in words, for the message: `in 0..=1`, `>= 1`, `> 0`.
    fn describe(self) -> String {
        let lower = match (self.min, self.exclusive_min) {
            (Some(v), _) => Some(format!(">= {v}")),
            (None, Some(v)) => Some(format!("> {v}")),
            (None, None) => None,
        };
        let upper = match (self.max, self.exclusive_max) {
            (Some(v), _) => Some(format!("<= {v}")),
            (None, Some(v)) => Some(format!("< {v}")),
            (None, None) => None,
        };
        match (self.min, self.max, lower, upper) {
            (Some(min), Some(max), _, _) => format!("in {min}..={max}"),
            (_, _, Some(lower), Some(upper)) => format!("{lower} and {upper}"),
            (_, _, Some(side), None) | (_, _, None, Some(side)) => side,
            (_, _, None, None) => "a number".to_string(),
        }
    }
}

/// GA settings the runner refused, named by the `ga_config` field it checked:
/// a number outside its range is `parameter_out_of_range` with the bounds as
/// the runner reported them (open ends as `exclusive_min` / `exclusive_max`),
/// a setting wrong only beside the others is `invalid_option`.
fn ga_settings_refused(e: ConfigError) -> WireError {
    let parameter = format!("ga_config.{}", e.parameter());
    let message = format!("ga_config.{e}");
    match e {
        ConfigError::OutOfRange {
            min,
            max,
            exclusive_min,
            exclusive_max,
            got,
            ..
        } => {
            let mut fields = json!({ "parameter": parameter, "got": got });
            Bounds {
                min,
                max,
                exclusive_min,
                exclusive_max,
            }
            .insert_into(&mut fields);
            WireError::new(Code::ParameterOutOfRange, message, fields)
        }
        _ => WireError::new(
            Code::InvalidOption,
            message,
            json!({ "parameter": parameter }),
        ),
    }
}

// ── whole numbers ───────────────────────────────────────────────────────────

/// A whole number exactly as the caller wrote it: any integer, or a number
/// with no fractional part.
///
/// Counts and seeds are read through this rather than straight into an
/// unsigned type. As a `usize`, `num_machines: -1` failed as a type mismatch
/// ("invalid value: integer `-1`, expected usize", `malformed_input`) while
/// `0` was refused by the range check -- two answers to one mistake. Read
/// signed and wide, every whole number reaches the range check, which names
/// the accepted range ([`whole_in`]). A fractional part is refused here:
/// whether a count is whole is a matter of its type, so `2.5` is
/// `malformed_input`.
struct Whole(i128);

impl<'de> Deserialize<'de> for Whole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WholeVisitor;

        impl serde::de::Visitor<'_> for WholeVisitor {
            type Value = Whole;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a whole number")
            }

            fn visit_i64<E>(self, v: i64) -> Result<Whole, E> {
                Ok(Whole(v.into()))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Whole, E> {
                Ok(Whole(v.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Whole, E> {
                if v.fract() == 0.0 {
                    // `as` saturates only past ±1.7e38, far outside every
                    // range a setting accepts.
                    Ok(Whole(v as i128))
                } else {
                    Err(E::custom(format_args!("must be a whole number, got {v}")))
                }
            }
        }

        deserializer.deserialize_any(WholeVisitor)
    }
}

/// Reads a whole-number field (`#[serde(deserialize_with = "whole")]`); see
/// [`Whole`].
fn whole<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i128, D::Error> {
    Whole::deserialize(deserializer).map(|w| w.0)
}

/// Reads an optional whole-number field: `null` is no value.
fn optional_whole<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i128>, D::Error> {
    Option::<Whole>::deserialize(deserializer).map(|w| w.map(|w| w.0))
}

/// A whole number from the request as the `T` the engine takes.
///
/// Below `min` -- a negative count as much as a too-small one -- it is
/// refused as `parameter_out_of_range` with `min` and no upper bound; above
/// the largest `T` (`max`, the type's own limit) with both bounds.
fn whole_in<T: TryFrom<i128>>(
    parameter: &str,
    value: i128,
    min: i128,
    max: i128,
) -> Result<T, WireError> {
    if value < min {
        return Err(WireError::out_of_range(
            parameter,
            min as f64..,
            value as f64,
        ));
    }
    T::try_from(value)
        .map_err(|_| WireError::out_of_range(parameter, min as f64..=max as f64, value as f64))
}

/// A count of at least `min`, as a `usize`; see [`whole_in`].
fn count(parameter: &str, value: i128, min: i128) -> Result<usize, WireError> {
    whole_in(parameter, value, min, usize::MAX as i128)
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// The checked problem's findings. The code and fields are the first one's:
/// they name one thing to fix, which is what a program branching on them can
/// act on. `errors` lists every finding the same way -- `code`, `message` and
/// its fields -- so a caller can point at all of them at once instead of
/// fixing one and re-running; the message joins their texts.
impl From<Vec<ValidationError>> for WireError {
    fn from(errors: Vec<ValidationError>) -> Self {
        let message = errors
            .iter()
            .map(|e| e.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        let Some(first) = errors.first() else {
            return WireError::new(
                Code::Internal,
                "internal: a problem refused with no finding".to_string(),
                json!({}),
            );
        };
        let findings: Vec<serde_json::Value> = errors
            .iter()
            .map(|e| {
                let mut finding = json!({ "code": e.kind.code(), "message": e.message });
                if let (Some(into), serde_json::Value::Object(from)) =
                    (finding.as_object_mut(), finding_fields(&e.kind))
                {
                    into.extend(from);
                }
                finding
            })
            .collect();
        let mut fields = finding_fields(&first.kind);
        if let Some(object) = fields.as_object_mut() {
            object.insert("errors".into(), serde_json::Value::Array(findings));
        }
        WireError::new(Code::of(&first.kind), message, fields)
    }
}

/// The values behind one finding: which entity, which setting, and the
/// numbers a range check compared.
fn finding_fields(kind: &ValidationErrorKind) -> serde_json::Value {
    match kind {
        ValidationErrorKind::DuplicateId { entity, id } => {
            json!({ "entity": entity.name(), "id": id })
        }
        ValidationErrorKind::InvalidResourceReference { activity, resource } => {
            json!({ "activity": activity, "resource": resource })
        }
        ValidationErrorKind::CyclicDependency { activity } => json!({ "activity": activity }),
        ValidationErrorKind::EmptyTask { task } => {
            json!({ "parameter": "operations", "task": task })
        }
        ValidationErrorKind::InvalidPredecessor {
            activity,
            predecessor,
        } => json!({ "activity": activity, "predecessor": predecessor }),
        // A weight must be > 0: zero itself is refused, so the bound is open.
        ValidationErrorKind::WeightOutOfRange { task, weight } => json!({
            "parameter": "weight", "task": task, "min": null, "max": null,
            "exclusive_min": 0.0, "got": weight
        }),
        ValidationErrorKind::NegativeDuration {
            activity,
            duration_ms,
        } => json!({
            "parameter": "processing_time", "activity": activity, "min": 0.0, "max": null,
            "got": ms_to_sec(*duration_ms),
        }),
        ValidationErrorKind::CapacityOutOfRange { resource, capacity } => json!({
            "parameter": "capacity", "resource": resource, "min": 1.0, "max": null,
            "got": capacity,
        }),
        ValidationErrorKind::TardinessWeightOutOfRange { weight } => json!({
            "parameter": "ga_config.tardiness_weight", "min": 0.0, "max": 1.0, "got": weight,
        }),
        // The wire input carries no calendars today; mapped so a future one reports it.
        ValidationErrorKind::ReversedCalendarPeriod {
            resource,
            period,
            index,
            start_ms,
            end_ms,
        } => json!({
            "parameter": format!("calendar.{}", period.name()), "resource": resource,
            "index": index, "start": ms_to_sec(*start_ms), "end": ms_to_sec(*end_ms),
        }),
    }
}

/// Every refusal crosses into JavaScript as an `Error` whose `message` is the
/// readable text and which carries `code` -- a stable reason -- and the values
/// behind it as further properties (`id`, `parameter`, `got`, ...). A program
/// branches on `err.code` and reads the fields; `err.message` reads as it
/// always did.
fn js_err(error: impl Into<WireError>) -> JsValue {
    let error = error.into();
    let js = js_sys::Error::new(&error.message);
    // `json_compatible` turns the map into a plain object; the default would
    // produce a JavaScript `Map`, which `Object.assign` does not read.
    if let Ok(fields) = error
        .fields
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    {
        js_sys::Object::assign(&js, &fields.into());
    }
    js.into()
}

/// Serializes a response; a failure is reported rather than unwrapped.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(|e| js_err(WireError::internal(e.to_string())))
}

/// A NaN or ±Infinity found in a JS argument, and where it sits.
///
/// JSON has no non-finite numbers, so on the way to the wire schema
/// `serde_json` turns one into `null` and the caller would be told a value has
/// the wrong type. [`find_non_finite`] looks before that happens, so the
/// refusal names the real reason and the place.
struct NonFinite {
    /// The argument's name, then `.key` and `[i]` steps down to the array or
    /// field that holds the number.
    parameter: String,
    /// The number's position, when it is an array element.
    index: Option<usize>,
    value: f64,
}

impl NonFinite {
    fn message(&self) -> String {
        let at = match self.index {
            Some(i) => format!("{}[{i}]", self.parameter),
            None => self.parameter.clone(),
        };
        let got = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        format!("{at}: expected a finite number, got {got}")
    }

    /// `parameter` and `index` (`null` when the number is not an array element).
    fn fields(&self) -> serde_json::Value {
        serde_json::json!({ "parameter": self.parameter, "index": self.index })
    }
}

/// The first NaN or ±Infinity in `value`, searching arrays, iterables and
/// plain objects. `allow_nan` lets NaN through for an input that reads it as a
/// missing value; it then arrives as `null`.
fn find_non_finite(value: &JsValue, parameter: &str, allow_nan: bool) -> Option<NonFinite> {
    let refused = |n: f64| !n.is_finite() && !(allow_nan && n.is_nan());
    let found = |index: Option<usize>, value: f64| NonFinite {
        parameter: parameter.to_string(),
        index,
        value,
    };
    if let Some(n) = value.as_f64() {
        return refused(n).then(|| found(None, n));
    }
    if !value.is_object() {
        return None;
    }
    if let Ok(Some(items)) = js_sys::try_iter(value) {
        for (i, item) in items.enumerate() {
            // An iterator that throws is left for serde to report.
            let item = item.ok()?;
            match item.as_f64() {
                Some(n) if refused(n) => return Some(found(Some(i), n)),
                Some(_) => {}
                None => {
                    let inner = find_non_finite(&item, &format!("{parameter}[{i}]"), allow_nan);
                    if inner.is_some() {
                        return inner;
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"), allow_nan);
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    let refuse = |message: String| js_err(WireError::malformed_input(param, message));
    if value.as_string().is_some() {
        return Err(refuse(format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    if let Some(found) = find_non_finite(&value, param, false) {
        return Err(js_err(WireError::new(
            Code::ValueNotFinite,
            found.message(),
            found.fields(),
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    from_json(json, param).map_err(js_err)
}

/// The half of [`from_js`] that enforces the wire schema, apart from the
/// `JsValue` (which cannot be built off `wasm32`) so tests walk the same path a
/// JS caller does. A refusal names where it stopped: see [`failure_site`].
fn from_json<T: serde::de::DeserializeOwned>(
    json: serde_json::Value,
    param: &str,
) -> Result<T, WireError> {
    serde_path_to_error::deserialize(json).map_err(|e| {
        let (parameter, index) = failure_site(param, e.path());
        let mut err =
            WireError::malformed_input(&parameter, format!("{param}: {}: {}", e.path(), e.inner()));
        if let Some(i) = index {
            err.fields["index"] = serde_json::json!(i);
        }
        err
    })
}

/// Where in the argument `param` a request stopped deserializing, as the
/// refusal reports it: the field (`design[1]`, `points[0].y`) and, when the
/// failure sits in an array, its position there -- the same `parameter` /
/// `index` a number array read element by element reports. Without it a
/// `null` three levels down was "invalid type: null, expected f64" with no
/// way to say which row (the gap `read_numbers` closed for bare arrays).
fn failure_site(param: &str, path: &serde_path_to_error::Path) -> (String, Option<usize>) {
    use serde_path_to_error::Segment;
    let segments: Vec<&Segment> = path.iter().collect();
    let index = segments.iter().rev().find_map(|s| match s {
        Segment::Seq { index } => Some(*index),
        _ => None,
    });
    // A trailing `[i]` is the index, not part of the name.
    let named = match segments.last() {
        Some(Segment::Seq { .. }) => &segments[..segments.len() - 1],
        _ => &segments[..],
    };
    let mut name = String::new();
    for segment in named {
        match segment {
            Segment::Seq { index } => name.push_str(&format!("[{index}]")),
            Segment::Map { key } | Segment::Enum { variant: key } => {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(key);
            }
            Segment::Unknown => name.push_str(".?"),
        }
    }
    // A top-level array argument (`data[1][2]`) or a failure at the root
    // (a missing field) is named by the argument itself.
    if name.is_empty() || name.starts_with('[') {
        name.insert_str(0, param);
    }
    (name, index)
}

// ══════════════════════════════════════════════════════════════════════════════
// run_schedule — dispatching rules (single / parallel machines)
// ══════════════════════════════════════════════════════════════════════════════

// ── input schema ─────────────────────────────────────────────────────────────

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct InputJob {
    id: String,
    processing_time: f64,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    due_date: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    release_time: Option<f64>,
    /// WSPT/ATC weight `w_j` (> 0, default 1): a heavier job goes earlier.
    #[serde(default = "default_weight")]
    #[tsify(optional)]
    weight: f64,
    /// Read by the `PRIORITY` rule only (higher goes first, default 0).
    #[serde(default)]
    #[tsify(optional)]
    priority: i32,
}

fn default_weight() -> f64 {
    1.0
}

fn default_rule() -> String {
    "SPT".to_string()
}

fn default_num_machines() -> i128 {
    1
}

/// What an omitted `config` means: the same values an empty `config: {}`
/// gets field by field. A derived `Default` would ignore the serde defaults
/// and give `rule: ""` and `num_machines: 0`.
impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            rule: default_rule(),
            num_machines: default_num_machines(),
            atc_k: None,
        }
    }
}

#[cfg(test)]
impl ScheduleConfig {
    fn default_for(rule: &str) -> Self {
        Self {
            rule: rule.to_string(),
            ..Self::default()
        }
    }
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct ScheduleConfig {
    #[serde(default = "default_rule")]
    #[tsify(optional)]
    #[tsify(
        type = "\"SPT\" | \"LPT\" | \"EDD\" | \"FCFS\" | \"CR\" | \"WSPT\" | \"MST\" | \"S/RO\" | \"SRO\" | \"ATC\" | \"LWKR\" | \"MWKR\" | \"PRIORITY\""
    )]
    rule: String,
    /// Number of identical parallel machines, at least 1 (default: 1 = single
    /// machine). Read signed so a negative count is a range refusal: [`Whole`].
    #[serde(default = "default_num_machines", deserialize_with = "whole")]
    #[tsify(optional, type = "number")]
    num_machines: i128,
    /// Lookahead parameter `k` of the ATC rule (> 0; default 2). Read by
    /// `ATC` only: given with any other rule it is refused (`invalid_option`)
    /// rather than ignored.
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    atc_k: Option<f64>,
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct ScheduleInput {
    jobs: Vec<InputJob>,
    #[serde(default)]
    #[tsify(optional)]
    config: ScheduleConfig,
}

// ── output schema ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, tsify::Tsify)]
struct OutputJob {
    id: String,
    start: f64,
    end: f64,
    tardiness: f64,
    /// Machine index (0-based). Always 0 for single-machine mode.
    machine: usize,
}

#[derive(Debug, Serialize, tsify::Tsify)]
struct MachineUtilization {
    machine: usize,
    busy_time: f64,
    utilization: f64,
}

#[derive(Debug, Serialize, tsify::Tsify)]
struct ScheduleOutput {
    schedule: Vec<OutputJob>,
    makespan: f64,
    total_tardiness: f64,
    /// Per-machine utilization. Present only when `num_machines > 1`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    machine_utilization: Vec<MachineUtilization>,
}

// ── conversion helpers ──────────────────────────────────────────────────────

/// Seconds -> milliseconds (i64), rounded to the nearest millisecond -- the
/// engine's time unit, so 0.0004 s is 0 ms and 0.0005 s is 1 ms (README).
fn sec_to_ms(secs: f64) -> i64 {
    (secs * 1_000.0).round() as i64
}

/// Milliseconds -> seconds (f64).
fn ms_to_sec(ms: i64) -> f64 {
    ms as f64 / 1_000.0
}

/// Build a `Task` with a single activity from an `InputJob`.
///
/// - `processing_time` (seconds) -> `ActivityDuration::fixed` (ms)
/// - `due_date` (seconds) -> `Task::deadline` (ms)
/// - `release_time` (seconds) -> `Task::release_time` (ms)
/// - `weight` -> `Task::weight`, `priority` -> `Task::priority`
fn build_task(job: &InputJob) -> Task {
    let duration_ms = sec_to_ms(job.processing_time);
    let activity = Activity::new(format!("{}_O1", job.id), &job.id, 0)
        .with_duration(ActivityDuration::fixed(duration_ms))
        .with_requirement(ResourceRequirement::new(MACHINE));

    let mut task = Task::new(&job.id)
        .with_priority(job.priority)
        .with_weight(job.weight)
        .with_activity(activity);

    if let Some(dd) = job.due_date {
        task.deadline = Some(sec_to_ms(dd));
    }
    if let Some(rt) = job.release_time {
        task.release_time = Some(sec_to_ms(rt));
    }

    task
}

/// Refuses a job `id` given to two jobs.
///
/// The output names jobs by `id` alone. A dispatching schedule would list two
/// entries under one name that the caller cannot tell apart, and the job-shop
/// solver keys its problem by task id, so one of the two jobs silently left
/// the schedule.
fn refuse_repeated_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<(), WireError> {
    let mut first_at: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (position, id) in ids.into_iter().enumerate() {
        if let Some(first) = first_at.insert(id, position) {
            return Err(WireError::new(
                Code::DuplicateId,
                format!(
                    "job {id:?}: the id is given twice, at positions {first} and {position} \
                     of jobs (counting from 0); the schedule names jobs by id, so every \
                     job needs its own"
                ),
                json!({ "entity": "job", "id": id, "first": first, "second": position }),
            ));
        }
    }
    Ok(())
}

/// Refuses a job value the schedule would otherwise use silently wrong: a
/// negative or non-finite processing time, or a weight that is not finite and
/// `> 0` (WSPT and ATC rank by `w / p`, so a zero or negative weight would
/// put the job where no weight puts it). Fields name the job by `index` and
/// `id`.
fn refuse_job_values(jobs: &[InputJob]) -> Result<(), WireError> {
    for (index, job) in jobs.iter().enumerate() {
        // A processing time of 0 is accepted (closed bound), a weight of 0 is
        // not (open bound).
        let (parameter, got, rule, bounds) =
            if !(job.processing_time.is_finite() && job.processing_time >= 0.0) {
                (
                    "processing_time",
                    job.processing_time,
                    "finite and >= 0",
                    Bounds::of(0.0..),
                )
            } else if !(job.weight.is_finite() && job.weight > 0.0) {
                (
                    "weight",
                    job.weight,
                    "finite and > 0",
                    Bounds::of((Bound::Excluded(0.0), Bound::Unbounded)),
                )
            } else {
                continue;
            };
        let mut fields = json!({
            "parameter": parameter, "got": got, "index": index, "id": job.id,
        });
        bounds.insert_into(&mut fields);
        return Err(WireError::new(
            Code::ParameterOutOfRange,
            format!(
                "job {:?} (position {index} of jobs, counting from 0): {parameter} must be {rule}, got {got}",
                job.id
            ),
            fields,
        ));
    }
    Ok(())
}

// ── rule selection ──────────────────────────────────────────────────────────

/// The rule names `build_engine` accepts, in the order the README lists them.
const RULES: [&str; 13] = [
    "SPT", "LPT", "EDD", "FCFS", "CR", "WSPT", "MST", "S/RO", "SRO", "ATC", "LWKR", "MWKR",
    "PRIORITY",
];

/// The settings a `run_schedule` request's `config` can carry besides `rule`.
/// Which of them a rule reads is [`rule_reads`].
const RULE_SETTINGS: [&str; 2] = ["num_machines", "atc_k"];

/// Whether `rule` reads `setting` (one of [`RULE_SETTINGS`]): the one table
/// both the refusal of a setting the rule does not read and the published
/// catalog ([`available_rules`]) come from. `num_machines` is read by every
/// rule, `atc_k` by `ATC` only.
fn rule_reads(rule: &str, setting: &str) -> bool {
    match setting {
        "num_machines" => true,
        "atc_k" => rule == "ATC",
        _ => false,
    }
}

/// One dispatching rule and the `config` settings it reads; a request giving
/// any other setting with this rule is refused (`invalid_option`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct RuleInfo {
    /// The rule, as a request's `config.rule` names it.
    #[tsify(
        type = "\"SPT\" | \"LPT\" | \"EDD\" | \"FCFS\" | \"CR\" | \"WSPT\" | \"MST\" | \"S/RO\" | \"ATC\" | \"LWKR\" | \"MWKR\" | \"PRIORITY\""
    )]
    name: String,
    /// The settings it reads, in [`RULE_SETTINGS`] order.
    #[tsify(type = "(\"num_machines\" | \"atc_k\")[]")]
    reads: Vec<String>,
}

/// Every rule, once each (`"SRO"` is another spelling of `"S/RO"`), with the
/// settings it reads.
fn rule_catalog() -> Vec<RuleInfo> {
    RULES
        .into_iter()
        .filter(|r| *r != "SRO")
        .map(|rule| RuleInfo {
            name: rule.to_string(),
            reads: RULE_SETTINGS
                .into_iter()
                .filter(|s| rule_reads(rule, s))
                .map(String::from)
                .collect(),
        })
        .collect()
}

/// Every dispatching rule `run_schedule` takes, with the `config` settings
/// each reads: `[{ name: "SPT", reads: ["num_machines"] }, ..., { name: "ATC",
/// reads: ["num_machines", "atc_k"] }, ...]`.
///
/// It is the table `run_schedule`'s refusal of a setting the rule does not
/// read (`invalid_option`) comes from, so a form can show only the settings
/// that change the chosen rule's run.
#[wasm_bindgen(unchecked_return_type = "RuleInfo[]")]
pub fn available_rules() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&rule_catalog())
        .map_err(|e| js_err(WireError::internal(e.to_string())))
}

// ── vocabularies ────────────────────────────────────────────────────────────

/// A refusal reason: the `code` an `Error` carries and the fields that can
/// stand beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct RefusalCode {
    code: String,
    fields: Vec<String>,
}

/// A violation a checker reports: its `code`, the fields beside it, and the
/// functions that report it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct ViolationCode {
    code: String,
    fields: Vec<String>,
    #[tsify(type = "\"check_jobshop\"[]")]
    checkers: Vec<String>,
}

/// What this package can answer with besides a result: every refusal `code`
/// and every checker violation `code`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct Codes {
    refusals: Vec<RefusalCode>,
    violations: Vec<ViolationCode>,
}

/// The vocabularies, from the tables the refusals and violations are built
/// from: [`Code`] and [`JobShopViolation::examples`].
fn code_catalog() -> Codes {
    let names = |fields: &[&str]| fields.iter().map(|f| f.to_string()).collect();
    Codes {
        refusals: Code::ALL
            .into_iter()
            .map(|c| RefusalCode {
                code: c.name().to_string(),
                fields: names(c.fields()),
            })
            .collect(),
        violations: JobShopViolation::examples()
            .iter()
            .map(|v| {
                let value = serde_json::to_value(v).expect("a violation serializes");
                let object = value.as_object().expect("a violation is an object");
                ViolationCode {
                    code: object["code"].as_str().expect("tagged").to_string(),
                    fields: object.keys().filter(|k| *k != "code").cloned().collect(),
                    checkers: vec!["check_jobshop".to_string()],
                }
            })
            .collect(),
    }
}

/// Every refusal `code` the functions of this package throw, with the fields
/// that can come with it, and every violation `code` `check_jobshop` reports:
/// `{ refusals: [{ code, fields }], violations: [{ code, fields, checkers }] }`.
///
/// It is built from the tables the refusals and violations come from, so a
/// caller that translates codes can check its list against the package's.
#[wasm_bindgen(unchecked_return_type = "Codes")]
pub fn codes() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&code_catalog())
        .map_err(|e| js_err(WireError::internal(e.to_string())))
}

/// Builds the rule engine `rule` names. A setting only some rules read
/// (`atc_k`) is refused when the chosen rule does not read it -- it used to
/// be ignored, so a caller tuning `k` under another rule saw no effect and
/// no error.
fn build_engine(rule: &str, config: &ScheduleConfig) -> Result<RuleEngine, WireError> {
    let engine = RuleEngine::new();
    let engine = match rule {
        "SPT" => engine.with_rule(rules::Spt),
        "LPT" => engine.with_rule(rules::Lpt),
        "EDD" => engine.with_rule(rules::Edd),
        "FCFS" => engine.with_rule(rules::Fifo),
        "CR" => engine.with_rule(rules::Cr),
        "WSPT" => engine.with_rule(rules::Wspt),
        "MST" => engine.with_rule(rules::Mst),
        "S/RO" | "SRO" => engine.with_rule(rules::Sro),
        "ATC" => engine.with_rule(atc_rule(config.atc_k)?),
        "LWKR" => engine.with_rule(rules::Lwkr),
        "MWKR" => engine.with_rule(rules::Mwkr),
        "PRIORITY" => engine.with_rule(rules::Priority),
        other => return Err(WireError::unknown_option("rule", other, &RULES)),
    };
    if config.atc_k.is_some() && !rule_reads(rule, "atc_k") {
        let readers: Vec<&str> = RULES
            .into_iter()
            .filter(|r| *r != "SRO" && rule_reads(r, "atc_k"))
            .collect();
        return Err(WireError::new(
            Code::InvalidOption,
            format!(
                "config.atc_k is read by {} -- rule \"{rule}\" does not read it; leave it out",
                readers.join(", ")
            ),
            json!({ "parameter": "config.atc_k", "rule": rule }),
        ));
    }
    Ok(engine)
}

/// The ATC rule with the given lookahead `k`, or the rule's default when it
/// is left out. `k` divides the slack, so it must be finite and `> 0`.
fn atc_rule(atc_k: Option<f64>) -> Result<rules::Atc, WireError> {
    match atc_k {
        None => Ok(rules::Atc::default()),
        Some(k) if k.is_finite() && k > 0.0 => Ok(rules::Atc::with_k(k)),
        Some(k) => Err(WireError::out_of_range(
            "config.atc_k",
            (Bound::Excluded(0.0), Bound::Unbounded),
            k,
        )),
    }
}

// ── simulation ──────────────────────────────────────────────────────────────

/// Resource type of the identical machines `run_schedule` dispatches on.
const MACHINE: &str = "Machine";

/// Dispatches the jobs on `num_machines` identical machines with the
/// scheduler every other path uses ([`SimpleScheduler`]): at each decision
/// the rule judges the jobs waiting then, on a context filled for that
/// moment, so a rule that reads the clock (CR, MST, S/RO, ATC) or the queue
/// (ATC's average) sees the state it is about.
///
/// [`SimpleScheduler`]: crate::scheduler::SimpleScheduler
fn dispatch(
    tasks: Vec<Task>,
    engine: RuleEngine,
    num_machines: usize,
) -> Result<Vec<OutputJob>, WireError> {
    let machines: Vec<Resource> = (0..num_machines)
        .map(|m| Resource::new(format!("M{m}"), MACHINE))
        .collect();
    let problem =
        Problem::new(tasks, machines).map_err(|e| WireError::internal(format!("{e:?}")))?;
    let schedule = crate::scheduler::SimpleScheduler::new()
        .with_rule_engine(engine)
        .schedule(&problem, 0);
    schedule
        .assignments
        .iter()
        .map(|a| {
            let task = problem
                .tasks()
                .iter()
                .find(|t| t.id == a.task_id)
                .expect("an assignment names a task of the problem");
            let machine = a.resource_id[1..]
                .parse()
                .expect("machines are named M<index>");
            Ok(OutputJob {
                id: task.id.clone(),
                start: ms_to_sec(a.start_ms),
                end: ms_to_sec(a.end_ms),
                tardiness: ms_to_sec(task.tardiness_ms(a.end_ms)),
                machine,
            })
        })
        .collect()
}

/// Compute per-machine utilization from a schedule and makespan.
fn compute_utilization(
    schedule: &[OutputJob],
    num_machines: usize,
    makespan: f64,
) -> Vec<MachineUtilization> {
    if makespan <= 0.0 {
        return Vec::new();
    }
    let mut busy = vec![0.0_f64; num_machines];
    for job in schedule {
        busy[job.machine] += job.end - job.start;
    }
    busy.iter()
        .enumerate()
        .map(|(i, &b)| MachineUtilization {
            machine: i,
            busy_time: b,
            utilization: b / makespan,
        })
        .collect()
}

// ── public API: run_schedule ────────────────────────────────────────────────

/// Run a dispatching schedule on single or parallel identical machines.
///
/// # Arguments
/// `jobs` -- A native JS object matching `ScheduleInput`.
///
/// # Returns
/// A JS object matching `ScheduleOutput` on success. A refusal throws an
/// `Error` carrying `code` (see the README's *Errors*).
///
/// # Backward Compatibility
/// When `config.num_machines` is omitted (defaults to 1), behavior is
/// identical to the original single-machine implementation.
#[wasm_bindgen(unchecked_return_type = "ScheduleOutput")]
pub fn run_schedule(
    #[wasm_bindgen(unchecked_param_type = "ScheduleInput")] jobs: JsValue,
) -> Result<JsValue, JsValue> {
    let input: ScheduleInput = from_js(jobs, "jobs")?;
    to_js(&schedule_jobs(&input).map_err(js_err)?)
}

/// The whole of `run_schedule` past the JS boundary: validate, then dispatch.
/// Tests call this, so they walk the path a JS caller does.
fn schedule_jobs(input: &ScheduleInput) -> Result<ScheduleOutput, WireError> {
    let num_machines = count("config.num_machines", input.config.num_machines, 1)?;
    let engine = build_engine(&input.config.rule, &input.config)?;
    if input.jobs.is_empty() {
        return Ok(ScheduleOutput {
            schedule: vec![],
            makespan: 0.0,
            total_tardiness: 0.0,
            machine_utilization: vec![],
        });
    }

    refuse_repeated_ids(input.jobs.iter().map(|j| j.id.as_str()))?;
    refuse_job_values(&input.jobs)?;
    let tasks: Vec<Task> = input.jobs.iter().map(build_task).collect();

    let schedule = dispatch(tasks, engine, num_machines)?;

    let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
    let total_tardiness: f64 = schedule.iter().map(|j| j.tardiness).sum();

    let machine_utilization = if num_machines > 1 {
        compute_utilization(&schedule, num_machines, makespan)
    } else {
        vec![]
    };

    Ok(ScheduleOutput {
        schedule,
        makespan,
        total_tardiness,
        machine_utilization,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// solve_jobshop — GA-based job-shop scheduling
// ══════════════════════════════════════════════════════════════════════════════

// ── input schema ────────────────────────────────────────────────────────────

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopOperation {
    /// Machine ID (e.g., "M1", "M2"). If multiple candidates, use an array.
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "string | null")]
    machine: Option<String>,
    /// Candidate machine IDs. Takes precedence over `machine`.
    #[serde(default)]
    #[tsify(optional)]
    machines: Vec<String>,
    /// Processing time in seconds.
    processing_time: f64,
}

impl JobShopOperation {
    /// Resolved candidate machine IDs.
    fn candidates(&self) -> Vec<String> {
        if !self.machines.is_empty() {
            self.machines.clone()
        } else if let Some(ref m) = self.machine {
            vec![m.clone()]
        } else {
            vec![]
        }
    }
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopJob {
    id: String,
    operations: Vec<JobShopOperation>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    due_date: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    release_time: Option<f64>,
}

fn default_population_size() -> i128 {
    100
}

fn default_max_generations() -> i128 {
    200
}

fn default_mutation_rate() -> f64 {
    0.1
}

fn default_tardiness_weight() -> f64 {
    0.5
}

fn default_crossover_type() -> String {
    "POX".to_string()
}

fn default_mutation_type() -> String {
    "Swap".to_string()
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopGaConfig {
    /// At least 2. Counts and the seed are read signed so a negative one is a
    /// range refusal: [`Whole`].
    #[serde(default = "default_population_size", deserialize_with = "whole")]
    #[tsify(optional, type = "number")]
    population_size: i128,
    /// At least 1.
    #[serde(default = "default_max_generations", deserialize_with = "whole")]
    #[tsify(optional, type = "number")]
    max_generations: i128,
    #[serde(default = "default_mutation_rate")]
    #[tsify(optional)]
    mutation_rate: f64,
    /// From 0 to 2^64 - 1.
    #[serde(default, deserialize_with = "optional_whole")]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    seed: Option<i128>,
    #[serde(default = "default_tardiness_weight")]
    #[tsify(optional)]
    tardiness_weight: f64,
    #[serde(default = "default_crossover_type")]
    #[tsify(optional)]
    #[tsify(type = "\"POX\" | \"LOX\" | \"JOX\"")]
    crossover: String,
    #[serde(default = "default_mutation_type")]
    #[tsify(optional)]
    #[tsify(type = "\"Swap\" | \"Insert\" | \"Invert\"")]
    mutation: String,
}

impl Default for JobShopGaConfig {
    fn default() -> Self {
        Self {
            population_size: default_population_size(),
            max_generations: default_max_generations(),
            mutation_rate: default_mutation_rate(),
            seed: None,
            tardiness_weight: default_tardiness_weight(),
            crossover: default_crossover_type(),
            mutation: default_mutation_type(),
        }
    }
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopInput {
    jobs: Vec<JobShopJob>,
    /// Number of machines. If omitted, inferred from operation machine IDs.
    #[serde(default, deserialize_with = "optional_whole")]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    num_machines: Option<i128>,
    #[serde(default)]
    #[tsify(optional)]
    ga_config: JobShopGaConfig,
}

// ── output schema ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize, tsify::Tsify)]
struct JobShopAssignment {
    job_id: String,
    /// Which step of its job this is, counting from 1 in the order the job's
    /// `operations` array lists them -- not the order they run on the timeline.
    operation: usize,
    machine: String,
    start: f64,
    end: f64,
}

/// How a job ends: when its last step finishes and how late that is.
#[derive(Debug, Serialize, tsify::Tsify)]
struct JobShopJobResult {
    id: String,
    /// When the job's last step ends (seconds).
    completion: f64,
    /// `max(0, completion − due_date)` in seconds; 0 without a `due_date`.
    tardiness: f64,
}

#[derive(Debug, Serialize, tsify::Tsify)]
struct JobShopOutput {
    schedule: Vec<JobShopAssignment>,
    /// Every job, in the order the request lists them -- the tardiness the
    /// search weighed (`ga_config.tardiness_weight`), per job.
    jobs: Vec<JobShopJobResult>,
    makespan: f64,
    /// The sum of `jobs[].tardiness`.
    total_tardiness: f64,
    fitness: f64,
    generations: usize,
    fitness_history: Vec<f64>,
    /// Why the search stopped: `"iteration_limit"` when it ran its
    /// `max_generations`, `"no_improvement"` when it stalled first;
    /// `"completed"` for a problem with no jobs.
    #[tsify(
        type = "\"completed\" | \"optimal\" | \"target_reached\" | \"iteration_limit\" | \"no_improvement\" | \"time_limit\" | \"cancelled\""
    )]
    stop_reason: StopReason,
}

// ── conversion helpers ──────────────────────────────────────────────────────

fn parse_crossover(s: &str) -> Result<CrossoverType, WireError> {
    match s {
        "POX" => Ok(CrossoverType::POX),
        "LOX" => Ok(CrossoverType::LOX),
        "JOX" => Ok(CrossoverType::JOX),
        other => Err(WireError::unknown_option(
            "ga_config.crossover",
            other,
            &["POX", "LOX", "JOX"],
        )),
    }
}

fn parse_mutation(s: &str) -> Result<MutationType, WireError> {
    match s {
        "Swap" => Ok(MutationType::Swap),
        "Insert" => Ok(MutationType::Insert),
        "Invert" => Ok(MutationType::Invert),
        other => Err(WireError::unknown_option(
            "ga_config.mutation",
            other,
            &["Swap", "Insert", "Invert"],
        )),
    }
}

// ── GA config validation ────────────────────────────────────────────────────

/// The whole-number GA settings, once [`validate_ga_config`] has accepted
/// them, as the types `GaConfig` takes.
#[derive(Debug, PartialEq)]
struct GaCounts {
    population_size: usize,
    max_generations: usize,
    seed: Option<u64>,
}

/// Validates `JobShopGaConfig` fields before they are passed to `GaConfig`.
///
/// This catches invalid values at the WASM boundary with clear error messages,
/// preventing panics deeper in the GA runner.
fn validate_ga_config(cfg: &JobShopGaConfig) -> Result<GaCounts, WireError> {
    let counts = GaCounts {
        population_size: count("ga_config.population_size", cfg.population_size, 2)?,
        max_generations: count("ga_config.max_generations", cfg.max_generations, 1)?,
        seed: cfg
            .seed
            .map(|seed| whole_in("ga_config.seed", seed, 0, u64::MAX.into()))
            .transpose()?,
    };
    if !(0.0..=1.0).contains(&cfg.mutation_rate) {
        return Err(WireError::out_of_range(
            "ga_config.mutation_rate",
            0.0..=1.0,
            cfg.mutation_rate,
        ));
    }
    if !(0.0..=1.0).contains(&cfg.tardiness_weight) {
        return Err(WireError::out_of_range(
            "ga_config.tardiness_weight",
            0.0..=1.0,
            cfg.tardiness_weight,
        ));
    }
    Ok(counts)
}

/// The checked scheduling problem a job-shop request describes: one task per
/// job, one activity per operation, one resource per machine id (named by the
/// operations, padded to `num_machines`).
fn jobshop_problem(input: &JobShopInput) -> Result<Problem, WireError> {
    refuse_repeated_ids(input.jobs.iter().map(|j| j.id.as_str()))?;

    // ── Collect all machine IDs ──
    let mut machine_ids: Vec<String> = Vec::new();
    for job in &input.jobs {
        for op in &job.operations {
            for m in op.candidates() {
                if !machine_ids.contains(&m) {
                    machine_ids.push(m);
                }
            }
        }
    }
    machine_ids.sort();

    // `num_machines` adds idle machines beyond the ones the operations name;
    // it cannot remove any of those.
    if let Some(n) = input.num_machines {
        let named = machine_ids.len().max(1);
        // A negative count is below the minimum like any other: one check.
        if n < named as i128 {
            return Err(WireError::new(
                Code::ParameterOutOfRange,
                format!(
                    "num_machines must be at least the {named} machine(s) the operations name, got {n}"
                ),
                json!({ "parameter": "num_machines", "min": named as f64, "max": null, "got": n as f64 }),
            ));
        }
        let n = count("num_machines", n, named as i128)?;
        let mut next = 1;
        while machine_ids.len() < n {
            let id = format!("M{next}");
            next += 1;
            if !machine_ids.contains(&id) {
                machine_ids.push(id);
            }
        }
    }

    if machine_ids.is_empty() {
        return Err(WireError::new(
            Code::NoMachines,
            "No machines specified in operations".to_string(),
            json!({}),
        ));
    }

    // ── Build domain Tasks ──
    let mut tasks = Vec::with_capacity(input.jobs.len());
    for job in &input.jobs {
        let mut task = Task::new(&job.id);

        if let Some(dd) = job.due_date {
            task.deadline = Some(sec_to_ms(dd));
        }
        if let Some(rt) = job.release_time {
            task.release_time = Some(sec_to_ms(rt));
        }

        for (i, op) in job.operations.iter().enumerate() {
            let candidates = op.candidates();
            if candidates.is_empty() {
                return Err(WireError::new(
                    Code::MissingMachine,
                    format!("Job '{}' operation {} has no machine specified", job.id, i),
                    json!({ "job": job.id, "operation": i }),
                ));
            }

            let activity = Activity::new(format!("{}_{}", job.id, i + 1), &job.id, i as i32)
                .with_duration(ActivityDuration::fixed(sec_to_ms(op.processing_time)))
                .with_requirement(ResourceRequirement::new("Machine").with_candidates(candidates));

            task = task.with_activity(activity);
        }

        tasks.push(task);
    }

    // ── Build Resources ──
    let resources: Vec<Resource> = machine_ids
        .iter()
        .map(|id| Resource::new(id, "Machine"))
        .collect();

    Ok(Problem::new(tasks, resources)?)
}

// ── public API: solve_jobshop ───────────────────────────────────────────────

/// Solve a job-shop scheduling problem using Genetic Algorithm.
///
/// # Arguments
/// `problem_json` -- A JS object matching `JobShopInput`.
///
/// # Returns
/// A JS object matching `JobShopOutput` on success. A refusal throws an
/// `Error` carrying `code` (see the README's *Errors*).
///
/// # Input Format
/// ```json
/// {
///   "jobs": [
///     {
///       "id": "J1",
///       "operations": [
///         { "machine": "M1", "processing_time": 3.0 },
///         { "machines": ["M2", "M3"], "processing_time": 2.0 }
///       ],
///       "due_date": 15.0,
///       "release_time": 0.0
///     }
///   ],
///   "num_machines": 3,
///   "ga_config": {
///     "population_size": 100,
///     "max_generations": 200,
///     "mutation_rate": 0.1,
///     "seed": 42,
///     "tardiness_weight": 0.5,
///     "crossover": "POX",
///     "mutation": "Swap"
///   }
/// }
/// ```
#[wasm_bindgen(unchecked_return_type = "JobShopOutput")]
pub fn solve_jobshop(
    #[wasm_bindgen(unchecked_param_type = "JobShopInput")] problem: JsValue,
) -> Result<JsValue, JsValue> {
    let input: JobShopInput = from_js(problem, "problem")?;
    to_js(&jobshop(&input).map_err(js_err)?)
}

/// The whole of `solve_jobshop` past the JS boundary. Tests call this, so
/// they walk the path a JS caller does.
fn jobshop(input: &JobShopInput) -> Result<JobShopOutput, WireError> {
    // The settings are refused or accepted whether or not there are jobs.
    let counts = validate_ga_config(&input.ga_config)?;
    let crossover_type = parse_crossover(&input.ga_config.crossover)?;
    let mutation_type = parse_mutation(&input.ga_config.mutation)?;

    if input.jobs.is_empty() {
        let output = JobShopOutput {
            schedule: vec![],
            jobs: vec![],
            makespan: 0.0,
            total_tardiness: 0.0,
            fitness: 0.0,
            generations: 0,
            fitness_history: vec![],
            stop_reason: StopReason::Completed,
        };
        return Ok(output);
    }

    let problem = jobshop_problem(input)?;

    // ── Build GA problem ──
    let ga_problem = SchedulingGaProblem::new(&problem)
        .with_tardiness_weight(input.ga_config.tardiness_weight)
        .map_err(|e| WireError::from(vec![e]))?
        .with_operators(GeneticOperators {
            crossover_type,
            mutation_type,
        });

    // ── Configure GA ──
    // Compute a safe elite_ratio: ensure at least 1 elite for any population_size.
    // Default 0.1 gives elite_count=0 when population_size < 10.
    let elite_ratio = {
        let pop = counts.population_size;
        let default_ratio = 0.1_f64;
        let min_ratio = 1.0 / pop as f64; // guarantees at least 1 elite
        default_ratio.max(min_ratio)
    };

    let mut config = GaConfig::default()
        .with_population_size(counts.population_size)
        .with_max_generations(counts.max_generations)
        .with_mutation_rate(input.ga_config.mutation_rate)
        .with_elite_ratio(elite_ratio)
        .with_parallel(false); // WASM is single-threaded

    if let Some(seed) = counts.seed {
        config = config.with_seed(seed);
    }

    // Defence-in-depth: call GaConfig's own validation as well.
    config.validate().map_err(ga_settings_refused)?;

    // ── Run GA ──
    let result = GaRunner::run(&ga_problem, &config).map_err(ga_settings_refused)?;

    // ── Decode best solution ──
    let best_schedule = ga_problem.decode(&result.best);

    let schedule: Vec<JobShopAssignment> = best_schedule
        .assignments
        .iter()
        .map(|a| {
            // The schedule says which step this is; it used to be recovered by
            // parsing the trailing number out of `activity_id`, with a fallback
            // of 1 -- so when the decoder stopped putting an activity id there,
            // every row silently read "operation 1".
            let operation = a.sequence.ok_or_else(|| {
                WireError::new(
                    Code::Internal,
                    format!(
                        "internal: assignment for job '{}' on '{}' carries no operation number",
                        a.task_id, a.resource_id
                    ),
                    json!({}),
                )
            })? as usize;

            Ok(JobShopAssignment {
                job_id: a.task_id.clone(),
                operation,
                machine: a.resource_id.clone(),
                start: ms_to_sec(a.start_ms),
                end: ms_to_sec(a.end_ms),
            })
        })
        .collect::<Result<_, WireError>>()?;

    let makespan = ms_to_sec(best_schedule.makespan_ms());

    // Completion and tardiness by the definition the fitness weighed.
    let mut total_tardiness_ms = 0;
    let jobs = problem
        .tasks()
        .iter()
        .map(|task| {
            let completion_ms = best_schedule
                .task_completion_time(&task.id)
                .ok_or_else(|| {
                    WireError::new(
                        Code::Internal,
                        format!("internal: job '{}' has no step in the schedule", task.id),
                        json!({}),
                    )
                })?;
            let tardiness_ms = task.tardiness_ms(completion_ms);
            total_tardiness_ms += tardiness_ms;
            Ok(JobShopJobResult {
                id: task.id.clone(),
                completion: ms_to_sec(completion_ms),
                tardiness: ms_to_sec(tardiness_ms),
            })
        })
        .collect::<Result<_, WireError>>()?;

    let output = JobShopOutput {
        schedule,
        jobs,
        makespan,
        total_tardiness: ms_to_sec(total_tardiness_ms),
        fitness: result.best_fitness,
        generations: result.generations,
        fitness_history: result.fitness_history,
        stop_reason: result.stop_reason,
    };

    Ok(output)
}

// ── checking a job-shop schedule ────────────────────────────────────────────

/// One placed step of a schedule to check -- an entry of a `solve_jobshop`
/// answer's `schedule`. `operation` is read as given: a number the job has no
/// step for (0, negative, past its last) is a violation, not a refusal.
#[derive(Debug, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopStep {
    job_id: String,
    #[serde(deserialize_with = "whole")]
    #[tsify(type = "number")]
    operation: i128,
    machine: String,
    start: f64,
    end: f64,
}

/// A schedule to check: the `schedule` of a `solve_jobshop` answer, or one
/// built elsewhere in its shape, and the `makespan` it reports, if any.
#[derive(Debug, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopPlan {
    schedule: Vec<JobShopStep>,
    /// The makespan the schedule claims, compared with its latest end.
    #[serde(default)]
    #[tsify(optional, type = "number | null")]
    makespan: Option<f64>,
}

/// The input of `check_jobshop`: the problem as given to `solve_jobshop`,
/// unchanged, and the schedule to check against it.
#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopCheckInput {
    problem: JobShopInput,
    solution: JobShopPlan,
}

/// One way a schedule breaks its problem, by `code`. `entity` is the step
/// (`"J1_2"` -- job and step number), job or machine it is about.
#[derive(Debug, Serialize, tsify::Tsify)]
#[serde(tag = "code", rename_all = "snake_case")]
enum JobShopViolation {
    /// A job or step the problem does not have, or a step placed twice.
    UnknownActivity { entity: String, message: String },
    /// A step left out, or put on a machine it cannot run on.
    RequirementUnfilled { entity: String, message: String },
    /// Two steps on one machine at once.
    CapacityExceeded { entity: String, message: String },
    /// A step that starts before the step ahead of it ends.
    PrecedenceViolation { entity: String, message: String },
    /// A step shorter than its processing time.
    DurationTooShort { entity: String, message: String },
    /// A step that starts before its job's `release_time`.
    EarlyStart { entity: String, message: String },
    /// A job that ends after its `due_date`.
    DeadlineMiss { entity: String, message: String },
    /// A step on a machine while the machine is unavailable.
    ResourceUnavailable { entity: String, message: String },
    /// A step outside its time window.
    TimeWindowViolation { entity: String, message: String },
    /// A reported `makespan` other than the schedule's latest end.
    MakespanMismatch { reported: f64, actual: f64 },
}

impl JobShopViolation {
    /// One of each violation, for the published vocabulary ([`codes`]): its
    /// code and fields are read off these, so they are the structs'.
    fn examples() -> Vec<Self> {
        let at = || (String::new(), String::new());
        let ((e0, m0), (e1, m1), (e2, m2)) = (at(), at(), at());
        let ((e3, m3), (e4, m4), (e5, m5)) = (at(), at(), at());
        let ((e6, m6), (e7, m7), (e8, m8)) = (at(), at(), at());
        vec![
            Self::UnknownActivity {
                entity: e0,
                message: m0,
            },
            Self::RequirementUnfilled {
                entity: e1,
                message: m1,
            },
            Self::CapacityExceeded {
                entity: e2,
                message: m2,
            },
            Self::PrecedenceViolation {
                entity: e3,
                message: m3,
            },
            Self::DurationTooShort {
                entity: e4,
                message: m4,
            },
            Self::EarlyStart {
                entity: e5,
                message: m5,
            },
            Self::DeadlineMiss {
                entity: e6,
                message: m6,
            },
            Self::ResourceUnavailable {
                entity: e7,
                message: m7,
            },
            Self::TimeWindowViolation {
                entity: e8,
                message: m8,
            },
            Self::MakespanMismatch {
                reported: 0.0,
                actual: 0.0,
            },
        ]
    }

    /// The wire violation for the library's own: job-shop problems have no
    /// skills, synchronized steps or custom checks, so those do not arise.
    fn of(v: Violation) -> Self {
        let (entity, message) = (v.entity_id, v.message);
        match v.violation_type {
            ViolationType::UnknownActivity => Self::UnknownActivity { entity, message },
            ViolationType::RequirementUnfilled | ViolationType::SkillMismatch => {
                Self::RequirementUnfilled { entity, message }
            }
            ViolationType::CapacityExceeded => Self::CapacityExceeded { entity, message },
            ViolationType::PrecedenceViolation | ViolationType::SynchronizeViolation => {
                Self::PrecedenceViolation { entity, message }
            }
            ViolationType::DurationTooShort => Self::DurationTooShort { entity, message },
            ViolationType::EarlyStart => Self::EarlyStart { entity, message },
            ViolationType::DeadlineMiss => Self::DeadlineMiss { entity, message },
            ViolationType::ResourceUnavailable | ViolationType::Custom(_) => {
                Self::ResourceUnavailable { entity, message }
            }
            ViolationType::TimeWindowViolation => Self::TimeWindowViolation { entity, message },
        }
    }
}

/// Whether a schedule keeps its problem, and every way it does not.
#[derive(Debug, Serialize, tsify::Tsify)]
struct JobShopCheckOutput {
    feasible: bool,
    violations: Vec<JobShopViolation>,
}

/// Checks a job-shop schedule against the problem it answers -- each step of
/// each job placed once, on a machine it can run on, for its processing time,
/// after the step ahead of it and the job's release, no two steps on one
/// machine at once, and the reported `makespan` equal to the latest end.
///
/// `problem` is read as `solve_jobshop` reads it (the same refusals);
/// `solution` is `{ schedule, makespan? }` -- the schedule of an answer, so a
/// caller passes `{ schedule: answer.schedule, makespan: answer.makespan }`.
/// A schedule that breaks the problem is a result, not an error.
#[wasm_bindgen(unchecked_return_type = "JobShopCheckOutput")]
pub fn check_jobshop(
    #[wasm_bindgen(unchecked_param_type = "JobShopCheckInput")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: JobShopCheckInput = from_js(input, "input")?;
    to_js(&check_jobshop_input(&input).map_err(js_err)?)
}

/// The whole of `check_jobshop` past the JS boundary.
fn check_jobshop_input(input: &JobShopCheckInput) -> Result<JobShopCheckOutput, WireError> {
    let problem = jobshop_problem(&input.problem)?;

    // Each (job, step) of the problem, by the activity it is.
    let mut activity_of: std::collections::HashMap<(&str, usize), &str> =
        std::collections::HashMap::new();
    for task in problem.tasks() {
        for activity in &task.activities {
            let step = usize::try_from(activity.sequence).unwrap_or(usize::MAX);
            activity_of.insert((task.id.as_str(), step + 1), activity.id.as_str());
        }
    }

    let mut violations = Vec::new();
    let mut schedule = Schedule::new();
    for a in &input.solution.schedule {
        let step = usize::try_from(a.operation).ok();
        let Some(activity_id) = step.and_then(|step| activity_of.get(&(a.job_id.as_str(), step)))
        else {
            violations.push(JobShopViolation::UnknownActivity {
                entity: a.job_id.clone(),
                message: format!(
                    "job '{}' has no operation {} in the problem",
                    a.job_id, a.operation
                ),
            });
            continue;
        };
        let mut assignment = Assignment::new(
            *activity_id,
            &a.job_id,
            &a.machine,
            sec_to_ms(a.start),
            sec_to_ms(a.end),
        );
        assignment.sequence = i32::try_from(a.operation).ok();
        schedule.assignments.push(assignment);
    }

    let feasibility = FeasibilityInput {
        tasks: problem.tasks(),
        resources: problem.resources(),
        constraints: &[],
    };
    violations.extend(
        check_schedule(&schedule, &feasibility)
            .into_iter()
            .map(JobShopViolation::of),
    );

    if let Some(reported) = input.solution.makespan {
        let latest = input
            .solution
            .schedule
            .iter()
            .map(|a| a.end)
            .fold(0.0_f64, f64::max);
        if sec_to_ms(reported) != sec_to_ms(latest) {
            violations.push(JobShopViolation::MakespanMismatch {
                reported,
                actual: latest,
            });
        }
    }

    Ok(JobShopCheckOutput {
        feasible: violations.is_empty(),
        violations,
    })
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;

    /// The published vocabulary is what the binding answers with: every
    /// violation has an example (a new variant fails to compile here until it
    /// has one), and the fields listed are the ones it serializes.
    #[test]
    fn the_code_catalog_covers_every_violation() {
        let examples = JobShopViolation::examples();
        let mut seen = [false; 10];
        for v in &examples {
            let i = match v {
                JobShopViolation::UnknownActivity { .. } => 0,
                JobShopViolation::RequirementUnfilled { .. } => 1,
                JobShopViolation::CapacityExceeded { .. } => 2,
                JobShopViolation::PrecedenceViolation { .. } => 3,
                JobShopViolation::DurationTooShort { .. } => 4,
                JobShopViolation::EarlyStart { .. } => 5,
                JobShopViolation::DeadlineMiss { .. } => 6,
                JobShopViolation::ResourceUnavailable { .. } => 7,
                JobShopViolation::TimeWindowViolation { .. } => 8,
                JobShopViolation::MakespanMismatch { .. } => 9,
            };
            assert!(!seen[i], "two examples of {v:?}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|s| *s), "a violation without an example");

        let catalog = code_catalog();
        // The one violation about the report rather than an entity; serde_json
        // orders an object's keys by name.
        let tag = serde_json::to_value(JobShopViolation::MakespanMismatch {
            reported: 0.0,
            actual: 0.0,
        })
        .expect("serializes")["code"]
            .clone();
        let mismatch = catalog
            .violations
            .iter()
            .find(|v| tag == v.code.as_str())
            .expect("listed");
        assert_eq!(mismatch.fields, ["actual", "reported"]);
        let late = catalog
            .violations
            .iter()
            .find(|v| v.code == "deadline_miss")
            .expect("listed");
        assert_eq!(late.fields, ["entity", "message"]);
        assert!(catalog
            .violations
            .iter()
            .all(|v| v.checkers == ["check_jobshop"]));
        let refusals: Vec<&str> = catalog.refusals.iter().map(|r| r.code.as_str()).collect();
        assert_eq!(refusals.len(), Code::ALL.len());
        assert!(refusals.contains(&"parameter_out_of_range"));
    }

    /// Every reason the problem check names is a code of this binding.
    #[test]
    fn every_problem_check_reason_is_a_listed_code() {
        use crate::validation::Entity;
        let kinds = [
            ValidationErrorKind::DuplicateId {
                entity: Entity::Task,
                id: String::new(),
            },
            ValidationErrorKind::EmptyTask {
                task: String::new(),
            },
            ValidationErrorKind::CyclicDependency {
                activity: String::new(),
            },
            ValidationErrorKind::InvalidResourceReference {
                activity: String::new(),
                resource: String::new(),
            },
            ValidationErrorKind::InvalidPredecessor {
                activity: String::new(),
                predecessor: String::new(),
            },
            ValidationErrorKind::TardinessWeightOutOfRange { weight: 2.0 },
        ];
        for kind in &kinds {
            assert_eq!(Code::of(kind).name(), kind.code());
        }
    }

    /// The README's code tables list exactly the published vocabulary, so the
    /// tables and `codes()` cannot drift apart.
    #[test]
    fn the_readme_documents_every_code_and_no_other() {
        let readme = include_str!("../README.md");
        let mut documented = std::collections::BTreeSet::new();
        let mut in_table = false;
        for line in readme.lines() {
            if line.starts_with("| `code` |") {
                in_table = true;
                continue;
            }
            if !line.starts_with('|') {
                in_table = false;
                continue;
            }
            if in_table && !line.starts_with("|---") {
                let first = line.split('|').nth(1).expect("a cell");
                for code in first.split('·') {
                    documented.insert(code.trim().trim_matches('`').to_string());
                }
            }
        }
        let catalog = code_catalog();
        let published: std::collections::BTreeSet<String> = catalog
            .refusals
            .iter()
            .map(|r| r.code.clone())
            .chain(catalog.violations.iter().map(|v| v.code.clone()))
            .collect();
        assert_eq!(documented, published);
    }

    fn make_input_job(
        id: &str,
        pt: f64,
        due: Option<f64>,
        release: Option<f64>,
        weight: f64,
    ) -> InputJob {
        InputJob {
            id: id.to_string(),
            processing_time: pt,
            due_date: due,
            release_time: release,
            weight,
            priority: 0,
        }
    }

    fn run(jobs: Vec<InputJob>, rule: &str) -> (Vec<OutputJob>, f64, f64) {
        let input = ScheduleInput {
            jobs,
            config: ScheduleConfig::default_for(rule),
        };
        let out = schedule_jobs(&input).expect("valid input");
        (out.schedule, out.makespan, out.total_tardiness)
    }

    fn order(schedule: &[OutputJob]) -> Vec<&str> {
        schedule.iter().map(|j| j.id.as_str()).collect()
    }

    /// Smith's rule orders by p/w ascending. Before the fix the
    /// binding stored w as `priority = 1000w` and WSPT read `1000/(priority+1)`
    /// (about 1/w) back, so every unequal pair came out reversed.
    #[test]
    fn wspt_and_atc_follow_smiths_rule() {
        for rule in ["WSPT", "ATC"] {
            let heavy_a = vec![
                make_input_job("A", 4.0, None, Some(0.0), 10.0),
                make_input_job("B", 2.0, None, Some(0.0), 1.0),
            ];
            assert_eq!(order(&run(heavy_a, rule).0), ["A", "B"], "{rule}");
            let heavy_b = vec![
                make_input_job("A", 4.0, None, Some(0.0), 1.0),
                make_input_job("B", 2.0, None, Some(0.0), 10.0),
            ];
            assert_eq!(order(&run(heavy_b, rule).0), ["B", "A"], "{rule}");
            let equal = vec![
                make_input_job("A", 4.0, None, Some(0.0), 1.0),
                make_input_job("B", 2.0, None, Some(0.0), 1.0),
            ];
            assert_eq!(order(&run(equal, rule).0), ["B", "A"], "{rule} = SPT");
        }
        // A weight below 1/1000 used to round to priority 0, the most
        // important; it is now the least.
        let tiny = vec![
            make_input_job("A", 4.0, None, Some(0.0), 0.0001),
            make_input_job("B", 4.0, None, Some(0.0), 1.0),
        ];
        assert_eq!(order(&run(tiny, "WSPT").0), ["B", "A"]);
    }

    #[test]
    fn a_weight_that_is_not_positive_is_refused_with_its_job() {
        for bad in [0.0, -1.0] {
            let jobs = vec![
                make_input_job("A", 4.0, None, None, 1.0),
                make_input_job("B", 2.0, None, None, bad),
            ];
            let err = schedule_jobs(&ScheduleInput {
                jobs,
                config: ScheduleConfig::default_for("WSPT"),
            })
            .expect_err("weight must be > 0");
            assert_eq!(
                err.fields,
                json!({
                    "code": "parameter_out_of_range",
                    "parameter": "weight", "min": null, "max": null, "exclusive_min": 0.0,
                    "got": bad, "index": 1, "id": "B",
                })
            );
        }
    }

    /// The model's own check reports a task weight the same way: `> 0` is an
    /// open bound.
    #[test]
    fn a_task_weight_finding_reports_its_bound_as_open() {
        let fields = finding_fields(&ValidationErrorKind::WeightOutOfRange {
            task: "T".to_string(),
            weight: 0.0,
        });
        assert_eq!(
            fields,
            json!({
                "parameter": "weight", "task": "T", "min": null, "max": null,
                "exclusive_min": 0.0, "got": 0.0,
            })
        );
    }

    /// A processing time of 0 is accepted, so its bound is closed: `min: 0`
    /// and no `exclusive_min`, unlike the weight beside it.
    #[test]
    fn a_negative_processing_time_is_refused() {
        let err = schedule_jobs(&ScheduleInput {
            jobs: vec![make_input_job("A", -1.0, None, None, 1.0)],
            config: ScheduleConfig::default_for("SPT"),
        })
        .expect_err("negative processing time");
        assert_eq!(
            err.fields,
            json!({
                "code": "parameter_out_of_range",
                "parameter": "processing_time", "min": 0.0, "max": null, "got": -1.0,
                "index": 0, "id": "A",
            })
        );
    }

    /// `num_machines: 0` used to become 1 without a word.
    #[test]
    fn zero_machines_is_refused_even_without_jobs() {
        for jobs in [vec![], vec![make_input_job("A", 1.0, None, None, 1.0)]] {
            let mut config = ScheduleConfig::default_for("SPT");
            config.num_machines = 0;
            let err = schedule_jobs(&ScheduleInput { jobs, config }).expect_err("0 machines");
            assert_eq!(
                err.fields,
                json!({
                    "code": "parameter_out_of_range",
                    "parameter": "config.num_machines", "min": 1.0, "max": null, "got": 0.0,
                })
            );
        }
    }

    /// The catalog and the refusal are one table: every rule once, and
    /// `atc_k` accepted exactly by the rules the catalog says read it.
    #[test]
    fn the_rule_catalog_is_what_run_schedule_accepts() {
        let catalog = rule_catalog();
        let mut names: Vec<&str> = catalog.iter().map(|r| r.name.as_str()).collect();
        names.push("SRO");
        names.sort_unstable();
        let mut rules = RULES.to_vec();
        rules.sort_unstable();
        assert_eq!(names, rules, "every rule once, SRO as a spelling of S/RO");
        let jobs = || vec![make_input_job("A", 2.0, Some(10.0), None, 1.0)];
        for info in &catalog {
            let mut config = ScheduleConfig::default_for(&info.name);
            config.num_machines = 2;
            config.atc_k = Some(3.0);
            let run = schedule_jobs(&ScheduleInput {
                jobs: jobs(),
                config,
            });
            if info.reads.iter().any(|r| r == "atc_k") {
                assert!(run.is_ok(), "{} refused atc_k it reads", info.name);
            } else {
                assert_eq!(run.expect_err(&info.name).code(), "invalid_option");
            }
            assert!(
                info.reads.iter().any(|r| r == "num_machines"),
                "{}",
                info.name
            );
        }
    }

    /// `atc_k` is read by ATC only. Under any other rule it used to be
    /// ignored without a word; now it is refused, naming the rule.
    #[test]
    fn atc_k_is_refused_by_a_rule_that_does_not_read_it() {
        let jobs = || vec![make_input_job("A", 2.0, Some(10.0), None, 1.0)];
        for rule in ["SPT", "EDD", "WSPT", "PRIORITY"] {
            let mut config = ScheduleConfig::default_for(rule);
            config.atc_k = Some(3.0);
            let err = schedule_jobs(&ScheduleInput {
                jobs: jobs(),
                config,
            })
            .expect_err("atc_k under a rule that does not read it");
            assert_eq!(
                err.fields,
                json!({ "code": "invalid_option", "parameter": "config.atc_k", "rule": rule }),
                "{rule}"
            );
            assert_eq!(
                err.message,
                format!(
                    "config.atc_k is read by ATC -- rule \"{rule}\" does not read it; leave it out"
                )
            );
        }
        // Refused even with no jobs: the setting is wrong whatever the input.
        let mut config = ScheduleConfig::default_for("SPT");
        config.atc_k = Some(3.0);
        let err = schedule_jobs(&ScheduleInput {
            jobs: vec![],
            config,
        })
        .expect_err("atc_k under SPT, no jobs");
        assert_eq!(err.code(), "invalid_option");
    }

    /// Under ATC the given `k` is used; left out, the default applies; a
    /// `k` that is not `> 0` is a range refusal with an open lower bound.
    #[test]
    fn atc_reads_atc_k_and_refuses_one_that_is_not_positive() {
        for atc_k in [None, Some(0.5), Some(4.0)] {
            let mut config = ScheduleConfig::default_for("ATC");
            config.atc_k = atc_k;
            let out = schedule_jobs(&ScheduleInput {
                jobs: vec![
                    make_input_job("A", 2.0, Some(10.0), None, 1.0),
                    make_input_job("B", 1.0, Some(3.0), None, 1.0),
                ],
                config,
            })
            .expect("ATC reads atc_k");
            assert_eq!(out.schedule.len(), 2, "{atc_k:?}");
        }
        let input: ScheduleInput = serde_json::from_value(json!({
            "jobs": [{ "id": "A", "processing_time": 1.0 }],
            "config": { "rule": "ATC", "atc_k": 1.5 },
        }))
        .expect("atc_k is an optional number");
        assert_eq!(input.config.atc_k, Some(1.5));

        for bad in [0.0, -1.0] {
            let mut config = ScheduleConfig::default_for("ATC");
            config.atc_k = Some(bad);
            let err = schedule_jobs(&ScheduleInput {
                jobs: vec![make_input_job("A", 1.0, None, None, 1.0)],
                config,
            })
            .expect_err("k must be > 0");
            assert_eq!(
                err.fields,
                json!({
                    "code": "parameter_out_of_range", "parameter": "config.atc_k",
                    "min": null, "max": null, "exclusive_min": 0.0, "got": bad,
                })
            );
        }
    }

    /// An omitted `config` and an empty one mean the same: SPT on one machine.
    #[test]
    fn an_omitted_config_is_the_documented_defaults() {
        for input in [
            json!({ "jobs": [{ "id": "A", "processing_time": 2.0 }, { "id": "B", "processing_time": 1.0 }] }),
            json!({ "jobs": [{ "id": "A", "processing_time": 2.0 }, { "id": "B", "processing_time": 1.0 }], "config": {} }),
        ] {
            let input: ScheduleInput = serde_json::from_value(input).expect("valid input");
            let out = schedule_jobs(&input).expect("defaults schedule");
            assert_eq!(order(&out.schedule), ["B", "A"]);
            assert!(out.machine_utilization.is_empty());
        }
    }

    #[test]
    fn an_unknown_rule_is_refused_even_without_jobs() {
        let err = schedule_jobs(&ScheduleInput {
            jobs: vec![],
            config: ScheduleConfig::default_for("FIFO"),
        })
        .expect_err("FIFO is spelled FCFS");
        assert_eq!(err.code(), "unknown_option");
    }

    // ── existing tests (backward compatibility) ──

    #[test]
    fn test_spt_order() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
            make_input_job("C", 8.0, None, None, 1.0),
        ];
        let (schedule, makespan, _) = run(jobs, "SPT");
        // B(2) -> A(5) -> C(8)
        assert_eq!(schedule[0].id, "B");
        assert_eq!(schedule[1].id, "A");
        assert_eq!(schedule[2].id, "C");
        assert!((makespan - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_edd_order() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(10.0), None, 1.0),
            make_input_job("B", 3.0, Some(5.0), None, 1.0),
            make_input_job("C", 3.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "EDD");
        assert_eq!(schedule[0].id, "B");
        assert_eq!(schedule[1].id, "A");
        assert_eq!(schedule[2].id, "C");
    }

    #[test]
    fn test_lpt_order() {
        let jobs = vec![
            make_input_job("S", 1.0, None, None, 1.0),
            make_input_job("L", 9.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "LPT");
        assert_eq!(schedule[0].id, "L");
    }

    #[test]
    fn test_tardiness_computed() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(2.0), None, 1.0),
            make_input_job("B", 3.0, Some(10.0), None, 1.0),
        ];
        let (schedule, _, total_tardiness) = run(jobs, "SPT");
        let a = schedule.iter().find(|j| j.id == "A").expect("A");
        assert!((a.tardiness - 1.0).abs() < 1e-9);
        let b = schedule.iter().find(|j| j.id == "B").expect("B");
        assert!((b.tardiness - 0.0).abs() < 1e-9);
        assert!((total_tardiness - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_release_time_respected() {
        let jobs = vec![
            make_input_job("A", 2.0, None, Some(5.0), 1.0),
            make_input_job("B", 3.0, None, Some(0.0), 1.0),
        ];
        let (schedule, makespan, _) = run(jobs, "SPT");
        let a = schedule.iter().find(|j| j.id == "A").expect("A");
        assert!(
            (a.start - 5.0).abs() < 1e-9,
            "A start should be 5.0 (release respected)"
        );
        assert!((a.end - 7.0).abs() < 1e-9);
        // B waits from 0 and A is not released before B would finish, so the
        // machine runs B first instead of idling until 5 for the shorter A.
        let b = schedule.iter().find(|j| j.id == "B").expect("B");
        assert!((b.start - 0.0).abs() < 1e-9, "B start {}", b.start);
        assert!((makespan - 7.0).abs() < 1e-9, "makespan {makespan}");
    }

    #[test]
    fn test_empty_jobs() {
        let (schedule, makespan, tardiness) = run(vec![], "SPT");
        assert!(schedule.is_empty());
        assert!((makespan - 0.0).abs() < 1e-9);
        assert!((tardiness - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_unknown_rule_error() {
        let config = ScheduleConfig {
            rule: "UNKNOWN".to_string(),
            num_machines: 1,
            atc_k: None,
        };
        assert!(build_engine("UNKNOWN", &config).is_err());
    }

    #[test]
    fn test_wspt_weight() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "WSPT");
        assert_eq!(
            schedule[0].id, "B",
            "equal weight: shorter job goes first (SPT-like)"
        );
    }

    // ── new rule tests ──

    #[test]
    fn test_mst_rule() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(10.0), None, 1.0),
            make_input_job("B", 3.0, Some(5.0), None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "MST");
        // B has tighter deadline → less slack → higher priority
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_atc_rule() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(20.0), None, 1.0),
            make_input_job("B", 3.0, Some(4.0), None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "ATC");
        // B has very tight deadline → urgency boost → scheduled first
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_lwkr_rule() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "LWKR");
        // LWKR prefers least remaining work → B first (shorter)
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_mwkr_rule() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "MWKR");
        // MWKR prefers most remaining work → A first (longer)
        assert_eq!(schedule[0].id, "A");
    }

    #[test]
    fn test_priority_rule() {
        // PRIORITY reads `priority` (higher first) and ignores `weight`.
        let mut a = make_input_job("A", 3.0, None, None, 10.0);
        let mut b = make_input_job("B", 3.0, None, None, 1.0);
        a.priority = 1;
        b.priority = 5;
        let (schedule, _, _) = run(vec![a, b], "PRIORITY");
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_sro_alias() {
        // "SRO" alias for "S/RO" should work
        let config = ScheduleConfig {
            rule: "SRO".to_string(),
            num_machines: 1,
            atc_k: None,
        };
        assert!(build_engine("SRO", &config).is_ok());
    }

    // ── parallel machine tests ──

    #[test]
    fn test_parallel_two_machines() {
        // Two identical jobs on 2 machines should run in parallel.
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 5.0, None, None, 1.0),
        ];
        let config = ScheduleConfig {
            rule: "SPT".to_string(),
            num_machines: 2,
            atc_k: None,
        };
        let tasks: Vec<Task> = jobs.iter().map(build_task).collect();
        let engine = build_engine(&config.rule, &config).expect("valid rule");
        let schedule = dispatch(tasks, engine, 2).expect("dispatches");

        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        // Both on separate machines → makespan = 5, not 10
        assert!((makespan - 5.0).abs() < 1e-9);
        // Jobs should be on different machines
        assert_ne!(schedule[0].machine, schedule[1].machine);
    }

    #[test]
    fn test_parallel_utilization() {
        let jobs = vec![
            make_input_job("A", 4.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
            make_input_job("C", 6.0, None, None, 1.0),
        ];
        let config = ScheduleConfig {
            rule: "LPT".to_string(),
            num_machines: 2,
            atc_k: None,
        };
        let tasks: Vec<Task> = jobs.iter().map(build_task).collect();
        let engine = build_engine(&config.rule, &config).expect("valid rule");
        let schedule = dispatch(tasks, engine, 2).expect("dispatches");

        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        let util = compute_utilization(&schedule, 2, makespan);

        assert_eq!(util.len(), 2);
        // Total busy time should equal sum of processing times (12s)
        let total_busy: f64 = util.iter().map(|u| u.busy_time).sum();
        assert!((total_busy - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_machine_no_utilization() {
        // num_machines=1 → machine_utilization should be empty
        let jobs = vec![make_input_job("A", 3.0, None, None, 1.0)];
        let config = ScheduleConfig {
            rule: "SPT".to_string(),
            num_machines: 1,
            atc_k: None,
        };
        let tasks: Vec<Task> = jobs.iter().map(build_task).collect();
        let engine = build_engine(&config.rule, &config).expect("valid rule");
        let schedule = dispatch(tasks, engine, 1).expect("dispatches");

        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        // Single machine → no utilization reported
        let util = compute_utilization(&schedule, 1, makespan);
        // (utilization is computed but for 1 machine it would just be 1.0)
        assert_eq!(util.len(), 1);
        assert!((util[0].utilization - 1.0).abs() < 1e-9);
    }

    // ── jobshop tests ──

    #[test]
    fn a_repeated_job_id_is_refused_naming_both_positions() {
        let err = refuse_repeated_ids(["A", "B", "A"]).expect_err("A is given twice");
        assert!(err.message.contains("job \"A\""), "{err}");
        assert!(err.message.contains("positions 0 and 2"), "{err}");
        assert!(
            !err.message.contains("  "),
            "a wrapped literal left a run of spaces: {err}"
        );
        assert_eq!(err.code(), "duplicate_id");
        assert_eq!(
            err.fields,
            json!({ "code": "duplicate_id", "entity": "job", "id": "A", "first": 0, "second": 2 })
        );
        assert!(refuse_repeated_ids(["A", "B", "C"]).is_ok());
    }

    /// Every refusal on the job-shop path names its reason as a `code` and
    /// carries the values behind it -- through the function `solve_jobshop`
    /// itself calls, not a copy of it.
    /// The GA runner's own check names the `ga_config` field it refused, with
    /// the bounds when it is a range.
    #[test]
    fn ga_settings_the_runner_refuses_name_their_field() {
        let range = GaConfig::default()
            .with_population_size(1)
            .validate()
            .expect_err("a population of one");
        let err = ga_settings_refused(range);
        assert_eq!(
            err.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "ga_config.population_size",
                    "min": 2.0, "max": null, "got": 1.0 })
        );

        // An open end the runner reports stays open on the wire: (0, 1] is
        // `exclusive_min` beside `max`, never `min: 0`.
        let open = u_metaheur::alns::AlnsConfig::default()
            .with_reaction_factor(0.0)
            .validate()
            .expect_err("a reaction factor of zero");
        let err = ga_settings_refused(open);
        assert_eq!(
            err.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "ga_config.reaction_factor",
                    "min": null, "max": 1.0, "exclusive_min": 0.0, "got": 0.0 })
        );

        let together = GaConfig::default()
            .with_population_size(2)
            .with_elite_ratio(0.1)
            .validate()
            .expect_err("no elite in a population of two");
        let err = ga_settings_refused(together);
        assert_eq!(
            err.fields,
            json!({ "code": "invalid_option", "parameter": "ga_config.elite_ratio" })
        );
    }

    #[test]
    fn jobshop_refusals_carry_their_code_and_values() {
        let op = |machine: Option<&str>| JobShopOperation {
            machine: machine.map(str::to_string),
            machines: vec![],
            processing_time: 1.0,
        };
        let job = |id: &str, ops: Vec<JobShopOperation>| JobShopJob {
            id: id.to_string(),
            operations: ops,
            due_date: None,
            release_time: None,
        };
        let input = |jobs: Vec<JobShopJob>| JobShopInput {
            jobs,
            num_machines: None,
            ga_config: JobShopGaConfig::default(),
        };

        let missing = jobshop_problem(&input(vec![job("J1", vec![op(Some("M1")), op(None)])]))
            .expect_err("an operation without a machine");
        assert_eq!(
            missing.fields,
            json!({ "code": "missing_machine", "job": "J1", "operation": 1 })
        );

        let none = jobshop_problem(&input(vec![job("J1", vec![])])).expect_err("no machines");
        assert_eq!(none.code(), "no_machines");

        let empty = jobshop_problem(&input(vec![
            job("J1", vec![op(Some("M1"))]),
            job("J2", vec![]),
        ]))
        .expect_err("a job with no operations");
        assert_eq!(
            empty.fields,
            json!({ "code": "empty_input", "parameter": "operations", "task": "J2", "errors": [
                { "code": "empty_input", "message": empty.message, "parameter": "operations", "task": "J2" }
            ] })
        );

        // Every finding is listed with its own code and values, not only the
        // first: a caller can point at all of them in one pass.
        let both = jobshop_problem(&input(vec![
            job("J1", vec![op(Some("M1"))]),
            job("J2", vec![]),
            job("J3", vec![]),
        ]))
        .expect_err("two jobs with no operations");
        assert_eq!(both.code(), "empty_input");
        assert_eq!(both.fields["task"], "J2");
        let tasks: Vec<_> = both.fields["errors"]
            .as_array()
            .expect("errors is a list")
            .iter()
            .map(|e| (e["code"].clone(), e["task"].clone()))
            .collect();
        assert_eq!(
            tasks,
            vec![
                (json!("empty_input"), json!("J2")),
                (json!("empty_input"), json!("J3"))
            ]
        );

        // `num_machines` can add idle machines, never remove named ones.
        let mut few = input(vec![job("J1", vec![op(Some("M1")), op(Some("M2"))])]);
        few.num_machines = Some(1);
        let few = jobshop_problem(&few).expect_err("fewer machines than named");
        assert_eq!(
            few.fields,
            json!({ "code": "parameter_out_of_range", "parameter": "num_machines",
                    "min": 2.0, "max": null, "got": 1.0 })
        );
        // Padding never repeats a name the operations already use.
        let mut padded = input(vec![job("J1", vec![op(Some("M2"))])]);
        padded.num_machines = Some(3);
        let problem = jobshop_problem(&padded).expect("padded");
        let mut ids: Vec<&str> = problem.resources().iter().map(|r| r.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, ["M1", "M2", "M3"]);

        // Settings are refused with or without jobs.
        let mut no_jobs = input(vec![]);
        no_jobs.ga_config.crossover = "OX".to_string();
        let bad = jobshop(&no_jobs).expect_err("unknown crossover, no jobs");
        assert_eq!(bad.fields["parameter"], "ga_config.crossover");
        let mut no_jobs = input(vec![]);
        no_jobs.ga_config.population_size = 0;
        assert_eq!(
            jobshop(&no_jobs).expect_err("population 0, no jobs").code(),
            "parameter_out_of_range"
        );
        assert!(jobshop(&input(vec![]))
            .expect("defaults, no jobs")
            .schedule
            .is_empty());

        // A negative processing time is refused, not scheduled.
        let negative = JobShopOperation {
            processing_time: -1.0,
            ..op(Some("M1"))
        };
        let negative = jobshop_problem(&input(vec![job("J1", vec![negative])]))
            .expect_err("negative processing time");
        assert_eq!(negative.fields["code"], "parameter_out_of_range");
        assert_eq!(negative.fields["parameter"], "processing_time");
        assert_eq!(negative.fields["activity"], "J1_1");

        let rule = build_engine(
            "FIFO",
            &ScheduleConfig {
                rule: "FIFO".to_string(),
                num_machines: 1,
                atc_k: None,
            },
        )
        .expect_err("FIFO is spelled FCFS here");
        assert_eq!(rule.code(), "unknown_option");
        assert_eq!(rule.fields["parameter"], "rule");
        assert_eq!(rule.fields["got"], "FIFO");

        let crossover = parse_crossover("OX").expect_err("unknown crossover");
        assert_eq!(crossover.fields["parameter"], "ga_config.crossover");

        let population = validate_ga_config(&JobShopGaConfig {
            population_size: 1,
            ..JobShopGaConfig::default()
        })
        .expect_err("population of one");
        assert_eq!(
            population.fields,
            json!({
                "code": "parameter_out_of_range",
                "parameter": "ga_config.population_size",
                "min": 2.0,
                "max": null,
                "got": 1.0,
            })
        );
    }

    /// A three-job, three-machine problem; `flexible` lets the second step
    /// of each job run on either of two machines.
    fn check_problem(seed: i128, flexible: bool) -> JobShopInput {
        let step = |machine: &str, alt: Option<&str>, t: f64| JobShopOperation {
            machine: if flexible && alt.is_some() {
                None
            } else {
                Some(machine.to_string())
            },
            machines: match (flexible, alt) {
                (true, Some(other)) => vec![machine.to_string(), other.to_string()],
                _ => vec![],
            },
            processing_time: t,
        };
        JobShopInput {
            jobs: vec![
                JobShopJob {
                    id: "J1".into(),
                    operations: vec![
                        step("M1", None, 3.0),
                        step("M2", Some("M3"), 2.0),
                        step("M3", None, 2.0),
                    ],
                    due_date: None,
                    release_time: None,
                },
                JobShopJob {
                    id: "J2".into(),
                    operations: vec![
                        step("M2", None, 4.0),
                        step("M1", Some("M3"), 1.0),
                        step("M3", None, 3.0),
                    ],
                    due_date: None,
                    release_time: Some(1.0),
                },
                JobShopJob {
                    id: "J3".into(),
                    operations: vec![
                        step("M3", None, 2.0),
                        step("M2", Some("M1"), 3.0),
                        step("M1", None, 2.0),
                    ],
                    due_date: None,
                    release_time: None,
                },
            ],
            num_machines: Some(3),
            ga_config: JobShopGaConfig {
                population_size: 20,
                max_generations: 30,
                mutation_rate: 0.1,
                seed: Some(seed),
                tardiness_weight: 0.0,
                crossover: "POX".to_string(),
                mutation: "Swap".to_string(),
            },
        }
    }

    /// The answer's schedule as a schedule to check.
    fn steps(schedule: &[JobShopAssignment]) -> Vec<JobShopStep> {
        schedule
            .iter()
            .map(|a| JobShopStep {
                job_id: a.job_id.clone(),
                operation: a.operation as i128,
                machine: a.machine.clone(),
                start: a.start,
                end: a.end,
            })
            .collect()
    }

    fn codes(
        problem: JobShopInput,
        schedule: Vec<JobShopStep>,
        makespan: Option<f64>,
    ) -> Vec<String> {
        let out = check_jobshop_input(&JobShopCheckInput {
            problem,
            solution: JobShopPlan { schedule, makespan },
        })
        .expect("a valid problem");
        assert_eq!(out.feasible, out.violations.is_empty());
        out.violations
            .iter()
            .map(|v| {
                serde_json::to_value(v).expect("serializes")["code"]
                    .as_str()
                    .expect("a code")
                    .to_string()
            })
            .collect()
    }

    /// Every schedule `solve_jobshop` returns checks clean.
    #[test]
    fn a_solved_schedule_has_no_violations() {
        for seed in 0..12 {
            for flexible in [false, true] {
                let answer = jobshop(&check_problem(seed, flexible)).expect("solved");
                let found = codes(
                    check_problem(seed, flexible),
                    steps(&answer.schedule),
                    Some(answer.makespan),
                );
                assert!(
                    found.is_empty(),
                    "seed {seed} flexible {flexible}: {found:?}"
                );
            }
        }
    }

    /// Each way of breaking a solved schedule is named.
    #[test]
    fn a_broken_schedule_names_each_violation() {
        let solved = || steps(&jobshop(&check_problem(3, false)).expect("solved").schedule);
        let find = |s: &[JobShopStep], job: &str, op: i128| {
            s.iter()
                .position(|a| a.job_id == job && a.operation == op)
                .expect("placed")
        };
        let run = |s: Vec<JobShopStep>, m: Option<f64>| codes(check_problem(3, false), s, m);
        let has = |found: Vec<String>, code: &str| found.iter().any(|c| c == code);

        // A step left out.
        let mut s = solved();
        s.remove(find(&s, "J1", 2));
        assert!(has(run(s, None), "requirement_unfilled"));
        // A job the problem does not have; a step number it does not have,
        // read as given (past the last, and negative).
        let mut s = solved();
        s[0].job_id = "J9".into();
        assert!(has(run(s, None), "unknown_activity"));
        for op in [4, -1] {
            let mut s = solved();
            let i = find(&s, "J2", 3);
            s[i].operation = op;
            assert!(has(run(s, None), "unknown_activity"), "operation {op}");
        }
        // A step placed twice.
        let mut s = solved();
        let i = find(&s, "J3", 1);
        let copy = JobShopStep {
            job_id: s[i].job_id.clone(),
            operation: s[i].operation,
            machine: s[i].machine.clone(),
            start: 100.0,
            end: 102.0,
        };
        s.push(copy);
        assert!(has(run(s, None), "unknown_activity"));
        // On a machine the step cannot run on.
        let mut s = solved();
        let i = find(&s, "J1", 1);
        s[i].machine = "M2".into();
        assert!(has(run(s, None), "requirement_unfilled"));
        // Shorter than its processing time.
        let mut s = solved();
        let i = find(&s, "J1", 1);
        s[i].end = s[i].start + 1.0;
        assert!(has(run(s, None), "duration_too_short"));
        // Before the job's release.
        let mut s = solved();
        let i = find(&s, "J2", 1);
        let length = s[i].end - s[i].start;
        s[i].start = 0.0;
        s[i].end = length;
        assert!(has(run(s, None), "early_start"));
        // Two steps on one machine at once, and a step before its predecessor.
        let mut s = solved();
        for a in &mut s {
            let length = a.end - a.start;
            a.start = 50.0;
            a.end = 50.0 + length;
        }
        let found = run(s, None);
        assert!(has(found.clone(), "capacity_exceeded"), "{found:?}");
        assert!(has(found.clone(), "precedence_violation"), "{found:?}");
        // A makespan that is not the latest end.
        let answer = jobshop(&check_problem(3, false)).expect("solved");
        let out = check_jobshop_input(&JobShopCheckInput {
            problem: check_problem(3, false),
            solution: JobShopPlan {
                schedule: steps(&answer.schedule),
                makespan: Some(answer.makespan + 1.0),
            },
        })
        .expect("a valid problem");
        assert!(
            matches!(
                out.violations[..],
                [JobShopViolation::MakespanMismatch { reported, actual }]
                    if reported == answer.makespan + 1.0 && actual == answer.makespan
            ),
            "{:?}",
            out.violations
        );
    }

    /// The problem is read as `solve_jobshop` reads it: the same refusals.
    #[test]
    fn a_check_refuses_the_problems_a_solve_refuses() {
        let mut problem = check_problem(1, false);
        problem.jobs[0].operations[0].machine = None;
        let refused = check_jobshop_input(&JobShopCheckInput {
            problem,
            solution: JobShopPlan {
                schedule: vec![],
                makespan: None,
            },
        })
        .expect_err("no machine");
        assert_eq!(refused.code(), "missing_machine");
    }

    /// The answer says why the search stopped -- the reason the runner
    /// recorded, not one derived here.
    #[test]
    fn a_jobshop_answer_says_why_the_search_stopped() {
        let input = |jobs: usize, max_generations: i128| JobShopInput {
            jobs: (0..jobs)
                .map(|j| JobShopJob {
                    id: format!("J{j}"),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M1".to_string()),
                            machines: vec![],
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M2".to_string()),
                            machines: vec![],
                            processing_time: 2.0,
                        },
                    ],
                    due_date: None,
                    release_time: None,
                })
                .collect(),
            num_machines: Some(2),
            ga_config: JobShopGaConfig {
                population_size: 20,
                max_generations,
                mutation_rate: 0.1,
                seed: Some(42),
                tardiness_weight: 0.0,
                crossover: "POX".to_string(),
                mutation: "Swap".to_string(),
            },
        };
        let stop = |jobs, gens| jobshop(&input(jobs, gens)).expect("solved").stop_reason;
        assert_eq!(stop(2, 2), StopReason::IterationLimit);
        assert_eq!(stop(2, 100_000), StopReason::NoImprovement);
        assert_eq!(stop(0, 2), StopReason::Completed);
    }

    #[test]
    fn test_jobshop_basic() {
        // Classic 2-job, 2-machine job-shop instance
        let input = JobShopInput {
            jobs: vec![
                JobShopJob {
                    id: "J1".to_string(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M1".to_string()),
                            machines: vec![],
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M2".to_string()),
                            machines: vec![],
                            processing_time: 2.0,
                        },
                    ],
                    due_date: None,
                    release_time: None,
                },
                JobShopJob {
                    id: "J2".to_string(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M2".to_string()),
                            machines: vec![],
                            processing_time: 4.0,
                        },
                        JobShopOperation {
                            machine: Some("M1".to_string()),
                            machines: vec![],
                            processing_time: 1.0,
                        },
                    ],
                    due_date: None,
                    release_time: None,
                },
            ],
            num_machines: Some(2),
            ga_config: JobShopGaConfig {
                population_size: 20,
                max_generations: 30,
                mutation_rate: 0.1,
                seed: Some(42),
                tardiness_weight: 0.0,
                crossover: "POX".to_string(),
                mutation: "Swap".to_string(),
            },
        };

        let checked = jobshop_problem(&input).expect("valid input");
        let problem = SchedulingGaProblem::new(&checked)
            .with_tardiness_weight(input.ga_config.tardiness_weight)
            .expect("weight in [0, 1]");

        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(30)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        let schedule = problem.decode(&result.best);

        assert!(schedule.makespan_ms() > 0);
        assert_eq!(schedule.assignments.len(), 4); // 2 jobs * 2 ops
        assert!(result.best_fitness.is_finite());
    }

    /// The answer says how late each job is, by the definition the search
    /// weighed: with `tardiness_weight` 1 the fitness is the total tardiness.
    #[test]
    fn a_jobshop_answer_reports_each_jobs_completion_and_tardiness() {
        let op = |m: &str, pt: f64| JobShopOperation {
            machine: Some(m.to_string()),
            machines: vec![],
            processing_time: pt,
        };
        let job = |id: &str, ops: Vec<JobShopOperation>, due: Option<f64>| JobShopJob {
            id: id.to_string(),
            operations: ops,
            due_date: due,
            release_time: None,
        };
        let input = JobShopInput {
            jobs: vec![
                job("J1", vec![op("M1", 3.0), op("M2", 2.0)], Some(2.0)),
                job("J2", vec![op("M2", 4.0), op("M1", 1.0)], None),
                job("J3", vec![op("M1", 2.0)], Some(100.0)),
            ],
            num_machines: Some(2),
            ga_config: JobShopGaConfig {
                population_size: 20,
                max_generations: 20,
                seed: Some(7),
                tardiness_weight: 1.0,
                ..JobShopGaConfig::default()
            },
        };
        let out = jobshop(&input).expect("solved");

        let ids: Vec<&str> = out.jobs.iter().map(|j| j.id.as_str()).collect();
        assert_eq!(ids, ["J1", "J2", "J3"], "request order");
        for (j, due) in out.jobs.iter().zip([Some(2.0), None, Some(100.0)]) {
            let last_end = out
                .schedule
                .iter()
                .filter(|a| a.job_id == j.id)
                .map(|a| a.end)
                .fold(f64::MIN, f64::max);
            assert_eq!(j.completion, last_end, "{}", j.id);
            let expected = due.map_or(0.0, |d: f64| (last_end - d).max(0.0));
            assert_eq!(j.tardiness, expected, "{}", j.id);
        }
        // J1's two steps take 5 s against a due date of 2 s.
        assert!(out.jobs[0].tardiness >= 3.0);
        let total: f64 = out.jobs.iter().map(|j| j.tardiness).sum();
        assert_eq!(out.total_tardiness, total);
        assert_eq!(
            out.fitness,
            out.total_tardiness * 1000.0,
            "fitness is in ms"
        );
    }

    #[test]
    fn test_jobshop_flexible() {
        // Flexible job-shop: operations can go on multiple machines
        let input = JobShopInput {
            jobs: vec![JobShopJob {
                id: "J1".to_string(),
                operations: vec![JobShopOperation {
                    machine: None,
                    machines: vec!["M1".to_string(), "M2".to_string()],
                    processing_time: 5.0,
                }],
                due_date: None,
                release_time: None,
            }],
            num_machines: Some(2),
            ga_config: JobShopGaConfig::default(),
        };

        let tasks = build_jobshop_tasks(&input).expect("valid input");
        assert_eq!(tasks[0].activities.len(), 1);
        // Activity should have 2 candidate machines
        let candidates = &tasks[0].activities[0].resource_requirements[0].candidates;
        assert_eq!(candidates.len(), 2);
    }

    /// The tasks of a job-shop request, through the same path `solve_jobshop` takes.
    fn build_jobshop_tasks(input: &JobShopInput) -> Result<Vec<Task>, WireError> {
        jobshop_problem(input).map(|p| p.into_parts().0)
    }

    /// The reporter's job shop: two jobs of three operations each. Every row
    /// used to read `operation: 1`, while machines, times and makespan were
    /// all correct -- a silent wrong column.
    #[test]
    fn jobshop_numbers_each_operation_within_its_job() {
        use u_metaheur::ga::{GaConfig, GaRunner};

        let input = JobShopInput {
            jobs: vec![
                JobShopJob {
                    id: "J1".into(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M1".into()),
                            machines: Vec::new(),
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M2".into()),
                            machines: Vec::new(),
                            processing_time: 2.0,
                        },
                        JobShopOperation {
                            machine: Some("M3".into()),
                            machines: Vec::new(),
                            processing_time: 4.0,
                        },
                    ],
                    due_date: Some(20.0),
                    release_time: Some(0.0),
                },
                JobShopJob {
                    id: "J2".into(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M2".into()),
                            machines: Vec::new(),
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M3".into()),
                            machines: Vec::new(),
                            processing_time: 2.0,
                        },
                        JobShopOperation {
                            machine: Some("M1".into()),
                            machines: Vec::new(),
                            processing_time: 5.0,
                        },
                    ],
                    due_date: Some(18.0),
                    release_time: Some(0.0),
                },
            ],
            num_machines: Some(3),
            ga_config: JobShopGaConfig {
                population_size: 30,
                max_generations: 20,
                mutation_rate: 0.1,
                seed: Some(42),
                ..JobShopGaConfig::default()
            },
        };

        let checked = jobshop_problem(&input).expect("every operation names a machine");
        let problem = SchedulingGaProblem::new(&checked);
        let config = GaConfig::default()
            .with_population_size(30)
            .with_max_generations(20)
            .with_seed(42)
            .with_parallel(false);
        let result = GaRunner::run(&problem, &config).expect("a valid GA problem");
        let schedule = problem.decode(&result.best);

        assert_eq!(schedule.assignment_count(), 6, "two jobs of three steps");
        for job in ["J1", "J2"] {
            let mut steps: Vec<i32> = schedule
                .assignments
                .iter()
                .filter(|a| a.task_id == job)
                .map(|a| a.sequence.expect("the schedule says the step"))
                .collect();
            steps.sort_unstable();
            assert_eq!(steps, vec![1, 2, 3], "{job} must number its own steps");
        }

        // The step order is the order the job lists its machines, whatever
        // order the solver puts them in on the timeline.
        for (job, machines) in [("J1", ["M1", "M2", "M3"]), ("J2", ["M2", "M3", "M1"])] {
            for (idx, machine) in machines.iter().enumerate() {
                let a = schedule
                    .assignments
                    .iter()
                    .find(|a| a.task_id == job && a.sequence == Some(idx as i32 + 1))
                    .unwrap_or_else(|| panic!("{job} step {}", idx + 1));
                assert_eq!(&a.resource_id, machine, "{job} step {}", idx + 1);
            }
        }
    }

    // ── GA config validation tests ──

    #[test]
    fn test_validate_ga_config_defaults_ok() {
        let cfg = JobShopGaConfig::default();
        assert!(validate_ga_config(&cfg).is_ok());
    }

    #[test]
    fn test_validate_ga_config_population_size_zero() {
        let cfg = JobShopGaConfig {
            population_size: 0,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("population_size"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_population_size_one() {
        let cfg = JobShopGaConfig {
            population_size: 1,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("population_size"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_population_size_two_ok() {
        let cfg = JobShopGaConfig {
            population_size: 2,
            ..Default::default()
        };
        assert!(validate_ga_config(&cfg).is_ok());
    }

    #[test]
    fn test_validate_ga_config_max_generations_zero() {
        let cfg = JobShopGaConfig {
            max_generations: 0,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("max_generations"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_negative() {
        let cfg = JobShopGaConfig {
            mutation_rate: -0.1,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("mutation_rate"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_above_one() {
        let cfg = JobShopGaConfig {
            mutation_rate: 1.5,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("mutation_rate"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_boundary_ok() {
        // 0.0 and 1.0 are both valid
        let cfg0 = JobShopGaConfig {
            mutation_rate: 0.0,
            ..Default::default()
        };
        assert!(validate_ga_config(&cfg0).is_ok());

        let cfg1 = JobShopGaConfig {
            mutation_rate: 1.0,
            ..Default::default()
        };
        assert!(validate_ga_config(&cfg1).is_ok());
    }

    #[test]
    fn test_validate_ga_config_tardiness_weight_invalid() {
        let cfg = JobShopGaConfig {
            tardiness_weight: -0.5,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("tardiness_weight"), "got: {}", err);

        let cfg2 = JobShopGaConfig {
            tardiness_weight: 2.0,
            ..Default::default()
        };
        let err2 = validate_ga_config(&cfg2).unwrap_err();
        assert!(err2.message.contains("tardiness_weight"), "got: {}", err2);
    }

    #[test]
    fn test_jobshop_invalid_population_returns_error() {
        // Verify that invalid GA config is caught as an error, not a panic.
        // We test via validate_ga_config since solve_jobshop requires JsValue (WASM).
        let cfg = JobShopGaConfig {
            population_size: 0,
            max_generations: 10,
            mutation_rate: 0.1,
            seed: Some(42),
            tardiness_weight: 0.0,
            crossover: "POX".to_string(),
            mutation: "Swap".to_string(),
        };
        assert!(validate_ga_config(&cfg).is_err());
    }

    #[test]
    fn test_jobshop_edge_case_single_job_single_op() {
        // Minimal valid jobshop: 1 job, 1 operation, 1 machine
        let input = JobShopInput {
            jobs: vec![JobShopJob {
                id: "J1".to_string(),
                operations: vec![JobShopOperation {
                    machine: Some("M1".to_string()),
                    machines: vec![],
                    processing_time: 5.0,
                }],
                due_date: None,
                release_time: None,
            }],
            num_machines: Some(1),
            ga_config: JobShopGaConfig {
                population_size: 4,
                max_generations: 5,
                mutation_rate: 0.1,
                seed: Some(1),
                tardiness_weight: 0.0,
                crossover: "POX".to_string(),
                mutation: "Swap".to_string(),
            },
        };

        let checked = jobshop_problem(&input).expect("valid input");
        let problem = SchedulingGaProblem::new(&checked);
        let config = GaConfig::default()
            .with_population_size(4)
            .with_max_generations(5)
            // population 4 × default elite_ratio 0.1 floors to 0 elites, which
            // GaConfig::validate rejects; 0.25 keeps exactly 1 elite.
            .with_elite_ratio(0.25)
            .with_seed(1)
            .with_parallel(false);

        config.validate().expect("config should be valid");
        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        let schedule = problem.decode(&result.best);

        assert_eq!(schedule.assignments.len(), 1);
        assert!(schedule.makespan_ms() > 0);
    }
}

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn schedule_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::ScheduleInput>(json!({
            "jobs": [{ "id": "j1", "processing_time": 1.0 }],
            "rule": "spt"
        }));
    }

    #[test]
    fn schedule_nested_job_and_config_reject_unknown_keys() {
        assert_rejects_unknown::<super::ScheduleInput>(json!({
            "jobs": [{ "id": "j1", "processing_time": 1.0, "importance": 2 }]
        }));
        assert_rejects_unknown::<super::ScheduleInput>(json!({
            "jobs": [{ "id": "j1", "processing_time": 1.0 }],
            "config": { "rule": "spt", "machines": 2 }
        }));
    }

    #[test]
    fn jobshop_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::JobShopInput>(json!({
            "jobs": [{ "id": "j1", "operations": [{ "machine": "M1", "processing_time": 1.0 }] }],
            "time_limit": 5
        }));
    }

    #[test]
    fn jobshop_nested_structs_reject_unknown_keys() {
        assert_rejects_unknown::<super::JobShopJob>(json!({
            "id": "j1", "operations": [], "priority": 1
        }));
        assert_rejects_unknown::<super::JobShopOperation>(json!({
            "machine": "M1", "processing_time": 1.0, "setup_time": 0.5
        }));
        assert_rejects_unknown::<super::JobShopGaConfig>(json!({
            "population_size": 10, "elitism": true
        }));
    }
}

#[cfg(test)]
mod path_tests {
    //! A value of the wrong type deep inside an argument is refused where it
    //! sits: `parameter` names the field, `index` its array position.

    #[test]
    fn a_bad_field_is_refused_at_its_job() {
        let err = super::from_json::<super::ScheduleInput>(
            serde_json::json!({ "jobs": [
                { "id": "a", "processing_time": 3.0 },
                { "id": "b", "processing_time": "three" }
            ] }),
            "jobs",
        )
        .err()
        .expect("a string is not a number");
        assert_eq!(err.fields["code"], "malformed_input");
        assert_eq!(err.fields["parameter"], "jobs[1].processing_time");
        assert_eq!(err.fields["index"], 1);
    }
}

#[cfg(test)]
mod whole_number_tests {
    //! Counts and the seed are whole numbers read signed: a negative one is
    //! refused by the same range check as a too-small one, a fractional one
    //! as malformed.
    //!
    //! `from_json` is the half of `from_js` after serde-wasm-bindgen has read
    //! the JS value into JSON. That reader turns a JS number that is a safe
    //! integer (`3`, `-1`, `3.0` -- the same number in JS) into a JSON integer
    //! and any other number (`2.5`) into a JSON float, so these JSON values are
    //! what a JS caller's arguments arrive as.

    use super::*;

    fn schedule(config: serde_json::Value) -> Result<ScheduleOutput, WireError> {
        let input = json!({
            "jobs": [{ "id": "a", "processing_time": 1.0 }, { "id": "b", "processing_time": 2.0 }],
            "config": config,
        });
        schedule_jobs(&from_json::<ScheduleInput>(input, "jobs")?)
    }

    fn jobshop_with(extra: serde_json::Value) -> Result<JobShopOutput, WireError> {
        let mut input = json!({
            "jobs": [{ "id": "J1", "operations": [{ "machine": "M1", "processing_time": 1.0 }] }],
            "ga_config": { "population_size": 4, "max_generations": 2, "seed": 1 },
        });
        for (key, value) in extra.as_object().expect("an object") {
            match (key.as_str(), value) {
                ("ga_config", serde_json::Value::Object(settings)) => input["ga_config"]
                    .as_object_mut()
                    .expect("an object")
                    .extend(settings.clone()),
                _ => input[key] = value.clone(),
            }
        }
        jobshop(&from_json::<JobShopInput>(input, "problem")?)
    }

    fn out_of_range(parameter: &str, min: f64, got: f64) -> serde_json::Value {
        json!({ "code": "parameter_out_of_range", "parameter": parameter,
                "min": min, "max": null, "got": got })
    }

    #[test]
    fn a_negative_count_is_refused_like_a_too_small_one() {
        for got in [-1, 0] {
            let err = schedule(json!({ "num_machines": got })).expect_err("below 1");
            assert_eq!(
                err.fields,
                out_of_range("config.num_machines", 1.0, got as f64)
            );
        }
        let err = schedule(json!({ "num_machines": -1 })).expect_err("negative");
        assert_eq!(err.message, "config.num_machines must be >= 1, got -1");

        for got in [-1, 1] {
            let err = jobshop_with(json!({ "ga_config": { "population_size": got } }))
                .expect_err("below 2");
            assert_eq!(
                err.fields,
                out_of_range("ga_config.population_size", 2.0, got as f64)
            );
        }
        for got in [-1, 0] {
            let err = jobshop_with(json!({ "ga_config": { "max_generations": got } }))
                .expect_err("below 1");
            assert_eq!(
                err.fields,
                out_of_range("ga_config.max_generations", 1.0, got as f64)
            );
        }
        // The operations name one machine, so at least 1: -1 and 0 alike.
        for got in [-1, 0] {
            let err = jobshop_with(json!({ "num_machines": got })).expect_err("below 1");
            assert_eq!(err.fields, out_of_range("num_machines", 1.0, got as f64));
        }
    }

    #[test]
    fn a_negative_seed_is_out_of_range() {
        let err = jobshop_with(json!({ "ga_config": { "seed": -1 } })).expect_err("negative");
        assert_eq!(err.fields, out_of_range("ga_config.seed", 0.0, -1.0));
        // The whole 64-bit range is a seed; past it is refused with its top.
        assert!(jobshop_with(json!({ "ga_config": { "seed": u64::MAX } })).is_ok());
        let err = whole_in::<u64>(
            "ga_config.seed",
            i128::from(u64::MAX) + 1,
            0,
            u64::MAX.into(),
        )
        .expect_err("past u64");
        assert_eq!(err.fields["max"], json!(u64::MAX as f64));
        assert_eq!(err.fields["min"], json!(0.0));
    }

    #[test]
    fn a_fractional_count_is_malformed_at_its_field() {
        let err = schedule(json!({ "num_machines": 2.5 })).expect_err("not whole");
        assert_eq!(
            err.fields,
            json!({ "code": "malformed_input", "parameter": "config.num_machines" })
        );
        assert!(
            err.message.contains("must be a whole number, got 2.5"),
            "{}",
            err.message
        );
        for (extra, parameter) in [
            (
                json!({ "ga_config": { "population_size": 2.5 } }),
                "ga_config.population_size",
            ),
            (
                json!({ "ga_config": { "max_generations": 0.5 } }),
                "ga_config.max_generations",
            ),
            (json!({ "ga_config": { "seed": 1.5 } }), "ga_config.seed"),
            (json!({ "num_machines": 1.5 }), "num_machines"),
        ] {
            let err = jobshop_with(extra).expect_err("not whole");
            assert_eq!(err.fields["code"], "malformed_input", "{parameter}");
            assert_eq!(err.fields["parameter"], parameter);
        }
    }

    #[test]
    fn a_whole_count_reads_as_itself() {
        let out = schedule(json!({ "num_machines": 2 })).expect("two machines");
        assert_eq!(out.machine_utilization.len(), 2);
        // `2.0` is the number 2 in JS.
        let out = schedule(json!({ "num_machines": 2.0 })).expect("two machines");
        assert_eq!(out.machine_utilization.len(), 2);
        // Omitted: one machine.
        let out = schedule(json!({})).expect("default");
        assert!(out.machine_utilization.is_empty());

        let config: JobShopGaConfig = from_json(
            json!({ "population_size": 3, "max_generations": 1, "seed": 42 }),
            "ga_config",
        )
        .expect("whole numbers");
        assert_eq!(
            validate_ga_config(&config).expect("accepted"),
            GaCounts {
                population_size: 3,
                max_generations: 1,
                seed: Some(42),
            }
        );
        assert_eq!(
            validate_ga_config(&JobShopGaConfig::default()).expect("defaults"),
            GaCounts {
                population_size: 100,
                max_generations: 200,
                seed: None,
            }
        );
        assert!(jobshop_with(json!({ "num_machines": 3, "ga_config": { "seed": null } })).is_ok());
    }

    #[test]
    fn a_count_that_is_not_a_number_stays_malformed() {
        for bad in [json!(null), json!("2"), json!(true)] {
            let err = schedule(json!({ "num_machines": bad })).expect_err("not a number");
            assert_eq!(err.fields["code"], "malformed_input", "{bad}");
            assert_eq!(err.fields["parameter"], "config.num_machines", "{bad}");
        }
    }
}
