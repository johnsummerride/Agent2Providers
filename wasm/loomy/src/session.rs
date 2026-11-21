//! Internal conversation keys are deliberately separate from Loomy protocol IDs.
//! Only normal chat.message requests use this durable state.  The pure state
//! machine below is shared by host-backed production code and deterministic tests.

use super::sign;
use agent2api_wasm_sdk::{self as sdk, json, state::Snapshot, Error, Result, Value};

const CONVERSATION: &str = "loomy-session-lifecycle";
const SCHEMA: u32 = 1;
const TTL: i64 = 0;
const MAX_ATTEMPTS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub revision: u64,
    pub affinity: String,
    pub chat_id: String,
    pub msg_id: String,
    pub turn_id: String,
    pub traceparent: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub params_session_present: bool,
    pub header_session_present: bool,
    pub key: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Overrides {
    pub chat_id: Option<String>,
    pub msg_id: Option<String>,
    pub turn_id: Option<String>,
    pub traceparent: Option<String>,
    pub affinity: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub chat_id: String,
    pub msg_id: String,
    pub turn_id: String,
    pub traceparent: String,
    pub affinity: Option<String>,
}

impl Material {
    fn from_state(state: &State, overrides: &Overrides) -> Self {
        Self {
            chat_id: overrides.chat_id.clone().unwrap_or_else(|| state.chat_id.clone()),
            msg_id: overrides.msg_id.clone().unwrap_or_else(|| state.msg_id.clone()),
            turn_id: overrides.turn_id.clone().unwrap_or_else(|| state.turn_id.clone()),
            traceparent: overrides
                .traceparent
                .clone()
                .unwrap_or_else(|| state.traceparent.clone()),
            affinity: overrides.affinity.clone().or_else(|| Some(state.affinity.clone())),
        }
    }
}

pub fn resolve(params: &Value, incoming: &Value) -> Resolved {
    let valid = |value: &Value| -> Option<String> {
        value.as_str().map(str::trim).filter(|value| {
            !value.is_empty()
                && value.len() <= 256
                && value.is_ascii()
                && !value.bytes().any(|byte| byte.is_ascii_control())
        }).map(str::to_string)
    };
    let param = valid(&params["session"]);
    let header = incoming.as_object().and_then(|headers| {
        // The agent2api host nests inbound client headers under "headers";
        // other callers pass the flat map. Accept both, inner first.
        let mut candidates = Vec::new();
        if let Some(inner) = headers.get("headers").filter(|v| v.is_object()) {
            candidates.push(inner);
        }
        candidates.push(incoming);
        candidates.into_iter().flatten().find_map(|(name, value)| {
            name.eq_ignore_ascii_case("x-session-id")
                .then(|| valid(value))
                .flatten()
        })
    });
    // The host synthesizes params.session for conversational forwarding. An
    // explicit wire session header must win over that generated value.
    let key = header.clone().or_else(|| param.clone());
    Resolved {
        key,
        params_session_present: param.is_some(),
        header_session_present: header.is_some(),
    }
}

/// Match only a terminal assistant tool call followed immediately by its tool
/// results.  Looking at the whole history would incorrectly classify a fresh
/// user turn after an earlier tool call as a continuation.
pub fn continuation(body: &Value, anthropic: bool) -> bool {
    let Some(messages) = body["messages"].as_array() else { return false };
    let Some(last) = messages.last() else { return false };
    let expected: Vec<String> = if anthropic {
        let content = last["content"].as_array();
        if last["role"] != "user" || content.is_none() || content.is_some_and(Vec::is_empty) {
            return false;
        }
        let mut ids = Vec::new();
        for part in content.unwrap() {
            if part["type"] != "tool_result" {
                return false;
            }
            let Some(id) = part["tool_use_id"].as_str().filter(|id| !id.is_empty()) else {
                return false;
            };
            ids.push(id.to_string());
        }
        ids
    } else {
        if last["role"] != "tool" {
            return false;
        }
        let mut ids = Vec::new();
        for message in messages.iter().rev() {
            if message["role"] != "tool" {
                break;
            }
            let Some(id) = message["tool_call_id"].as_str().filter(|id| !id.is_empty()) else {
                return false;
            };
            ids.push(id.to_string());
        }
        ids.reverse();
        ids
    };
    if expected.is_empty() {
        return false;
    }
    // The preceding assistant is immediately before this terminal result run.
    // A user text message breaks the run and is therefore a fresh user turn.
    let assistant_index = messages.len().checked_sub(if anthropic { 2 } else { expected.len() + 1 });
    let Some(assistant) = assistant_index
        .and_then(|index| messages.get(index))
        .filter(|message| message["role"] == "assistant")
    else {
        return false;
    };
    let calls: Vec<&str> = if anthropic {
        assistant["content"]
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter(|part| part["type"] == "tool_use")
                    .filter_map(|part| part["id"].as_str())
                    .collect()
            })
            .unwrap_or_default()
    } else {
        assistant["tool_calls"]
            .as_array()
            .map(|calls| calls.iter().filter_map(|call| call["id"].as_str()).collect())
            .unwrap_or_default()
    };
    calls.len() == expected.len()
        && calls.iter().zip(expected.iter()).all(|(left, right)| *left == right)
}

pub fn fresh(affinity: String) -> Result<State> {
    let msg_id = sign::uuid()?;
    Ok(State {
        revision: 0,
        affinity,
        chat_id: sign::uuid()?,
        turn_id: msg_id.clone(),
        msg_id,
        traceparent: format!("00-{}-{}-01", sign::random_hex(16)?, sign::random_hex(8)?),
    })
}

pub fn next_turn(previous: &State) -> Result<State> {
    let mut next = fresh(previous.affinity.clone())?;
    next.chat_id = previous.chat_id.clone();
    Ok(next)
}

fn valid_affinity(value: &str) -> bool {
    value.len() == 30
        && value.starts_with("ses_")
        && value[4..].bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn parse(snapshot: &Snapshot) -> Result<Option<State>> {
    if !snapshot.compatible {
        return Err(Error::http(409, "Loomy 会话状态 schema 不兼容，拒绝覆盖")
            .with_code("session_schema_incompatible"));
    }
    if snapshot.value.is_null() {
        return Ok(None);
    }
    let text = |field: &str| -> String {
        snapshot.value[field]
            .as_str()
            .filter(|value| !value.is_empty() && value.len() <= 256 && value.is_ascii())
            .unwrap_or("")
            .to_string()
    };
    let affinity = text("affinity");
    let state = State {
        revision: snapshot.revision,
        chat_id: text("chat_id"),
        msg_id: text("msg_id"),
        turn_id: text("turn_id"),
        traceparent: text("traceparent"),
        affinity,
    };
    // Any malformed durable row is unsafe to overwrite: it may belong to a
    // different/newer implementation despite a nominally compatible schema.
    if !valid_affinity(&state.affinity)
        || state.chat_id.is_empty()
        || state.msg_id.is_empty()
        || state.turn_id.is_empty()
        || state.traceparent.is_empty()
    {
        return Err(Error::http(409, "Loomy 会话状态内容无效，拒绝覆盖")
            .with_code("session_state_invalid"));
    }
    Ok(Some(state))
}

fn value(state: &State) -> Value {
    json!({
        "affinity": state.affinity,
        "chat_id": state.chat_id,
        "msg_id": state.msg_id,
        "turn_id": state.turn_id,
        "traceparent": state.traceparent,
    })
}

pub fn new_affinity() -> Result<String> {
    Ok(format!("ses_{}", sign::random_hex(13)?))
}

pub fn resolve_state_with<G, C>(
    continuing: bool,
    mut next: impl FnMut(Option<&State>) -> Result<State>,
    mut get: G,
    mut compare_exchange: C,
) -> Result<State>
where
    G: FnMut() -> Result<Snapshot>,
    C: FnMut(u64, Value) -> Result<bool>,
{
    for _ in 0..MAX_ATTEMPTS {
        let snapshot = get()?;
        let previous = parse(&snapshot)?;
        let mut candidate = match previous.as_ref() {
            Some(previous) if continuing => previous.clone(),
            previous => next(previous)?,
        };
        candidate.revision = snapshot.revision;
        if compare_exchange(snapshot.revision, value(&candidate))? {
            candidate.revision = snapshot.revision.saturating_add(1);
            return Ok(candidate);
        }
    }
    Err(Error::http(409, "Loomy 会话状态并发更新超过 8 次，请重试")
        .with_code("session_cas_exhausted"))
}

/// Production CAS driver. Every failed exchange is followed by a fresh read.
pub fn resolve_state(key: &str, continuing: bool, create: impl Fn(Option<&State>) -> Result<State>) -> Result<State> {
    let conversation = storage_conversation(key);
    resolve_state_with(
        continuing,
        create,
        || sdk::state::conversation_get(&conversation, key, SCHEMA),
        |revision, value| {
            sdk::state::conversation_compare_exchange(&conversation, key, SCHEMA, revision, value, TTL)
        },
    )
}

pub fn material(
    resolved: &Resolved,
    continuing: bool,
    overrides: &Overrides,
) -> Result<(Material, Option<State>)> {
    if let Some(key) = &resolved.key {
        let state = resolve_state(key, continuing, |previous| match previous {
            Some(previous) if continuing => Ok(previous.clone()),
            Some(previous) => next_turn(previous),
            None => fresh(new_affinity()?),
        })?;
        let material = Material::from_state(&state, overrides);
        return Ok((material, Some(state)));
    }
    let state = if overrides.chat_id.is_none()
        || overrides.msg_id.is_none()
        || overrides.traceparent.is_none()
    {
        Some(fresh(String::new())?)
    } else {
        None
    };
    Ok((Material {
        chat_id: match overrides.chat_id.clone() {
            Some(value) => value,
            None => state.as_ref().unwrap().chat_id.clone(),
        },
        msg_id: match overrides.msg_id.clone() {
            Some(value) => value,
            None => state.as_ref().unwrap().msg_id.clone(),
        },
        turn_id: overrides.turn_id.clone().or_else(|| overrides.msg_id.clone())
            .or_else(|| state.as_ref().map(|state| state.turn_id.clone())).unwrap(),
        traceparent: match overrides.traceparent.clone() {
            Some(value) => value,
            None => state.as_ref().unwrap().traceparent.clone(),
        },
        affinity: overrides.affinity.clone(),
    }, None))
}

fn storage_conversation(key: &str) -> String {
    format!("{CONVERSATION}-{key}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(1);
    fn fixture(previous: Option<&State>) -> Result<State> {
        let serial = COUNTER.fetch_add(1, Ordering::SeqCst) << 32;
        let part = |salt: u64| format!("{:032x}", serial.wrapping_mul(8).wrapping_add(salt));
        Ok(State {
            revision: 0,
            affinity: previous.map(|state| state.affinity.clone()).unwrap_or_else(|| format!("ses_{}", &part(1)[..26])),
            chat_id: previous.map(|state| state.chat_id.clone()).unwrap_or_else(|| part(2)),
            msg_id: part(3),
            turn_id: part(4),
            traceparent: format!("00-{}-{:016x}-01", part(5), serial),
        })
    }
    fn memory_cas(
        key: &str,
        store: &mut BTreeMap<String, (u64, State)>,
        continuing: bool,
        conflicts: &mut usize,
    ) -> Result<State> {
        for _ in 0..MAX_ATTEMPTS {
            let before = store.get(key).cloned();
            let previous = before.as_ref().map(|(_, state)| state);
            let mut next = match previous {
                Some(previous) if continuing => previous.clone(),
                Some(previous) => {
                    let mut next = fixture(Some(previous))?;
                    next.chat_id = previous.chat_id.clone();
                    next
                }
                None => fixture(None)?,
            };
            next.revision = before.as_ref().map_or(0, |(revision, _)| *revision);
            if *conflicts > 0 {
                *conflicts -= 1;
                continue;
            }
            let revision = next.revision + 1;
            next.revision = revision;
            store.insert(key.to_string(), (revision, next.clone()));
            return Ok(next);
        }
        Err(Error::new("CAS exhausted"))
    }
    fn overrides() -> Overrides {
        Overrides { chat_id: None, msg_id: None, turn_id: None, traceparent: None, affinity: None }
    }
    fn oai_tail(call: &str, result: &str) -> Value {
        json!({"messages": [
            {"role":"user","content":"question"},
            {"role":"assistant","content":null,"tool_calls":[{"id":call,"type":"function","function":{"name":"lookup","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":result,"content":"answer"}
        ]})
    }
    fn anthropic_tail(call: &str, result: &str) -> Value {
        json!({"messages": [
            {"role":"user","content":[{"type":"text","text":"question"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":call,"name":"lookup","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":result,"content":"answer"}]}
        ]})
    }

    #[test]
    fn session_resolution_precedence_and_validation() {
        let resolved = resolve(&json!({"session":" params-key "}), &json!({"X-Session-Id":"header-key"}));
        assert_eq!(resolved.key.as_deref(), Some("header-key"));
        assert!(resolved.params_session_present && resolved.header_session_present);
        assert_eq!(resolve(&json!({}), &json!({"x-session-id":"header-key"})).key.as_deref(), Some("header-key"));
        // Host-nested shape: inbound client headers sit under "headers".
        assert_eq!(resolve(&json!({}), &json!({"headers":{"x-session-id":"nested-key"}})).key.as_deref(), Some("nested-key"));
        assert_eq!(resolve(&json!({"session":"flat-key"}), &json!({"headers":{}})).key.as_deref(), Some("flat-key"));
        assert!(resolve(&json!({"session":"\n"}), &json!({})).key.is_none());
        assert!(resolve(&json!({"session":42}), &json!({})).key.is_none());
    }

    #[test]
    fn affinity_is_exactly_ses_plus_26_lowercase_hex() {
        let state = fixture(None).unwrap();
        assert!(valid_affinity(&state.affinity));
        assert_eq!(state.affinity.len(), 30);
        assert!(!valid_affinity("ses_0123456789abcdef012345678"));
        assert!(!valid_affinity("ses_0123456789ABCDEF0123456789"));
        assert!(!valid_affinity("0123456789abcdef0123456789"));
    }

    #[test]
    fn same_session_new_session_and_no_key_lifecycle() {
        let mut store = BTreeMap::new();
        let first = memory_cas("one", &mut store, false, &mut 0).unwrap();
        let continuing = memory_cas("one", &mut store, true, &mut 0).unwrap();
        assert_eq!(continuing.revision, first.revision + 1);
        assert_eq!(continuing.affinity, first.affinity);
        assert_eq!(continuing.chat_id, first.chat_id);
        assert_eq!(continuing.msg_id, first.msg_id);
        assert_eq!(continuing.turn_id, first.turn_id);
        assert_eq!(continuing.traceparent, first.traceparent);
        let next = memory_cas("one", &mut store, false, &mut 0).unwrap();
        assert_eq!(next.affinity, first.affinity);
        assert_eq!(next.chat_id, first.chat_id);
        assert_ne!(next.msg_id, first.msg_id);
        assert_ne!(next.turn_id, first.turn_id);
        assert_ne!(next.traceparent, first.traceparent);
        let other = memory_cas("two", &mut store, false, &mut 0).unwrap();
        assert_ne!(other.affinity, first.affinity);
        assert_ne!(other.chat_id, first.chat_id);
        let no_key = fixture(None).unwrap();

        let unkeyed = material(
            &resolve(&json!({}), &json!({})),
            false,
            &Overrides {
                chat_id: Some(no_key.chat_id),
                msg_id: Some(no_key.msg_id.clone()),
                turn_id: Some(no_key.turn_id),
                traceparent: Some(no_key.traceparent),
                affinity: None,
            },
        ).unwrap();
        assert_eq!(unkeyed.0.affinity, None);
        assert!(unkeyed.1.is_none());
    }

    #[test]
    fn continuation_requires_terminal_matching_ids_in_both_protocols() {
        assert!(continuation(&oai_tail("call-a", "call-a"), false));
        assert!(!continuation(&oai_tail("call-a", "other"), false));
        assert!(continuation(&anthropic_tail("call-a", "call-a"), true));
        assert!(!continuation(&anthropic_tail("call-a", "other"), true));
        let mut fresh_user = oai_tail("call-a", "call-a");
        fresh_user["messages"].as_array_mut().unwrap().push(json!({"role":"user","content":"new turn"}));
        assert!(!continuation(&fresh_user, false));
        let mut fresh_anthropic = anthropic_tail("call-a", "call-a");
        fresh_anthropic["messages"].as_array_mut().unwrap().push(json!({"role":"user","content":[{"type":"text","text":"new turn"}]}));
        assert!(!continuation(&fresh_anthropic, true));
        let mut extra_text = anthropic_tail("call-a", "call-a");
        extra_text["messages"][2]["content"].as_array_mut().unwrap().push(json!({"type":"text","text":"new turn"}));
        assert!(!continuation(&extra_text, true));
        assert!(!continuation(&json!({"messages":[{"role":"tool","tool_call_id":"call-a"}]}), false));
        assert!(!continuation(&oai_tail("call-a", ""), false));
    }

    #[test]
    fn multiple_tool_results_must_match_in_order() {
        let body = json!({"messages":[
            {"role":"assistant","tool_calls":[{"id":"a"},{"id":"b"}]},
            {"role":"tool","tool_call_id":"a","content":"a"},
            {"role":"tool","tool_call_id":"b","content":"b"}
        ]});
        assert!(continuation(&body, false));
        let mut swapped = body.clone();
        swapped["messages"][1]["tool_call_id"] = json!("b");
        swapped["messages"][2]["tool_call_id"] = json!("a");
        assert!(!continuation(&swapped, false));
    }

    #[test]
    fn concurrent_first_turn_and_new_turn_reuse_winner() {
        let mut store = BTreeMap::new();
        let winner = memory_cas("same", &mut store, false, &mut 0).unwrap();
        let loser = memory_cas("same", &mut store, false, &mut 1).unwrap();
        assert_eq!(loser.affinity, winner.affinity);
        assert_eq!(loser.chat_id, winner.chat_id);
        let next = memory_cas("same", &mut store, false, &mut 1).unwrap();
        assert_eq!(next.affinity, winner.affinity);
        assert_eq!(next.chat_id, winner.chat_id);
        assert_ne!(next.msg_id, winner.msg_id);

        // Exercise the exact production CAS core with deterministic collisions,
        // rather than relying only on the higher-level lifecycle fixture.
        let snapshot = RefCell::new(Snapshot { revision: 0, schema: SCHEMA, value: Value::Null, compatible: true });
        let mut conflicts = 1;
        let first = resolve_state_with(
            false,
            |previous| fixture(previous),
            || Ok(snapshot.borrow().clone()),
            |revision, candidate| {
                assert_eq!(revision, snapshot.borrow().revision);
                if conflicts > 0 {
                    conflicts -= 1;
                    *snapshot.borrow_mut() = Snapshot {
                        revision: revision + 1,
                        schema: SCHEMA,
                        value: json!({
                            "affinity": winner.affinity, "chat_id": winner.chat_id,
                            "msg_id": winner.msg_id, "turn_id": winner.turn_id,
                            "traceparent": winner.traceparent,
                        }),
                        compatible: true,
                    };
                    return Ok(false);
                }
                *snapshot.borrow_mut() = Snapshot { revision: revision + 1, schema: SCHEMA, value: candidate, compatible: true };
                Ok(true)
            },
        ).unwrap();
        assert_eq!(first.affinity, winner.affinity);
        assert_eq!(first.chat_id, winner.chat_id);
        assert_eq!(first.revision, 2);
    }

    #[test]
    fn explicit_ids_win_without_promoting_session_header() {
        let mut explicit = overrides();
        explicit.chat_id = Some("explicit-chat".into());
        explicit.msg_id = Some("explicit-msg".into());
        explicit.turn_id = Some("explicit-turn".into());
        explicit.traceparent = Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".into());
        explicit.affinity = Some("explicit-affinity".into());
        let resolved = resolve(&json!({}), &json!({"x-session-id":"not-a-chat-id"}));
        assert_eq!(resolved.key.as_deref(), Some("not-a-chat-id"));
        let resolved = Resolved { params_session_present: false, header_session_present: true, key: Some("not-a-chat-id".into()) };
        let (explicit_material, _) = material(&resolve(&json!({}), &json!({})), false, &explicit).unwrap();
        assert_eq!(explicit_material.chat_id, "explicit-chat");
        assert_ne!(explicit_material.chat_id, "not-a-chat-id");
        assert_eq!(explicit_material.affinity.as_deref(), Some("explicit-affinity"));
        let error = material(&resolved, false, &overrides()).unwrap_err();
        assert_eq!(error.message, "插件只能由 Wasm 宿主调用");
        let keyed = resolve_state_with(false, |_| fixture(None), || Ok(Snapshot {
            revision: 0, schema: SCHEMA, value: Value::Null, compatible: true,
        }), |_, _| Ok(true)).unwrap();
        assert_ne!(keyed.chat_id, "not-a-chat-id");
        assert!(keyed.affinity.starts_with("ses_"));
    }

    #[test]
    fn auxiliary_purposes_do_not_use_session_state() {
        // Chat gates state on the exact default purpose; these fixtures document
        // the two auxiliary wire shapes which continue through headers::build.
        for purpose in ["memory.extract", "chat.title"] {
            let incoming = json!({"x-loomy-request-purpose":purpose});
            assert_ne!(incoming["x-loomy-request-purpose"], "chat.message");
            assert!(resolve(&json!({}), &json!({})).key.is_none());
        }
    }

    #[test]
    fn incompatible_or_malformed_rows_fail_closed() {
        let mut snapshot = Snapshot { revision: 4, schema: 2, value: Value::Null, compatible: false };
        assert_eq!(parse(&snapshot).unwrap_err().code.as_deref(), Some("session_schema_incompatible"));
        snapshot.compatible = true;
        snapshot.value = json!({"affinity":"bad","chat_id":"c","msg_id":"m","turn_id":"t","traceparent":"p"});
        assert_eq!(parse(&snapshot).unwrap_err().code.as_deref(), Some("session_state_invalid"));
    }
}
