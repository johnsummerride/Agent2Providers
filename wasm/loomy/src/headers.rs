//! 官方 0.9.39 chat/completions 头对齐（23 条真实抓包取证）。
//!
//! 入站头只认白名单：chatid, x-session-id, msgid, x-request-id, turnid,
//! traceparent, invokeorigin, x-loomy-request-purpose, x-session-affinity,
//! x-parent-session-id。客户端 Authorization/token/cookie/自定义 UA 一律不读不转。
//! 官方没有 token 头，ChatId/MsgId/TurnId 一律带 chat-/msg- 前缀，affinity 与
//! parent 只透传调用方给的值、绝不自行合成。

use agent2api_wasm_sdk::{json, Result, Value};

use super::{sign, CLIENT_VERSION};

/// 官方 opencode 客户端 UA，chat 与 image 两个分支都原样发送。
const USER_AGENT: &str = "opencode/1.14.22 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13";

const CHAT_PURPOSE: &str = "chat.message";
const IMAGE_PURPOSE: &str = "image.generate";
const MEMORY_PURPOSE: &str = "memory.extract";
const TITLE_PURPOSE: &str = "chat.title";

const CHAT_PREFIX: &str = "chat-";
const MSG_PREFIX: &str = "msg-";

fn header(headers: &Value, name: &str) -> Option<String> {
    let value = headers
        .as_object()?
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))?
        .1
        .as_str()?
        .trim();
    (!value.is_empty()
        && value.len() <= 256
        && value.is_ascii()
        && !value.bytes().any(|byte| byte.is_ascii_control()))
    .then(|| value.to_string())
}

fn valid_traceparent(value: &str) -> bool {
    let parts: Vec<&str> = value.split('-').collect();
    let hex = |value: &str, length: usize| {
        value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    };
    parts.len() == 4
        && parts[0] == "00"
        && hex(parts[1], 32)
        && hex(parts[2], 16)
        && hex(parts[3], 2)
        && parts[1].bytes().any(|byte| byte != b'0')
        && parts[2].bytes().any(|byte| byte != b'0')
}

/// 官方裸 UUID 入站也会被补前缀；已带前缀的不重复补。
fn prefixed(value: &str, prefix: &str) -> String {
    if value.starts_with(prefix) {
        value.to_string()
    } else {
        format!("{prefix}{value}")
    }
}

fn purpose(incoming: &Value, image: bool) -> String {
    header(incoming, "x-loomy-request-purpose").unwrap_or_else(|| {
        if image {
            IMAGE_PURPOSE
        } else {
            CHAT_PURPOSE
        }
        .to_string()
    })
}

/// 纯函数：不取随机、不碰宿主，相同输入必得相同头集合。
fn assemble(
    session: &str,
    incoming: &Value,
    image: bool,
    chat_id: &str,
    msg_id: &str,
    traceparent: &str,
) -> Value {
    let chat_id = prefixed(chat_id, CHAT_PREFIX);
    let msg_id = prefixed(msg_id, MSG_PREFIX);
    let purpose = purpose(incoming, image);
    let turn_id = match header(incoming, "turnid") {
        Some(value) => prefixed(&value, MSG_PREFIX),
        None => msg_id.clone(),
    };
    // 官方只接受 user|system，其它取值（如 nexus）丢弃并回落 user。
    let origin = match header(incoming, "invokeorigin").as_deref() {
        Some("system") => "system",
        _ => "user",
    };
    let mut headers = json!({
        "Content-Type": "application/json",
        "Accept": if image { "application/json" } else { "*/*" },
        "User-Agent": USER_AGENT,
        "Authorization": format!("Bearer {session}"),
        "traceparent": traceparent,
        "ChatId": chat_id,
        "MsgId": msg_id,
        "InvokeOrigin": origin,
        "X-Loomy-Request-Purpose": purpose
    });
    if !image {
        headers["loomy-version"] = json!(CLIENT_VERSION);
    }
    // 官方 memory.extract 抓包里没有 turnid。
    if purpose != MEMORY_PURPOSE {
        headers["TurnId"] = json!(turn_id);
    }
    // 不提供就不输出，绝不自行随机或伪造。
    if let Some(affinity) = header(incoming, "x-session-affinity") {
        headers["x-session-affinity"] = json!(affinity);
    }
    // 官方 chat.title 不带 parent；parent 只透传调用方给的值。
    if purpose != TITLE_PURPOSE {
        if let Some(parent) = header(incoming, "x-parent-session-id") {
            headers["x-parent-session-id"] = json!(parent);
        }
    }
    headers
}

pub fn build(session: &str, incoming: &Value, image: bool) -> Result<Value> {
    let chat_id = match header(incoming, "chatid").or_else(|| header(incoming, "x-session-id")) {
        Some(value) => value,
        None => sign::uuid()?,
    };
    let msg_id = match header(incoming, "msgid").or_else(|| header(incoming, "x-request-id")) {
        Some(value) => value,
        None => sign::uuid()?,
    };
    let traceparent = match header(incoming, "traceparent").filter(|value| valid_traceparent(value))
    {
        Some(value) => value.to_ascii_lowercase(),
        None => format!("00-{}-{}-01", sign::random_hex(16)?, sign::random_hex(8)?),
    };
    Ok(assemble(
        session,
        incoming,
        image,
        &chat_id,
        &msg_id,
        &traceparent,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = "session-token-abc";
    const CHAT: &str = "11111111-2222-4333-8444-555555555555";
    const MSG: &str = "66666666-7777-4888-9999-aaaaaaaaaaaa";
    const TRACE: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const UA: &str = "opencode/1.14.22 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.13";

    fn assemble_with(incoming: Value, image: bool) -> Value {
        assemble(SESSION, &incoming, image, CHAT, MSG, TRACE)
    }

    fn get<'a>(headers: &'a Value, name: &str) -> Option<&'a str> {
        headers.get(name).and_then(Value::as_str)
    }

    fn has_key(headers: &Value, name: &str) -> bool {
        headers
            .as_object()
            .map(|object| object.keys().any(|key| key.eq_ignore_ascii_case(name)))
            .unwrap_or(false)
    }

    fn len_of(headers: &Value) -> usize {
        headers.as_object().map(|object| object.len()).unwrap_or(0)
    }

    // (a) chat.message 默认头集合逐项对齐官方抓包。
    #[test]
    fn chat_default_head_set() {
        let headers = assemble_with(json!({}), false);
        assert!(get(&headers, "token").is_none(), "官方 23/23 无 token 头");
        assert!(!has_key(&headers, "token"));
        assert_eq!(get(&headers, "Content-Type"), Some("application/json"));
        assert_eq!(get(&headers, "Accept"), Some("*/*"));
        assert_eq!(get(&headers, "User-Agent"), Some(UA));
        assert_eq!(
            get(&headers, "Authorization"),
            Some("Bearer session-token-abc")
        );
        assert_eq!(get(&headers, "loomy-version"), Some(CLIENT_VERSION));
        assert_eq!(get(&headers, "ChatId").unwrap(), format!("chat-{CHAT}"));
        assert_eq!(get(&headers, "ChatId").unwrap().len(), 41);
        assert_eq!(get(&headers, "MsgId").unwrap(), format!("msg-{MSG}"));
        assert_eq!(get(&headers, "MsgId").unwrap().len(), 40);
        assert_eq!(get(&headers, "TurnId"), get(&headers, "MsgId"));
        assert_eq!(get(&headers, "InvokeOrigin"), Some("user"));
        assert_eq!(
            get(&headers, "X-Loomy-Request-Purpose"),
            Some("chat.message")
        );
        assert_eq!(get(&headers, "traceparent"), Some(TRACE));
        assert!(!has_key(&headers, "x-session-affinity"));
        assert!(!has_key(&headers, "x-parent-session-id"));
        // 键集封闭：默认 11 个头，多一个就说明夹带了客户端头。
        assert_eq!(len_of(&headers), 11);
    }

    // (b) 裸 UUID 补前缀；已带前缀不重复补；入站名大小写不敏感且做合法性校验。
    #[test]
    fn prefix_normalization_is_idempotent() {
        let bare = assemble_with(json!({ "chatid": CHAT, "msgid": MSG }), false);
        assert_eq!(get(&bare, "ChatId").unwrap(), format!("chat-{CHAT}"));
        assert_eq!(get(&bare, "MsgId").unwrap(), format!("msg-{MSG}"));

        let tagged = build(
            SESSION,
            &json!({ "ChatId": "chat-already", "MsgId": "msg-already", "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01" }),
            false,
        )
        .expect("入站提供 ID 不需要宿主随机");
        assert_eq!(get(&tagged, "ChatId"), Some("chat-already"));
        assert_eq!(get(&tagged, "MsgId"), Some("msg-already"));

        assert_eq!(prefixed("chat-x", CHAT_PREFIX), "chat-x");
        assert_eq!(prefixed("msg-x", MSG_PREFIX), "msg-x");
        assert_eq!(prefixed("x", MSG_PREFIX), "msg-x");

        assert_eq!(
            header(&json!({ "X-Session-Id": "sess-1" }), "x-session-id").as_deref(),
            Some("sess-1")
        );
        assert_eq!(
            header(&json!({ "X-Request-Id": "req-1" }), "x-request-id").as_deref(),
            Some("req-1")
        );
        assert!(header(&json!({ "chatid": "  " }), "chatid").is_none());
        assert!(header(&json!({ "chatid": "中文" }), "chatid").is_none());
        assert!(header(&json!({ "chatid": "line\nbreak" }), "chatid").is_none());
    }

    // (c) 入站 turnid 覆盖默认值，并且同样补 msg- 前缀。
    #[test]
    fn incoming_turn_id_wins() {
        let tagged = assemble_with(json!({ "turnid": "msg-turn-override" }), false);
        assert_eq!(get(&tagged, "TurnId"), Some("msg-turn-override"));

        let bare = assemble_with(
            json!({ "turnid": "abcdefae-0000-4000-8000-000000000001" }),
            false,
        );
        assert_eq!(
            get(&bare, "TurnId"),
            Some("msg-abcdefae-0000-4000-8000-000000000001")
        );
        assert_ne!(get(&bare, "TurnId"), get(&bare, "MsgId"));
    }

    // (d) memory.extract 无 TurnId，affinity 与 parent 原样透传。
    #[test]
    fn memory_extract_drops_turn_id_keeps_parent() {
        let headers = assemble_with(
            json!({
                "x-loomy-request-purpose": "memory.extract",
                "x-session-affinity": "affinity-parent",
                "x-parent-session-id": "parent-session"
            }),
            false,
        );
        assert_eq!(
            get(&headers, "X-Loomy-Request-Purpose"),
            Some("memory.extract")
        );
        assert!(!has_key(&headers, "TurnId"));
        assert_eq!(get(&headers, "x-session-affinity"), Some("affinity-parent"));
        assert_eq!(get(&headers, "x-parent-session-id"), Some("parent-session"));
        assert_eq!(get(&headers, "loomy-version"), Some(CLIENT_VERSION));
    }

    // (e) chat.title 有 TurnId、只带自身 affinity，入站 parent 被强制丢弃。
    #[test]
    fn chat_title_never_sends_parent() {
        let headers = assemble_with(
            json!({
                "x-loomy-request-purpose": "chat.title",
                "x-session-affinity": "affinity-self",
                "x-parent-session-id": "parent-session"
            }),
            false,
        );
        assert_eq!(get(&headers, "X-Loomy-Request-Purpose"), Some("chat.title"));
        assert_eq!(get(&headers, "TurnId"), get(&headers, "MsgId"));
        assert_eq!(get(&headers, "x-session-affinity"), Some("affinity-self"));
        assert!(!has_key(&headers, "x-parent-session-id"));
    }

    // (f) invokeorigin 只认 user|system，nexus 回落 user。
    #[test]
    fn invoke_origin_whitelist() {
        assert_eq!(
            get(
                &assemble_with(json!({ "invokeorigin": "system" }), false),
                "InvokeOrigin"
            ),
            Some("system")
        );
        assert_eq!(
            get(
                &assemble_with(json!({ "invokeorigin": "user" }), false),
                "InvokeOrigin"
            ),
            Some("user")
        );
        assert_eq!(
            get(
                &assemble_with(json!({ "invokeorigin": "nexus" }), false),
                "InvokeOrigin"
            ),
            Some("user")
        );
        assert_eq!(
            get(&assemble_with(json!({}), false), "InvokeOrigin"),
            Some("user")
        );
    }

    // (g) image 分支：JSON 响应、无 loomy-version、purpose 默认 image.generate。
    #[test]
    fn image_branch_headers() {
        let headers = assemble_with(json!({}), true);
        assert_eq!(get(&headers, "Accept"), Some("application/json"));
        assert!(!has_key(&headers, "loomy-version"));
        assert_eq!(
            get(&headers, "X-Loomy-Request-Purpose"),
            Some("image.generate")
        );
        assert_eq!(get(&headers, "User-Agent"), Some(UA));
        assert_eq!(
            get(&headers, "Authorization"),
            Some("Bearer session-token-abc")
        );
        assert_eq!(get(&headers, "Content-Type"), Some("application/json"));
        assert_eq!(get(&headers, "traceparent"), Some(TRACE));
    }

    // (h) 不泄露客户端鉴权头与自定义 UA。
    #[test]
    fn client_credentials_never_forwarded() {
        let headers = assemble_with(
            json!({
                "Authorization": "Bearer client-own-token",
                "token": "client-token",
                "Cookie": "sid=client-cookie",
                "user-agent": "curl/8.4.0"
            }),
            false,
        );
        assert_eq!(
            get(&headers, "Authorization"),
            Some("Bearer session-token-abc")
        );
        assert_eq!(get(&headers, "User-Agent"), Some(UA));
        assert!(!has_key(&headers, "token"));
        assert!(!has_key(&headers, "cookie"));
        assert_eq!(len_of(&headers), 11);
    }

    // (i) 合法 W3C traceparent 透传为小写；非法值不会被当成合法值。
    #[test]
    fn traceparent_passthrough_and_validation() {
        let upper = "00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01";
        assert!(valid_traceparent(upper));
        let headers = assemble_with(json!({ "traceparent": upper }), false);
        assert_eq!(
            get(&headers, "traceparent"),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        );
        assert!(!valid_traceparent(
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01"
        ));
        assert!(!valid_traceparent(
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        ));
        assert!(!valid_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7"
        ));
    }

    // build 在入站齐备时不碰宿主随机：traceparent 转小写后原样透传。
    #[test]
    fn build_without_host_randomness() {
        let headers = build(
            SESSION,
            &json!({
                "chatid": CHAT,
                "msgid": MSG,
                "traceparent": "00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01"
            }),
            false,
        )
        .expect("入站头齐备时不需要宿主随机");
        assert_eq!(
            get(&headers, "traceparent"),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        );
        assert_eq!(get(&headers, "ChatId").unwrap(), format!("chat-{CHAT}"));
        assert_eq!(get(&headers, "MsgId").unwrap(), format!("msg-{MSG}"));
    }

    // 缺 ChatId 时 build 必须依赖宿主随机能力（非 wasm 目标直接报错，不做伪随机）。
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn build_requires_host_random_when_ids_missing() {
        assert!(build(SESSION, &json!({}), false).is_err());
    }
}