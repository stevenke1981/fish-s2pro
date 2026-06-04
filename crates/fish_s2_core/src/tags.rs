#[derive(Debug, Clone)]
pub struct ControlTag {
    pub label: &'static str,
    pub token: &'static str,
}

pub const CONTROL_TAGS: &[ControlTag] = &[
    ControlTag {
        label: "停頓",
        token: "[pause]",
    },
    ControlTag {
        label: "強調",
        token: "[emphasis]",
    },
    ControlTag {
        label: "笑",
        token: "[laughing]",
    },
    ControlTag {
        label: "輕笑",
        token: "[chuckle]",
    },
    ControlTag {
        label: "耳語",
        token: "[whisper]",
    },
    ControlTag {
        label: "低聲",
        token: "[low voice]",
    },
    ControlTag {
        label: "興奮",
        token: "[excited]",
    },
    ControlTag {
        label: "悲傷",
        token: "[sad]",
    },
    ControlTag {
        label: "憤怒",
        token: "[angry]",
    },
    ControlTag {
        label: "驚訝",
        token: "[surprised]",
    },
    ControlTag {
        label: "嘆息",
        token: "[sigh]",
    },
    ControlTag {
        label: "吸氣",
        token: "[inhale]",
    },
    ControlTag {
        label: "呼氣",
        token: "[exhale]",
    },
    ControlTag {
        label: "音量↑",
        token: "[volume up]",
    },
    ControlTag {
        label: "音量↓",
        token: "[volume down]",
    },
    ControlTag {
        label: "唱歌",
        token: "[singing]",
    },
];

pub fn insert_tag_at_cursor(text: &str, cursor: usize, tag: &str) -> (String, usize) {
    let cursor = cursor.min(text.len());
    let mut out = String::with_capacity(text.len() + tag.len() + 1);
    out.push_str(&text[..cursor]);
    if !tag.is_empty() && !out.ends_with(' ') && !out.is_empty() {
        out.push(' ');
    }
    out.push_str(tag);
    if cursor < text.len() && !text[cursor..].starts_with(' ') {
        out.push(' ');
    }
    out.push_str(&text[cursor..]);
    let new_cursor = out.len();
    (out, new_cursor)
}
