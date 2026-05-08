#[derive(Clone, Copy)]
pub(crate) struct ResponseBudget {
    pub(crate) max_tokens: u32,
    pub(crate) mode: &'static str,
}

pub(crate) fn response_budget(
    max_tokens: u32,
    backlog_ms: u64,
    low_water_ms: u64,
    high_water_ms: u64,
) -> ResponseBudget {
    if backlog_ms >= high_water_ms {
        ResponseBudget {
            max_tokens: max_tokens.min(24),
            mode: "very_low",
        }
    } else if backlog_ms >= low_water_ms {
        ResponseBudget {
            max_tokens: max_tokens.min(40),
            mode: "low",
        }
    } else {
        ResponseBudget {
            max_tokens,
            mode: "normal",
        }
    }
}

pub(crate) fn response_instructions_for_budget(
    base_instructions: &str,
    budget: ResponseBudget,
    backlog_ms: u64,
) -> String {
    match budget.mode {
        "very_low" => format!(
            "{base_instructions} Backlog is {backlog_ms}ms. Use at most 6 words. Prefer a direct question. Stop immediately after the question."
        ),
        "low" => format!(
            "{base_instructions} Backlog is {backlog_ms}ms. Use at most 10 words. One simple sentence only."
        ),
        _ => format!(
            "{base_instructions} Use at most 14 words. One sentence. Stop as soon as the next user action is clear."
        ),
    }
}
