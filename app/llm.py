"""LLM backends for the assistant. The orchestrator (assistant.py) owns all safety; backends only decide
"which tool next" or "what to say", so swapping providers never weakens a guardrail.

Internal message format (provider-neutral):
  {"role":"user","text":...}
  {"role":"assistant","text":..., "tool_calls":[ToolCall...], "raw":<provider payload to replay verbatim>}
  {"role":"tool","name":..., "result":{...}}

Backends: GeminiLLM (REST function calling, needs GEMINI_API_KEY) and RuleBasedLLM (offline, deterministic).
"""
import os
import re
import time
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class LLMTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict | None = None


class LLM:
    name = "base"

    def generate(self, system: str, messages: list[dict], tool_specs: list[dict]) -> LLMTurn:
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
# Gemini (REST)
# --------------------------------------------------------------------------------------------------
def _gemini_schema(node):
    """JSON-schema (lowercase types) -> Gemini Schema (uppercase type enum)."""
    if isinstance(node, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else _gemini_schema(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_gemini_schema(x) for x in node]
    return node


class GeminiLLM(LLM):
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None, model: str | None = None, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0, retries: int = 2):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.name = f"gemini:{self.model}"
        self.retries = retries
        self._client = httpx.Client(timeout=timeout, transport=transport)

    # -- translation ---------------------------------------------------------------------------
    @staticmethod
    def to_contents(messages: list[dict]) -> list[dict]:
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["text"]}]})
            elif m["role"] == "assistant":
                # Replay the provider's own content verbatim (keeps thought signatures intact for function calling).
                contents.append(m["raw"] if m.get("raw") else {"role": "model", "parts": [{"text": m.get("text", "")}]})
            elif m["role"] == "tool":
                part = {"functionResponse": {"name": m["name"], "response": {"untrusted_data": m["result"]}}}
                if contents and contents[-1]["role"] == "user" and contents[-1]["parts"] and "functionResponse" in contents[-1]["parts"][0]:
                    contents[-1]["parts"].append(part)  # parallel results go in one turn
                else:
                    contents.append({"role": "user", "parts": [part]})
        return contents

    def build_request(self, system: str, messages: list[dict], tool_specs: list[dict]) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": self.to_contents(messages),
            "tools": [{"functionDeclarations": [
                {"name": s["name"], "description": s["description"], "parameters": _gemini_schema(s["parameters"])}
                for s in tool_specs]}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }

    @staticmethod
    def parse_response(data: dict) -> LLMTurn:
        cands = data.get("candidates") or []
        if not cands or "content" not in cands[0]:
            reason = (data.get("promptFeedback") or {}).get("blockReason") or (cands[0].get("finishReason") if cands else "no candidates")
            raise LLMError(f"empty Gemini response ({reason})")
        content = cands[0]["content"]
        parts = content.get("parts", [])
        calls = [ToolCall(p["functionCall"]["name"], dict(p["functionCall"].get("args") or {})) for p in parts if "functionCall" in p]
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
        return LLMTurn(text=text.strip(), tool_calls=calls, raw={"role": "model", "parts": parts})

    # -- call ----------------------------------------------------------------------------------
    def generate(self, system, messages, tool_specs) -> LLMTurn:
        body = self.build_request(system, messages, tool_specs)
        url = f"{self.BASE}/models/{self.model}:generateContent"
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(url, json=body, headers={"x-goog-api-key": self.api_key})
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}"
            else:
                if r.status_code == 200:
                    return self.parse_response(r.json())
                last = f"HTTP {r.status_code}"
                if r.status_code not in (429, 500, 503):
                    break  # not retryable
            time.sleep(min(2 ** attempt * 0.5, 4))
        raise LLMError(f"Gemini request failed ({last})")


# --------------------------------------------------------------------------------------------------
# Offline rule-based backend (deterministic fallback; understands the demo intents only)
# --------------------------------------------------------------------------------------------------
_LOANS = re.compile(r"\b(borrowed|borrowing|due|loans?|fines?|overdue|penalt\w+|my (books|items|account))\b", re.I)
_RENEW = re.compile(r"\brenew\w*\b", re.I)
_RESERVE = re.compile(r"\b(reserve|hold|waitlist)\b", re.I)
_RECOMMEND = re.compile(r"\b(recommend\w*|suggest\w*|what should i read|reading list|what to read)\b", re.I)
_ANALYTICS = [  # (report, pattern) in priority order; only used when the librarian_analytics tool is offered (librarians)
    ("catalog_gaps", re.compile(r"catalog gaps?|gaps? in the catalog|not in (the )?catalog|unanswered|can'?t find|couldn'?t find|searching for|missing titles|unmet", re.I)),
    ("forecast", re.compile(r"forecast|what (should|to) (we )?(acquire|buy)|acquisitions?|extra copies|buy more|demand (next|forecast)", re.I)),
    ("utilization", re.compile(r"under-?used|utili[sz]ation|never (been )?borrowed|unused|low circulation|not (being )?used", re.I)),
    ("overdue", re.compile(r"overdue (trends?|rate|items|summary|books)|late returns?|fines? (trends?|total|outstanding)", re.I)),
    ("circulation", re.compile(r"circulation (patterns?|trends?)|most borrowed|top borrowed|busiest|borrowing trends?", re.I)),
    ("waitlists", re.compile(r"wait-?lists?", re.I)),
    ("dashboard", re.compile(r"dashboard|overview|kpis?|snapshot|how is the library doing", re.I)),
]
_AVAIL_Q = re.compile(r"\b(?:is|are)\s+(?P<t>.+?)\s+available\b|\bavailability of\s+(?P<t2>.+)|\bcopies of\s+(?P<t3>.+)|\bin stock\b", re.I)
_AVAIL_FLAG = re.compile(r"\b(currently |now )?available\b", re.I)


def _clean_title(t: str) -> str:
    return re.sub(r"[?.!\"'“”]+", "", t).strip()


def _render_analytics(res: dict) -> str:
    """Deterministic text for an analytics result; titles are [#id] markers (rendered/verified by the orchestrator)."""
    if res["status"] != "ok":
        return res.get("message", "I couldn't run that report.")
    rep, rows = res["report"], res.get("rows", [])
    if rep == "waitlists":
        if not rows:
            return "No titles currently have a waitlist."
        return (f"{res['titles_with_waitlist']} titles have waitlists ({res['total_waiting']} readers waiting). Longest:\n" + "\n".join(
            f"{n}. [#{r['resource_id']}]: {r['waiting']} waiting for {r['copies']} cop{'y' if r['copies'] == 1 else 'ies'} (oldest {r['oldest_wait_days']} days) - {r['action']}"
            for n, r in enumerate(rows, 1)))
    if rep == "overdue":
        sm = res["summary"]
        out = [f"{sm['overdue_now']} of {sm['open_loans']} open loans are overdue ({sm['overdue_rate']:.0%}); fines accruing {sm['fines_accruing']:.2f} {sm['currency']}."]
        out += [f"- [#{i['resource_id']}] loan {i['loan_id']} (user {i['user_id']}): {i['days_overdue']} days overdue, fine {i['fine']:.2f}" for i in res["items"]]
        out += ["Late-return trend:"] + [f"- {t['month']}: {t['late']} of {t['returned']} returns late" + (f" ({t['late_rate']:.0%})" if t["late_rate"] is not None else "")
                                         for t in res["trend"][-3:]]
        return "\n".join(out)
    if rep == "circulation":
        out = [f"{res['total_loans']} loans in the last {len(res['monthly'])} months. Most borrowed:"]
        out += [f"{n}. [#{r['resource_id']}]: {r['loans']} loans by {r['unique_readers']} readers" for n, r in enumerate(rows, 1)]
        out += ["Recent months: " + ", ".join(f"{m['month']}: {m['loans']}" for m in res["monthly"][-3:])]
        return "\n".join(out)
    if rep == "utilization":
        sm = res["summary"]
        out = [f"{sm['copies_in_use_now_pct']}% of physical copies are in use now; {sm['never_borrowed_titles']} physical and {sm['digital_unused_titles']} digital titles had no use in {sm['window_days']} days."]
        out += [f"- Never borrowed: [#{r['resource_id']}] ({r['copies']} copies)" for r in res["never_borrowed"]]
        out += [f"- Underused: [#{r['resource_id']}] ({r['loans_per_copy']} loans per copy)" for r in res["underused"][:3]]
        out += [f"- Unused online: [#{r['resource_id']}]" for r in res["digital_unused"]]
        return "\n".join(out)
    if rep == "catalog_gaps":
        if not rows:
            return f"No repeated unanswered searches in the last {res['window_days']} days."
        rate = f" Search success rate: {res['search_success_rate']:.0%} over {res['searches']} searches." if res.get("search_success_rate") is not None else ""
        return "Repeated searches the catalog could not answer:\n" + "\n".join(
            f"{n}. {r['query']} - {r['unanswered_searches']} searches by {r['distinct_users']} users ({r['action']})" for n, r in enumerate(rows, 1)) + "\n" + rate
    if rep == "forecast":
        need = [r for r in rows if r["suggested_extra_copies"] > 0]
        bt = res["backtest"]
        head = ("Suggested acquisitions for the next %d days:\n" % res["horizon_days"] + "\n".join(
            f"{n}. [#{r['resource_id']}]: +{r['suggested_extra_copies']} copy (expected demand {r['expected_demand']} vs capacity {r['capacity']}; confidence {r['confidence']})"
            for n, r in enumerate(need, 1))) if need else f"No title is forecast to exceed capacity in the next {res['horizon_days']} days."
        err = f"\nBacktest error (WAPE) {bt['wape']} vs {bt['baseline_last_month_wape']} for a last-month baseline; treat forecasts as indicative." if bt.get("wape") is not None else ""
        return head + err
    k = res["kpis"]      # dashboard
    out = [f"Library snapshot as of {res['as_of']}:",
           f"- Loans in last 30 days: {k['loans_last_30_days']}; active loans {k['active_loans']}; copies in use {k['copies_in_use_pct']}%",
           f"- Overdue now: {k['overdue_now']} ({k['overdue_rate']:.0%}); fines accruing {k['fines_accruing']:.2f} {k['currency']}",
           f"- {k['titles_with_waitlist']} titles with waitlists ({k['readers_waiting']} readers waiting)"]
    if res.get("rows"):
        out.append(f"- Longest waitlist: [#{res['rows'][0]['resource_id']}] with {res['rows'][0]['waiting']} waiting")
    if res.get("top_gaps"):
        out.append(f"- Top unmet search: {res['top_gaps'][0]['query']} ({res['top_gaps'][0]['unanswered_searches']} searches)")
    out.append(f"- Never-borrowed physical titles (365d): {k['never_borrowed_titles_365d']}; unused digital titles: {k['digital_unused_titles_365d']}")
    return "\n".join(out)


class RuleBasedLLM(LLM):
    name = "rule-based-offline"

    def generate(self, system, messages, tool_specs) -> LLMTurn:
        i = max(k for k, m in enumerate(messages) if m["role"] == "user")
        text = messages[i]["text"]
        results = [(m["name"], m["result"]) for m in messages[i + 1:] if m["role"] == "tool"]

        def call(name, **args):
            return LLMTurn(tool_calls=[ToolCall(name, args)])

        # ---- librarian analytics (only offered to librarians) ----
        if any(sp["name"] == "librarian_analytics" for sp in tool_specs):
            rep = next((name for name, pat in _ANALYTICS if pat.search(text)), None)
            if rep:
                if not results:
                    return call("librarian_analytics", report=rep, top_n=5)
                return LLMTurn(text=_render_analytics(results[-1][1]))

        # ---- renew: my_loans -> renew(loan_id) ----
        if _RENEW.search(text):
            if not results:
                return call("my_loans")
            name, res = results[-1]
            if name == "my_loans":
                if res["status"] != "ok":
                    return LLMTurn(text=res.get("message", "I couldn't read your loans."))
                low = text.lower()
                match = [l for l in res["loans"] if l["title"].lower() in low or any(w in low for w in l["title"].lower().split() if len(w) > 5)]
                pick = match[0] if len(match) == 1 else (res["loans"][0] if len(res["loans"]) == 1 else None)
                if pick:
                    return call("renew", loan_id=pick["loan_id"])
                if not res["loans"]:
                    return LLMTurn(text="You have no items on loan to renew.")
                return LLMTurn(text="Which item would you like to renew? You currently have: " +
                               "; ".join(f"{l['title']} (due {l['due_at']})" for l in res["loans"]) + ".")
            return LLMTurn(text=res.get("message", "Done."))

        # ---- loans / fines / due dates ----
        if _LOANS.search(text) and not _RESERVE.search(text):
            if not results:
                return call("my_loans", **({"due_within_days": 7} if re.search(r"this week|due soon|next 7", text, re.I) else {}))
            _, res = results[-1]
            if res["status"] != "ok":
                return LLMTurn(text=("Please sign in to see your borrowing details." if res["status"] == "denied" else res.get("message", "")))
            if not res["loans"]:
                return LLMTurn(text="You have no matching items on loan.")
            lines = [f"- {l['title']}: due {l['due_at']}" + (f" (OVERDUE by {l['days_overdue']} days, fine {l['fine']:.2f} {res['currency']})" if l["overdue"]
                                                          else f" ({l['days_left']} days left)") + ("; renewable" if l["renewable"] else "") for l in res["loans"]]
            tail = f"\nTotal fines: {res['total_fine']:.2f} {res['currency']}." if res["total_fine"] else ""
            return LLMTurn(text="Your items:\n" + "\n".join(lines) + tail)

        # ---- reserve: search/check -> check_availability ... -> reserve ----
        if _RESERVE.search(text):
            query = re.sub(r"\b(please|can you|could you|and|reserve|hold|it|me|for|an?|the|put|on|place|find)\b", " ", text, flags=re.I)
            query = re.sub(r"\bavailable\b", " ", query, flags=re.I).strip()
            if not results:
                return call("search_resources", query=query or text, top_k=5)
            if results[-1][0] == "reserve":  # the tool already answered (e.g. denied for anonymous users)
                return LLMTurn(text=results[-1][1].get("message", "I couldn't place that reservation."))
            name, res = results[0]
            if name != "search_resources" or res["status"] != "ok":
                return LLMTurn(text=res.get("message", "I couldn't find a matching title in the catalog."))
            named = [h for h in res["results"] if h["title"].lower() in text.lower()]
            if named:  # the user named a specific title: reserve exactly that one
                return call("reserve", resource_id=named[0]["resource_id"])
            hits = [h for h in res["results"] if h["access_url"] is None] or res["results"]
            checks = [r for n, r in results if n == "check_availability"]
            for h, chk in zip(hits, checks):
                if chk.get("available"):
                    return call("reserve", resource_id=h["resource_id"])
            if len(checks) < min(len(hits), 5):
                return call("check_availability", resource_id=hits[len(checks)]["resource_id"])
            return call("reserve", resource_id=hits[0]["resource_id"])  # none free: join the waitlist for the best match

        # ---- recommendations (personalised, explainable) ----
        if _RECOMMEND.search(text):
            if not results:
                lvl = re.search(r"\b(beginner|intermediate|advanced)\b", text, re.I)
                topic = re.sub(r"\b(recommend\w*|suggest\w*|what should i read|reading list|what to read|me|some|for|my|the|"
                               r"(that|which) are|currently|available|now|resources?|books?|beginner|intermediate|advanced|learning|to|an?)\b", " ", text, flags=re.I)
                topic = re.sub(r"[^\w\s-]", " ", topic).strip()
                args = {"top_k": 5}
                if len(topic.split()) >= 1 and topic:
                    args["topic"] = " ".join(topic.split())
                if lvl:
                    args["level"] = lvl.group(1).lower()
                if _AVAIL_FLAG.search(text):
                    args["available_only"] = True
                return call("recommend", **args)
            _, res = results[-1]
            if res["status"] != "ok":
                return LLMTurn(text=res.get("message") or "I couldn't find recommendations matching that in the library's authorized sources (not_found).")
            lines = []
            for n, h in enumerate(res["results"][:5], 1):
                where = {"digital": "online", "available": "available now", "waitlist": f"{h['waitlist']} waiting"}[h["availability"]]
                lines.append(f"{n}. [#{h['resource_id']}] ({h['level']}, {where})\n   Why: " + "; ".join(h["reasons"][:2]))
            p = res["personalization"]
            note = f"\n\n{p['note']}" if p.get("note") else "\n\nPersonalised using: " + ", ".join(p["used"]).replace("_", " ") + "."
            return LLMTurn(text="Recommended resources:\n" + "\n".join(lines) + note + "\nNext action: say \"reserve <title>\" to hold one, or open the online ones.")

        # ---- availability of a named title ----
        m = _AVAIL_Q.search(text)
        title = _clean_title((m.group("t") or m.group("t2") or m.group("t3") or "") if m else "")
        if title.lower() in {"currently", "now", "still", "all", "they", "these", "those", "it"}:
            title = ""  # "...that are currently available" is a filter on a search, not a question about a title
        if title:
            if not results:
                return call("check_availability", title=title)
            _, res = results[-1]
            if res["status"] == "ok":
                if res["format"] == "digital":
                    return LLMTurn(text=f"[#{res['resource_id']}] is a digital resource, available now: {res['access_url']}")
                where = "; ".join(f"{l['branch']} ({l['shelf']})" for l in res["locations"])
                return LLMTurn(text=(f"Yes: [#{res['resource_id']}] has {res['copies_available']} of {res['copies_total']} copies available at {where}."
                                     if res["available"] else
                                     f"[#{res['resource_id']}] is currently checked out ({res['waitlist_length']} waiting). I can place a reservation if you'd like."))
            if res["status"] == "ambiguous":
                return LLMTurn(text="Several titles match: " + "; ".join(f"[#{c['resource_id']}]" for c in res["candidates"][:5]) + ". Which one do you mean?")
            return LLMTurn(text="I couldn't find that title in any authorized source (not_found).")

        # ---- default: search / recommend (optionally only what is available now) ----
        want_avail = bool(_AVAIL_FLAG.search(text))
        if not results:
            return call("search_resources", query=text, top_k=5)
        name, res = results[0]
        if res["status"] != "ok":
            return LLMTurn(text="I couldn't find anything matching that in the library's authorized sources (not_found).")
        hits = res["results"]
        if want_avail:
            checks = [r for n, r in results if n == "check_availability"]
            physical = [h for h in hits if h["access_url"] is None]
            if len(checks) < min(len(physical), 5):
                return call("check_availability", resource_id=physical[len(checks)]["resource_id"])
            free = {c["resource_id"] for c in checks if c.get("available")}
            hits = [h for h in hits if h["access_url"] or h["resource_id"] in free]
            if not hits:
                return LLMTurn(text="Matching titles exist but none is currently available; I can reserve one for you.")
        lines = [f"{n}. [#{h['resource_id']}] ({h['type'].replace('_', ' ')}, {h['level']}) — " +
                 (f"online: {h['access_url']}" if h["access_url"] else "physical copy") for n, h in enumerate(hits[:5], 1)]
        why = "Why these: matched your terms (" + ", ".join(sorted({t for h in hits[:5] for t in h['matched_terms']})) + ") in the catalog metadata."
        return LLMTurn(text="Here is what I found:\n" + "\n".join(lines) + "\n\n" + why)


def get_llm(kind: str | None = None) -> LLM:
    """LLM=auto|gemini|rules. auto = Gemini when GEMINI_API_KEY is set, else the offline rule-based backend."""
    kind = (kind or os.environ.get("LLM", "auto")).lower()
    if kind == "rules" or (kind == "auto" and not os.environ.get("GEMINI_API_KEY")):
        return RuleBasedLLM()
    return GeminiLLM()
