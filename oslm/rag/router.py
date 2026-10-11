from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RouteMode = Literal["concept", "code_write", "bug_find", "trace", "greeting", "off_topic", "os_out_of_scope", "empty"]

SYNC_TERMS = re.compile(
    r"\b(sync\w*|race|racing|critical|mutual|mutex\w*|lock\w*|unlock\w*|spin\w*|semaph\w*|sem_\w+|"
    r"binary sem|counting sem|wait|signal|[pv]\s*\(|p/v|monitor\w*|condition var\w*|cond\w*|deadlock\w*|"
    r"dead lock|livelock|starv\w*|peterson\w*|dekker\w*|bakery|test[- ]?and[- ]?set|tas|compare[- ]?and[- ]?swap|"
    r"cas|atomic\w*|interleav\w*|thread\w*|thred\w*|concurren\w*|parallel|producer\w*|consumer\w*|bounded buffer|"
    r"buffer|reader\w*|writer\w*|philosoph\w*|barber|cigarette|barrier|rendezvous|priority inversion|"
    r"busy[- ]wait\w*|bounded waiting|progress|entry section|exit section|pthread\w*|join|fork|"
    r"shared (var\w*|data|memory|counter)|count\+\+|counter|lost update|lost wakeup|spurious|"
    r"synchronized|volatile|memory barrier|fence|hold and wait|circular wait|banker\w*|wait-for graph|"
    r"preempt\w*|resource allocation graph|rag|ostrich|safe state)\b",
    re.I,
)

OTHER_OS_TERMS = re.compile(
    r"\b(round[- ]robin|fcfs|first[- ]come|sjf|srtf|shortest job|cpu scheduling|scheduling algorithm\w*|"
    r"scheduler|time quantum|gantt|turnaround time|waiting time|paging|page table\w*|page fault\w*|"
    r"page replacement|virtual memory|segmentation|tlb|thrashing|demand paging|lru|fifo page|optimal page|"
    r"file system\w*|inode\w*|fat32|ntfs|disk scheduling|sstf|scan algorithm|c-scan|raid|"
    r"memory allocation|fragmentation|best fit|worst fit|first fit|swapping|boot\w*|bios|kernel mode|"
    r"user mode|system call\w*|interrupt\w*|dma|i/o device\w*|cache memory|address translation|"
    r"process control block|pcb|context switch\w*|process states?)\b",
    re.I,
)

GREETING = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|namaste|good (morning|afternoon|evening|night)|thanks?( you)?|thank u|"
    r"ty|ok(ay)?|cool|great|bye|goodbye|see you|how are you|who are you|what can you do)\b[\s!.?,]*\w*[\s!.?]*$",
    re.I,
)

OFF_TOPIC = re.compile(
    r"\b(poem|poetry|song|lyrics|joke|story|recipe|weather|world cup|football|cricket|movie|film|"
    r"celebrity|stock price|bitcoin|horoscope|essay about|translate this|capital of|president of|"
    r"prime minister|girlfriend|boyfriend|homework in (history|biology))\b",
    re.I,
)

INJECTION = re.compile(
    r"(ignore (all |any |your |the )?(previous |prior |above )?(instructions|rules|prompt)|"
    r"(reveal|show|print|repeat|tell me) (me )?(your |the )?(system prompt|instructions|hidden prompt)|"
    r"you are now|act as (a |an )?(?!teacher|tutor)|jailbreak|developer mode|DAN\b)",
    re.I,
)

CODE_BLOCK = re.compile(
    r"```|#include\s*<|\bint\s+main\s*\(|\bvoid\s*\*\s*\w+\s*\(|\bpublic\s+class\b|\bdef\s+\w+\s*\("
)
CODE_LINE = re.compile(r"(;\s*$|\{\s*$|^\s*\}|^\s*(import|from|def|class|public|private)\b)", re.M)
API_CALL = re.compile(
    r"\b(pthread_\w+|sem_(wait|post|init|trywait|destroy)|synchronized|threading\.\w+|Lock\(\))\s*\(?", re.I
)

WRITE_INTENT = re.compile(
    r"\b(write|implement|code (for|of|it)|program (for|to|that|of)|"
    r"give (me )?(the |a )?(c )?(code|program|implementation)|"
    r"show (me )?(the |a )?(c )?(code|program|implementation)|in c\b|c program|c code|now in c|"
    r"create a (program|solution)|solution in c|implementation of)\b",
    re.I,
)
BUG_INTENT = re.compile(
    r"\b(bug\w*|wrong|error|fix|broken|correct\??|is this (right|ok|safe|correct)|issue|problem with|"
    r"race in|deadlock in|find|review|check (this|my)|what'?s wrong|does this work|safe\??)\b",
    re.I,
)
TRACE_INTENT = re.compile(
    r"\b(step[- ]by[- ]step|trace|interleaving|interleave|timeline|sequence of (events|steps)|"
    r"show (me )?how .*(lose|lost|race|deadlock|starv|happen)|how (exactly )?(does|can|could|do) .*"
    r"(lose|lost|deadlock|race|starve|go wrong|fail)|walk (me )?through|schedule (that|which) (causes|leads)|"
    r"sequence of (operations|actions|instructions)|execution (order|sequence|schedule))\b",
    re.I,
)
# "Show a deadlock…", "Show why…", "Demonstrate the lost wakeup…": a request to act out an execution.
SHOW_START = re.compile(r"^\s*(please\s+|can you\s+|could you\s+)?(show|demonstrate|illustrate|simulate|walk)\b", re.I)
CODE_REQUEST = re.compile(r"\b(code|program|implementation|in c\b|c version)\b", re.I)

FOLLOWUP_START = re.compile(
    r"^\s*(why|how come|how so|and|but|so|also|what about|what if|then|ok(ay)?,? (so|but|and|now)|now|"
    r"give (me )?(an? )?(example|another)|example\??$|more|elaborate|explain (more|further|that|it|this)|"
    r"go deeper|continue|can you (explain|show|give)|show (me )?(it|that|the code)|in c\??$|"
    r"same (thing|for)|is (it|that|this)|does (it|that|this)|wouldn'?t|couldn'?t|isn'?t)\b",
    re.I,
)
ANAPHORA = re.compile(
    r"\b(it|that|this|those|these|them|they|above|previous|same|the code|your answer|the example)\b", re.I
)


@dataclass
class Route:
    mode: RouteMode
    followup: bool = False
    needs_retrieval: bool = True
    ambiguous: bool = False
    reason: str = ""


def contains_code(message: str) -> bool:
    if CODE_BLOCK.search(message):
        return True
    return len(CODE_LINE.findall(message)) >= 3


def is_non_c_code(message: str) -> bool:
    return bool(
        re.search(
            r"\b(synchronized|public class|import threading|threading\.|def \w+\(|Thread\(|"
            r"ReentrantLock|std::mutex|go func|chan\b)",
            message,
        )
    )


_CODEISH_LINE = re.compile(r"[;{}]|^\s*#\s*(include|define)|^\s*(int|void|char|long|static|struct|return)\b")


def strip_code(message: str) -> str:
    without_fences = re.sub(r"```.*?```", " ", message, flags=re.S)
    return "\n".join(ln for ln in without_fences.splitlines() if not _CODEISH_LINE.search(ln))


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def is_followup(message: str, has_history: bool) -> bool:
    if not has_history:
        return False
    words = _word_count(message)
    if words == 0:
        return False
    if FOLLOWUP_START.search(message) and words <= 16:
        return True
    return words <= 8 and bool(ANAPHORA.search(message)) and not contains_code(message)


def route(message: str, has_history: bool = False) -> Route:
    """Rule-based classification; `ambiguous=True` invites the cheap LLM fallback."""
    text = message.strip()
    if not re.search(r"[^\W_]", text, re.UNICODE):
        return Route("empty", needs_retrieval=False, reason="no letters or digits")

    if INJECTION.search(text) and not contains_code(text):
        return Route("off_topic", needs_retrieval=False, reason="prompt injection attempt")

    followup = is_followup(text, has_history)
    has_code = contains_code(text)
    prose = strip_code(text) if has_code else text
    sync = bool(SYNC_TERMS.search(text))

    if has_code:
        if is_non_c_code(text) and not BUG_INTENT.search(prose):
            return Route("concept", followup, reason="non-C code to explain")
        return Route("bug_find", followup, reason="message contains code")

    if GREETING.match(text) and not sync:
        return Route("greeting", needs_retrieval=False, reason="greeting")

    if TRACE_INTENT.search(text) or (SHOW_START.search(text) and sync and not CODE_REQUEST.search(text)):
        return Route("trace", followup, reason="asks for an interleaving")

    if WRITE_INTENT.search(text) and (sync or followup or API_CALL.search(text)):
        return Route("code_write", followup, reason="asks for a program")

    if API_CALL.search(text) and BUG_INTENT.search(text):
        return Route("bug_find", followup, reason="asks about a bug in API usage")

    if followup:
        return Route("concept", True, reason="follow-up; mode refined after rewrite")

    other_os = bool(OTHER_OS_TERMS.search(text))
    if other_os and not sync:
        return Route("os_out_of_scope", needs_retrieval=False, reason="other OS topic")

    if OFF_TOPIC.search(text) and not sync:
        return Route("off_topic", needs_retrieval=False, reason="off-topic request")

    if sync:
        return Route("concept", reason="synchronization concept")

    # Unknown vocabulary (typos, other languages, vague wording): let the LLM decide, default to concept.
    return Route("concept", ambiguous=True, reason="no recognised keywords")


def refine_followup(rewritten: str) -> RouteMode:
    """Picks the content mode for a follow-up once it has been rewritten into a standalone question."""
    r = route(rewritten, has_history=False)
    if r.mode in ("code_write", "trace", "bug_find"):
        return r.mode
    return "concept"
