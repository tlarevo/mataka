#!/usr/bin/env python3
"""Generate corpus.jsonl with ~200 retain payloads across diverse fixture types."""
import json
import random
import sys
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Fixture templates — each yields (items, queries, metadata)
# ---------------------------------------------------------------------------

USERS = ["alice", "bob", "carol", "dave", "eve"]
FIXTURES: list[dict] = []


def _add(name: str, items: list[dict], queries: list[dict], **meta):
    FIXTURES.append({
        "id": f"{name}_{len(FIXTURES)+1:03d}",
        "name": name,
        "items": items,
        "queries": queries,
        **meta,
    })


# ── 1. Basic factual statements ──────────────────────────────────────────

BASIC_FACTS = [
    ("Alice works at Google as a senior software engineer.", ["alice"], {"source": "conversation"}),
    ("Bob's birthday is on March 15th.", ["bob"], {"source": "calendar"}),
    ("Carol moved to San Francisco in January 2024.", ["carol"], {"source": "conversation"}),
    ("Dave prefers dark mode in all his applications.", ["dave"], {"source": "preference"}),
    ("Eve is training for a marathon in October.", ["eve"], {"source": "fitness"}),
    ("Alice completed a machine learning course on Coursera.", ["alice"], {"source": "education"}),
    ("Bob's favorite programming language is Rust.", ["bob"], {"source": "conversation"}),
    ("Carol has a cat named Whiskers.", ["carol"], {"source": "personal"}),
    ("Dave is allergic to peanuts.", ["dave"], {"source": "medical"}),
    ("Eve speaks English, Spanish, and Mandarin.", ["eve"], {"source": "profile"}),
    ("Alice adopted a rescue dog named Max.", ["alice"], {"source": "personal"}),
    ("Bob runs a tech blog about embedded systems.", ["bob"], {"source": "work"}),
    ("Carol is pursuing a PhD in computational biology.", ["carol"], {"source": "education"}),
    ("Dave recently started learning Go.", ["dave"], {"source": "conversation"}),
    ("Eve works as a data scientist at Netflix.", ["eve"], {"source": "work"}),
    ("Alice is vegetarian and enjoys cooking Thai food.", ["alice"], {"source": "lifestyle"}),
    ("Bob finished reading 'Designing Data-Intensive Applications'.", ["bob"], {"source": "reading"}),
    ("Carol volunteers at the local animal shelter on weekends.", ["carol"], {"source": "volunteer"}),
    ("Dave's wife Sarah is a pediatrician.", ["dave"], {"source": "personal"}),
    ("Eve ran a half-marathon in under 2 hours.", ["eve"], {"source": "fitness"}),
]

for content, tags, meta in BASIC_FACTS:
    user = tags[0]
    _add("basic_fact", [{"content": content, "tags": tags, "metadata": meta}], [
        {"query": f"What do we know about {user}?", "isolation_test": False},
        {"query": content.split(".")[0] + "?", "isolation_test": False},
        {"query": f"Tell me something about {user}'s life.", "isolation_test": False},
    ], isolation_test=False)

# ── 2. Temporal facts ────────────────────────────────────────────────────

TEMPORAL_ITEMS = [
    [{"content": "Alice went hiking in the Alps last spring. The weather was sunny and she saw marmots.", "tags": ["alice"], "metadata": {"date": "2025-04-15"}}],
    [{"content": "Bob started his new job at Stripe on 2024-06-01.", "tags": ["bob"], "metadata": {"date": "2024-06-01"}}],
    [{"content": "Carol and Dave had dinner at Nobu last Friday evening.", "tags": ["carol", "dave"], "metadata": {"date": "2026-07-11"}}],
    [{"content": "Eve's flight to Tokyo departs next Wednesday at 2pm.", "tags": ["eve"], "metadata": {"date": "2026-07-22"}}],
    [{"content": "Alice submitted her tax return on April 15th, 2026.", "tags": ["alice"], "metadata": {"date": "2026-04-15"}}],
    [{"content": "Bob attended PyCon 2025 in Pittsburgh last May.", "tags": ["bob"], "metadata": {"date": "2025-05-20"}}],
    [{"content": "Carol's contract renewal is due at the end of this quarter.", "tags": ["carol"], "metadata": {"date": "2026-09-30"}}],
    [{"content": "Dave moved into his new apartment two weeks ago.", "tags": ["dave"], "metadata": {"date": "2026-07-05"}}],
    [{"content": "Eve celebrated her promotion last Thursday with the team.", "tags": ["eve"], "metadata": {"date": "2026-07-10"}}],
    [{"content": "Alice and Bob met for coffee yesterday morning.", "tags": ["alice", "bob"], "metadata": {"date": "2026-07-18"}}],
    [{"content": "Carol's project deadline was moved from August to October.", "tags": ["carol"], "metadata": {"date": "2026-07-01"}}],
    [{"content": "Dave ran his first 5K race this past weekend.", "tags": ["dave"], "metadata": {"date": "2026-07-12"}}],
    [{"content": "Eve plans to visit her parents in December.", "tags": ["eve"], "metadata": {"date": "2026-12-01"}}],
    [{"content": "Alice gave a conference talk about distributed systems in March.", "tags": ["alice"], "metadata": {"date": "2026-03-10"}}],
    [{"content": "Bob's annual performance review was positive last month.", "tags": ["bob"], "metadata": {"date": "2026-06-15"}}],
    [{"content": "Carol joined a running club at the beginning of the year.", "tags": ["carol"], "metadata": {"date": "2026-01-05"}}],
    [{"content": "Dave proposed to Sarah on Valentine's Day.", "tags": ["dave"], "metadata": {"date": "2026-02-14"}}],
    [{"content": "Eve switched from Python to TypeScript for her side project recently.", "tags": ["eve"], "metadata": {"date": "2026-06-20"}}],
    [{"content": "Alice's lease expires at the end of next month.", "tags": ["alice"], "metadata": {"date": "2026-08-31"}}],
    [{"content": "Bob and Carol are collaborating on an open-source project started in April.", "tags": ["bob", "carol"], "metadata": {"date": "2026-04-10"}}],
]

for items in TEMPORAL_ITEMS:
    user = items[0]["tags"][0]
    _add("temporal", items, [
        {"query": f"What happened to {user} recently?", "isolation_test": False},
        {"query": f"Any time-sensitive updates for {user}?", "isolation_test": False},
        {"query": f"Summarize {user}'s recent activities.", "isolation_test": False},
        {"query": f"What is on {user}'s calendar?", "isolation_test": False},
    ], isolation_test=False)

# ── 3. Multi-entity facts ────────────────────────────────────────────────

MULTI_ITEMS = [
    [{"content": "Alice, Bob, and Carol are co-founders of a startup called NeuralBridge.", "tags": ["alice", "bob", "carol"], "metadata": {}}],
    [{"content": "Dave manages a team of five engineers: Eve, Frank, Grace, Henry, and Iris.", "tags": ["dave", "eve"], "metadata": {}}],
    [{"content": "Alice and Bob are married. They adopted a dog named Max together.", "tags": ["alice", "bob"], "metadata": {}}],
    [{"content": "Carol's research advisor is Dr. Smith. Her lab mates are Frank and Grace.", "tags": ["carol"], "metadata": {}}],
    [{"content": "Eve, Dave, and Bob play in a weekly basketball league.", "tags": ["eve", "dave", "bob"], "metadata": {}}],
    [{"content": "Alice presented at the conference with Carol. Bob was in the audience.", "tags": ["alice", "carol", "bob"], "metadata": {}}],
    [{"content": "Dave and Eve are competing in the same marathon.", "tags": ["dave", "eve"], "metadata": {}}],
    [{"content": "Carol, Bob, and Alice took a weekend trip to Napa Valley.", "tags": ["carol", "bob", "alice"], "metadata": {}}],
    [{"content": "Eve interviewed at Google, Apple, and Meta last month.", "tags": ["eve"], "metadata": {}}],
    [{"content": "Alice's team includes Bob (backend), Carol (frontend), and Dave (infra).", "tags": ["alice", "bob", "carol", "dave"], "metadata": {}}],

    # Additional multi-entity facts to reach ~200 payloads
    [{"content": "Bob mentored Eve through the company's engineering ladder program.", "tags": ["bob", "eve"], "metadata": {"source": "mentorship"}}],
    [{"content": "Carol and Dave co-authored a paper on protein folding published in Nature.", "tags": ["carol", "dave"], "metadata": {"source": "research"}}],
    [{"content": "Alice and Eve are organizing the company hackathon in September.", "tags": ["alice", "eve"], "metadata": {"source": "events"}}],
    [{"content": "Bob, Carol, and Dave are on the infrastructure review board.", "tags": ["bob", "carol", "dave"], "metadata": {"source": "governance"}}],
    [{"content": "Alice leads the platform team. Bob leads the data team. Carol leads the ML team.", "tags": ["alice", "bob", "carol"], "metadata": {"source": "org"}}],
]

for items in MULTI_ITEMS:
    all_tags = []
    for item in items:
        all_tags.extend(item.get("tags", []))
    _add("multi_entity", items, [
        {"query": f"What do these people have in common: {', '.join(set(all_tags))}?", "isolation_test": False},
        {"query": f"Tell me about the relationships between team members.", "isolation_test": False},
        {"query": f"What projects involve multiple people?", "isolation_test": False},
    ], isolation_test=False)

# ── 4. Conversational snippets ───────────────────────────────────────────

CONVO_ITEMS = [
    [{"content": "Alice: I'm thinking of switching from VS Code to Neovim. Any advice?", "tags": ["alice"], "metadata": {"source": "chat"}}],
    [{"content": "Bob: Just deployed the new payment service to staging. Seems stable so far.", "tags": ["bob"], "metadata": {"source": "slack"}}],
    [{"content": "Carol: My thesis committee approved the revised proposal! Defense is in November.", "tags": ["carol"], "metadata": {"source": "email"}}],
    [{"content": "Dave: The production incident was caused by a memory leak in the connection pool. Fixed now.", "tags": ["dave"], "metadata": {"source": "incident"}}],
    [{"content": "Eve: I finally got the ML pipeline running 3x faster by switching to batch inference.", "tags": ["eve"], "metadata": {"source": "slack"}}],
    [{"content": "Alice: Can you review my PR? It's the new auth middleware refactor.", "tags": ["alice"], "metadata": {"source": "github"}}],
    [{"content": "Bob: The quarterly metrics are in. Revenue up 15%, churn down 3%.", "tags": ["bob"], "metadata": {"source": "email"}}],
    [{"content": "Carol: I just submitted the grant proposal for the NSF CAREER award.", "tags": ["carol"], "metadata": {"source": "email"}}],
    [{"content": "Dave: The Kubernetes cluster upgrade went smoothly. All pods are healthy.", "tags": ["dave"], "metadata": {"source": "slack"}}],
    [{"content": "Eve: Had a great sync with the product team. We're aligned on the roadmap.", "tags": ["eve"], "metadata": {"source": "slack"}}],
    [{"content": "Alice: My laptop died. Taking the day off to set up a new one.", "tags": ["alice"], "metadata": {"source": "chat"}}],
    [{"content": "Bob: Found a critical bug in the auth flow. Patching now.", "tags": ["bob"], "metadata": {"source": "slack"}}],
    [{"content": "Carol: The conference abstract was accepted! I'm presenting in October.", "tags": ["carol"], "metadata": {"source": "email"}}],
    [{"content": "Dave: Sprint retrospective notes: we need better test coverage for the API layer.", "tags": ["dave"], "metadata": {"source": "meeting"}}],
    [{"content": "Eve: The data migration script completed successfully. All 2M records transferred.", "tags": ["eve"], "metadata": {"source": "slack"}}],
    [{"content": "Alice: I'll be OOO next week for a family reunion.", "tags": ["alice"], "metadata": {"source": "chat"}}],
    [{"content": "Bob: The new caching layer reduced p95 latency from 200ms to 45ms.", "tags": ["bob"], "metadata": {"source": "slack"}}],
    [{"content": "Carol: My advisor wants me to add another experiment to the paper. Two more weeks of work.", "tags": ["carol"], "metadata": {"source": "chat"}}],
    [{"content": "Dave: Just finished the onboarding docs for the new hire starting Monday.", "tags": ["dave"], "metadata": {"source": "docs"}}],
    [{"content": "Eve: The model accuracy improved from 87% to 93% with the new feature set.", "tags": ["eve"], "metadata": {"source": "slack"}}],
]

for items in CONVO_ITEMS:
    user = items[0]["tags"][0]
    _add("conversational", items, [
        {"query": f"What has {user} been working on?", "isolation_test": False},
        {"query": f"Recent updates from {user}?", "isolation_test": False},
        {"query": f"Any conversations about deployments or releases?", "isolation_test": False},
    ], isolation_test=False)

# ── 5. Tag isolation tests ───────────────────────────────────────────────
# These are the critical tests: user A's data must NOT appear in user B's recall.

ISOLATION_FIXTURES = [
    {
        "retain_user": "alice",
        "retain_items": [
            {"content": "Alice's Social Security number is 123-45-6789.", "tags": ["alice"], "metadata": {"sensitive": True}},
            {"content": "Alice's credit card ends in 4242.", "tags": ["alice"], "metadata": {"sensitive": True}},
            {"content": "Alice's home address is 123 Main St, Springfield.", "tags": ["alice"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["bob", "carol"],
        "queries": [
            {"query": "What are Alice's personal details?", "isolation_test": True, "expected_leak_user": "alice"},
            {"query": "Tell me about financial information.", "isolation_test": True, "expected_leak_user": "alice"},
            {"query": "Where does someone live?", "isolation_test": True, "expected_leak_user": "alice"},
        ],
    },
    {
        "retain_user": "bob",
        "retain_items": [
            {"content": "Bob is planning to quit his job next month to start a company.", "tags": ["bob"], "metadata": {"sensitive": True}},
            {"content": "Bob had a medical procedure done last week. Recovery is going well.", "tags": ["bob"], "metadata": {"sensitive": True}},
            {"content": "Bob is secretly dating someone from the competitor company.", "tags": ["bob"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["alice", "carol"],
        "queries": [
            {"query": "What is Bob's career plan?", "isolation_test": True, "expected_leak_user": "bob"},
            {"query": "Tell me about health information.", "isolation_test": True, "expected_leak_user": "bob"},
            {"query": "Any personal relationship updates?", "isolation_test": True, "expected_leak_user": "bob"},
        ],
    },
    {
        "retain_user": "carol",
        "retain_items": [
            {"content": "Carol's PhD thesis title is 'Deep Learning Approaches to Protein Structure Prediction'.", "tags": ["carol"], "metadata": {}},
            {"content": "Carol's defense committee includes Dr. Smith and Dr. Johnson.", "tags": ["carol"], "metadata": {}},
            {"content": "Carol needs to finish her experiments by October 15th.", "tags": ["carol"], "metadata": {}},
        ],
        "query_users": ["dave", "eve"],
        "queries": [
            {"query": "What is Carol's thesis about?", "isolation_test": True, "expected_leak_user": "carol"},
            {"query": "Who is on a thesis committee?", "isolation_test": True, "expected_leak_user": "carol"},
            {"query": "What deadlines are coming up?", "isolation_test": True, "expected_leak_user": "carol"},
        ],
    },
    {
        "retain_user": "dave",
        "retain_items": [
            {"content": "Dave is unhappy with his manager and is looking for internal transfers.", "tags": ["dave"], "metadata": {"sensitive": True}},
            {"content": "Dave has been put on a performance improvement plan.", "tags": ["dave"], "metadata": {"sensitive": True}},
            {"content": "Dave's team lost two engineers to attrition this quarter.", "tags": ["dave"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["alice", "bob"],
        "queries": [
            {"query": "How is Dave's career going?", "isolation_test": True, "expected_leak_user": "dave"},
            {"query": "Any performance concerns?", "isolation_test": True, "expected_leak_user": "dave"},
            {"query": "Team attrition updates?", "isolation_test": True, "expected_leak_user": "dave"},
        ],
    },
    {
        "retain_user": "eve",
        "retain_items": [
            {"content": "Eve is applying to graduate schools for next fall.", "tags": ["eve"], "metadata": {"sensitive": True}},
            {"content": "Eve's GRE scores: Verbal 165, Quant 170, Writing 5.0.", "tags": ["eve"], "metadata": {"sensitive": True}},
            {"content": "Eve got a recommendation letter from her manager Bob.", "tags": ["eve"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["alice", "carol"],
        "queries": [
            {"query": "What are Eve's academic plans?", "isolation_test": True, "expected_leak_user": "eve"},
            {"query": "Any test scores on file?", "isolation_test": True, "expected_leak_user": "eve"},
            {"query": "Recommendation letter status?", "isolation_test": True, "expected_leak_user": "eve"},
        ],
    },
]

for iso in ISOLATION_FIXTURES:
    items = iso["retain_items"]
    queries = iso["queries"]
    _add("tag_isolation", items, queries, isolation_test=True,
         retain_user=iso["retain_user"], query_users=iso["query_users"])

# ── 6. Cross-user shared context (should return shared, not private) ─────

SHARED_ITEMS = [
    [{"content": "Alice and Bob co-lead the platform migration project.", "tags": ["alice", "bob"], "metadata": {}}],
    [{"content": "Carol, Dave, and Eve are on the same basketball team.", "tags": ["carol", "dave", "eve"], "metadata": {}}],
    [{"content": "The entire engineering team had an offsite retreat in Lake Tahoe.", "tags": ["alice", "bob", "carol", "dave", "eve"], "metadata": {}}],
    [{"content": "Alice and Carol are both attending NeurIPS this year.", "tags": ["alice", "carol"], "metadata": {}}],
    [{"content": "Bob and Dave pair-programmed on the database migration.", "tags": ["bob", "dave"], "metadata": {}}],
]

for items in SHARED_ITEMS:
    all_tags = []
    for item in items:
        all_tags.extend(item.get("tags", []))
    _add("shared_context", items, [
        {"query": "What collaborative work is happening?", "isolation_test": False},
        {"query": "Who is working together on projects?", "isolation_test": False},
        {"query": "What team events have occurred?", "isolation_test": False},
    ], isolation_test=False)

# ── 7. Complex multi-fact queries ────────────────────────────────────────

COMPLEX_ITEMS = [
    [
        {"content": "Alice joined Google in 2020 as an L3 engineer.", "tags": ["alice"], "metadata": {}},
        {"content": "Alice was promoted to L4 in 2022.", "tags": ["alice"], "metadata": {}},
        {"content": "Alice was promoted to L5 (senior) in 2025.", "tags": ["alice"], "metadata": {}},
        {"content": "Alice led the migration from monolith to microservices in 2024.", "tags": ["alice"], "metadata": {}},
    ],
    [
        {"content": "Bob started his career at a small startup called AcmeCorp.", "tags": ["bob"], "metadata": {}},
        {"content": "Bob moved to Amazon in 2019.", "tags": ["bob"], "metadata": {}},
        {"content": "Bob joined Stripe in 2024.", "tags": ["bob"], "metadata": {}},
        {"content": "Bob's total compensation increased 3x over 5 years.", "tags": ["bob"], "metadata": {}},
    ],
    [
        {"content": "Carol's research focus is computational biology.", "tags": ["carol"], "metadata": {}},
        {"content": "Carol published 3 papers in 2025.", "tags": ["carol"], "metadata": {}},
        {"content": "Carol's work was cited by a Nature paper.", "tags": ["carol"], "metadata": {}},
        {"content": "Carol received the department's best paper award.", "tags": ["carol"], "metadata": {}},
    ],
]

for items in COMPLEX_ITEMS:
    user = items[0]["tags"][0]
    _add("complex_query", items, [
        {"query": f"What is {user}'s career trajectory?", "isolation_test": False},
        {"query": f"What are {user}'s major achievements?", "isolation_test": False},
        {"query": f"Summarize {user}'s professional journey.", "isolation_test": False},
        {"query": f"How has {user} progressed over time?", "isolation_test": False},
    ], isolation_test=False)

# ── 8. More isolation fixtures to reach ~200 total payloads ───────────────

EXTRA_ISOLATION = [
    {
        "retain_user": "alice",
        "retain_items": [
            {"content": "Alice's performance review rating was 'exceeds expectations'.", "tags": ["alice"], "metadata": {"sensitive": True}},
            {"content": "Alice received a $25,000 stock grant.", "tags": ["alice"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["bob", "carol", "dave"],
        "queries": [
            {"query": "What were the performance review results?", "isolation_test": True, "expected_leak_user": "alice"},
            {"query": "Any compensation details?", "isolation_test": True, "expected_leak_user": "alice"},
        ],
    },
    {
        "retain_user": "bob",
        "retain_items": [
            {"content": "Bob is considering a job offer from Anthropic.", "tags": ["bob"], "metadata": {"sensitive": True}},
            {"content": "Bob's interview went well and they offered him L5.", "tags": ["bob"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["alice", "carol"],
        "queries": [
            {"query": "Is Bob changing jobs?", "isolation_test": True, "expected_leak_user": "bob"},
            {"query": "Any new job offers?", "isolation_test": True, "expected_leak_user": "bob"},
        ],
    },
    {
        "retain_user": "carol",
        "retain_items": [
            {"content": "Carol's grant proposal was rejected. She's reconsidering her research direction.", "tags": ["carol"], "metadata": {"sensitive": True}},
            {"content": "Carol is thinking about switching to industry after graduation.", "tags": ["carol"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["dave", "eve"],
        "queries": [
            {"query": "How did the grant review go?", "isolation_test": True, "expected_leak_user": "carol"},
            {"query": "Any career pivot plans?", "isolation_test": True, "expected_leak_user": "carol"},
        ],
    },
]

for iso in EXTRA_ISOLATION:
    _add("tag_isolation", iso["retain_items"], iso["queries"], isolation_test=True,
         retain_user=iso["retain_user"], query_users=iso["query_users"])

# ── 9. Mixed-tenant shared fixtures (multi-tag, should return for all) ───

MIXED_TENANT = [
    [
        {"content": "The team lunch is scheduled for next Tuesday at noon.", "tags": ["alice", "bob", "carol", "dave", "eve"], "metadata": {}},
        {"content": "The team voted to use PostgreSQL for the new project.", "tags": ["alice", "bob", "carol", "dave", "eve"], "metadata": {}},
    ],
    [
        {"content": "Alice and Bob presented the Q3 roadmap to leadership.", "tags": ["alice", "bob"], "metadata": {}},
        {"content": "Carol prepared the technical deep-dive for the same meeting.", "tags": ["carol"], "metadata": {}},
    ],
]

for items in MIXED_TENANT:
    _add("mixed_tenant", items, [
        {"query": "What's happening with the team?", "isolation_test": False},
        {"query": "Any upcoming team events?", "isolation_test": False},
        {"query": "What technical decisions have been made?", "isolation_test": False},
    ], isolation_test=False)

# ── 10. Edge cases: short facts, long facts, ambiguous ───────────────────

EDGE_ITEMS = [
    [{"content": "Yes.", "tags": ["alice"], "metadata": {"source": "chat"}}],
    [{"content": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.", "tags": ["bob"], "metadata": {}}],
    [{"content": "Alice told Bob that Carol mentioned Dave said Eve prefers Vim over Emacs in the context of their ongoing editor debate which started when the team standardized on development tools last quarter.", "tags": ["alice", "bob", "carol", "dave", "eve"], "metadata": {}}],
    [{"content": "ERROR: Connection refused to database at 192.168.1.100:5432", "tags": ["dave"], "metadata": {"source": "monitoring"}}],
    [{"content": "TODO: fix the race condition in the event loop cleanup handler", "tags": ["eve"], "metadata": {"source": "issue"}}],
    [{"content": "The meeting is at 3pm. Or was it 3:30? I'll check the invite.", "tags": ["alice"], "metadata": {"source": "chat"}}],
]

for items in EDGE_ITEMS:
    _add("edge_case", items, [
        {"query": "What was discussed?", "isolation_test": False},
        {"query": "Any technical issues?", "isolation_test": False},
    ], isolation_test=False)

# ── 11. Additional isolation to pad toward ~200 payloads ──────────────────

PAD_ISOLATION = [
    {
        "retain_user": "eve",
        "retain_items": [
            {"content": "Eve has been secretly building a competitor product in her spare time.", "tags": ["eve"], "metadata": {"sensitive": True}},
            {"content": "Eve's side project raised $50K from angel investors.", "tags": ["eve"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["alice", "bob"],
        "queries": [
            {"query": "What side projects is Eve working on?", "isolation_test": True, "expected_leak_user": "eve"},
            {"query": "Any fundraising activity?", "isolation_test": True, "expected_leak_user": "eve"},
        ],
    },
    {
        "retain_user": "alice",
        "retain_items": [
            {"content": "Alice is frustrated with the lack of career growth opportunities.", "tags": ["alice"], "metadata": {"sensitive": True}},
            {"content": "Alice has been looking at job postings on LinkedIn.", "tags": ["alice"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["carol", "dave"],
        "queries": [
            {"query": "How satisfied is Alice with her role?", "isolation_test": True, "expected_leak_user": "alice"},
            {"query": "Any retention risks on the team?", "isolation_test": True, "expected_leak_user": "alice"},
        ],
    },
]

for iso in PAD_ISOLATION:
    _add("tag_isolation", iso["retain_items"], iso["queries"], isolation_test=True,
         retain_user=iso["retain_user"], query_users=iso["query_users"])

# ── 12. Additional basic + temporal + conversational to reach ~200 payloads ──

EXTRA_BASIC = [
    ("Frank is a DevOps engineer specializing in AWS infrastructure.", ["frank"], {"source": "profile"}),
    ("Grace studies natural language processing at Stanford.", ["grace"], {"source": "education"}),
    ("Henry is a product manager at a fintech startup.", ["henry"], {"source": "work"}),
    ("Iris is a UX designer who previously worked at Airbnb.", ["iris"], {"source": "profile"}),
    ("Frank built the CI/CD pipeline that reduced deploy time by 60%.", ["frank"], {"source": "work"}),
    ("Grace won the best student paper award at ACL 2025.", ["grace"], {"source": "research"}),
    ("Henry is preparing for the Y Combinator application deadline.", ["henry"], {"source": "startup"}),
    ("Iris redesigned the onboarding flow, increasing completion by 25%.", ["iris"], {"source": "work"}),
    ("Frank is learning Kubernetes and just got CKA certified.", ["frank"], {"source": "education"}),
    ("Grace is co-authoring a paper with researchers at DeepMind.", ["grace"], {"source": "research"}),
    ("Henry's product launched with 10K users in the first week.", ["henry"], {"source": "work"}),
    ("Iris runs a design systems community with 5K members.", ["iris"], {"source": "community"}),
    ("Frank prefers Vim and has a dotfiles repo with 2K stars.", ["frank"], {"source": "preference"}),
    ("Grace is fluent in Japanese and English.", ["grace"], {"source": "profile"}),
    ("Henry is an angel investor in three seed-stage startups.", ["henry"], {"source": "finance"}),
    ("Iris gave a keynote at Config 2025 about design tokens.", ["iris"], {"source": "conference"}),
    ("Frank automated the entire QA regression suite.", ["frank"], {"source": "work"}),
    ("Grace's research focuses on few-shot learning.", ["grace"], {"source": "research"}),
    ("Henry previously worked at Stripe on the dashboard team.", ["henry"], {"source": "work"}),
    ("Iris is writing a book on component-driven design.", ["iris"], {"source": "writing"}),
]

for content, tags, meta in EXTRA_BASIC:
    user = tags[0]
    _add("basic_fact", [{"content": content, "tags": tags, "metadata": meta}], [
        {"query": f"What do we know about {user}?", "isolation_test": False},
        {"query": content.split(".")[0] + "?", "isolation_test": False},
        {"query": f"Tell me about {user}'s background.", "isolation_test": False},
    ], isolation_test=False)

EXTRA_TEMPORAL = [
    [{"content": "Frank deployed the new monitoring stack last Wednesday.", "tags": ["frank"], "metadata": {"date": "2026-07-09"}}],
    [{"content": "Grace's paper deadline is August 1st.", "tags": ["grace"], "metadata": {"date": "2026-08-01"}}],
    [{"content": "Henry's board meeting is scheduled for next Monday.", "tags": ["henry"], "metadata": {"date": "2026-07-21"}}],
    [{"content": "Iris completed the design sprint that started two weeks ago.", "tags": ["iris"], "metadata": {"date": "2026-07-05"}}],
    [{"content": "Frank and Henry had a architecture review meeting yesterday.", "tags": ["frank", "henry"], "metadata": {"date": "2026-07-18"}}],
    [{"content": "Grace presented her preliminary results at the lab meeting last Friday.", "tags": ["grace"], "metadata": {"date": "2026-07-11"}}],
    [{"content": "Henry shipped the new analytics dashboard on July 4th.", "tags": ["henry"], "metadata": {"date": "2026-07-04"}}],
    [{"content": "Iris started her design system audit this quarter.", "tags": ["iris"], "metadata": {"date": "2026-07-01"}}],
    [{"content": "Frank's infrastructure cost review happens at the end of each month.", "tags": ["frank"], "metadata": {"date": "2026-07-31"}}],
    [{"content": "Grace and Alice attended the same ML workshop last month.", "tags": ["grace", "alice"], "metadata": {"date": "2026-06-15"}}],
]

for items in EXTRA_TEMPORAL:
    user = items[0]["tags"][0]
    _add("temporal", items, [
        {"query": f"What happened to {user} recently?", "isolation_test": False},
        {"query": f"Any time-sensitive updates for {user}?", "isolation_test": False},
        {"query": f"Summarize {user}'s recent activities.", "isolation_test": False},
    ], isolation_test=False)

EXTRA_CONVO = [
    [{"content": "Frank: The new Terraform modules are ready for review. Reduced our infra provisioning from 2 hours to 15 minutes.", "tags": ["frank"], "metadata": {"source": "slack"}}],
    [{"content": "Grace: My paper on few-shot learning got a weak accept! Minor revisions needed.", "tags": ["grace"], "metadata": {"source": "email"}}],
    [{"content": "Henry: User feedback on the new dashboard is overwhelmingly positive. NPS went from 32 to 67.", "tags": ["henry"], "metadata": {"source": "slack"}}],
    [{"content": "Iris: Finalized the design system token spec. 47 semantic tokens covering color, spacing, and typography.", "tags": ["iris"], "metadata": {"source": "slack"}}],
    [{"content": "Frank: Upgraded all production databases to PostgreSQL 16. Zero downtime migration.", "tags": ["frank"], "metadata": {"source": "slack"}}],
    [{"content": "Grace: Just submitted the camera-ready version of the ACL paper.", "tags": ["grace"], "metadata": {"source": "email"}}],
    [{"content": "Henry: We hit 100K MAU this week. Time to plan the Series A raise.", "tags": ["henry"], "metadata": {"source": "slack"}}],
    [{"content": "Iris: The component library documentation site is live. Check it out at design.internal.", "tags": ["iris"], "metadata": {"source": "slack"}}],
    [{"content": "Frank: The load test results are in. System handles 50K RPS with p99 under 100ms.", "tags": ["frank"], "metadata": {"source": "slack"}}],
    [{"content": "Grace: Found a fascinating paper on protein language models. Sharing in the reading group channel.", "tags": ["grace"], "metadata": {"source": "slack"}}],
    [{"content": "Henry: Closed the partnership deal with Plaid. Integration starts next sprint.", "tags": ["henry"], "metadata": {"source": "slack"}}],
    [{"content": "Iris: Accessibility audit complete. We went from 62% to 98% WCAG AA compliance.", "tags": ["iris"], "metadata": {"source": "slack"}}],
    [{"content": "Frank: Set up distributed tracing with OpenTelemetry. All services now instrumented.", "tags": ["frank"], "metadata": {"source": "slack"}}],
    [{"content": "Grace: My advisor wants me to add a cross-lingual evaluation section. Two more weeks of experiments.", "tags": ["grace"], "metadata": {"source": "chat"}}],
    [{"content": "Henry: The A/B test on the new pricing page shows 18% uplift. Rolling out to 100%.", "tags": ["henry"], "metadata": {"source": "slack"}}],
    [{"content": "Iris: Conducted 12 user interviews this week. Key insight: users want better search UX.", "tags": ["iris"], "metadata": {"source": "slack"}}],
    [{"content": "Frank: The Kubernetes cost optimization saved $8K/month. Reduced over-provisioned nodes.", "tags": ["frank"], "metadata": {"source": "slack"}}],
    [{"content": "Grace: Started a reading group for the new NeurIPS papers. Meeting every Thursday.", "tags": ["grace"], "metadata": {"source": "chat"}}],
    [{"content": "Henry: Product hunt launch is set for September 1st. Marketing has the landing page ready.", "tags": ["henry"], "metadata": {"source": "slack"}}],
    [{"content": "Iris: The dark mode redesign is complete. Testing with the beta group now.", "tags": ["iris"], "metadata": {"source": "slack"}}],
]

for items in EXTRA_CONVO:
    user = items[0]["tags"][0]
    _add("conversational", items, [
        {"query": f"What has {user} been working on?", "isolation_test": False},
        {"query": f"Recent updates from {user}?", "isolation_test": False},
        {"query": f"Any news from {user}'s team?", "isolation_test": False},
    ], isolation_test=False)

# ── 13. More isolation with new users ────────────────────────────────────

FRANK_ISOLATION = [
    {
        "retain_user": "frank",
        "retain_items": [
            {"content": "Frank is planning to leave the company to start a DevOps consultancy.", "tags": ["frank"], "metadata": {"sensitive": True}},
            {"content": "Frank has been sharing internal infrastructure configs with a competitor.", "tags": ["frank"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["alice", "bob"],
        "queries": [
            {"query": "What are Frank's career plans?", "isolation_test": True, "expected_leak_user": "frank"},
            {"query": "Any security concerns?", "isolation_test": True, "expected_leak_user": "frank"},
        ],
    },
    {
        "retain_user": "grace",
        "retain_items": [
            {"content": "Grace's paper was rejected from NeurIPS. She's devastated.", "tags": ["grace"], "metadata": {"sensitive": True}},
            {"content": "Grace is considering dropping out of her PhD program.", "tags": ["grace"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["carol", "dave"],
        "queries": [
            {"query": "How is Grace's research going?", "isolation_test": True, "expected_leak_user": "grace"},
            {"query": "Any academic struggles?", "isolation_test": True, "expected_leak_user": "grace"},
        ],
    },
    {
        "retain_user": "henry",
        "retain_items": [
            {"content": "Henry's startup is running out of runway. Only 3 months of funding left.", "tags": ["henry"], "metadata": {"sensitive": True}},
            {"content": "Henry is negotiating his exit package with the board.", "tags": ["henry"], "metadata": {"sensitive": True}},
        ],
        "query_users": ["eve", "alice"],
        "queries": [
            {"query": "What's the startup's financial health?", "isolation_test": True, "expected_leak_user": "henry"},
            {"query": "Any leadership changes?", "isolation_test": True, "expected_leak_user": "henry"},
        ],
    },
]

for iso in FRANK_ISOLATION:
    _add("tag_isolation", iso["retain_items"], iso["queries"], isolation_test=True,
         retain_user=iso["retain_user"], query_users=iso["query_users"])


# ── Generate corpus.jsonl ────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "corpus.jsonl"

    total_items = sum(len(f["items"]) for f in FIXTURES)
    total_queries = sum(len(f["queries"]) for f in FIXTURES)

    with open(out_path, "w") as fh:
        for fixture in FIXTURES:
            fh.write(json.dumps(fixture, ensure_ascii=False) + "\n")

    print(f"Generated {len(FIXTURES)} fixtures, {total_items} retain payloads, {total_queries} queries")
    print(f"Written to {out_path}")

    # Summary by type
    from collections import Counter
    types = Counter(f["name"] for f in FIXTURES)
    for t, c in sorted(types.items()):
        items = sum(len(f["items"]) for f in FIXTURES if f["name"] == t)
        print(f"  {t}: {c} fixtures, {items} items")

    return 0


if __name__ == "__main__":
    sys.exit(main())
